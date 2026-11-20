#!/usr/bin/env python3
"""
Literature Atlas builder
========================

Rebuilds an interactive literature atlas (topic map, citation network, co-author network, table and
trends) for one project: every paper on a simulation suite's publications page. Each project lives in
projects/<name>/ (config.json, tags.json, overrides.json, positions.json) and its site in <name>/index.html.
config.json holds what differs between projects: name, publications page, site address, analytics ID,
topic names, the hand-checked topic of every paper (topic_seed) and the allowed tag names.

Pipeline
  1. Publications list   pubs_url in the project's config.json  (every entry)
  2. arXiv API           titles, abstracts, authors, dates for entries with an arXiv link
  3. NASA ADS            full records for entries that link to ADS instead of arXiv, and total
                         citation counts (token read from ~/.ads/dev_key or $ADS_TOKEN; without a
                         token, citation counts come from Semantic Scholar)
  4. Semantic Scholar    SPECTER2 embeddings + references (citation links between the listed papers;
                         an optional free API key in $S2_API_KEY avoids rate limits),
                         matched by arXiv ID, DOI or exact title
  Entries with no arXiv or ADS record (e.g. theses) use the title and authors from the
  publications page; if no date is known, the year is estimated from the entry's position
  in the list (which is newest first) and marked as estimated on the page.
  5. Analysis            k-means topics (10, hand-named), 3D t-SNE, author-name merging
  6. Output              one self-contained index.html built from atlas.html (structure), atlas.css
                         (styles) and atlas.js (behaviour), with the data inserted at __DATA__.
                         Only Google Fonts and Google Analytics are loaded from outside.

Usage
  python3 build_atlas.py --project camels                # fetch everything fresh, write camels/index.html
  python3 build_atlas.py --project camels --offline      # reuse the cached downloads in atlas_cache/camels
  python3 build_atlas.py --project camels --recluster    # redo the topic clustering (then check/rename topics)

Layout positions are stored in projects/<name>/positions.json (--positions) and reused on later builds, so the
maps keep their shape as papers are added; only new papers and authors are placed. --relayout
recomputes every layout from scratch.

Safety: before writing anything, the script checks that it found at least 90% as many papers as the
previous build in --out, that arXiv returned >= 95% of the requested records and that >= 90% of
the arXiv papers have embeddings; after building, that the citation links, the links found by ADS and the
papers with ADS citation counts did not drop by more than 10% from the current page (an API that fails
part-way makes the build fall back silently). If a check fails it stops without touching the page or
the cache (--force overrides). ADS requests are retried on rate limits, server errors and timeouts.

Requirements: Python 3.9+, requests, numpy, scipy, scikit-learn.

Topics: k-means clusters of the SPECTER2 embeddings (one per entry in the config's "topics"), named by
hand after reading their titles. The topic of every paper checked then is stored in "topic_seed", so the
names keep matching their papers. New papers join the topic whose centre (mean embedding) is nearest, or,
without an embedding, the topic most common among the papers with the most similar words.
--recluster redoes the clustering from scratch (then check the printed titles and edit "topics"); a
project with an empty "topic_seed" (a new project) is clustered from scratch automatically.
"""
import argparse
import difflib
import html as html_mod
import json
import os
import re
import sys
import time
import unicodedata
import urllib.parse
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import date

import numpy as np
import requests
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

REPO_URL = "https://github.com/franciscovillaescusa/literature-atlas"
NEW_DAYS = 30   # papers added to the site within this many days are marked "New"
UA = {"User-Agent": "literature-atlas-builder/1.0"}
SEED = 0

# Set from the project's config.json by configure() before anything else runs.
NAME = PUBS_URL = SITE_URL = HOMEPAGE = GA_ID = None
N_TOPICS = UNASSIGNED = None
TOPICS = TOPIC_NAMES = TOPIC_SHORT = TOPIC_SEED = TAG_AXES = None


def configure(cfg):
    """Project settings from config.json:
      name, homepage, pubs_url, site_url (public address, used for link previews),
      ga_id (Google Analytics 4 measurement ID; "" disables it),
      topics: [{name, short (used where space is tight), probes (terms that match the name to a k-means
              cluster in --recluster)}], unassigned_short,
      topic_seed: {arXiv ID: index into topics}  (hand-checked topics, kept fixed on later builds),
      tag_axes: {suite: [...], data: [...], method: [...]}  (allowed tag names)."""
    global NAME, PUBS_URL, SITE_URL, HOMEPAGE, GA_ID, N_TOPICS, UNASSIGNED
    global TOPICS, TOPIC_NAMES, TOPIC_SHORT, TOPIC_SEED, TAG_AXES
    NAME, PUBS_URL, SITE_URL, HOMEPAGE = cfg["name"], cfg["pubs_url"], cfg["site_url"], cfg["homepage"]
    GA_ID = cfg.get("ga_id", "")
    # (name, probe terms used to match the name to a k-means cluster)
    TOPICS = [(t["name"], t.get("probes", [])) for t in cfg["topics"]]
    N_TOPICS = len(TOPICS)
    if not 1 <= N_TOPICS <= 10:   # the page has 10 topic colours (--t0 ... --t9) plus grey for "Unassigned"
        sys.exit(f"config: {N_TOPICS} topics; the page has colours for 1 to 10 (add colours to atlas.css for more)")
    # papers that fit no topic get this extra index (shown in gray as "Unassigned (new papers)")
    UNASSIGNED = len(TOPICS)
    TOPIC_NAMES = [n for n, _ in TOPICS] + ["Unassigned (new papers)"]
    TOPIC_SHORT = [t.get("short", t["name"]) for t in cfg["topics"]] + [cfg.get("unassigned_short", "Unassigned")]
    TOPIC_SEED = {k: int(v) for k, v in cfg.get("topic_seed", {}).items()}
    TAG_AXES = cfg["tag_axes"]


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------- fetching
def fetch_entries():
    """Every numbered entry on the publications page: title, authors, link (arXiv, ADS or other)."""
    page = requests.get(PUBS_URL, headers=UA, timeout=60).text
    entries = []
    for n, item in enumerate(re.findall(r'<li><div class="line-block">(.*?)</div>\s*</li>', page, re.S)):
        lines = [html_mod.unescape(re.sub(r"<[^>]+>", "", l)).strip()
                 for l in re.findall(r'<div class="line">(.*?)</div>', item, re.S)]
        hrefs = [html_mod.unescape(h) for h in re.findall(r'href="([^"]+)"', item)]
        ax = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})", item)
        bib = next((urllib.parse.unquote(m.group(1)) for h in hrefs
                    for m in [re.search(r"ui\.adsabs\.harvard\.edu/abs/([^/]+)", h)] if m), None)
        entries.append({
            "n": n + 1, "title": lines[0] if lines else "",
            "authors": [a.strip() for a in lines[1].split(",") if a.strip()] if len(lines) > 1 else [],
            "label": lines[2] if len(lines) > 2 else "", "url": hrefs[0] if hrefs else "",
            "arxiv": ax.group(1) if ax else None, "bibcode": None if ax else bib,
        })
    log(f"publications page: {len(entries)} entries, {sum(1 for e in entries if e['arxiv'])} with arXiv IDs")
    return entries


def fetch_arxiv(ids):
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = {}
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        for attempt in range(6):  # back off when arXiv rate-limits (HTTP 429) or is busy (503)
            r = requests.get("http://export.arxiv.org/api/query", headers=UA, timeout=120,
                             params={"id_list": ",".join(chunk), "max_results": len(chunk)})
            if r.status_code not in (429, 503):
                break
            log(f"arXiv HTTP {r.status_code}, retrying in {15 * (attempt + 1)} s...")
            time.sleep(15 * (attempt + 1))
        r.raise_for_status()
        for e in ET.fromstring(r.text).findall("a:entry", ns):
            aid = re.sub(r"v\d+$", "", e.find("a:id", ns).text.rsplit("/abs/", 1)[-1])
            out[aid] = {
                "title": " ".join(e.find("a:title", ns).text.split()),
                "abstract": " ".join(e.find("a:summary", ns).text.split()),
                "published": e.find("a:published", ns).text[:10],
                "authors": [a.find("a:name", ns).text for a in e.findall("a:author", ns)],
            }
        log(f"arXiv: {len(out)}/{len(ids)}")
        time.sleep(3)  # arXiv asks for >= 3 s between requests
    return out


def s2_get(url, method="GET", **kw):
    """Semantic Scholar request with back-off on rate limiting (HTTP 429)."""
    for attempt in range(8):
        hdr = dict(UA, **({"x-api-key": os.environ["S2_API_KEY"]} if os.environ.get("S2_API_KEY") else {}))
        r = requests.request(method, url, headers=hdr, timeout=120, **kw)
        if r.status_code in (200, 404):
            return r
        time.sleep(5 * (attempt + 1))
    return r


def norm_title(t):
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", t).lower())


def fetch_semantic_scholar(entries, ads_rec):
    """One Semantic Scholar record per entry: by arXiv ID, else DOI, else an exact title match."""
    keys = []
    for e in entries:
        doi = ((ads_rec.get(e["bibcode"]) or {}).get("doi") or [None])[0] if e["bibcode"] else None
        if e["arxiv"]:
            keys.append(f"ARXIV:{e['arxiv']}")
        elif doi:
            keys.append(f"DOI:{doi}")
        else:
            r = s2_get("https://api.semanticscholar.org/graph/v1/paper/search/match",
                       params={"query": e["title"], "fields": "title"})
            hit = r.json()["data"][0] if r.status_code == 200 and r.json().get("data") else None
            keys.append(hit["paperId"] if hit and norm_title(hit["title"]) == norm_title(e["title"]) else None)
            time.sleep(1.5)
    fields = "title,year,venue,citationCount,externalIds,references.externalIds,embedding.specter_v2"
    want = [k for k in keys if k]
    got = {}
    for i in range(0, len(want), 100):
        chunk = want[i:i + 100]
        r = s2_get("https://api.semanticscholar.org/graph/v1/paper/batch", method="POST",
                   params={"fields": fields}, json={"ids": chunk})
        r.raise_for_status()
        got.update(zip(chunk, r.json()))
        log(f"Semantic Scholar: {len(got)}/{len(want)}")
        time.sleep(3)
    return [got.get(k) if k else None for k in keys]


def ads_token():
    tok = os.environ.get("ADS_TOKEN")
    path = os.path.expanduser("~/.ads/dev_key")
    if not tok and os.path.exists(path):
        tok = open(path).read().strip()
    return tok or None


def ads_query(q, fl, rows=200, attempts=4):
    """One ADS search. Rate limits (429), server errors (5xx), timeouts and dropped connections are
    retried with growing waits (10, 20, 40 s); other errors (e.g. a bad token) fail at once."""
    tok = ads_token()
    for attempt in range(attempts):
        last = attempt == attempts - 1
        try:
            r = requests.get("https://api.adsabs.harvard.edu/v1/search/query", timeout=60,
                             headers=dict(UA, Authorization=f"Bearer {tok}"), params={"q": q, "fl": fl, "rows": rows})
        except requests.RequestException as err:
            if last:
                raise RuntimeError(f"ADS request failed: {err}") from err
            problem = type(err).__name__
        else:
            if r.status_code == 200:
                return r.json()["response"]["docs"]
            if r.status_code != 429 and r.status_code < 500 or last:
                raise RuntimeError(f"ADS HTTP {r.status_code}: {r.text[:200]}")
            problem = f"HTTP {r.status_code}"
        wait = 10 * 2 ** attempt
        log(f"ADS {problem}, retrying in {wait} s...")
        time.sleep(wait)


def fetch_ads_records(bibcodes):
    """Full ADS records for entries that link to ADS instead of arXiv."""
    if not bibcodes or not ads_token():
        return {}
    q = "bibcode:(" + " OR ".join(f'"{b}"' for b in bibcodes) + ")"
    docs = ads_query(q, "bibcode,title,author,abstract,pubdate,entry_date,citation_count,doi,identifier,pub")
    log(f"ADS records for non-arXiv entries: {len(docs)}/{len(bibcodes)}")
    return {d["bibcode"]: d for d in docs}


def fetch_ads_citations(ids):
    if not ads_token():
        log("ADS: no token in $ADS_TOKEN or ~/.ads/dev_key -> using Semantic Scholar citation counts")
        return {}
    out = {}
    try:
        for i in range(0, len(ids), 40):
            chunk = ids[i:i + 40]
            docs = ads_query("identifier:(" + " OR ".join(f'"arXiv:{a}"' for a in chunk) + ")",
                             "bibcode,identifier,citation_count")
            want = set(chunk)
            for d in docs:
                for ident in d.get("identifier", []):
                    a = ident.split(":", 1)[1] if ident.lower().startswith("arxiv:") else ident
                    if a in want:
                        c = d.get("citation_count") or 0
                        if a not in out or c > out[a]["citations"]:
                            out[a] = {"bibcode": d["bibcode"], "citations": c}
            time.sleep(1)
    except RuntimeError as err:
        log(f"{err} -> using Semantic Scholar counts")
        return {}
    log(f"ADS citations: matched {len(out)}/{len(ids)}")
    return out


def fetch_ads_references(bibcodes):
    """ADS reference lists (bibcodes) and alternative identifiers for every paper with an ADS record."""
    if not bibcodes or not ads_token():
        return {}
    out = {}
    try:
        for i in range(0, len(bibcodes), 50):
            chunk = bibcodes[i:i + 50]
            for d in ads_query("bibcode:(" + " OR ".join(f'"{b}"' for b in chunk) + ")",
                               "bibcode,identifier,reference", rows=len(chunk) + 10):
                out[d["bibcode"]] = {"ids": d.get("identifier", []), "refs": d.get("reference", [])}
            time.sleep(1)
    except RuntimeError as err:
        log(f"{err} -> citation links from Semantic Scholar only")
        return {}
    log(f"ADS reference lists: {sum(1 for v in out.values() if v['refs'])}/{len(bibcodes)} papers")
    return out


PENDING_CACHE = {}  # fresh downloads, written to the cache only after the sanity checks pass


def load_or_fetch(cache, name, fn, offline):
    path = os.path.join(cache, name)
    if offline:
        if not os.path.exists(path):
            sys.exit(f"--offline: missing cached file {path}; run once without --offline first")
        return json.load(open(path))
    data = fn()
    PENDING_CACHE[path] = data
    return data


def previous_data(path):
    """The data embedded in an earlier build of the page at `path` (None if there is none)."""
    try:
        h = open(path, encoding="utf-8").read()
        blob = h.split('id="atlas-data">', 1)[1].split("</script>", 1)[0].replace("<\\/", "</")
        return json.loads(blob)
    except (OSError, IndexError, ValueError, KeyError):
        return None


def fetch_date(cache, offline):
    """Date of the downloads: today for a fresh fetch (saved with the cache), else the saved date."""
    path = os.path.join(cache, "meta.json")
    if not offline:
        PENDING_CACHE[path] = {"fetched": date.today().isoformat()}
        return date.today().isoformat()
    if os.path.exists(path):
        return json.load(open(path))["fetched"]
    return date.fromtimestamp(os.path.getmtime(os.path.join(cache, "entries.json"))).isoformat()


def sanity_check(entries, arx, s2, prev_n):
    """Refuse to build a page that looks broken (e.g. the publications page changed its HTML,
    or an API returned partial results). Returns a list of problems (empty if all is well)."""
    problems = []
    n, ax = len(entries), [e["arxiv"] for e in entries if e["arxiv"]]
    if n == 0:
        problems.append("no entries found on the publications page (did its HTML structure change?)")
    elif prev_n and n < 0.9 * prev_n:
        problems.append(f"only {n} entries found, but the previous build had {prev_n} (more than 10% fewer)")
    got_ax = sum(1 for a in ax if a in arx)
    if ax and got_ax < 0.95 * len(ax):
        problems.append(f"arXiv returned only {got_ax} of {len(ax)} requested records")
    # embeddings are counted over the arXiv papers only: theses and other entries rarely have one, so with
    # many of them a healthy fetch could fall below the limit (Quijote: 4 of 30 theses have one)
    pool = [p for e, p in zip(entries, s2) if e["arxiv"]] or list(s2)
    n_emb = sum(1 for p in pool if p and p.get("embedding"))
    if pool and n_emb < 0.9 * len(pool):
        problems.append(f"Semantic Scholar embeddings for only {n_emb} of {len(pool)} "
                        f"{'arXiv papers' if ax else 'entries'} (below 90%)")
    return problems


def compare_with_previous(data, prev, drop=0.9):
    """Refuse to publish a page whose citation data collapsed compared with the current site. When an
    API fails part-way, the build falls back silently (e.g. ADS down -> Semantic Scholar counts and no
    ADS reference lists), so it would otherwise publish a site with far fewer links. Returns problems."""
    if not prev:
        return []
    problems = []
    def check(what, new, old, why):
        if old >= 20 and new < drop * old:
            problems.append(f"{what}: {new} now vs {old} on the current site (more than {round(100 * (1 - drop))}% fewer; {why})")
    links = lambda d: sum(len(p["r"]) for p in d["papers"])
    ads_links = lambda d: (d.get("refsrc") or {}).get("both", 0) + (d.get("refsrc") or {}).get("ads", 0)
    check("citation links", links(data), links(prev), "did ADS or Semantic Scholar return partial reference lists?")
    check("citation links found by ADS", ads_links(data), ads_links(prev), "did the ADS reference lists fail?")
    check("papers with ADS citation counts", data["nads"], prev.get("nads", 0), "did the ADS citation query fail?")
    return problems


# ----------------------------------------------------------------------------- publications-page problems
def surnames(names):
    return {re.sub(r"[^a-z]", "", unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode().lower().split()[-1])
            for n in names if n.split()}


def entry_problems(entries, arx):
    """Problems in the publications-page entries that a person should look at (the build goes on):
    arXiv IDs that arXiv does not know, arXiv links that probably point to a different paper (different
    title AND different authors; a paper retitled between arXiv versions keeps its authors), and duplicates."""
    out, seen = [], {}
    for e in entries:
        where = f"entry {e['n']} ({e['title'][:60]!r})"
        ax = e["arxiv"]
        if ax and ax not in arx:
            out.append(f"{where}: arXiv returned no record for {ax} (typo in the link?)")
        elif ax and e["title"] and e["authors"]:
            a = arx[ax]
            if similar_title(e["title"], a["title"]) < 0.6 and len(surnames(e["authors"]) & surnames(a["authors"])) < 0.5 * len(surnames(e["authors"])):
                out.append(f"{where}: the link {ax} is to {a['title'][:60]!r} by {', '.join(a['authors'][:2])}: wrong link?")
        key = ax or e["bibcode"] or "title:" + norm_title(e["title"])
        if key in seen:
            out.append(f"{where}: duplicate of entry {seen[key]}; only the first is used")
        seen.setdefault(key, e["n"])
    return out


def safe_url(url):
    """Links on the page must be http(s): relative links are resolved against the publications page,
    anything else (e.g. javascript:) is dropped. Returns (url, problem or None)."""
    url = (url or "").strip()
    if not url or re.match(r"https?://", url, re.I):
        return url, None
    if not re.match(r"[a-z][a-z0-9+.-]*:", url, re.I):   # relative
        return urllib.parse.urljoin(PUBS_URL, url), None
    return "", f"link {url[:60]!r} is not http(s); dropped"


# ----------------------------------------------------------------------------- analysis
def norm_name(s):
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"[^a-z ]", "", s.lower().replace("-", " ").replace(".", " ")).split()


def canonical_authors(names):
    """Merge spelling variants: same surname + first initial. Different full first names stay separate
    (e.g. Bitao vs Bonny Wang); abbreviated forms join the most frequent full form."""
    groups = defaultdict(list)
    for nm in names:
        t = norm_name(nm)
        groups[(t[0][0], t[-1])].append(nm)
    canon = {}
    for key, nms in groups.items():
        full, abbrev = defaultdict(list), []
        for nm in nms:
            f = norm_name(nm)[0]
            (full[f] if len(f) > 1 else abbrev).append(nm)
        # a two-part given name written as one word elsewhere: Jun-Qing Xia joins Junqing Xia
        for f in list(full):
            for nm in list(full[f]):
                t = norm_name(nm)
                if len(t) >= 3 and t[0] + t[1] in full and t[0] + t[1] != f:
                    full[f].remove(nm)
                    full[t[0] + t[1]].append(nm)
            if not full[f]:
                del full[f]
        if not full:
            full[key[0]], abbrev = abbrev, []
        best = max(full, key=lambda f: len(full[f]))
        full[best].extend(abbrev)
        for f, variants in full.items():
            cnt = Counter(variants)
            top = cnt.most_common(1)[0][1]
            # most frequent spelling; ties -> longest, then the accented form, then alphabetical
            disp = max((n for n in cnt if cnt[n] == top),
                       key=lambda n: (len(n), sum(ord(ch) > 127 for ch in n), n))
            for nm in variants:
                canon[nm] = disp
    return canon


def first_sentence(text):
    """First sentence of an abstract (markup removed), used where only an excerpt may be published."""
    text = " ".join(re.sub(r"<[^>]+>", " ", html_mod.unescape(text or "")).split())
    m = re.search(r"^(.{40,}?[.?!])\s+[A-Z(]", text)
    return m.group(1) if m else text[:300]


def ads_author(a):
    """'Last, First M.' -> 'First M. Last'"""
    last, _, first = a.partition(",")
    return f"{first.strip()} {last.strip()}".strip()


# ----------------------------------------------------------------------------- hand corrections
OVERRIDE_FIELDS = {"id", "topic", "date", "abstract", "url", "link", "authors", "title", "hide", "comment"}


def apply_overrides(recs, overrides):
    """Apply hand corrections. Keys are an arXiv ID, an ADS bibcode or the paper title; values may set
    topic (a topic name), date ("YYYY", "YYYY-MM" or "YYYY-MM-DD"), abstract, url, link (label),
    authors (list), title, hide (true leaves the entry out), comment (ignored) and id: the paper's ID on
    the site. Papers without an arXiv ID are known by their title, so a title edited on the publications
    page gives a new ID; "id" keeps the old one, and with it the paper's tags, topic and map position.
    Returns the remaining records, {record index: topic index} for topic overrides and {ID from the
    source: kept ID} for ID overrides."""
    by_key = {}
    for n, r in enumerate(recs):
        by_key[r["uid"]] = n
        by_key["title:" + norm_title(r["title"])] = n
    # first pass: ID changes, so that the other fields can also be keyed by the paper's kept ID
    kept = {}
    for key, ov in overrides.items():
        n = by_key.get(key, by_key.get("title:" + norm_title(key)))
        if key.startswith("_") or n is None or "id" not in ov:
            continue
        new = str(ov["id"])
        if any(r["uid"] == new for m, r in enumerate(recs) if m != n):
            log(f"overrides: id {new!r} for {key!r} is already the ID of another paper; ignored")
            continue
        log(f"overrides: paper {recs[n]['uid'][:60]!r} keeps the ID {new[:60]!r}")
        kept[recs[n]["uid"]] = new
        recs[n]["uid"] = new
        by_key[new] = n
    hidden, topic_of, used = set(), {}, 0
    for key, ov in overrides.items():
        if key.startswith("_"):
            continue  # help text
        n = by_key.get(key, by_key.get("title:" + norm_title(key)))
        if n is None:
            log(f"overrides: no paper matches {key!r}; check the arXiv ID, bibcode or title")
            continue
        bad = set(ov) - OVERRIDE_FIELDS
        if bad:
            log(f"overrides: unknown field(s) {sorted(bad)} for {key!r}; allowed: {sorted(OVERRIDE_FIELDS)}")
        r, used, done = recs[n], used + 1, []
        if ov.get("hide"):
            hidden.add(n)
            done.append("hide")
        for f in ("abstract", "url", "link", "authors", "title"):
            if f in ov:
                r[f] = ov[f]
                if f == "abstract":
                    r["excerpt"] = False  # your own text is published in full
                done.append(f)
        if "date" in ov:
            if not re.fullmatch(r"\d{4}(-\d{2}){0,2}", str(ov["date"])):
                log(f"overrides: date {ov['date']!r} for {key!r} must look like 2025, 2025-06 or 2025-06-30")
            else:
                r["date"], r["dq"] = str(ov["date"]), False
                done.append("date")
        if "topic" in ov:
            if ov["topic"] in TOPIC_NAMES:
                topic_of[r["uid"]] = TOPIC_NAMES.index(ov["topic"])
                done.append("topic")
            else:
                log(f"overrides: unknown topic {ov['topic']!r} for {key!r}; allowed: {TOPIC_NAMES}")
        if done:
            log(f"overrides: applied {done} to {r['title'][:60]}")
    keep = [r for n, r in enumerate(recs) if n not in hidden]
    return keep, {n: topic_of[r["uid"]] for n, r in enumerate(keep) if r["uid"] in topic_of}, kept


# ----------------------------------------------------------------------------- tags
# Tags on three axes, assigned by reading each abstract and stored in projects/<name>/tags.json (--tags),
# which you can edit by hand: {id: {"title", "suite": [...], "data": [...], "method": [...], "uncertain": [...]}}.
# The allowed names per axis are "tag_axes" in the project's config.json.

def similar_title(a, b):
    """How alike two titles are (0..1), ignoring case, accents and punctuation."""
    return difflib.SequenceMatcher(None, norm_title(a), norm_title(b)).ratio()


def report_orphans(uids, titles, tags_path, store):
    """Warn about tags, hand-checked topics and map positions whose paper ID matches no paper. This happens
    when the publications page edits the title of a paper without an arXiv ID (its ID is its title) or
    changes its link; the paper then shows up untagged with a new ID. For each such ID, suggest the
    current paper with the most similar title and the overrides.json line that keeps the old ID."""
    tags = json.load(open(tags_path)) if os.path.exists(tags_path) else {}
    known = set(uids)
    old = {}   # orphaned ID -> (where it is used, its title if known)
    for uid, t in tags.items():
        if uid not in known:
            old.setdefault(uid, [set(), t.get("title", "")])[0].add("tags")
    for uid in TOPIC_SEED:
        if uid not in known:
            old.setdefault(uid, [set(), ""])[0].add("topic_seed")
    for uid in store.get("topics2d", {}):
        if uid not in known:
            old.setdefault(uid, [set(), ""])[0].add("positions")
    if not old:
        return
    log(f"{len(old)} paper ID(s) in the tags, topic_seed or positions match no paper (renamed or removed?):")
    for uid, (where, title) in sorted(old.items()):
        title = title or (uid[6:] if uid.startswith("title:") else "")
        best = max(((similar_title(title, t), u, t) for u, t in zip(uids, titles)
                    if u not in tags and u not in TOPIC_SEED), default=None) if title else None
        log(f"      - {uid[:70]}  (in {', '.join(sorted(where))})")
        if best and best[0] >= 0.8:
            log(f"        probably renamed to: {best[2][:70]}\n"
                f"        to keep its tags, topic and position, add to overrides.json: "
                f"{json.dumps(best[1])}: {{\"id\": {json.dumps(uid)}}}")


def load_tags(path, uids, titles):
    """Tags per paper from the tags file; warns about unknown tag names and untagged papers."""
    tags = json.load(open(path)) if os.path.exists(path) else {}
    out, untagged = [], []
    for uid, title in zip(uids, titles):
        t = tags.get(uid)
        if not t:
            untagged.append(title)
            out.append(None)
            continue
        rec = {}
        for ax, short in (("suite", "s"), ("data", "d"), ("method", "m")):
            bad = [x for x in t.get(ax, []) if x not in TAG_AXES[ax]]
            if bad:
                log(f"tags file: unknown {ax} tag(s) {bad} for {uid}; allowed: {TAG_AXES[ax]}")
            rec[short] = [x for x in t.get(ax, []) if x in TAG_AXES[ax]]
        rec["u"] = t.get("uncertain", [])
        if t.get("note"):
            rec["n"] = t["note"]
        out.append(rec)
    if untagged:
        log(f"{len(untagged)} paper(s) without tags (add them to {path}):")
        for title in untagged:
            log(f"      - {title[:90]}")
    return out


# ----------------------------------------------------------------------------- layouts
# Positions are stored in a JSON file (--positions) and kept fixed on later builds, so the maps
# don't rearrange when papers are added. Only new papers/authors are placed.

def spiral(n, dim=3):
    """Deterministic start positions on a spherical (3D) or flat (2D) spiral."""
    if dim == 2:
        i = np.arange(n)
        r, th = 10 * np.sqrt(0.5 + i), i * np.pi * (3 - np.sqrt(5))
        return np.stack([r * np.cos(th), r * np.sin(th)], 1)
    return spiral3d(n)


def spiral3d(n):
    """Deterministic start positions on a spherical spiral."""
    i = np.arange(n)
    r = 12 * np.cbrt(0.5 + i)
    th = i * np.pi * (3 - np.sqrt(5))
    ph = np.arccos(1 - 2 * (i + 0.5) / max(n, 1))
    return np.stack([r * np.sin(ph) * np.cos(th), r * np.sin(ph) * np.sin(th), r * np.cos(ph)], 1)


def force3d(pos, s, t, length, k, r, charge, gravity, fixed, iters=300):
    """3D force layout (d3-force style: springs, many-body repulsion, collision, centering).
    Nodes with fixed=True keep their position but still push and pull the others."""
    pos, vel = pos.astype(float).copy(), np.zeros_like(pos, dtype=float)
    deg = np.bincount(np.concatenate([s, t]), minlength=len(pos)).astype(float)
    bias = deg[s] / np.maximum(deg[s] + deg[t], 1)
    # all nodes are updated at once, so links on highly connected nodes are weakened (as d3 does by
    # default) to keep their summed pull from overshooting; a step cap is a further safeguard
    k = np.minimum(k, 1.0 / np.maximum(np.minimum(deg[s], deg[t]), 1)) if len(s) else k
    free = ~fixed
    rr = r[:, None] + r[None, :] + 1.5
    alpha, decay = 1.0, 1 - 0.001 ** (1 / 300)
    for _ in range(iters):
        if len(s):
            d = pos[t] + vel[t] - pos[s] - vel[s]
            dist = np.maximum(np.linalg.norm(d, axis=1), 1e-6)
            dv = d * ((dist - length) / dist * alpha * k)[:, None]
            np.subtract.at(vel, t, dv * bias[:, None])
            np.add.at(vel, s, dv * (1 - bias)[:, None])
        diff = pos[None, :, :] - pos[:, None, :]          # [i, j] = pos_j - pos_i
        d2 = np.maximum((diff ** 2).sum(-1), 1.0)
        np.fill_diagonal(d2, np.inf)
        f = charge * alpha / d2
        dd = np.sqrt(d2)
        f = f + np.where(dd < rr, (rr - dd) / dd * 0.35, 0.0)  # collision
        vel -= (diff * f[..., None]).sum(1)
        vel -= pos * gravity * alpha
        vel *= 0.6
        step = np.linalg.norm(vel, axis=1, keepdims=True)
        vel *= np.minimum(1.0, 30.0 / np.maximum(step, 1e-9))
        vel[~free] = 0
        pos += vel
        alpha -= alpha * decay
    return pos


def network_positions(keys, s, t, length, k, r, charge, gravity, old, dim=3):
    """Positions for a network. Known keys keep their stored position; new nodes start next to
    their known neighbours (or on the outer spiral) and settle while the known ones stay fixed."""
    n = len(keys)
    fixed = np.array([key in old for key in keys], bool)
    if not fixed.any():
        return force3d(spiral(n, dim), s, t, length, k, r, charge, gravity, fixed)
    pos = spiral(n, dim) * 3.0
    nb = defaultdict(list)
    for a, b in zip(s, t):
        nb[a].append(b)
        nb[b].append(a)
    for i, key in enumerate(keys):
        if fixed[i]:
            pos[i] = old[key]
    for i in np.where(~fixed)[0]:
        known = [j for j in nb[i] if fixed[j]]
        if known:
            pos[i] = np.mean(pos[known], 0) + np.cos(np.arange(dim) + i) * 5
    if fixed.all():
        return pos
    return force3d(pos, s, t, length, k, r, charge, gravity, fixed)


def tsne(X, dim):
    Y = TSNE(dim, perplexity=12, metric="cosine", init="pca", random_state=1, learning_rate="auto").fit_transform(X)
    Y = Y - Y.mean(0)
    return Y / np.abs(Y).max()


def topic_positions(X, T, emb, uids, titles, store):
    """2D topic-map positions. First build: t-SNE of the embeddings. Later builds: stored
    positions; new papers go to the mean position of their 3 most similar placed papers."""
    old2 = store.get("topics2d", {})
    N, row = len(uids), {n: j for j, n in enumerate(emb)}
    pos2 = {}
    known = [n for n in range(N) if uids[n] in old2]
    if known:
        for n in known:
            pos2[n] = np.array(old2[uids[n]])
    else:
        Y2 = tsne(X, 2)
        for j, n in enumerate(emb):
            pos2[n] = Y2[j]
    placed = list(pos2)
    for n in range(N):
        if n in pos2:
            continue
        cand = [m for m in placed if m in row]
        if n in row and cand:      # by embedding similarity
            sims = X[[row[m] for m in cand]] @ X[row[n]]
        else:                      # no embedding: by title/abstract word similarity
            cand = placed
            sims = (T[cand] @ T[n].T).toarray().ravel()
        nn = [cand[i] for i in np.argsort(-sims)[:3]]
        pos2[n] = np.mean([pos2[m] for m in nn], 0) + 0.03 * np.cos(np.arange(2) + n)
        log(f"placed next to similar papers: {titles[n][:70]}")
    return pos2


def similarity(X, T, emb, N):
    """Paper-to-paper similarity: cosine of the SPECTER2 embeddings where both papers have one,
    otherwise cosine of their title+abstract word (TF-IDF) vectors. Returns (matrix, kind matrix)."""
    W = (T @ T.T).toarray()
    sim, kind = W.copy(), np.full((N, N), "w")
    row = np.array(emb)
    E = X @ X.T
    sim[np.ix_(row, row)] = E
    kind[np.ix_(row, row)] = "e"
    np.fill_diagonal(sim, -np.inf)
    return sim, kind


def similar_papers(sim, kind, n_best=5):
    """For each paper, its most similar papers [(index, score)], and whether by embedding or words."""
    out = []
    for i in range(len(sim)):
        own = kind[i] == "e"
        pool = np.where(own)[0] if own.sum() > 1 else np.arange(len(sim))  # compare like with like
        pool = pool[pool != i]
        best = pool[np.argsort(-sim[i, pool], kind="stable")[:n_best]]   # stable: ties ordered the same on every platform
        out.append({"k": "e" if own.sum() > 1 else "w",
                    "l": [[int(j), round(float(sim[i, j]), 3)] for j in best]})
    return out


def topic_graph_positions(sim, kind, uids, old, n_links=4):
    """3D topic map: link each paper to its n_links most similar papers and lay the network out like
    the citation and co-author networks. This keeps similar papers together more faithfully than a
    3D t-SNE (tested Nov 2026: 60% vs 47% of each paper's 5 most similar papers among its 10 nearest)."""
    N, pairs = len(uids), set()
    for i in range(N):
        own = np.where(kind[i] == "e")[0] if (kind[i] == "e").sum() > 1 else np.arange(N)
        own = own[own != i]
        for j in own[np.argsort(-sim[i, own])[:n_links]]:
            pairs.add((min(i, int(j)), max(i, int(j))))
    pairs = sorted(pairs)
    s = np.array([a for a, _ in pairs], int)
    t = np.array([b for _, b in pairs], int)
    return network_positions(uids, s, t, np.full(len(s), 30.0), np.full(len(s), 0.5), np.full(N, 6.0), 90, 0.1, old)


def recluster_topics(X, T, tv, emb):
    """Fresh k-means clustering. Names are matched to clusters by the probe keywords in TOPICS,
    so check the printed titles and edit TOPICS (and TOPIC_SEED) if the clusters have changed."""
    log("re-clustering topics from scratch (--recluster): check that the names still fit!")
    L = KMeans(N_TOPICS, n_init=50, random_state=SEED).fit(X).labels_
    V = {v: n for n, v in enumerate(tv.get_feature_names_out())}
    Te = T[emb]
    labs = sorted(set(L))
    score = np.zeros((len(labs), len(TOPICS)))
    for a, lab in enumerate(labs):
        m = L == lab
        diff = np.asarray(Te[m].mean(0)).ravel() - np.asarray(Te[~m].mean(0)).ravel()
        for b, (_, probes) in enumerate(TOPICS):
            score[a, b] = sum(diff[V[p]] for p in probes if p in V)
    r_, c_ = linear_sum_assignment(-score)
    lab2topic = {labs[a]: int(b) for a, b in zip(r_, c_)}
    return {n: lab2topic[L[j]] for j, n in enumerate(emb)}


def build_data(entries, arx, s2, ads_rec, ads_cit, ads_refs, store, fetched, tags_path, overrides,
               recluster=False):
    # ---- one record per entry on the publications page
    recs = []
    notes = entry_problems(entries, arx)
    seen_ids = set()
    for e, p in zip(entries, s2):
        p = p or {}
        ax, rec = e["arxiv"], ads_rec.get(e["bibcode"] or "")
        uid0 = ax or e["bibcode"] or "title:" + norm_title(e["title"])[:80]
        if uid0 in seen_ids:   # duplicate entry on the publications page (reported above)
            continue
        seen_ids.add(uid0)
        if ax and ax in arx:
            a = arx[ax]
            title, abstract, date_, authors = a["title"], a["abstract"], a["published"], a["authors"]
            link, url = f"arXiv:{ax}", f"https://arxiv.org/abs/{ax}"
        elif rec:
            title, abstract = (rec.get("title") or [e["title"]])[0], rec.get("abstract") or ""
            # date of first appearance: the earlier of the issue date and when ADS first listed it
            # (journal issues can be dated months after the paper came out online)
            issue = rec.get("pubdate", "")[:7].replace("-00", "")
            entry = (rec.get("entry_date") or "")[:10]
            date_ = min(d for d in (issue, entry) if d) if (issue or entry) else ""
            authors = e["authors"] or [ads_author(x) for x in rec.get("author", [])]
            link, url = rec.get("pub") or "ADS", e["url"]
        else:  # theses and other entries without an arXiv or ADS record
            title, abstract, authors = e["title"], "", e["authors"]
            date_ = str(p["year"]) if p.get("year") else ""
            link, url = e["label"] or "Link", e["url"]
        if ax in ads_cit:
            c, cs, bib = ads_cit[ax]["citations"], 0, ads_cit[ax]["bibcode"]
        elif rec:
            c, cs, bib = rec.get("citation_count") or 0, 0, rec["bibcode"]
        elif p.get("citationCount") is not None:
            c, cs, bib = p["citationCount"], 1, ""
        else:
            c, cs, bib = 0, 2, ""  # unknown
        # drop "names" without letters (arXiv records occasionally list e.g. " :" as an author)
        authors = [nm for nm in authors if norm_name(nm)]
        recs.append({"uid": uid0, "title": title, "abstract": abstract,
                     "date": date_, "authors": authors or ["Unknown"], "link": link, "url": url,
                     "c": c, "cs": cs, "bib": bib, "venue": p.get("venue") or (rec or {}).get("pub") or "",
                     "doi": (p.get("externalIds") or {}).get("DOI") or ((rec or {}).get("doi") or [""])[0],
                     "s2": p, "dq": False,
                     # journal abstracts from ADS are the publisher's copyright: publish only an excerpt
                     "excerpt": bool(rec) and not (ax and ax in arx)})

    # ---- hand corrections from the overrides file (applied before anything is derived from them)
    recs, topic_override, kept_ids = apply_overrides(recs, overrides)
    for r in recs:   # after the overrides, which can fix these
        where = f"{r['uid'][:40]} ({r['title'][:60]!r})"
        r["url"], bad = safe_url(r["url"])
        if bad:
            notes.append(f"{where}: {bad}")
        if not r["title"].strip() or re.match(r"(https?://|www\.)", r["title"].strip(), re.I):
            notes.append(f"{where}: the title is empty or a link; set \"title\" in overrides.json")
        if r["authors"] == ["Unknown"]:
            notes.append(f"{where}: no authors on the publications page or in arXiv/ADS; set \"authors\" in overrides.json")
    if notes:
        log(f"{len(notes)} problem(s) on the publications page (also listed in the \"needs tags\" issue):")
        for n_ in notes:
            log(f"      - {n_}")

    # ---- entries with no date: estimate from neighbours (the list is newest first)
    years = [int(r["date"][:4]) if r["date"] else None for r in recs]
    for n, r in enumerate(recs):
        if years[n] is None:
            prev = next((years[j] for j in range(n - 1, -1, -1) if years[j]), None)
            nxt = next((years[j] for j in range(n + 1, len(recs)) if years[j]), None)
            known = [y for y in (prev, nxt) if y]
            r["date"], r["dq"] = str(round(sum(known) / len(known))) if known else "2020", True
            log(f"date estimated from list position: {r['date']}  {r['title'][:70]}")

    N = len(recs)
    docs = [r["title"] + " " + r["abstract"] for r in recs]
    emb = [n for n, r in enumerate(recs) if r["s2"].get("embedding")]
    X = np.array([recs[n]["s2"]["embedding"]["vector"] for n in emb])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    tv = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.5, sublinear_tf=True)
    T = tv.fit_transform(docs)
    if recluster or not TOPIC_SEED:
        topic = recluster_topics(X, T, tv, emb)
    else:
        # fixed topics: known papers keep their hand-checked topic; new papers join the nearest topic centre
        seeded = [(j, TOPIC_SEED[recs[n]["uid"]]) for j, n in enumerate(emb) if recs[n]["uid"] in TOPIC_SEED]
        C = np.array([X[[j for j, k in seeded if k == t]].mean(0) for t in range(N_TOPICS)])
        C /= np.linalg.norm(C, axis=1, keepdims=True)
        # cutoff: the lowest similarity of any known paper to its own topic centre. A new paper less
        # similar than that to every topic fits none of them and goes to "Unassigned".
        cutoff = min(float(C[k] @ X[j]) for j, k in seeded)
        topic = {}
        for j, n in enumerate(emb):
            uid = recs[n]["uid"]
            if uid in TOPIC_SEED:
                topic[n] = TOPIC_SEED[uid]
                continue
            sims = C @ X[j]
            best = int(np.argmax(sims))
            if sims[best] < cutoff:
                topic[n] = UNASSIGNED
                log(f"new paper fits no topic (similarity {sims[best]:.3f} < cutoff {cutoff:.3f}) -> Unassigned: "
                    f"{recs[n]['title'][:60]}")
            else:
                topic[n] = best
                log(f"new paper -> topic '{TOPICS[best][0]}' (similarity {sims[best]:.3f}, cutoff {cutoff:.3f}): "
                    f"{recs[n]['title'][:50]}")

    for n in range(N):  # no embedding: most common topic among the 3 papers with the most similar words
        if n not in topic:
            sims = (T[emb] @ T[n].T).toarray().ravel()
            nn = np.argsort(-sims)[:3]
            topic[n] = TOPIC_SEED.get(recs[n]["uid"], Counter(topic[emb[j]] for j in nn).most_common(1)[0][0])
            log(f"no embedding: topic from word similarity  {recs[n]['title'][:60]}")
    for n, k in topic_override.items():
        topic[n] = k

    # ---- citation links between listed entries (match by Semantic Scholar ID, arXiv ID or DOI)
    by_pid, by_ax, by_doi = {}, {}, {}
    for n, r in enumerate(recs):
        ext = r["s2"].get("externalIds") or {}
        if r["s2"].get("paperId"):
            by_pid[r["s2"]["paperId"]] = n
        if ext.get("ArXiv"):
            by_ax[ext["ArXiv"]] = n
        if ext.get("DOI"):
            by_doi[ext["DOI"].lower()] = n
    # ADS: references are bibcodes; match them against every identifier of every listed paper
    by_bib = {}
    for n, r in enumerate(recs):
        if r["bib"]:
            by_bib[r["bib"]] = n
            for ident in (ads_refs.get(r["bib"]) or {}).get("ids", []):
                by_bib.setdefault(ident, n)
    refs, n_s2, n_ads, n_both = [], 0, 0, 0
    for n, r in enumerate(recs):
        s2_set, ads_set = set(), set()
        for ref in r["s2"].get("references") or []:
            ext = ref.get("externalIds") or {}
            j = by_pid.get(ref.get("paperId"))
            j = j if j is not None else by_ax.get(ext.get("ArXiv"))
            j = j if j is not None else by_doi.get((ext.get("DOI") or "").lower())
            if j is not None and j != n:
                s2_set.add(j)
        for b in (ads_refs.get(r["bib"]) or {}).get("refs", []) if r["bib"] else []:
            j = by_bib.get(b)
            if j is not None and j != n:
                ads_set.add(j)
        n_both += len(s2_set & ads_set)
        n_s2 += len(s2_set - ads_set)
        n_ads += len(ads_set - s2_set)
        refs.append(sorted(s2_set | ads_set))
        # reference data missing: neither source knows 10 or more of this paper's references (e.g. ADS has
        # not processed a new preprint yet, or a thesis is in neither database)
        n_ads_refs = len((ads_refs.get(r["bib"]) or {}).get("refs", [])) if r["bib"] else 0
        n_s2_refs = sum(1 for ref in (r["s2"].get("references") or []) if ref.get("paperId"))
        r["refmissing"] = n_ads_refs < 10 and n_s2_refs < 10
    log(f"citation links: {n_both} in both sources, {n_s2} Semantic Scholar only, {n_ads} ADS only")
    missing = [r["title"] for r in recs if r["refmissing"]]
    if missing:
        log(f"{len(missing)} paper(s) without a usable reference list in ADS or Semantic Scholar "
            "(their 'Cites' list may be empty or incomplete):")
        for title in missing:
            log(f"      - {title[:90]}")

    canon = canonical_authors([nm for r in recs for nm in r["authors"]])
    authors = sorted(set(canon.values()))
    aidx = {a: n for n, a in enumerate(authors)}
    au_of = []
    for r in recs:
        au = []
        for nm in r["authors"]:
            if aidx[canon[nm]] not in au:
                au.append(aidx[canon[nm]])
        au_of.append(au)

    # ---- layouts (stored positions are kept; only new items are placed)
    uids = [r["uid"] for r in recs]
    # a paper that keeps an earlier ID through an override may have been stored under its new ID by a build
    # made before the override was added: forget that ID, so it is not kept or reported as an orphan
    for key in ("topics2d", "topics3d_graph", "citations", "citations2d", "first_seen"):
        for u in kept_ids:
            store.get(key, {}).pop(u, None)
            store.get("retired", {}).get(key, {}).pop(u, None)
    report_orphans(uids, [r["title"] for r in recs], tags_path, store)
    # positions of papers whose ID no longer matches any paper are kept under "retired", so that an "id"
    # override added later (see apply_overrides) puts the paper back where it was
    for key, old in store.get("retired", {}).items():
        for u, v in old.items():
            store.setdefault(key, {}).setdefault(u, v)
    tags = load_tags(tags_path, uids, [r["title"] for r in recs])
    pos2 = topic_positions(X, T, emb, uids, [r["title"] for r in recs], store)
    sim, kind = similarity(X, T, emb, N)
    similar = similar_papers(sim, kind)
    graph3 = topic_graph_positions(sim, kind, uids, store.get("topics3d_graph", {}))
    cs_ = np.array([n for n in range(N) for j in refs[n]], int)
    ct_ = np.array([j for n in range(N) for j in refs[n]], int)
    citers = np.bincount(ct_, minlength=N)
    cit_args = (uids, cs_, ct_, np.full(len(cs_), 40.0), np.full(len(cs_), 0.35), 3 + np.sqrt(citers) * 2.2, 90, 0.18)
    cit = network_positions(*cit_args, store.get("citations", {}))
    cit2 = network_positions(*cit_args, store.get("citations2d", {}), dim=2)
    npap = Counter(a for au in au_of for a in au)
    pairs = Counter()
    for au in au_of:
        for x in range(len(au)):
            for y in range(x + 1, len(au)):
                pairs[(min(au[x], au[y]), max(au[x], au[y]))] += 1
    co, co2 = {}, {}
    for mode, keep in (("1", list(range(len(authors)))),):   # all authors
        loc = {a: i for i, a in enumerate(keep)}
        pl = [(loc[a], loc[b], c) for (a, b), c in pairs.items() if a in loc and b in loc]
        s_ = np.array([x[0] for x in pl], int)
        t_ = np.array([x[1] for x in pl], int)
        c_ = np.array([x[2] for x in pl], float)
        co_args = ([authors[a] for a in keep], s_, t_, 60 / np.sqrt(c_), np.ones(len(c_)),   # strength = 1/min(degree) via force3d (d3 default): small teams stay together
                   2.5 + np.sqrt(np.array([npap[a] for a in keep], float)) * 2, 70, 0.12)
        pos = network_positions(*co_args, store.get("coauthors" + mode, {}))
        pos2d = network_positions(*co_args, store.get("coauthors" + mode + "_2d", {}), dim=2)
        co[mode] = {a: pos[i] for i, a in enumerate(keep)}
        co2[mode] = {a: pos2d[i] for i, a in enumerate(keep)}
    rd = lambda v, nd=2: [round(float(x), nd) for x in v]
    new_store = {
        "topics2d": {uids[n]: rd(pos2[n], 4) for n in range(N)},
        "topics3d_graph": {uids[n]: rd(graph3[n]) for n in range(N)},
        "citations": {uids[n]: rd(cit[n]) for n in range(N)},
        "coauthors1": {authors[a]: rd(v) for a, v in co["1"].items()},
        "citations2d": {uids[n]: rd(cit2[n]) for n in range(N)},
        "coauthors1_2d": {authors[a]: rd(v) for a, v in co2["1"].items()},
    }
    uidset = set(uids)
    retired = {key: {u: v for u, v in store.get(key, {}).items() if u not in uidset}
               for key in ("topics2d", "topics3d_graph", "citations", "citations2d")}
    if any(retired.values()):
        new_store["retired"] = {k: v for k, v in retired.items() if v}
    # when each paper first appeared on the site ("launch" = already there at the first recorded build);
    # papers that appeared in the last NEW_DAYS days are marked "New"
    old_seen = store.get("first_seen", {})
    new_store["first_seen"] = {**{u: s for u, s in old_seen.items() if u not in uidset},   # kept, like "retired"
                               **{u: old_seen.get(u, fetched if old_seen else "launch") for u in uids}}
    def is_new(u):
        s = new_store["first_seen"][u]
        return s != "launch" and (date.fromisoformat(fetched) - date.fromisoformat(s)).days <= NEW_DAYS
    n_new = sum(is_new(u) for u in uids)
    if n_new:
        log(f"{n_new} paper(s) marked New (added to the site in the last {NEW_DAYS} days)")

    papers = []
    for n, r in enumerate(recs):
        au = []
        for nm in r["authors"]:
            k = aidx[canon[nm]]
            if k not in au:
                au.append(k)
        papers.append({
            "id": r["uid"], "t": r["title"], "ab": first_sentence(r["abstract"]) if r.get("excerpt") else r["abstract"],
            "abx": int(bool(r.get("excerpt"))), "nw": int(is_new(r["uid"])), "d": r["date"], "dq": int(r["dq"]),
            "y": int(r["date"][:4]), "au": au, "c": r["c"], "cs": r["cs"], "bib": r["bib"], "v": r["venue"],
            "lk": r["link"], "url": r["url"], "doi": r["doi"], "tg": tags[n],
            "x": round(float(pos2[n][0]), 4), "yy": round(float(pos2[n][1]), 4),
            "p3": [round(float(v) / 420, 4) for v in graph3[n]], "sim": similar[n], "c3": new_store["citations"][uids[n]], "c2": new_store["citations2d"][uids[n]],
            "k": topic[n], "r": refs[n], "rm": int(r["refmissing"]),
        })

    log(f"\n{len(papers)} papers, {len(authors)} authors, {sum(len(p['r']) for p in papers)} citation links")
    for k, name in enumerate(TOPIC_NAMES):
        mem = sorted((p for p in papers if p["k"] == k), key=lambda p: -p["c"])
        if k == UNASSIGNED and not mem:
            continue
        log(f"\n[{k}] {name}  ({len(mem)} papers)")
        for p in mem[:4]:
            log(f"      - {p['t'][:90]}")
    data = {"topics": TOPIC_NAMES, "topicsShort": TOPIC_SHORT, "authors": authors, "papers": papers,
            "co": {"1": [new_store["coauthors1"].get(a) for a in authors]},   # all authors
            "co2d": {"1": [new_store["coauthors1_2d"].get(a) for a in authors]},
            "refsrc": {"both": n_both, "s2": n_s2, "ads": n_ads}, "tagaxes": TAG_AXES,
            "fetched": fetched, "nads": sum(1 for p in papers if p["cs"] == 0),
            "proj": {"name": NAME, "pubs": PUBS_URL, "repo": REPO_URL, "home": HOMEPAGE}, "notes": notes}
    return data, new_store


# ----------------------------------------------------------------------------- link-preview image
def make_preview(data, template, path):
    """1200x630 image shown when the site link is shared: the 2D topic map plus title and counts.
    Colors are read from the page template (light theme) so they always match the site."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = template[template.index(":root {"):]
    root = root[:root.index("}")]
    col = {int(k): v for k, v in re.findall(r"--t(\d+): (#[0-9a-fA-F]{6})", root)}
    bg, ink, ink2 = (re.search(r"--%s: (#[0-9a-fA-F]{6})" % n, root).group(1) for n in ("bg", "ink", "ink-2"))
    papers = data["papers"]
    xy = np.array([[p["x"], p["yy"]] for p in papers])
    xy -= xy.mean(0)
    w, v = np.linalg.eigh(np.cov(xy.T))            # turn the long axis horizontal, as on the site
    xy = xy @ v[:, ::-1]
    size = np.array([(3 + 2.2 * np.log1p(p["c"])) ** 2 * 1.6 for p in papers])
    colors = [col.get(p["k"], "#8b9099") for p in papers]
    fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor=bg)
    ax = fig.add_axes([0.47, 0.06, 0.50, 0.88]); ax.set_facecolor(bg); ax.axis("off")
    ax.scatter(xy[:, 0], xy[:, 1], s=size * 9, c=colors, alpha=0.10, linewidths=0)   # soft glow
    ax.scatter(xy[:, 0], xy[:, 1], s=size, c=colors, alpha=0.95, linewidths=0.8, edgecolors=bg)
    pad = 0.06 * (xy.max(0) - xy.min(0))
    ax.set_xlim(xy[:, 0].min() - pad[0], xy[:, 0].max() + pad[0]); ax.set_ylim(xy[:, 1].min() - pad[1], xy[:, 1].max() + pad[1])
    ax.set_aspect("equal", adjustable="box")   # keep the map inside its own area, never behind the text
    fig.text(0.05, 0.74, NAME, fontsize=46, fontweight="bold", color=ink, family="DejaVu Sans")
    fig.text(0.05, 0.645, "Literature Atlas", fontsize=46, fontweight="bold", color=ink, family="DejaVu Sans")
    fig.text(0.05, 0.50, f"Every paper that uses the {NAME}\nsimulations, mapped by topic,\ncitations and collaborations.",
             fontsize=19, color=ink2, linespacing=1.45, va="top", family="DejaVu Sans")
    n_links = sum(len(p["r"]) for p in papers)
    fig.text(0.05, 0.20, f"{len(papers)} papers · {len(data['authors'])} authors · {n_links} citation links",
             fontsize=16, color=ink, family="DejaVu Sans")
    fig.text(0.05, 0.13, SITE_URL.replace("https://", "").rstrip("/"), fontsize=14, color=col.get(6, ink2), family="DejaVu Sans")
    fig.savefig(path, dpi=100, facecolor=bg, metadata={"Software": None})   # no version stamp: same data, same bytes
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True, help="project name: a folder in projects/ (e.g. camels, quijote)")
    ap.add_argument("--out", help="output HTML file (default: <project>/index.html, served by GitHub Pages)")
    ap.add_argument("--cache", help="folder for downloaded data (default: atlas_cache/<project>)")
    ap.add_argument("--offline", action="store_true", help="rebuild from the cached downloads, no network")
    ap.add_argument("--recluster", action="store_true",
                    help="redo the k-means topic clustering instead of using the fixed TOPIC_SEED assignments")
    ap.add_argument("--template", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "atlas.html"),
                    help="page template; its styles and code are read from atlas.css and atlas.js next to it")
    ap.add_argument("--overrides", help="file with hand corrections per paper (default: projects/<project>/overrides.json)")
    ap.add_argument("--tags", help="file with the suite/data/method tags of each paper (default: projects/<project>/tags.json)")
    ap.add_argument("--positions", help="file with the stored layout positions (default: projects/<project>/positions.json)")
    ap.add_argument("--relayout", action="store_true",
                    help="ignore the stored positions and compute all layouts from scratch")
    ap.add_argument("--force", action="store_true",
                    help="build even if the sanity checks fail (e.g. papers were really removed from the list)")
    args = ap.parse_args()
    pdir = os.path.join("projects", args.project)
    if not os.path.exists(os.path.join(pdir, "config.json")):
        sys.exit(f"no project {args.project!r}: {pdir}/config.json not found")
    configure(json.load(open(os.path.join(pdir, "config.json"), encoding="utf-8")))
    args.out = args.out or os.path.join(args.project, "index.html")
    args.cache = args.cache or os.path.join("atlas_cache", args.project)
    args.overrides = args.overrides or os.path.join(pdir, "overrides.json")
    args.tags = args.tags or os.path.join(pdir, "tags.json")
    args.positions = args.positions or os.path.join(pdir, "positions.json")
    os.makedirs(args.cache, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    entries = load_or_fetch(args.cache, "entries.json", fetch_entries, args.offline)
    ax_ids = [e["arxiv"] for e in entries if e["arxiv"]]
    bibs = [e["bibcode"] for e in entries if e["bibcode"]]
    arx = load_or_fetch(args.cache, "arxiv.json", lambda: fetch_arxiv(ax_ids), args.offline)
    ads_rec = load_or_fetch(args.cache, "ads_records.json", lambda: fetch_ads_records(bibs), args.offline)
    s2 = load_or_fetch(args.cache, "s2.json", lambda: fetch_semantic_scholar(entries, ads_rec), args.offline)
    ads_cit = load_or_fetch(args.cache, "ads.json", lambda: fetch_ads_citations(ax_ids), args.offline)
    all_bibs = sorted({v["bibcode"] for v in ads_cit.values()} | set(ads_rec))
    refs_path = os.path.join(args.cache, "ads_refs.json")
    if args.offline and not os.path.exists(refs_path):
        ads_refs = {}  # caches made before ADS references were used
    else:
        ads_refs = load_or_fetch(args.cache, "ads_refs.json", lambda: fetch_ads_references(all_bibs), args.offline)

    fetched = fetch_date(args.cache, args.offline)
    prev = previous_data(args.out)
    problems = sanity_check(entries, arx, s2, len(prev["papers"]) if prev else None)
    if problems:
        msg = "\n  - ".join(["sanity checks failed:"] + problems)
        if not args.force:
            sys.exit(f"{msg}\nNothing was written. Rerun with --force if this is expected.")
        log(f"{msg}\n--force given: building anyway.")

    store = {}
    if os.path.exists(args.positions):
        store = json.load(open(args.positions))
        if args.relayout:   # recompute layouts, but keep the record of when papers first appeared
            store = {"first_seen": store.get("first_seen", {})}
    overrides = json.load(open(args.overrides)) if os.path.exists(args.overrides) else {}
    data, new_store = build_data(entries, arx, s2, ads_rec, ads_cit, ads_refs, store, fetched, args.tags, overrides,
                                 recluster=args.recluster)
    problems = compare_with_previous(data, prev)
    if problems:
        msg = "\n  - ".join(["checks against the current site failed:"] + problems)
        if not args.force:
            sys.exit(f"{msg}\nNothing was written. Rerun with --force if this is expected.")
        log(f"{msg}\n--force given: publishing anyway.")
    # the downloads are kept only once both checks passed, so a failed fetch never replaces a good cache
    for path, d in PENDING_CACHE.items():
        with open(path, "w") as f:
            json.dump(d, f)
    # same papers, citations and positions as the current page: keep its date, so nothing changes
    strip = lambda d: json.dumps({k: v for k, v in d.items() if k != "fetched"}, sort_keys=True)
    if prev and strip(prev) == strip(data):
        data["fetched"] = prev["fetched"]
    blob = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    # page source: atlas.html with __CSS__ / __JS__ replaced by atlas.css / atlas.js (one self-contained file)
    base = os.path.splitext(args.template)[0]
    read = lambda path: open(path, encoding="utf-8").read()
    template = read(args.template).replace("__CSS__", read(base + ".css")).replace("__JS__", read(base + ".js"))
    template = template.replace("__NAME__", html_mod.escape(NAME)).replace("__HOME__", html_mod.escape(HOMEPAGE))
    html = template.replace("__DATA__", blob)
    # complete document: <head> with description, link-preview and analytics tags
    head, body = html.split("<header>", 1)
    n = len(data["papers"])
    desc = (f"Interactive map of the {n} papers that use the {NAME} simulations: "
            "topics, citation network and co-author network.")
    html = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            f'<meta name="description" content="{desc}">\n'
            f'<meta property="og:title" content="{NAME} Literature Atlas">\n'
            f'<meta property="og:description" content="{desc}">\n'
            '<meta property="og:type" content="website">\n'
            f'<meta property="og:url" content="{SITE_URL}">\n'
            f'<meta property="og:image" content="{SITE_URL}preview.png">\n'
            '<meta property="og:image:width" content="1200">\n<meta property="og:image:height" content="630">\n'
            '<meta name="twitter:card" content="summary_large_image">\n'
            + (f'<!-- Google tag (gtag.js) -->\n<script async src="https://www.googletagmanager.com/gtag/js?id={GA_ID}"></script>\n'
               f"<script>window.dataLayer = window.dataLayer || []; function gtag(){{dataLayer.push(arguments);}} "
               f"gtag('js', new Date()); gtag('config', '{GA_ID}');</script>\n" if GA_ID else "")
            + head.strip() + '\n<style>body{margin:0}</style>\n</head>\n<body>\n<header>' + body + '\n</body>\n</html>\n')
    with open(args.positions, "w") as f:
        json.dump(new_store, f, separators=(",", ":"), sort_keys=True)
    make_preview(data, template, os.path.join(os.path.dirname(os.path.abspath(args.out)), "preview.png"))
    old = open(args.out, encoding="utf-8").read() if os.path.exists(args.out) else None
    if old == html:
        log(f"\nno changes: {args.out} left as it was (data as of {data['fetched']}).")
        return
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    log(f"\nwrote {args.out} ({len(html) / 1e3:.0f} kB, data as of {data['fetched']}) and {args.positions}.")


if __name__ == '__main__':
    main()
