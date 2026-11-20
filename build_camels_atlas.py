#!/usr/bin/env python3
"""
CAMELS Literature Atlas builder
================================

Rebuilds the interactive "CAMELS Literature Atlas" web page (topic map, citation
network and co-author network, all in rotatable 3D) from scratch.

Pipeline
  1. Publications list   https://camels.readthedocs.io/en/latest/publications.html  (every entry)
  2. arXiv API           titles, abstracts, authors, dates for entries with an arXiv link
  3. NASA ADS            full records for entries that link to ADS instead of arXiv, and total
                         citation counts (token read from ~/.ads/dev_key or $ADS_TOKEN; without a
                         token, citation counts come from Semantic Scholar)
  4. Semantic Scholar    SPECTER2 embeddings + references (citation links between CAMELS papers),
                         matched by arXiv ID, DOI or exact title
  Entries with no arXiv or ADS record (e.g. theses) use the title and authors from the
  publications page; if no date is known, the year is estimated from the entry's position
  in the list (which is newest first) and marked as estimated on the page.
  5. Analysis            k-means topics (10, hand-named), 3D t-SNE, author-name merging
  6. Output              one self-contained HTML file (only Google Fonts is loaded from outside)

Usage
  python3 build_camels_atlas.py                 # fetch everything fresh, write camels_atlas.html
  python3 build_camels_atlas.py --offline       # reuse the cached downloads in ./atlas_cache
  python3 build_camels_atlas.py --out page.html --cache mycache
  python3 build_camels_atlas.py --artifact      # body-only HTML, for publishing as a claude.ai artifact
  python3 build_camels_atlas.py --recluster     # redo the topic clustering (then check/rename topics)

Layout positions are stored in atlas_positions.json (--positions) and reused on later builds, so the
maps keep their shape as papers are added; only new papers and authors are placed. --relayout
recomputes every layout from scratch.

Safety: before writing anything, the script checks that it found at least 90% as many papers as the
previous build in --out, that arXiv returned >= 95% of the requested records and that >= 90% of
entries have embeddings. If not, it stops without touching the page or the cache (--force overrides).

Requirements: Python 3.9+, requests, numpy, scipy, scikit-learn.

Topics: 10 k-means clusters of the SPECTER2 embeddings, named by hand after reading their titles
(Nov 2026). The topic of every paper checked then is stored in TOPIC_SEED, so the names keep
matching their papers. New papers join the topic whose centre (mean embedding) is nearest, or,
without an embedding, the topic most common among the papers with the most similar words.
--recluster redoes the clustering from scratch (then check the printed titles and edit TOPICS).
"""
import argparse
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

PUBS_URL = "https://camels.readthedocs.io/en/latest/publications.html"
UA = {"User-Agent": "camels-atlas-builder/1.0"}
N_TOPICS = 10
SEED = 0

# (name, probe terms used to match the name to a k-means cluster)
TOPICS = [
    ("HI & multifield maps", ["neutral hydrogen", "maps", "hi"]),
    ("Baryonic feedback: CGM, IGM, FRBs & SZ", ["agn feedback", "baryonic feedback", "feedback"]),
    ("Simulation-based inference methods", ["sbi", "loss", "inference neural"]),
    ("Primordial non-Gaussianity & analytic models", ["primordial non", "non gaussianity", "b_φ"]),
    ("Fast & differentiable simulators", ["particle mesh", "differentiable", "fast"]),
    ("Diffusion models & field generation", ["diffusion", "diffusion models", "generative"]),
    ("Halos, clusters & galaxy environments", ["haloes", "cluster", "ray"]),
    ("Galaxy properties & CAMELS releases", ["galaxy properties", "single galaxy", "properties"]),
    ("Field-level inference & graph networks", ["graph neural", "field level", "galaxy catalogs"]),
    ("Halo concentration & baryonic imprints", ["concentration", "concentration mass", "imprints"]),
]

# Papers that fit no topic get this extra index (shown in gray as "Unassigned (new papers)").
UNASSIGNED = len(TOPICS)
TOPIC_NAMES = [n for n, _ in TOPICS] + ["Unassigned (new papers)"]

# Hand-checked topic of every paper in the Nov 2026 clustering (arXiv ID -> index into TOPICS).
# These stay fixed so the hand-written names keep matching their papers. New papers join the topic
# whose centre (mean embedding) is nearest. Run with --recluster to redo the clustering from scratch.
TOPIC_SEED = {
    "2010.00619": 7, "2011.05992": 8, "2109.04484": 6, "2109.07070": 0, "2109.09747": 8, "2109.10360": 8,
    "2109.10915": 7, "2110.02983": 0, "2111.08683": 8, "2111.14874": 8, "2201.01300": 7, "2201.01305": 3,
    "2201.01663": 1, "2201.02202": 7, "2201.02708": 1, "2201.04142": 1, "2204.02408": 6, "2204.13713": 8,
    "2205.02245": 6, "2207.05509": 4, "2208.08927": 0, "2209.00657": 6, "2209.02075": 1, "2209.06843": 8,
    "2210.02467": 1, "2211.05000": 0, "2211.12444": 5, "2211.12724": 0, "2211.15675": 1, "2211.16461": 7,
    "2212.05964": 9, "2301.02186": 1, "2301.02231": 1, "2301.02253": 6, "2301.09655": 4, "2302.01363": 6,
    "2302.14101": 8, "2302.14591": 8, "2303.07473": 7, "2304.02096": 7, "2304.06084": 7, "2305.00992": 9,
    "2305.07531": 4, "2306.07728": 6, "2306.10943": 2, "2307.06360": 1, "2307.06967": 3, "2307.11832": 1,
    "2307.14362": 0, "2308.09751": 2, "2308.14822": 6, "2309.07912": 6, "2309.12048": 7, "2310.04499": 6,
    "2310.08634": 7, "2310.15234": 8, "2311.00799": 0, "2311.00833": 5, "2311.01588": 8, "2311.03491": 9,
    "2311.08558": 5, "2311.10088": 3, "2311.18007": 0, "2311.18017": 4, "2312.07534": 5, "2401.12318": 6,
    "2401.15891": 9, "2402.10997": 0, "2403.02313": 1, "2403.08611": 1, "2403.10609": 3, "2403.10648": 5,
    "2404.04327": 6, "2405.04447": 7, "2405.05255": 5, "2407.18349": 1, "2408.04864": 1, "2410.10942": 6,
    "2410.16361": 1, "2410.21722": 7, "2411.02496": 8, "2411.08957": 8, "2411.10515": 0, "2411.13960": 1,
    "2411.17682": 3, "2412.04559": 1, "2412.05131": 5, "2501.16709": 7, "2501.17922": 1, "2502.06954": 3,
    "2502.13239": 6, "2502.13589": 6, "2502.17568": 7, "2503.21736": 3, "2503.22654": 7, "2504.11794": 5,
    "2504.17839": 8, "2504.18745": 1, "2505.02920": 3, "2505.13620": 4, "2505.21215": 4, "2505.23904": 8,
    "2506.06087": 2, "2506.08783": 3, "2507.01820": 6, "2507.03707": 4, "2507.12530": 8, "2507.16816": 1,
    "2507.21682": 0, "2508.05744": 2, "2508.21152": 1, "2509.01881": 3, "2509.18260": 1, "2510.19224": 4,
    "2510.26593": 4, "2510.26851": 4, "2511.08438": 5, "2511.21881": 9, "2512.03038": 1, "2512.10222": 8,
    "2512.17964": 9, "2601.06258": 1, "2601.06290": 6, "2601.14377": 5, "2602.07651": 3, "2602.12357": 3,
    "2602.15938": 1, "2602.15951": 4, "2603.13011": 7, "2604.21790": 3, "2605.26210": 8, "2605.28516": 2,
    "2606.10024": 3, "2606.10038": 7, "2607.18229": 3, "2609.00807": 9, "2609.09504": 8, "2609.10709": 8,
    "2609.10710": 1, "2609.19295": 5, "2609.23982": 7, "2610.00725": 8, "2610.03544": 9, "2610.03946": 9,
    "2610.10741": 1, "2610.33955": 4, "2610.37825": 5, "2610.40218": 8,
}


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
        r = requests.request(method, url, headers=UA, timeout=120, **kw)
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


def ads_query(q, fl, rows=200):
    tok = ads_token()
    r = requests.get("https://api.adsabs.harvard.edu/v1/search/query", timeout=60,
                     headers=dict(UA, Authorization=f"Bearer {tok}"), params={"q": q, "fl": fl, "rows": rows})
    if r.status_code != 200:
        raise RuntimeError(f"ADS HTTP {r.status_code}: {r.text[:200]}")
    return r.json()["response"]["docs"]


def fetch_ads_records(bibcodes):
    """Full ADS records for entries that link to ADS instead of arXiv."""
    if not bibcodes or not ads_token():
        return {}
    q = "bibcode:(" + " OR ".join(f'"{b}"' for b in bibcodes) + ")"
    docs = ads_query(q, "bibcode,title,author,abstract,pubdate,citation_count,doi,identifier,pub")
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
    n_emb = sum(1 for p in s2 if p and p.get("embedding"))
    if n and n_emb < 0.9 * n:
        problems.append(f"Semantic Scholar embeddings for only {n_emb} of {n} entries (below 90%)")
    return problems


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
OVERRIDE_FIELDS = {"topic", "date", "abstract", "url", "link", "authors", "title", "hide", "comment"}


def apply_overrides(recs, overrides):
    """Apply hand corrections. Keys are an arXiv ID, an ADS bibcode or the paper title; values may set
    topic (a topic name), date ("YYYY", "YYYY-MM" or "YYYY-MM-DD"), abstract, url, link (label),
    authors (list), title, hide (true leaves the entry out) and comment (ignored).
    Returns the remaining records and {record index: topic index} for topic overrides."""
    by_key = {}
    for n, r in enumerate(recs):
        by_key[r["uid"]] = n
        by_key["title:" + norm_title(r["title"])] = n
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
    return keep, {n: topic_of[r["uid"]] for n, r in enumerate(keep) if r["uid"] in topic_of}


# ----------------------------------------------------------------------------- tags
# Tags on three axes, assigned by reading each abstract and stored in atlas_tags.json (--tags), which
# you can edit by hand: {id: {"title", "suite": [...], "data": [...], "method": [...], "uncertain": [...]}}.
TAG_AXES = {
    "suite": ["IllustrisTNG", "SIMBA", "Astrid", "Swift-EAGLE", "CROCODILE", "Magneticum", "Ramses",
              "N-body (dark matter only)", "CAMELS-SAM", "Zoom-ins (zoomGZ)", "CAMELS (suite not stated)",
              "Non-CAMELS simulations"],
    "data": ["2D maps", "3D fields", "Particle snapshots", "Halo & galaxy catalogs", "Merger trees",
             "Power spectra & summary statistics", "Profiles", "Synthetic observables"],
    "method": ["CNN", "Graph neural networks", "Simulation-based inference", "Normalizing flows", "Generative models",
               "Symbolic regression", "Emulators & regression ML", "Other ML", "No ML"],
}


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
    """2D and 3D topic-map positions. First build: t-SNE of the embeddings. Later builds: stored
    positions; new papers go to the mean position of their 3 most similar placed papers."""
    old2, old3 = store.get("topics2d", {}), store.get("topics3d", {})
    N, row = len(uids), {n: j for j, n in enumerate(emb)}
    pos2, pos3 = {}, {}
    known = [n for n in range(N) if uids[n] in old2 and uids[n] in old3]
    if known:
        for n in known:
            pos2[n], pos3[n] = np.array(old2[uids[n]]), np.array(old3[uids[n]])
    else:
        Y2, Y3 = tsne(X, 2), tsne(X, 3)
        for j, n in enumerate(emb):
            pos2[n], pos3[n] = Y2[j], Y3[j]
    placed = list(pos3)
    for n in range(N):
        if n in pos3:
            continue
        cand = [m for m in placed if m in row]
        if n in row and cand:      # by embedding similarity
            sims = X[[row[m] for m in cand]] @ X[row[n]]
        else:                      # no embedding: by title/abstract word similarity
            cand = placed
            sims = (T[cand] @ T[n].T).toarray().ravel()
        nn = [cand[i] for i in np.argsort(-sims)[:3]]
        jit = 0.03 * np.cos(np.arange(3) + n)
        pos3[n] = np.mean([pos3[m] for m in nn], 0) + jit
        pos2[n] = np.mean([pos2[m] for m in nn], 0) + jit[:2]
        log(f"placed next to similar papers: {titles[n][:70]}")
    return pos2, pos3


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
    for e, p in zip(entries, s2):
        p = p or {}
        ax, rec = e["arxiv"], ads_rec.get(e["bibcode"] or "")
        if ax and ax in arx:
            a = arx[ax]
            title, abstract, date_, authors = a["title"], a["abstract"], a["published"], a["authors"]
            link, url = f"arXiv:{ax}", f"https://arxiv.org/abs/{ax}"
        elif rec:
            title, abstract = (rec.get("title") or [e["title"]])[0], rec.get("abstract") or ""
            date_ = rec.get("pubdate", "")[:7].replace("-00", "")
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
        recs.append({"uid": ax or e["bibcode"] or "title:" + norm_title(e["title"])[:80], "title": title, "abstract": abstract,
                     "date": date_, "authors": authors or ["Unknown"], "link": link, "url": url,
                     "c": c, "cs": cs, "bib": bib, "venue": p.get("venue") or (rec or {}).get("pub") or "",
                     "s2": p, "dq": False,
                     # journal abstracts from ADS are the publisher's copyright: publish only an excerpt
                     "excerpt": bool(rec) and not (ax and ax in arx)})

    # ---- hand corrections from the overrides file (applied before anything is derived from them)
    recs, topic_override = apply_overrides(recs, overrides)

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
    if recluster:
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
    log(f"citation links: {n_both} in both sources, {n_s2} Semantic Scholar only, {n_ads} ADS only")

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
    tags = load_tags(tags_path, uids, [r["title"] for r in recs])
    pos2, pos3 = topic_positions(X, T, emb, uids, [r["title"] for r in recs], store)
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
    for mode, keep in (("2", [a for a in range(len(authors)) if npap[a] >= 2]), ("1", list(range(len(authors))))):
        loc = {a: i for i, a in enumerate(keep)}
        pl = [(loc[a], loc[b], c) for (a, b), c in pairs.items() if a in loc and b in loc]
        s_ = np.array([x[0] for x in pl], int)
        t_ = np.array([x[1] for x in pl], int)
        c_ = np.array([x[2] for x in pl], float)
        co_args = ([authors[a] for a in keep], s_, t_, 60 / np.sqrt(c_), np.minimum(0.6, 0.04 * c_),
                   2.5 + np.sqrt(np.array([npap[a] for a in keep], float)) * 2, 240 if mode == "2" else 70, 0.12)
        pos = network_positions(*co_args, store.get("coauthors" + mode, {}))
        pos2d = network_positions(*co_args, store.get("coauthors" + mode + "_2d", {}), dim=2)
        co[mode] = {a: pos[i] for i, a in enumerate(keep)}
        co2[mode] = {a: pos2d[i] for i, a in enumerate(keep)}
    rd = lambda v, nd=2: [round(float(x), nd) for x in v]
    new_store = {
        "topics2d": {uids[n]: rd(pos2[n], 4) for n in range(N)},
        "topics3d": {uids[n]: rd(pos3[n], 4) for n in range(N)},
        "citations": {uids[n]: rd(cit[n]) for n in range(N)},
        "coauthors2": {authors[a]: rd(v) for a, v in co["2"].items()},
        "coauthors1": {authors[a]: rd(v) for a, v in co["1"].items()},
        "citations2d": {uids[n]: rd(cit2[n]) for n in range(N)},
        "coauthors2_2d": {authors[a]: rd(v) for a, v in co2["2"].items()},
        "coauthors1_2d": {authors[a]: rd(v) for a, v in co2["1"].items()},
    }

    papers = []
    for n, r in enumerate(recs):
        au = []
        for nm in r["authors"]:
            k = aidx[canon[nm]]
            if k not in au:
                au.append(k)
        papers.append({
            "id": r["uid"], "t": r["title"], "ab": first_sentence(r["abstract"]) if r.get("excerpt") else r["abstract"],
            "abx": int(bool(r.get("excerpt"))), "d": r["date"], "dq": int(r["dq"]),
            "y": int(r["date"][:4]), "au": au, "c": r["c"], "cs": r["cs"], "bib": r["bib"], "v": r["venue"],
            "lk": r["link"], "url": r["url"], "tg": tags[n],
            "x": round(float(pos2[n][0]), 4), "yy": round(float(pos2[n][1]), 4),
            "p3": [round(float(v), 4) for v in pos3[n]], "c3": new_store["citations"][uids[n]], "c2": new_store["citations2d"][uids[n]],
            "k": topic[n], "r": refs[n],
        })

    log(f"\n{len(papers)} papers, {len(authors)} authors, {sum(len(p['r']) for p in papers)} citation links")
    for k, name in enumerate(TOPIC_NAMES):
        mem = sorted((p for p in papers if p["k"] == k), key=lambda p: -p["c"])
        if k == UNASSIGNED and not mem:
            continue
        log(f"\n[{k}] {name}  ({len(mem)} papers)")
        for p in mem[:4]:
            log(f"      - {p['t'][:90]}")
    data = {"topics": TOPIC_NAMES, "authors": authors, "papers": papers,
            "co": {m: [new_store["coauthors" + m].get(a) for a in authors] for m in ("1", "2")},
            "co2d": {m: [new_store["coauthors" + m + "_2d"].get(a) for a in authors] for m in ("1", "2")},
            "refsrc": {"both": n_both, "s2": n_s2, "ads": n_ads}, "tagaxes": TAG_AXES,
            "fetched": fetched, "nads": sum(1 for p in papers if p["cs"] == 0)}
    return data, new_store


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="camels_atlas.html", help="output HTML file (default: camels_atlas.html)")
    ap.add_argument("--cache", default="atlas_cache", help="folder for downloaded data (default: atlas_cache)")
    ap.add_argument("--offline", action="store_true", help="rebuild from the cached downloads, no network")
    ap.add_argument("--recluster", action="store_true",
                    help="redo the k-means topic clustering instead of using the fixed TOPIC_SEED assignments")
    ap.add_argument("--overrides", default="atlas_overrides.json",
                    help="file with hand corrections per paper (default: atlas_overrides.json)")
    ap.add_argument("--tags", default="atlas_tags.json",
                    help="file with the suite/data/method tags of each paper (default: atlas_tags.json)")
    ap.add_argument("--positions", default="atlas_positions.json",
                    help="file with the stored layout positions (default: atlas_positions.json)")
    ap.add_argument("--relayout", action="store_true",
                    help="ignore the stored positions and compute all layouts from scratch")
    ap.add_argument("--force", action="store_true",
                    help="build even if the sanity checks fail (e.g. papers were really removed from the list)")
    ap.add_argument("--artifact", action="store_true",
                    help="write the page body only (no <html>/<head>), for publishing as a claude.ai artifact")
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)

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
    for path, d in PENDING_CACHE.items():
        with open(path, "w") as f:
            json.dump(d, f)

    store = {}
    if not args.relayout and os.path.exists(args.positions):
        store = json.load(open(args.positions))
    overrides = json.load(open(args.overrides)) if os.path.exists(args.overrides) else {}
    data, new_store = build_data(entries, arx, s2, ads_rec, ads_cit, ads_refs, store, fetched, args.tags, overrides,
                                 recluster=args.recluster)
    # same papers, citations and positions as the current page: keep its date, so nothing changes
    strip = lambda d: json.dumps({k: v for k, v in d.items() if k != "fetched"}, sort_keys=True)
    if prev and strip(prev) == strip(data):
        data["fetched"] = prev["fetched"]
    blob = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    html = TEMPLATE.replace("__DATA__", blob)
    if not args.artifact:  # a complete document that opens directly in a browser or on GitHub Pages
        head, body = html.split("<header>", 1)
        head = head.replace('<meta name="claude:viewport-fit-sides" content="cover">\n', "")  # claude.ai-only
        n = len(data["papers"])
        desc = (f"Interactive 3D map of the {n} papers that use the CAMELS simulations: "
                "topics, citation network and co-author network.")
        html = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                f'<meta name="description" content="{desc}">\n'
                '<meta property="og:title" content="CAMELS Literature Atlas">\n'
                f'<meta property="og:description" content="{desc}">\n'
                '<meta property="og:type" content="website">\n'
                + head.strip() + '\n<style>body{margin:0}</style>\n</head>\n<body>\n<header>' + body + '\n</body>\n</html>\n')
    with open(args.positions, "w") as f:
        json.dump(new_store, f, separators=(",", ":"), sort_keys=True)
    old = open(args.out, encoding="utf-8").read() if os.path.exists(args.out) else None
    if old == html:
        log(f"\nno changes: {args.out} left as it was (data as of {data['fetched']}).")
        return
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    log(f"\nwrote {args.out} ({len(html) / 1e3:.0f} kB, data as of {data['fetched']}) and {args.positions}.")


# ----------------------------------------------------------------------------- page template
# The page (HTML + CSS + JS). __DATA__ is replaced by the JSON built above.
TEMPLATE = r'''<title>CAMELS Literature Atlas</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: one working screen — header + toolbar on top, canvas map left, detail panel right (stacks below on phones). */
:root {
  --bg: #f4f6f9; --surface: #ffffff; --ink: #12151c; --ink-2: #4c5361; --muted: #7d8492;
  --line: #e1e5eb; --edge: rgba(18,21,28,0.13); --edge-hi: #12151c; --accent: #2a78d6; --wash: rgba(42,120,214,0.08);
  --t0: #2a78d6; --t1: #eb6834; --t2: #1baf7a; --t3: #eda100; --t4: #e87ba4; --t5: #008300; --t6: #6250d6; --t7: #e34948; --t8: #8c6d1f; --t9: #00a5b5; --t10: #8b9099;
  --f-display: "Space Grotesk", "Helvetica Neue", Arial, sans-serif;
  --f-body: Inter, system-ui, -apple-system, "Segoe UI", sans-serif;
  --f-mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #0e1116; --surface: #161a21; --ink: #eef0f4; --ink-2: #b6bcc8; --muted: #858c99;
  --line: #2a303b; --edge: rgba(238,240,244,0.12); --edge-hi: #eef0f4; --accent: #3987e5; --wash: rgba(57,135,229,0.14);
  --t0: #3987e5; --t1: #d95926; --t2: #199e70; --t3: #c98500; --t4: #d55181; --t5: #008300; --t6: #9085e9; --t7: #e66767; --t8: #9a7b26; --t9: #0f9fae; --t10: #7b808a; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #0e1116; --surface: #161a21; --ink: #eef0f4; --ink-2: #b6bcc8; --muted: #858c99;
  --line: #2a303b; --edge: rgba(238,240,244,0.12); --edge-hi: #eef0f4; --accent: #3987e5; --wash: rgba(57,135,229,0.14);
  --t0: #3987e5; --t1: #d95926; --t2: #199e70; --t3: #c98500; --t4: #d55181; --t5: #008300; --t6: #9085e9; --t7: #e66767; --t8: #9a7b26; --t9: #0f9fae; --t10: #7b808a; color-scheme: dark; }

html, body { height: 100%; }
body { background: var(--bg); color: var(--ink); font-family: var(--f-body); font-size: 14px; line-height: 1.5;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
  padding-block: 14px 16px; box-sizing: border-box; display: flex; flex-direction: column; gap: 12px; }
*, *::before, *::after { box-sizing: border-box; }
button, input, select { font: inherit; color: inherit; }
[hidden] { display: none !important; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

header { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 12px 24px; }
.brand { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
h1 { font-family: var(--f-display); font-weight: 700; font-size: 22px; letter-spacing: -0.01em; margin: 0; text-wrap: balance; }
.stats { color: var(--ink-2); font-size: 13px; font-variant-numeric: tabular-nums; }
.stats b { color: var(--ink); font-weight: 600; }

.modes { display: inline-flex; background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 3px; gap: 2px; }
.modes button { border: 0; background: transparent; padding: 7px 14px; border-radius: 7px; cursor: pointer; font-weight: 500; color: var(--ink-2);
  display: inline-flex; align-items: center; gap: 8px; }
.modes button svg { width: 16px; height: 16px; flex: none; }
.modes button[aria-pressed="true"] { background: var(--ink); color: var(--surface); }
.modes button:not([aria-pressed="true"]):hover { background: var(--wash); color: var(--ink); }

.toolbar { display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center; }
.field { display: flex; align-items: center; gap: 6px; color: var(--ink-2); font-size: 13px; }
.search { flex: 1 1 260px; max-width: 380px; position: relative; }
.search input { width: 100%; padding: 7px 10px 7px 32px; border: 1px solid var(--line); border-radius: 8px; background: var(--surface); }
.search svg { position: absolute; left: 10px; top: 50%; transform: translateY(-50%); width: 15px; height: 15px; color: var(--muted); }
select { padding: 6px 8px; border: 1px solid var(--line); border-radius: 8px; background: var(--surface); font-variant-numeric: tabular-nums; }
.toolbar select { max-width: 190px; }
.tablewrap { position: absolute; inset: 0; overflow: auto; }
.ptable { width: 100%; border-collapse: collapse; font-size: 13px; }
.ptable caption { text-align: left; padding: 10px 12px; color: var(--muted); font-size: 12px; }
.ptable th { position: sticky; top: 0; background: var(--surface); text-align: left; font-weight: 600; font-size: 12px;
  color: var(--ink-2); border-bottom: 1px solid var(--line); padding: 0; z-index: 1; white-space: nowrap; }
.ptable th button { all: unset; display: block; padding: 9px 10px; cursor: pointer; }
.ptable th button:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
.ptable th[aria-sort="ascending"] button::after { content: " ▲"; font-size: 9px; }
.ptable th[aria-sort="descending"] button::after { content: " ▼"; font-size: 9px; }
.ptable td { padding: 7px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }
.ptable td.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.ptable td.tt { min-width: 260px; color: var(--ink); }
.ptable td.sm { color: var(--ink-2); font-size: 12px; min-width: 110px; }
.ptable tbody tr { cursor: pointer; }
.ptable tbody tr:hover { background: var(--wash); }
.ptable tbody tr[aria-selected="true"] { background: var(--wash); box-shadow: inset 3px 0 0 var(--accent); }
.ptable tbody tr:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
.ptable i { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
.tagrow { display: grid; grid-template-columns: 58px minmax(0, 1fr); gap: 8px; align-items: baseline; margin-bottom: 4px; }
.tagax { font-size: 12px; color: var(--muted); }
.tag { border: 1px solid var(--line); border-radius: 6px; padding: 1px 7px; font-size: 12px; color: var(--ink); background: var(--wash); }
.tag.unsure { background: transparent; border-style: dashed; color: var(--ink-2); }
.tagnote { font-size: 12px; margin-top: 4px; }
.chk { display: flex; align-items: center; gap: 6px; font-size: 13px; color: var(--ink-2); cursor: pointer; }
.ghost { border: 1px solid var(--line); background: var(--surface); padding: 6px 10px; border-radius: 8px; cursor: pointer; font-size: 13px; color: var(--ink-2); }
.ghost:hover { color: var(--ink); border-color: var(--muted); }

.legend { display: flex; flex-wrap: wrap; gap: 6px; }
.chip { display: inline-flex; align-items: center; gap: 7px; padding: 4px 10px 4px 8px; border-radius: 999px; border: 1px solid var(--line);
  background: var(--surface); cursor: pointer; font-size: 12.5px; color: var(--ink); }
.chip i { width: 10px; height: 10px; border-radius: 50%; flex: none; }
.chip[aria-pressed="false"] { color: var(--muted); }
.chip[aria-pressed="false"] i { background: transparent !important; box-shadow: inset 0 0 0 1.5px var(--muted); }
.chip .n { color: var(--muted); font-variant-numeric: tabular-nums; }

main { flex: 1; min-height: 0; display: grid; grid-template-columns: minmax(0, 1fr) 360px; gap: 12px; }
.stage { position: relative; background: var(--surface); border: 1px solid var(--line); border-radius: 12px; overflow: hidden; min-height: 420px; }
canvas { display: block; width: 100%; height: 100%; cursor: grab; touch-action: none; }
canvas.hovering { cursor: pointer; }
.stage-note { position: absolute; left: 12px; bottom: 10px; font-size: 12px; color: var(--muted); pointer-events: none;
  background: color-mix(in srgb, var(--surface) 85%, transparent); padding: 2px 6px; border-radius: 6px; }
.zoom { position: absolute; right: 10px; top: 10px; display: flex; flex-direction: column; gap: 4px; }
.zoom button { width: 30px; height: 30px; border: 1px solid var(--line); background: var(--surface); border-radius: 8px; cursor: pointer; font-size: 16px; line-height: 1; color: var(--ink-2); }
.tip { position: absolute; pointer-events: none; max-width: 300px; background: var(--ink); color: var(--surface); padding: 7px 10px; border-radius: 8px;
  font-size: 12.5px; line-height: 1.35; box-shadow: 0 4px 16px rgba(0,0,0,.18); }
.tip small { display: block; opacity: .75; margin-top: 3px; font-variant-numeric: tabular-nums; }

aside { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; overflow: auto; min-width: 0; padding: 18px 18px 22px;
  display: flex; flex-direction: column; gap: 14px; }
aside h2 { font-family: var(--f-display); font-weight: 600; font-size: 18px; line-height: 1.3; margin: 0; text-wrap: balance; }
aside h3 { font-size: 11px; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin: 0 0 6px; font-weight: 600; }
aside p { margin: 0; color: var(--ink-2); }
.eyebrow { display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--ink-2); }
.eyebrow i { width: 10px; height: 10px; border-radius: 50%; }
.meta { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 13px; color: var(--ink-2); font-variant-numeric: tabular-nums; }
.meta a, .mono { font-family: var(--f-mono); font-size: 12.5px; }
a { color: var(--accent); text-decoration: none; } a:hover { text-decoration: underline; }
.abstract { font-size: 13px; line-height: 1.55; color: var(--ink-2); }
.names { display: flex; flex-wrap: wrap; gap: 4px; }
.names button { border: 1px solid var(--line); background: transparent; border-radius: 6px; padding: 2px 7px; font-size: 12.5px; cursor: pointer; color: var(--ink); }
.names button:hover { background: var(--wash); border-color: var(--accent); }
.list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; }
.list button { width: 100%; text-align: left; border: 0; background: transparent; padding: 7px 6px; border-radius: 6px; cursor: pointer;
  display: grid; grid-template-columns: 10px minmax(0, 1fr) auto; gap: 9px; align-items: baseline; }
.list button:hover { background: var(--wash); }
.list i { width: 8px; height: 8px; border-radius: 50%; transform: translateY(-1px); }
.list .tt { font-size: 13px; color: var(--ink); line-height: 1.35; }
.list .num { font-size: 12px; color: var(--muted); font-variant-numeric: tabular-nums; white-space: nowrap; }
.back { align-self: flex-start; }
.how { display: grid; gap: 6px; font-size: 13px; color: var(--ink-2); }
.how div { display: grid; grid-template-columns: 74px 1fr; gap: 8px; }
.how b { color: var(--ink); font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: .06em; }
.bars { display: flex; height: 8px; border-radius: 4px; overflow: hidden; gap: 2px; }
footer { font-size: 12px; color: var(--muted); }

@media (max-width: 880px) {
  body { height: auto; }
  main { grid-template-columns: minmax(0, 1fr); }
  .stage { height: 62vh; min-height: 380px; }
  aside { overflow: visible; }
  .modes button span { display: none; }
  .modes button { padding: 8px 12px; }
  .legend { flex-wrap: nowrap; overflow-x: auto; padding-bottom: 4px; scrollbar-width: thin; }
  .chip { flex: none; }
  .tapcard:not([hidden]) { display: flex; }
}
@media (max-width: 600px) {
  .field label[for^="f-"] { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); }
  .toolbar .field:has(select[id^="f-"]) { flex: 1 1 30%; min-width: 0; }
  .toolbar select[id^="f-"] { width: 100%; max-width: none; }
  .search { flex-basis: 100%; max-width: none; }
}
@media (pointer: coarse) {
  .zoom button { width: 40px; height: 40px; }
  .chip, .ghost, .names button { min-height: 34px; }
}
.tapcard { display: none; position: absolute; left: 10px; right: 10px; bottom: 10px; gap: 10px; align-items: center;
  background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px; box-shadow: 0 4px 16px rgba(0,0,0,.12); }
.tapcard .tt { flex: 1; min-width: 0; font-size: 13px; line-height: 1.35; color: var(--ink); overflow: hidden;
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; }
.tapcard button { flex: none; }
@media (min-width: 881px) { .modes button span.short { display: none; } }
</style>

<header>
  <div class="brand">
    <h1>CAMELS Literature Atlas</h1>
    <div class="stats" id="stats"></div>
  </div>
  <div class="modes" role="group" aria-label="View">
    <button id="m-topics" data-view="topics" aria-pressed="true">
      <svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><circle cx="4" cy="5" r="2"/><circle cx="10" cy="3.5" r="1.5"/><circle cx="12" cy="10" r="2.5"/><circle cx="5" cy="11.5" r="1.5"/></svg>
      <span>Topic map</span></button>
    <button id="m-citations" data-view="citations" aria-pressed="false">
      <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><circle cx="3.5" cy="12" r="2"/><circle cx="12.5" cy="12" r="2"/><circle cx="8" cy="3.5" r="2"/><path d="M4.6 10.3 7 5.3M11.4 10.3 9 5.3M5.5 12h5"/></svg>
      <span>Citations</span></button>
    <button id="m-coauthors" data-view="coauthors" aria-pressed="false">
      <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><circle cx="5.5" cy="5" r="2.3"/><circle cx="11.5" cy="6" r="1.8"/><path d="M1.5 14c.5-2.7 2-4 4-4s3.5 1.3 4 4M10 10.3c.5-.2 1-.3 1.5-.3 1.6 0 2.7 1.2 3 3.5"/></svg>
      <span>Co-authors</span></button>
    <button id="m-table" data-view="table" aria-pressed="false">
      <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><rect x="2" y="2.5" width="12" height="11" rx="1.5"/><path d="M2 6h12M2 9.5h12M6.5 6v7.5"/></svg>
      <span>Table</span></button>
  </div>
</header>

<div class="toolbar">
  <label class="search"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><circle cx="7" cy="7" r="4.5"/><path d="m10.5 10.5 3.5 3.5"/></svg>
    <input id="q" type="search" placeholder="Search titles, authors or arXiv IDs" aria-label="Search papers"></label>
  <span class="field"><label for="y0">Years</label> <select id="y0"></select> <span aria-hidden="true">–</span> <select id="y1" aria-label="To year"></select></span>
  <span class="field"><label for="f-suite">Suite</label> <select id="f-suite"></select></span>
  <span class="field"><label for="f-data">Data</label> <select id="f-data"></select></span>
  <span class="field"><label for="f-method">Method</label> <select id="f-method"></select></span>
  <label class="chk" id="selfwrap" hidden><input type="checkbox" id="selfc" checked> Count self-citations</label>
  <label class="chk" id="minwrap" hidden><input type="checkbox" id="min2" checked> Only authors with 2+ papers</label>
  <button class="ghost" id="reset">Clear filters</button>
</div>
<div class="legend" id="legend" aria-label="Topics"></div>

<main>
  <div class="stage" id="stage">
    <canvas id="cv" aria-label="Interactive map of CAMELS publications" role="img"></canvas>
    <div class="zoom"><button id="zin" aria-label="Zoom in">+</button><button id="zout" aria-label="Zoom out">−</button><button id="zfit" aria-label="Fit to view">⤢</button><button id="zdim" aria-label="Switch to 2D view">2D</button><button id="zspin" aria-label="Pause rotation" aria-pressed="true" hidden>⏸</button></div>
    <div class="stage-note" id="note"></div>
    <div class="tip" id="tip" hidden></div>
    <div class="tablewrap" id="tablewrap" hidden></div>
    <div class="tapcard" id="tapcard" hidden><span class="tt" id="tapcard-t"></span><button class="ghost" id="tapcard-go">Details ↓</button><button class="ghost" id="tapcard-x" aria-label="Close">×</button></div>
  </div>
  <aside id="panel" aria-live="polite"></aside>
</main>
<footer id="foot"></footer>

<script type="application/json" id="atlas-data">__DATA__</script>
<script>
(() => {
const D = JSON.parse(document.getElementById('atlas-data').textContent);
const P = D.papers, TOP = D.topics;
const tex = s => s.replace(/\$([^$]*)\$/g, (m, x) => x
  .replace(/\\text\{([^}]*)\}/g, '$1').replace(/\\rm\s*/g, '').replace(/\\Omega/g, 'Ω').replace(/\\sigma/g, 'σ')
  .replace(/\\alpha/g, 'α').replace(/\\phi/g, 'φ').replace(/\\mu/g, 'μ').replace(/\\sim/g, '~').replace(/\\times/g, '×')
  .replace(/[{}\\]/g, '').replace(/\^/g, '').replace(/_/g, '')).replace(/\s+/g, ' ').trim();
P.forEach((p, i) => { p.i = i; p.t = tex(p.t); p.citers = []; });
P.forEach(p => p.r.forEach(j => P[j].citers.push(p.i)));
// self-citation: the citing and cited paper share at least one author
const shares = (i, j) => P[i].au.some(a => P[j].au.includes(a));
P.forEach(p => { p.citersInd = p.citers.filter(i => !shares(i, p.i)); });
const citeCount = p => S.selfc ? p.citers.length : p.citersInd.length;
const citeNote = () => S.selfc ? `${P.reduce((s, p) => s + p.r.length, 0)} citation links between CAMELS papers · drag to rotate`
  : `${P.reduce((s, p) => s + p.citersInd.length, 0)} of ${P.reduce((s, p) => s + p.r.length, 0)} citation links shown (papers without shared authors) · drag to rotate`;
const AU = D.authors.map((name, i) => ({ i, name, papers: [] }));
P.forEach(p => p.au.forEach(a => AU[a].papers.push(p.i)));
AU.forEach(a => {
  const c = new Array(TOP.length).fill(0); a.papers.forEach(j => c[P[j].k]++);
  a.tc = c; a.topic = c.indexOf(Math.max(...c)); a.lc = a.name.toLowerCase();
});
P.forEach(p => { p.hay = (p.t + ' ' + p.id + ' ' + p.au.map(a => AU[a].name).join(' ')).toLowerCase(); });
const EM = new Map();
P.forEach(p => { for (let x = 0; x < p.au.length; x++) for (let y = x + 1; y < p.au.length; y++) {
  const a = Math.min(p.au[x], p.au[y]), b = Math.max(p.au[x], p.au[y]), key = a * 10000 + b;
  if (!EM.has(key)) EM.set(key, { s: a, t: b, papers: [] }); EM.get(key).papers.push(p.i);
} });
const CE = []; P.forEach(p => p.r.forEach(j => CE.push({ s: p.i, t: j, self: shares(p.i, j) })));
const years = [...new Set(P.map(p => p.y))].sort();

// ---------- state ----------
const S = { view: 'topics', sel: null, hover: null, q: '', y0: years[0], y1: years[years.length - 1], on: new Set(TOP.map((_, k) => k)), min2: true, selfc: true, dim: 3, tag: { s: '', d: '', m: '' }, nTop: { topics: 25, citations: 25, coauthors: 25 } };
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
let C = {};
const readColors = () => { C = { surface: css('--surface'), ink: css('--ink'), ink2: css('--ink-2'), muted: css('--muted'), edge: css('--edge'), edgeHi: css('--edge-hi'), accent: css('--accent'), t: TOP.map((_, k) => css('--t' + k)) }; };

// ---------- filters ----------
let pOn = [], aOn = [], nShown = 0;
function applyFilters() {
  const q = S.q.trim().toLowerCase();
  const tagOk = p => ['s', 'd', 'm'].every(ax => !S.tag[ax] || (p.tg && p.tg[ax].includes(S.tag[ax])));
  pOn = P.map(p => p.y >= S.y0 && p.y <= S.y1 && S.on.has(p.k) && (!q || p.hay.includes(q)) && tagOk(p));
  const nameHit = q ? AU.filter(a => a.lc.includes(q)) : [];
  aOn = AU.map(a => nameHit.length ? a.lc.includes(q) : a.papers.some(j => pOn[j]));
  nShown = pOn.filter(Boolean).length;
}

// ---------- layouts (world coords) ----------
const L = {};
// centre on the mean and return a framing radius from the bulk (90th percentile), so far islands don't shrink the view
function centre3d(nodes) {
  const N = nodes.length, c = ['x3', 'y3', 'z3'].map(k => nodes.reduce((s, n) => s + n[k], 0) / N);
  nodes.forEach(n => { n.x3 -= c[0]; n.y3 -= c[1]; n.z3 -= c[2]; });
  const ds = nodes.map(n => Math.hypot(n.x3, n.y3, n.z3)).sort((a, b) => a - b);
  return Math.max(1, ds[Math.floor(0.9 * (N - 1))]);
}
const med = a => { const s = a.slice().sort((u, v) => u - v); return s[s.length >> 1]; };
function topicLayout() {
  if (S.dim === 2) {
    const nodes = P.map(p => ({ id: p.i, kind: 'p', x: p.x * 420, y: -p.yy * 420, r: Math.min(16, 3 + Math.sqrt(p.c) * 0.9) }));
    const labels = TOP.map((name, k) => { const pts = nodes.filter(n => P[n.id].k === k);
      return { k, name, x: med(pts.map(n => n.x)), y: med(pts.map(n => n.y)) }; }).filter(l => l.x !== undefined);
    return { nodes, links: [], labels, is3d: false };
  }
  const nodes = P.map(p => ({ id: p.i, kind: 'p', x3: p.p3[0] * 420, y3: -p.p3[1] * 420, z3: p.p3[2] * 420, r: Math.min(16, 3 + Math.sqrt(p.c) * 0.9) }));
  const R = centre3d(nodes);
  const labels = TOP.map((name, k) => {
    const pts = nodes.filter(n => P[n.id].k === k);
    return { k, name, x3: med(pts.map(n => n.x3)), y3: med(pts.map(n => n.y3)), z3: med(pts.map(n => n.z3)) };
  }).filter(l => l.x3 !== undefined);
  return { nodes, links: [], labels, R, is3d: true };
}
// Network positions are computed by the Python build script and shipped with the data.
function citationLayout() {
  const d2 = S.dim === 2;
  const nodes = P.map(p => ({ id: p.i, kind: 'p', r: 3 + Math.sqrt(citeCount(p)) * 2.2,
    ...(d2 ? { x: p.c2[0], y: p.c2[1] } : { x3: p.c3[0], y3: p.c3[1], z3: p.c3[2] }) }));
  const links = CE.map(e => ({ source: nodes[e.s], target: nodes[e.t], s: e.s, t: e.t, w: 0.45, self: e.self }));
  return d2 ? { nodes, links, labels: [], is3d: false } : { nodes, links, labels: [], R: centre3d(nodes), is3d: true };
}
function coauthorLayout(min2) {
  const d2 = S.dim === 2, pos = (d2 ? D.co2d : D.co)[min2 ? '2' : '1'], keep = AU.filter(a => pos[a.i]);
  const nodes = keep.map(a => ({ id: a.i, kind: 'a', r: 2.5 + Math.sqrt(a.papers.length) * 2,
    ...(d2 ? { x: pos[a.i][0], y: pos[a.i][1] } : { x3: pos[a.i][0], y3: pos[a.i][1], z3: pos[a.i][2] }) }));
  const byId = new Map(nodes.map(n => [n.id, n]));
  const links = [...EM.values()].filter(e => byId.has(e.s) && byId.has(e.t)).map(e => ({
    source: byId.get(e.s), target: byId.get(e.t), s: e.s, t: e.t, papers: e.papers,
    w: e.papers.reduce((s, j) => s + 1 / (P[j].au.length - 1), 0) }));  // Newman collaboration strength
  return d2 ? { nodes, links, labels: [], is3d: false } : { nodes, links, labels: [], R: centre3d(nodes), is3d: true };
}
// camera for the 3D view
const CAM = { yaw: 0.6, pitch: -0.25, spin: !matchMedia('(prefers-reduced-motion: reduce)').matches, pv: [0, 0, 0], snap: true };
// Rotation centre: the selected node; for a selection of the other kind, the mean of its related nodes
// (an author's papers in the paper views, a paper's authors in the co-author view); else the graph centre.
function pivotTarget(lay) {
  if (!S.sel || !lay.nodes.length) return [0, 0, 0];
  const isP = lay.nodes[0].kind === 'p', ids = new Set();
  if (S.sel.kind === (isP ? 'p' : 'a')) ids.add(S.sel.i);
  else if (S.sel.kind === 'a') AU[S.sel.i].papers.forEach(j => ids.add(j));
  else P[S.sel.i].au.forEach(a => ids.add(a));
  const ns = lay.nodes.filter(n => ids.has(n.id));
  if (!ns.length) return [0, 0, 0];
  return ['x3', 'y3', 'z3'].map(k => ns.reduce((s, n) => s + n[k], 0) / ns.length);
}
function project(lay) {
  // glide the rotation centre toward its target (instantly on view switch or with reduced motion)
  const tg = pivotTarget(lay), pv = CAM.pv, inst = CAM.snap || matchMedia('(prefers-reduced-motion: reduce)').matches;
  let moving = false;
  for (let i = 0; i < 3; i++) { const d = tg[i] - pv[i]; if (inst || Math.abs(d) < 0.05) pv[i] = tg[i]; else { pv[i] += d * 0.15; moving = true; } }
  CAM.snap = false; if (moving) draw();
  const cy = Math.cos(CAM.yaw), sy_ = Math.sin(CAM.yaw), cp = Math.cos(CAM.pitch), sp = Math.sin(CAM.pitch), Dc = lay.R * 3;
  for (const n of lay.nodes.concat(lay.labels)) {
    const X = n.x3 - pv[0], Y = n.y3 - pv[1], Z = n.z3 - pv[2];
    const x1 = X * cy + Z * sy_, z1 = -X * sy_ + Z * cy;
    const y2 = Y * cp - z1 * sp, z2 = Y * sp + z1 * cp, f = Dc / Math.max(Dc + z2, Dc * 0.3);
    n.x = x1 * f; n.y = y2 * f; n.s = Math.min(f, 2); n.dz = z2; n.fog = Math.max(0.35, Math.min(1, 0.45 + 0.55 * (1 - (z2 + lay.R) / (2 * lay.R))));
  }
}
const layout = () => {
  const key = (S.view === 'coauthors' ? 'co' + (S.min2 ? 2 : 1) : S.view) + '-' + S.dim + 'd';
  if (!L[key]) L[key] = S.view === 'topics' ? topicLayout() : S.view === 'citations' ? citationLayout() : coauthorLayout(S.min2);
  return L[key];
};

// ---------- canvas ----------
const cv = $('cv'), ctx = cv.getContext('2d'), stage = $('stage');
let W = 0, H = 0, raf = 0;
// ---------- zoom & pan (own small implementation: screen = (x, y) + k * world) ----------
const T = { k: 1, x: 0, y: 0, applyX(v) { return this.x + this.k * v; }, applyY(v) { return this.y + this.k * v; } };
const ZMIN = 0.05, ZMAX = 40;
let zAnim = 0, zTarget = null, userMoved = false;   // userMoved: stop auto-fitting once the user zooms or pans   // zTarget: where a running zoom animation will end
function setView2(k, x, y) { T.k = k; T.x = x; T.y = y; hideTip(); draw(); }
function animateTo(k, x, y, ms) {
  cancelAnimationFrame(zAnim);
  zTarget = null;
  if (!ms || matchMedia('(prefers-reduced-motion: reduce)').matches) return setView2(k, x, y);
  zTarget = { k, x, y };
  const k0 = T.k, x0 = T.x, y0 = T.y, t0 = performance.now();
  const step = now => {
    const u = Math.min(1, (now - t0) / ms), e = u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;
    setView2(k0 + (k - k0) * e, x0 + (x - x0) * e, y0 + (y - y0) * e);
    if (u < 1) zAnim = requestAnimationFrame(step); else zTarget = null;
  };
  zAnim = requestAnimationFrame(step);
}
function zoomAt(f, px, py, ms) {   // zoom by factor f keeping screen point (px, py) fixed
  const b = zTarget || T;            // build on an unfinished animation's end point, so quick clicks add up
  const k = Math.max(ZMIN, Math.min(ZMAX, b.k * f)), r = k / b.k;
  animateTo(k, px - (px - b.x) * r, py - (py - b.y) * r, ms);
}
cv.addEventListener('wheel', e => {
  e.preventDefault(); userMoved = true;
  const r = cv.getBoundingClientRect();
  zoomAt(Math.pow(2, -e.deltaY * (e.deltaMode ? 0.05 : 0.002)), e.clientX - r.left, e.clientY - r.top);
}, { passive: false });
const pointers = new Map();   // two-finger pinch zoom
let pinch = null;
cv.addEventListener('pointerdown', e => { pointers.set(e.pointerId, [e.clientX, e.clientY]); if (pointers.size === 2) pinch = null; });
cv.addEventListener('pointermove', e => {
  if (!pointers.has(e.pointerId)) return;
  pointers.set(e.pointerId, [e.clientX, e.clientY]);
  if (pointers.size !== 2) return;
  const [a, b] = [...pointers.values()], r = cv.getBoundingClientRect();
  const d = Math.hypot(a[0] - b[0], a[1] - b[1]), mx = (a[0] + b[0]) / 2 - r.left, my = (a[1] + b[1]) / 2 - r.top;
  if (pinch) { userMoved = true; zoomAt(d / pinch.d, mx, my); }
  pinch = { d };
});
const dropPointer = e => { pointers.delete(e.pointerId); if (pointers.size < 2) pinch = null; };
cv.addEventListener('pointerup', dropPointer); cv.addEventListener('pointercancel', dropPointer);
function resize() {
  const r = stage.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  W = r.width; H = r.height; cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); draw();
}
function fit(animate) {
  if (S.view === 'table') return;
  const lay = layout(), { nodes } = lay; if (!nodes.length || !W) return;
  const reduce0 = matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (lay.is3d) {
    animateTo(Math.min(W - 40, H - 40) / (2 * lay.R * 1.15), W / 2, H / 2, animate && !reduce0 ? 450 : 0); return;
  }
  const xs = nodes.map(n => n.x), ys = nodes.map(n => n.y);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const k = Math.min((W - 60) / (x1 - x0 || 1), (H - 60) / (y1 - y0 || 1));
  animateTo(k, W / 2 - k * (x0 + x1) / 2, H / 2 - k * (y0 + y1) / 2, animate && !reduce0 ? 450 : 0);
}
const draw = () => { if (!raf) raf = requestAnimationFrame(() => { raf = 0; render(); }); };

// which nodes/edges are emphasised by the current selection or hover
function focusSets() {
  const f = S.hover || S.sel; if (!f) return null;
  const papers = new Set(), authors = new Set();
  if (f.kind === 'p') { papers.add(f.i); P[f.i].au.forEach(a => authors.add(a));
    if (S.view === 'citations') { P[f.i].r.forEach(j => papers.add(j)); P[f.i].citers.forEach(j => papers.add(j)); } }
  else { authors.add(f.i); AU[f.i].papers.forEach(j => papers.add(j));
    if (S.view === 'coauthors') AU[f.i].papers.forEach(j => P[j].au.forEach(a => authors.add(a))); }
  return { f, papers, authors };
}
function render() {
  if (S.view === 'table') return;
  readColors();
  const lay = layout(), { nodes, links, labels } = lay;
  if (lay.is3d) project(lay);
  const rad = n => n.r * (n.s || 1);
  ctx.clearRect(0, 0, W, H); ctx.fillStyle = C.surface; ctx.fillRect(0, 0, W, H);
  const F = focusSets(), isP = S.view !== 'coauthors';
  const on = n => isP ? pOn[n.id] : aOn[n.id];
  const hot = n => !F || (isP ? F.papers.has(n.id) : F.authors.has(n.id));
  const sx = n => T.applyX(n.x), sy = n => T.applyY(n.y);
  // edges
  if (links.length) {
    // thin hairlines; opacity/width follow collaboration strength and depth, drawn in 10 buckets
    const NB = 10, paths = Array.from({ length: NB }, () => new Path2D());
    for (const l of links) { const a = l.source, b = l.target; if (!on(a) || !on(b) || (l.papers && !l.papers.some(j => pOn[j])) || (l.self && !S.selfc)) continue;
      const v = (0.05 + 0.55 * Math.min(1, l.w)) * ((a.fog ?? 1) + (b.fog ?? 1)) / 2, q = Math.min(NB - 1, Math.floor(v / 0.06));
      paths[q].moveTo(sx(a), sy(a)); paths[q].lineTo(sx(b), sy(b)); }
    ctx.strokeStyle = C.edgeHi;
    paths.forEach((p, q) => { ctx.globalAlpha = (q + 0.5) * 0.06 * (F ? 0.35 : 1); ctx.lineWidth = 0.4 + q * 0.12; ctx.stroke(p); });
    ctx.globalAlpha = 1;
  } else if (links.length) {
    ctx.lineWidth = 1; ctx.strokeStyle = C.edge; ctx.globalAlpha = F ? 0.45 : 1; ctx.beginPath();
    for (const l of links) { const a = l.source, b = l.target; if (!on(a) || !on(b)) continue;
      if (!isP && !l.papers.some(j => pOn[j])) continue;
      ctx.moveTo(sx(a), sy(a)); ctx.lineTo(sx(b), sy(b)); }
    ctx.stroke(); ctx.globalAlpha = 1;
  }
  if (links.length) {
    if (F) {
      ctx.strokeStyle = C.edgeHi; ctx.fillStyle = C.edgeHi;
      for (const l of links) { const a = l.source, b = l.target;
        const lit = isP ? (F.f.kind === 'p' ? (l.s === F.f.i || l.t === F.f.i) : (F.papers.has(l.s) && F.papers.has(l.t)))
                        : (F.f.kind === 'a' ? (l.s === F.f.i || l.t === F.f.i) : (F.authors.has(l.s) && F.authors.has(l.t)));
        if (!lit || (l.self && !S.selfc)) continue;
        const x1 = sx(a), y1 = sy(a), x2 = sx(b), y2 = sy(b);
        ctx.lineWidth = isP ? 1.4 : Math.min(4, 0.8 + 0.6 * l.papers.length);
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
        if (isP) { // arrowhead at the cited paper
          const ang = Math.atan2(y2 - y1, x2 - x1), rr = rad(b) + 2, ax = x2 - Math.cos(ang) * rr, ay = y2 - Math.sin(ang) * rr;
          ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(ax - 7 * Math.cos(ang - 0.4), ay - 7 * Math.sin(ang - 0.4));
          ctx.lineTo(ax - 7 * Math.cos(ang + 0.4), ay - 7 * Math.sin(ang + 0.4)); ctx.closePath(); ctx.fill(); }
      }
    }
  }
  // nodes: dimmed first, then emphasised on top
  const order = nodes.slice().sort(lay.is3d
    ? (a, b) => (F ? (on(a) && hot(a)) - (on(b) && hot(b)) : 0) || b.dz - a.dz
    : (a, b) => (on(a) && hot(a)) - (on(b) && hot(b)) || b.r - a.r);
  for (const n of order) {
    const x = sx(n), y = sy(n); if (x < -30 || y < -30 || x > W + 30 || y > H + 30) continue;
    const k = isP ? P[n.id].k : AU[n.id].topic, lit = on(n) && hot(n);
    ctx.globalAlpha = (!on(n) ? 0.08 : lit ? 1 : 0.22) * (lay.is3d && lit && !F ? n.fog : 1);
    ctx.beginPath(); ctx.arc(x, y, rad(n), 0, 6.2832); ctx.fillStyle = C.t[k]; ctx.fill();
    ctx.lineWidth = 1.5; ctx.strokeStyle = C.surface; ctx.stroke();
  }
  ctx.globalAlpha = 1;
  // selection rings
  const ring = (n, w) => { ctx.beginPath(); ctx.arc(sx(n), sy(n), rad(n) + 3, 0, 6.2832); ctx.lineWidth = w; ctx.strokeStyle = C.ink; ctx.stroke(); };
  const byId = new Map(nodes.map(n => [n.id, n]));
  if (S.sel && (S.sel.kind === 'p') === isP && byId.has(S.sel.i)) ring(byId.get(S.sel.i), 2);
  if (S.hover && (S.hover.kind === 'p') === isP && byId.has(S.hover.i)) ring(byId.get(S.hover.i), 1.25);
  // labels
  ctx.textBaseline = 'middle'; ctx.lineJoin = 'round';
  const boxes = [];
  const hits = b => boxes.some(o => b.x0 < o.x1 && b.x1 > o.x0 && b.y0 < o.y1 && b.y1 > o.y0);
  // place a label unless it collides with one already placed; `tries` lists vertical nudges to attempt
  const label = (txt, x, y, font, align, tries, force) => {
    ctx.font = font; ctx.textAlign = align || 'center';
    const w = ctx.measureText(txt).width, h = 15;
    for (const dy of tries || [0]) {
      const x0 = align === 'left' ? x : x - w / 2, b = { x0: x0 - 3, x1: x0 + w + 3, y0: y + dy - h / 2, y1: y + dy + h / 2 };
      if (!force && hits(b)) continue;
      boxes.push(b); ctx.lineWidth = 4; ctx.strokeStyle = C.surface; ctx.strokeText(txt, x, y + dy); ctx.fillStyle = C.ink; ctx.fillText(txt, x, y + dy); return;
    }
  };
  const fB = css('--f-body');
  if (S.view === 'topics' && !F) labels.forEach(l => { if (S.on.has(l.k)) label(l.name, T.applyX(l.x), T.applyY(l.y) - 24, '600 13px ' + fB, 'center', [0, -18, 18, -36, 36, -54, 54, -72, 72]); });
  if (S.view === 'coauthors') {
    const cand = nodes.filter(n => aOn[n.id] && (F ? F.authors.has(n.id) : AU[n.id].papers.length >= Math.max(4, 22 / T.k)))
      .sort((a, b) => (F && b.id === F.f.i) - (F && a.id === F.f.i) || AU[b.id].papers.length - AU[a.id].papers.length);
    cand.forEach(n => label(AU[n.id].name, sx(n) + rad(n) + 4, sy(n), (F && F.f.kind === 'a' && F.f.i === n.id ? '600 ' : '500 ') + '11.5px ' + fB, 'left', [0], F && n.id === F.f.i));
  }
  if (S.view === 'citations' && F && F.f.kind === 'p') {
    const cand = nodes.filter(n => n.id === F.f.i || (T.k > 1.6 && F.papers.has(n.id))).sort((a, b) => (b.id === F.f.i) - (a.id === F.f.i));
    cand.forEach(n => { const t = P[n.id].t; label(t.length > 48 ? t.slice(0, 46) + '…' : t, sx(n) + rad(n) + 5, sy(n), '500 11.5px ' + fB, 'left', [0], n.id === F.f.i); });
  }
}

// ---------- hit testing, tooltip ----------
function pick(mx, my) {
  const { nodes } = layout(), isP = S.view !== 'coauthors'; let best = null, bd = 1e9;
  for (const n of nodes) { if (!(isP ? pOn[n.id] : aOn[n.id])) continue;
    const d = Math.hypot(T.applyX(n.x) - mx, T.applyY(n.y) - my); if (d < n.r * (n.s || 1) + 5 && d < bd) { bd = d; best = n; } }
  return best ? { kind: isP ? 'p' : 'a', i: best.id } : null;
}
const tip = $('tip');
function hideTip() { tip.hidden = true; }
function showTip(h, mx, my) {
  if (h.kind === 'p') { const p = P[h.i], a = p.au.map(x => AU[x].name);
    tip.innerHTML = `${esc(p.t)}<small>${esc(TOP[p.k])}</small><small>${p.y} · ${esc(a[0] || '')}${a.length > 1 ? ' et al.' : ''} · ${p.cs === 2 ? 'citations unknown' : p.c + ' citations'}${S.view === 'citations' ? ` · cited by ${p.citers.length} CAMELS papers, ${p.citersInd.length} without shared authors` : ''}</small>`; }
  else { const a = AU[h.i]; tip.innerHTML = `${esc(a.name)}<small>${a.papers.length} CAMELS paper${a.papers.length > 1 ? 's' : ''}</small>`; }
  tip.hidden = false; const tw = tip.offsetWidth, th = tip.offsetHeight;
  tip.style.left = Math.min(W - tw - 8, mx + 14) + 'px'; tip.style.top = (my + th + 20 > H ? my - th - 12 : my + 14) + 'px';
}
cv.addEventListener('pointermove', e => {
  if (e.buttons) return; const r = cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top, h = pick(mx, my);
  const same = (h && S.hover && h.kind === S.hover.kind && h.i === S.hover.i) || (!h && !S.hover);
  S.hover = h; cv.classList.toggle('hovering', !!h); if (h) showTip(h, mx, my); else hideTip(); if (!same) draw();
});
cv.addEventListener('pointerleave', () => { S.hover = null; hideTip(); draw(); });
// 3D view: drag rotates (wheel/pinch zoom above), arrow keys rotate, gentle auto-spin
let rot = null, dragged = false;
cv.addEventListener('pointerdown', e => {
  dragged = false; if (e.button) return;
  rot = { x: e.clientX, y: e.clientY, x0: e.clientX, y0: e.clientY }; cv.setPointerCapture(e.pointerId);
});
cv.addEventListener('pointermove', e => {
  if (!rot || !e.buttons) return;
  if (pointers.size > 1) { rot.x = e.clientX; rot.y = e.clientY; dragged = true; return; }
  if (S.dim === 2) { cancelAnimationFrame(zAnim); zTarget = null; userMoved = true; T.x += e.clientX - rot.x; T.y += e.clientY - rot.y; }
  else { CAM.yaw += (e.clientX - rot.x) * 0.008; CAM.pitch = Math.max(-1.45, Math.min(1.45, CAM.pitch + (e.clientY - rot.y) * 0.008)); }
  rot.x = e.clientX; rot.y = e.clientY; if (Math.hypot(e.clientX - rot.x0, e.clientY - rot.y0) > 4) dragged = true;
  hideTip(); draw();
});
const endRot = () => { rot = null; };
cv.addEventListener('pointerup', endRot); cv.addEventListener('pointercancel', endRot);
cv.addEventListener('keydown', e => {
  const m = { ArrowLeft: [-0.12, 0], ArrowRight: [0.12, 0], ArrowUp: [0, -0.12], ArrowDown: [0, 0.12] }[e.key]; if (!m) return;
  e.preventDefault();
  if (S.dim === 2) { T.x -= m[0] * 300; T.y -= m[1] * 300; } else { CAM.yaw += m[0]; CAM.pitch = Math.max(-1.45, Math.min(1.45, CAM.pitch + m[1])); }
  draw();
});
// auto-rotation pauses after 20 s without interaction and while the canvas is off-screen (saves battery);
// any click, drag, scroll, key press or pointer movement over the map restarts it
const IDLE_MS = 20000;
let lastUse = performance.now(), onScreen = true;
['pointerdown', 'wheel', 'keydown'].forEach(ev => document.addEventListener(ev, () => { lastUse = performance.now(); }, { passive: true }));
cv.addEventListener('pointermove', () => { lastUse = performance.now(); }, { passive: true });
if ('IntersectionObserver' in window) new IntersectionObserver(es => { onScreen = es[0].isIntersecting; }).observe(stage);
(function spin() {
  if (CAM.spin && !rot && !S.hover && !document.hidden && S.view !== 'table' && S.dim === 3 && onScreen && performance.now() - lastUse < IDLE_MS) { CAM.yaw += 0.0025; render(); }
  requestAnimationFrame(spin);
})();
cv.addEventListener('click', e => { if (dragged) { dragged = false; return; } const r = cv.getBoundingClientRect(); select(pick(e.clientX - r.left, e.clientY - r.top)); });

// ---------- panel ----------
const panel = $('panel');
const dot = k => `<i style="background:var(--t${k})"></i>`;
const pItem = (j, sh) => `<li><button data-p="${j}">${dot(P[j].k)}<span class="tt">${esc(P[j].t)}</span><span class="num">${P[j].y}${sh ? ' · shared author' : ''}</span></button></li>`;
const VIEWINFO = {
  topics: ['Each dot is a paper, laid out in 3D. Papers with similar abstracts sit close together. Nearer papers look larger and brighter.', (D.nads ? 'Total citations (NASA ADS)' : 'Total citations (Semantic Scholar)'), 'Topic cluster'],
  citations: ['Each dot is a paper, laid out in 3D. A line means one CAMELS paper cites another. Select a paper to see arrows pointing to the work it cites. Nearer papers look larger and brighter.', 'Citations from other CAMELS papers', 'Topic cluster'],
  table: ['Every paper as a row of a table. Click a column header to sort; click a row (or select it with Tab and press Enter) to see its details. The search box and filters apply here too.', 'Not used', 'Topic shown by the colored dot'],
  coauthors: ['Each dot is an author, laid out in 3D. A line joins people who wrote a CAMELS paper together. Darker lines mean a closer collaboration: each shared paper counts more when it has fewer authors, so links from 40-author papers stay faint. Nearer authors look larger and brighter.', 'Number of CAMELS papers', 'Author’s most frequent topic'],
};
const USE_TXT = 'Drag to rotate (or use the arrow keys), scroll or pinch to zoom. Rotation pauses while you hover. Click a dot for details; your selection stays highlighted when you switch views.';
const USE2 = 'Drag to pan (or use the arrow keys), scroll or pinch to zoom. Click a dot for details; your selection stays highlighted when you switch views.';
const USE = { topics: USE_TXT, citations: USE_TXT, coauthors: USE_TXT };
function overview() {
  let [what, size, color] = VIEWINFO[S.view];
  if (S.dim === 2) what = what.replace(', laid out in 3D', '').replace(/ Nearer (papers|authors) look larger and brighter\./, '');
  if (S.view === 'citations' && !S.selfc) size += ', not counting papers that share an author (self-citations hidden)';
  let list = '';
  if (S.view === 'table') {
    panel.innerHTML = `<div><h3>How to read this view</h3><p>${what}</p></div><p>${nShown} of ${P.length} papers shown.</p>`;
    return;
  }
  if (S.view === 'coauthors') {
    const all = AU.filter(a => aOn[a.i]).sort((a, b) => b.papers.length - a.papers.length || a.name.localeCompare(b.name));
    const top = all.slice(0, S.nTop[S.view]), left = all.length - top.length;
    list = `<div><h3>Most frequent authors <span style="text-transform:none;letter-spacing:0;font-weight:400">· ${top.length} of ${all.length}</span></h3><ul class="list">${top.map((a, r) => `<li><button data-a="${a.i}">${dot(a.topic)}<span class="tt">${r + 1}. ${esc(a.name)}</span><span class="num">${a.papers.length} paper${a.papers.length > 1 ? 's' : ''}</span></button></li>`).join('')}</ul>
      ${left > 0 ? `<div style="display:flex;gap:8px;margin-top:8px"><button class="ghost" data-more="25">Show ${Math.min(25, left)} more</button><button class="ghost" data-more="all">Show all ${all.length}</button></div>` : ''}</div>`;
  } else {
    const key = S.view === 'citations' ? citeCount : p => p.c;
    const all = P.filter(p => pOn[p.i]).sort((a, b) => key(b) - key(a) || b.d.localeCompare(a.d));
    const top = all.slice(0, S.nTop[S.view]), left = all.length - top.length;
    list = `<div><h3>${S.view === 'citations' ? (S.selfc ? 'Most cited within CAMELS' : 'Most cited within CAMELS, no self-citations') : (D.nads ? 'Most cited overall (ADS)' : 'Most cited overall')} <span style="text-transform:none;letter-spacing:0;font-weight:400">· ${top.length} of ${all.length}</span></h3><ul class="list">${top.map((p, r) =>
      `<li><button data-p="${p.i}">${dot(p.k)}<span class="tt">${r + 1}. ${esc(p.t)}</span><span class="num">${key(p)}</span></button></li>`).join('')}</ul>
      ${left > 0 ? `<div style="display:flex;gap:8px;margin-top:8px"><button class="ghost" data-more="25">Show ${Math.min(25, left)} more</button><button class="ghost" data-more="all">Show all ${all.length}</button></div>` : ''}</div>`;
  }
  panel.innerHTML = `<div><h3>How to read this view</h3><p>${what}</p></div>
    <div class="how"><div><b>Size</b><span>${size}</span></div><div><b>Color</b><span>${color}</span></div>
    <div><b>Use</b><span>${S.dim === 2 && S.view !== 'table' ? USE2 : USE[S.view]}</span></div></div>${list}`;
}
function tagBlock(p) {
  if (!p.tg) return '<div><h3>Tags</h3><p>Not tagged yet.</p></div>';
  const row = (label, ax) => p.tg[ax].length ? `<div class="tagrow"><span class="tagax">${label}</span><span class="names">${p.tg[ax].map(t =>
    `<span class="tag${p.tg.u.includes(t) ? ' unsure' : ''}"${p.tg.u.includes(t) ? ' title="Uncertain: not clearly stated in the abstract"' : ''}>${esc(t)}${p.tg.u.includes(t) ? ' ?' : ''}</span>`).join('')}</span></div>` : '';
  return `<div><h3>Tags</h3>${row('Suite', 's')}${row('Data', 'd')}${row('Method', 'm')}${p.tg.n ? `<p class="tagnote">${esc(p.tg.n)}</p>` : ''}</div>`;
}
function paperPanel(i) {
  const p = P[i];
  const venue = p.v && p.v !== p.lk ? `<span>${esc(p.v)}</span>` : '';
  panel.innerHTML = `<button class="ghost back" data-back>← Overview</button>
    <div class="eyebrow">${dot(p.k)}${esc(TOP[p.k])}</div>
    <h2>${esc(p.t)}</h2>
    <div class="meta">${p.url ? `<a href="${esc(p.url)}" target="_blank" rel="noopener">${esc(p.lk)}</a>` : `<span>${esc(p.lk)}</span>`}<span>${p.dq ? p.y + ' (estimated from list position)' : p.d}</span>${venue}${p.bib ? `<a href="https://ui.adsabs.harvard.edu/abs/${encodeURIComponent(p.bib)}/abstract" target="_blank" rel="noopener">${p.c} citations (ADS)</a>` : `<span>${p.cs === 2 ? 'Citations unknown' : p.c + ' citations (Semantic Scholar)'}</span>`}</div>
    ${tagBlock(p)}
    <div><h3>${p.au.length} author${p.au.length > 1 ? 's' : ''}</h3><div class="names">${p.au.map(a => `<button data-a="${a}">${esc(AU[a].name)}</button>`).join('')}</div></div>
    <div><h3>Abstract</h3>${p.ab && p.abx ? `<p class="abstract">${esc(p.ab)} … <a href="https://ui.adsabs.harvard.edu/abs/${encodeURIComponent(p.bib)}/abstract" target="_blank" rel="noopener">Read the full abstract on ADS</a></p>` : p.ab ? `<p class="abstract">${esc(p.ab)}</p>` : '<p>No abstract available. Use the link above to read the full text.</p>'}</div>
    <div><h3>Cites ${p.r.length} CAMELS paper${p.r.length === 1 ? '' : 's'}</h3>${p.r.length ? `<ul class="list">${p.r.slice().sort((a, b) => P[b].d.localeCompare(P[a].d)).map(j => pItem(j, shares(j, p.i))).join('')}</ul>` : '<p>None found.</p>'}</div>
    <div><h3>Cited by ${p.citers.length} CAMELS paper${p.citers.length === 1 ? '' : 's'}${p.citers.length ? `, ${p.citersInd.length} without shared authors` : ''}</h3>${p.citers.length ? `<ul class="list">${p.citers.slice().sort((a, b) => P[b].d.localeCompare(P[a].d)).map(j => pItem(j, shares(j, p.i))).join('')}</ul>` : '<p>None yet.</p>'}</div>`;
}
function authorPanel(i) {
  const a = AU[i], co = new Map();
  a.papers.forEach(j => P[j].au.forEach(b => { if (b !== i) co.set(b, (co.get(b) || 0) + 1); }));
  const top = [...co].sort((x, y) => y[1] - x[1] || AU[x[0]].name.localeCompare(AU[y[0]].name)).slice(0, 12);
  const bars = a.tc.map((c, k) => c ? `<i title="${esc(TOP[k])}: ${c}" style="flex:${c};background:var(--t${k})"></i>` : '').join('');
  const tl = a.tc.map((c, k) => c ? `<span>${dot(k)} ${esc(TOP[k])} <span class="num">${c}</span></span>` : '').filter(Boolean);
  panel.innerHTML = `<button class="ghost back" data-back>← Overview</button>
    <div class="eyebrow">Author</div><h2>${esc(a.name)}</h2>
    <div class="meta"><span>${a.papers.length} CAMELS paper${a.papers.length > 1 ? 's' : ''}</span><span>${co.size} co-authors</span></div>
    <div><h3>Topics</h3><div class="bars">${bars}</div><div class="how" style="margin-top:8px">${tl.map(t => `<div style="grid-template-columns:1fr;gap:0" class="eyebrow">${t}</div>`).join('')}</div></div>
    ${top.length ? `<div><h3>Frequent co-authors</h3><div class="names">${top.map(([b, n]) => `<button data-a="${b}">${esc(AU[b].name)} · ${n}</button>`).join('')}</div></div>` : ''}
    <div><h3>Papers</h3><ul class="list">${a.papers.slice().sort((x, y) => P[y].d.localeCompare(P[x].d)).map(pItem).join('')}</ul></div>`;
}
function renderPanel() { if (!S.sel) overview(); else if (S.sel.kind === 'p') paperPanel(S.sel.i); else authorPanel(S.sel.i); panel.scrollTop = 0; }
panel.addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return;
  if (b.dataset.more) { const y = panel.scrollTop; S.nTop[S.view] = b.dataset.more === 'all' ? 1e9 : S.nTop[S.view] + 25; overview(); panel.scrollTop = y; }
  else if (b.dataset.back !== undefined) select(null);
  else if (b.dataset.p) select({ kind: 'p', i: +b.dataset.p });
  else if (b.dataset.a) select({ kind: 'a', i: +b.dataset.a });
});
function select(s) { S.sel = s; renderPanel(); draw(); if (S.view === 'table') markTableSelection(); tapCard(); }
// narrow screens: the details panel is below the map, so show a small card on the map after a tap
const narrow = () => matchMedia('(max-width: 880px)').matches;
function tapCard() {
  const c = $('tapcard');
  if (!S.sel || !narrow() || S.view === 'table') { c.hidden = true; return; }
  $('tapcard-t').textContent = S.sel.kind === 'p' ? P[S.sel.i].t : AU[S.sel.i].name + ' · ' + AU[S.sel.i].papers.length + ' papers';
  c.hidden = false;
}
$('tapcard-go').addEventListener('click', () => { panel.scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start' }); });
$('tapcard-x').addEventListener('click', () => { $('tapcard').hidden = true; });

// ---------- table view (accessible alternative to the canvas views)
const COLS = [
  { k: 't', label: 'Title', v: p => p.t, cls: 'tt' },
  { k: 'a', label: 'First author', v: p => p.au.length ? AU[p.au[0]].name : '', cls: 'sm' },
  { k: 'y', label: 'Year', v: p => p.d, show: p => p.dq ? p.y + '*' : p.y, cls: 'num' },
  { k: 'k', label: 'Topic', v: p => TOP[p.k], show: p => `<i style="background:var(--t${p.k})"></i>${esc(TOP[p.k])}`, cls: 'sm', html: true },
  { k: 'm', label: 'Methods', v: p => p.tg ? p.tg.m.join(', ') : '', cls: 'sm' },
  { k: 'c', label: 'Citations', v: p => p.cs === 2 ? -1 : p.c, show: p => p.cs === 2 ? '–' : p.c, cls: 'num', num: true },
  { k: 'ci', label: 'Cited in CAMELS', v: p => citeCount(p), cls: 'num', num: true },
];
S.sort = { k: 'c', dir: -1 };
function renderTable() {
  const col = COLS.find(c => c.k === S.sort.k);
  const rows = P.filter(p => pOn[p.i]).sort((a, b) => {
    const x = col.v(a), y = col.v(b);
    return (typeof x === 'number' ? x - y : String(x).localeCompare(String(y))) * S.sort.dir || b.c - a.c;
  });
  const head = COLS.map(c => `<th scope="col" aria-sort="${c.k === S.sort.k ? (S.sort.dir > 0 ? 'ascending' : 'descending') : 'none'}"><button data-sort="${c.k}">${c.label}</button></th>`).join('');
  const body = rows.map(p => `<tr tabindex="0" data-p="${p.i}" aria-selected="${!!(S.sel && S.sel.kind === 'p' && S.sel.i === p.i)}">${COLS.map(c => {
    const val = c.show ? c.show(p) : c.v(p);
    return `<td class="${c.cls}">${c.html ? val : esc(val)}</td>`; }).join('')}</tr>`).join('');
  $('tablewrap').innerHTML = `<table class="ptable"><caption>${rows.length} of ${P.length} papers. Click a header to sort. * year estimated.</caption><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}
function markTableSelection() {
  document.querySelectorAll('#tablewrap tbody tr').forEach(tr => tr.setAttribute('aria-selected', !!(S.sel && S.sel.kind === 'p' && S.sel.i === +tr.dataset.p)));
}
$('tablewrap').addEventListener('click', e => {
  const b = e.target.closest('button[data-sort]');
  if (b) { const k = b.dataset.sort; S.sort = { k, dir: S.sort.k === k ? -S.sort.dir : (COLS.find(c => c.k === k).num ? -1 : 1) }; renderTable();
    const nb = document.querySelector(`#tablewrap button[data-sort="${k}"]`); if (nb) nb.focus(); return; }
  const tr = e.target.closest('tr[data-p]'); if (tr) select({ kind: 'p', i: +tr.dataset.p });
});
$('tablewrap').addEventListener('keydown', e => {
  const tr = e.target.closest('tr[data-p]'); if (!tr) return;
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select({ kind: 'p', i: +tr.dataset.p }); }
  if (e.key === 'ArrowDown' || e.key === 'ArrowUp') { e.preventDefault(); const n = e.key === 'ArrowDown' ? tr.nextElementSibling : tr.previousElementSibling; if (n) n.focus(); }
});

// ---------- controls ----------
function setView(v, animate) {
  S.view = v; CAM.snap = true; ['topics', 'citations', 'coauthors', 'table'].forEach(x => $('m-' + x).setAttribute('aria-pressed', x === v));
  $('minwrap').hidden = v !== 'coauthors'; $('selfwrap').hidden = v !== 'citations'; $('zspin').hidden = S.dim === 2; spinBtn();
  const tbl = v === 'table';
  cv.hidden = tbl; $('note').hidden = tbl; document.querySelector('.zoom').hidden = tbl; $('tablewrap').hidden = !tbl;
  if (tbl) renderTable();
  cv.tabIndex = 0;
  $('note').textContent = noteText(v);
  try { if (location.hash !== '#' + v) history.replaceState(null, '', '#' + v); } catch (e) {}
  S.hover = null; hideTip(); userMoved = false; fit(animate); if (!S.sel) overview(); draw(); updateStats(); tapCard();
}
document.querySelectorAll('.modes button').forEach(b => b.addEventListener('click', () => setView(b.dataset.view, true)));
window.addEventListener('hashchange', () => { const v = location.hash.slice(1); if (VIEWINFO[v] && v !== S.view) setView(v, true); });
const legend = $('legend');
function renderLegend() {
  legend.innerHTML = TOP.map((t, k) => !P.some(p => p.k === k) ? '' : `<button class="chip" data-k="${k}" aria-pressed="${S.on.has(k)}"><i style="background:var(--t${k})"></i>${esc(t)} <span class="n">${P.filter(p => p.k === k).length}</span></button>`).join('');
}
legend.addEventListener('click', e => { const b = e.target.closest('.chip'); if (!b) return; const k = +b.dataset.k;
  if (e.altKey || e.metaKey) { S.on = new Set([k]); } else if (S.on.has(k)) S.on.delete(k); else S.on.add(k);
  renderLegend(); refresh(); });
[$('y0'), $('y1')].forEach((sel, n) => { sel.innerHTML = years.map(y => `<option ${y === (n ? S.y1 : S.y0) ? 'selected' : ''}>${y}</option>`).join('');
  sel.addEventListener('change', () => { S.y0 = +$('y0').value; S.y1 = +$('y1').value; if (S.y0 > S.y1) { [S.y0, S.y1] = [S.y1, S.y0]; $('y0').value = S.y0; $('y1').value = S.y1; } refresh(); }); });
let qt; $('q').addEventListener('input', e => { clearTimeout(qt); qt = setTimeout(() => { S.q = e.target.value; refresh(); }, 120); });
const noteText = v => (v === 'topics' ? `Layout: ${S.dim}D t-SNE of SPECTER2 abstract embeddings · axes have no units`
  : v === 'citations' ? citeNote().replace(' · drag to rotate', '') : `${AU.length} authors · ${EM.size} co-author pairs`)
  + (S.dim === 3 ? ' · drag to rotate' : ' · drag to pan');
$('zdim').addEventListener('click', () => {
  S.dim = S.dim === 3 ? 2 : 3;
  const b = $('zdim'); b.textContent = S.dim === 3 ? '2D' : '3D'; b.setAttribute('aria-label', `Switch to ${S.dim === 3 ? '2D' : '3D'} view`);
  $('zspin').hidden = S.dim === 2; CAM.snap = true;
  $('note').textContent = noteText(S.view); S.hover = null; hideTip(); userMoved = false; fit(false); if (!S.sel) overview(); draw();
});
$('min2').addEventListener('change', e => { S.min2 = e.target.checked; fit(false); draw(); });
$('selfc').addEventListener('change', e => {
  S.selfc = e.target.checked;
  Object.keys(L).filter(k => k.startsWith('citations')).forEach(k => L[k].nodes.forEach(n => { n.r = 3 + Math.sqrt(citeCount(P[n.id])) * 2.2; }));
  $('note').textContent = noteText('citations'); refresh(); if (S.sel) renderPanel();
});
$('reset').addEventListener('click', () => { S.q = ''; $('q').value = ''; S.y0 = years[0]; S.y1 = years[years.length - 1]; $('y0').value = S.y0; $('y1').value = S.y1;
  S.on = new Set(TOP.map((_, k) => k)); S.tag = { s: '', d: '', m: '' }; ['suite', 'data', 'method'].forEach(a => { $('f-' + a).value = ''; });
  renderLegend(); refresh(); });
// tag filters: one dropdown per axis, with paper counts
[['suite', 's'], ['data', 'd'], ['method', 'm']].forEach(([axis, ax]) => {
  const sel = $('f-' + axis), names = { suite: 'All suites', data: 'All data', method: 'All methods' };
  const n = t => P.filter(p => p.tg && p.tg[ax].includes(t)).length;
  sel.innerHTML = `<option value="">${names[axis]}</option>` + (D.tagaxes ? D.tagaxes[axis] : []).filter(n)
    .map(t => `<option value="${esc(t)}">${esc(t)} (${n(t)})</option>`).join('');
  sel.addEventListener('change', () => { S.tag[ax] = sel.value; refresh(); });
});
$('zin').addEventListener('click', () => { userMoved = true; zoomAt(1.5, W / 2, H / 2, 200); });
$('zout').addEventListener('click', () => { userMoved = true; zoomAt(1 / 1.5, W / 2, H / 2, 200); });
$('zfit').addEventListener('click', () => { userMoved = false; fit(true); });
const spinBtn = () => { const b = $('zspin'); b.textContent = CAM.spin ? '⏸' : '⟳'; b.setAttribute('aria-label', CAM.spin ? 'Pause rotation' : 'Start rotation'); b.setAttribute('aria-pressed', CAM.spin); };
$('zspin').addEventListener('click', () => { CAM.spin = !CAM.spin; spinBtn(); });
function updateStats() {
  const filtered = nShown < P.length;
  $('stats').innerHTML = `<b>${filtered ? nShown + ' of ' : ''}${P.length}</b> papers · <b>${AU.length}</b> authors · <b>${CE.length}</b> citation links · ${years[0]}–${years[years.length - 1]}`;
}
function refresh() { applyFilters(); updateStats(); if (!S.sel) overview(); if (S.view === 'table') renderTable(); draw(); }
$('foot').innerHTML = `Papers listed on the <a href="https://camels.readthedocs.io/en/latest/publications.html" target="_blank" rel="noopener">CAMELS publications page</a> (all ${P.length} entries). Abstracts and authors from arXiv, or from ADS and the publications page for entries without an arXiv ID; ${D.nads ? 'citation counts from NASA ADS (Semantic Scholar where ADS has no record)' : 'citation counts from Semantic Scholar'}; SPECTER2 embeddings from Semantic Scholar; citation links between papers from ${D.refsrc && D.refsrc.ads ? `Semantic Scholar and ADS combined (${D.refsrc.both} in both, ${D.refsrc.s2} only in Semantic Scholar, ${D.refsrc.ads} only in ADS)` : 'Semantic Scholar'}; data as of ${D.fetched}. Topics are k-means clusters of the embeddings, named by hand. Suite, data and method tags were assigned with AI assistance from each abstract and have not all been checked by hand; dashed tags marked "?" are uncertain.`;

// ---------- boot ----------
renderLegend(); applyFilters(); renderPanel();
new ResizeObserver(() => { resize(); if (!userMoved) fit(false); }).observe(stage);
const mq = matchMedia('(prefers-color-scheme: dark)'); mq.addEventListener && mq.addEventListener('change', draw);
new MutationObserver(draw).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
const h = location.hash.slice(1); setView(VIEWINFO[h] ? h : 'topics', false);
document.fonts && document.fonts.ready.then(draw);
})();
</script>
'''


if __name__ == '__main__':
    main()
