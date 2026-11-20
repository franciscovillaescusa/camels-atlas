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
  4. Semantic Scholar    SPECTER2 embeddings + references (citation links between CAMELS papers;
                         an optional free API key in $S2_API_KEY avoids rate limits),
                         matched by arXiv ID, DOI or exact title
  Entries with no arXiv or ADS record (e.g. theses) use the title and authors from the
  publications page; if no date is known, the year is estimated from the entry's position
  in the list (which is newest first) and marked as estimated on the page.
  5. Analysis            k-means topics (10, hand-named), 3D t-SNE, author-name merging
  6. Output              one self-contained HTML file built from atlas_template.html (the page's
                         HTML, CSS and JavaScript; __DATA__ is replaced by the data). Only Google
                         Fonts is loaded from outside.

Usage
  python3 build_camels_atlas.py                 # fetch everything fresh, write index.html
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
# short forms used where space is tight (e.g. the table view); same order as TOPIC_NAMES
TOPIC_SHORT = ["HI & multifield", "Baryonic feedback", "SBI methods", "Primordial NG", "Fast simulators",
               "Diffusion models", "Halos & clusters", "Galaxy properties", "Field-level inference",
               "Halo concentration", "Unassigned"]

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
        best = pool[np.argsort(-sim[i, pool])[:n_best]]
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
    tags = load_tags(tags_path, uids, [r["title"] for r in recs])
    pos2, pos3 = topic_positions(X, T, emb, uids, [r["title"] for r in recs], store)  # pos3 (3D t-SNE) no longer shown
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
        "topics3d_graph": {uids[n]: rd(graph3[n]) for n in range(N)},
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
            "co": {m: [new_store["coauthors" + m].get(a) for a in authors] for m in ("1", "2")},
            "co2d": {m: [new_store["coauthors" + m + "_2d"].get(a) for a in authors] for m in ("1", "2")},
            "refsrc": {"both": n_both, "s2": n_s2, "ads": n_ads}, "tagaxes": TAG_AXES,
            "fetched": fetched, "nads": sum(1 for p in papers if p["cs"] == 0)}
    return data, new_store


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="index.html", help="output HTML file (default: index.html, the GitHub Pages site)")
    ap.add_argument("--cache", default="atlas_cache", help="folder for downloaded data (default: atlas_cache)")
    ap.add_argument("--offline", action="store_true", help="rebuild from the cached downloads, no network")
    ap.add_argument("--recluster", action="store_true",
                    help="redo the k-means topic clustering instead of using the fixed TOPIC_SEED assignments")
    ap.add_argument("--template", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "atlas_template.html"),
                    help="page template (HTML + CSS + JS); default: atlas_template.html next to this script")
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
    with open(args.template, encoding="utf-8") as f:
        html = f.read().replace("__DATA__", blob)
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


if __name__ == '__main__':
    main()
