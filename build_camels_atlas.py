#!/usr/bin/env python3
"""
CAMELS Literature Atlas builder
================================

Rebuilds the interactive "CAMELS Literature Atlas" web page (topic map, citation
network and co-author network, all in rotatable 3D) from scratch.

Pipeline
  1. Publications list   https://camels.readthedocs.io/en/latest/publications.html  (arXiv IDs)
  2. arXiv API           titles, abstracts, authors, dates
  3. Semantic Scholar    SPECTER2 embeddings + references (citation links between CAMELS papers)
  4. NASA ADS            total citation counts (token read from ~/.ads/dev_key or $ADS_TOKEN);
                         falls back to Semantic Scholar counts if no token is found
  5. Analysis            k-means topics (10, hand-named), 3D t-SNE, author-name merging
  6. Output              one self-contained HTML file (loads d3 from cdnjs and Google Fonts)

Usage
  python3 build_camels_atlas.py                 # fetch everything fresh, write camels_atlas.html
  python3 build_camels_atlas.py --offline       # reuse the cached downloads in ./atlas_cache
  python3 build_camels_atlas.py --out page.html --cache mycache
  python3 build_camels_atlas.py --artifact      # body-only HTML, for publishing as a claude.ai artifact

Requirements: Python 3.9+, requests, numpy, scipy, scikit-learn.

Topic names: the 10 names below were written by hand after reading the titles in each
cluster (Nov 2026). They are matched to the k-means clusters through a few probe keywords each.
When many new papers are added the clusters can shift; the script prints the top titles of
each topic so you can check that the names still fit and edit TOPICS if needed.
"""
import argparse
import json
import os
import re
import sys
import time
import unicodedata
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


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------- fetching
def fetch_arxiv_ids():
    html = requests.get(PUBS_URL, headers=UA, timeout=60).text
    ids = []
    for m in re.finditer(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})", html):
        if m.group(1) not in ids:
            ids.append(m.group(1))
    log(f"publications page: {len(ids)} unique arXiv IDs")
    return ids


def fetch_arxiv(ids):
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = {}
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        r = requests.get("http://export.arxiv.org/api/query", headers=UA, timeout=120,
                         params={"id_list": ",".join(chunk), "max_results": len(chunk)})
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


def fetch_semantic_scholar(ids):
    fields = "title,year,venue,citationCount,externalIds,references.externalIds,embedding.specter_v2"
    papers = []
    for i in range(0, len(ids), 100):
        chunk = ids[i:i + 100]
        for attempt in range(8):
            r = requests.post("https://api.semanticscholar.org/graph/v1/paper/batch", params={"fields": fields},
                              json={"ids": [f"ARXIV:{a}" for a in chunk]}, headers=UA, timeout=120)
            if r.status_code == 200:
                break
            log(f"Semantic Scholar HTTP {r.status_code}, retrying...")
            time.sleep(5 * (attempt + 1))
        r.raise_for_status()
        papers.extend(r.json())
        log(f"Semantic Scholar: {len(papers)}/{len(ids)}")
        time.sleep(3)
    return {"ids": ids, "papers": papers}


def ads_token():
    tok = os.environ.get("ADS_TOKEN")
    path = os.path.expanduser("~/.ads/dev_key")
    if not tok and os.path.exists(path):
        tok = open(path).read().strip()
    return tok or None


def fetch_ads(ids):
    tok = ads_token()
    if not tok:
        log("ADS: no token in $ADS_TOKEN or ~/.ads/dev_key -> using Semantic Scholar citation counts")
        return {}
    H = dict(UA, Authorization=f"Bearer {tok}")
    out = {}
    for i in range(0, len(ids), 40):
        chunk = ids[i:i + 40]
        q = "identifier:(" + " OR ".join(f'"arXiv:{a}"' for a in chunk) + ")"
        r = requests.get("https://api.adsabs.harvard.edu/v1/search/query", headers=H, timeout=60,
                         params={"q": q, "fl": "bibcode,identifier,citation_count", "rows": 200})
        if r.status_code != 200:
            log(f"ADS HTTP {r.status_code}: {r.text[:200]} -> using Semantic Scholar counts")
            return {}
        want = set(chunk)
        for d in r.json()["response"]["docs"]:
            for ident in d.get("identifier", []):
                a = ident.split(":", 1)[1] if ident.lower().startswith("arxiv:") else ident
                if a in want:
                    c = d.get("citation_count") or 0
                    if a not in out or c > out[a]["citations"]:
                        out[a] = {"bibcode": d["bibcode"], "citations": c}
        time.sleep(1)
    log(f"ADS: matched {len(out)}/{len(ids)}")
    return out


def load_or_fetch(cache, name, fn, offline):
    path = os.path.join(cache, name)
    if offline:
        if not os.path.exists(path):
            sys.exit(f"--offline: missing cached file {path}; run once without --offline first")
        return json.load(open(path))
    data = fn()
    json.dump(data, open(path, "w"))
    return data


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


def build_data(arx, s2, ads):
    ids = [i for i in s2["ids"] if i in arx]
    S = {i: p for i, p in zip(s2["ids"], s2["papers"])}
    idx = {i: n for n, i in enumerate(ids)}

    emb_ids = [i for i in ids if S.get(i) and S[i].get("embedding")]
    X = np.array([S[i]["embedding"]["vector"] for i in emb_ids])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    L = KMeans(N_TOPICS, n_init=50, random_state=SEED).fit(X).labels_

    # name clusters: match probe keywords to each cluster's distinctive TF-IDF terms
    docs = [arx[i]["title"] + " " + arx[i]["abstract"] for i in ids]
    tv = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.5, sublinear_tf=True)
    T = tv.fit_transform(docs)
    V = {v: n for n, v in enumerate(tv.get_feature_names_out())}
    Te = T[[idx[i] for i in emb_ids]]
    labs = sorted(set(L))
    score = np.zeros((len(labs), len(TOPICS)))
    for a, lab in enumerate(labs):
        m = L == lab
        diff = np.asarray(Te[m].mean(0)).ravel() - np.asarray(Te[~m].mean(0)).ravel()
        for b, (_, probes) in enumerate(TOPICS):
            score[a, b] = sum(diff[V[p]] for p in probes if p in V)
    r, c = linear_sum_assignment(-score)
    lab2topic = {labs[a]: int(b) for a, b in zip(r, c)}
    topic = {i: lab2topic[L[n]] for n, i in enumerate(emb_ids)}

    def tsne(dim):
        Y = TSNE(dim, perplexity=12, metric="cosine", init="pca", random_state=1,
                 learning_rate="auto").fit_transform(X)
        Y = Y - Y.mean(0)
        return Y / np.abs(Y).max()
    Y2, Y3 = tsne(2), tsne(3)
    pos2 = {i: Y2[n] for n, i in enumerate(emb_ids)}
    pos3 = {i: Y3[n] for n, i in enumerate(emb_ids)}
    for i in ids:  # papers without an embedding: place by TF-IDF nearest neighbours
        if i in pos3:
            continue
        sims = (T[[idx[j] for j in emb_ids]] @ T[idx[i]].T).toarray().ravel()
        nn = np.argsort(-sims)[:3]
        pos2[i] = np.mean([Y2[n] for n in nn], 0) + 0.03
        pos3[i] = np.mean([Y3[n] for n in nn], 0) + 0.03
        topic[i] = Counter(topic[emb_ids[n]] for n in nn).most_common(1)[0][0]
        log(f"no embedding for {i}: placed next to its TF-IDF nearest neighbours")

    canon = canonical_authors([nm for i in ids for nm in arx[i]["authors"]])
    authors = sorted(set(canon.values()))
    aidx = {a: n for n, a in enumerate(authors)}

    papers = []
    for i in ids:
        p = S.get(i) or {}
        refs = []
        for ref in p.get("references") or []:
            x = (ref.get("externalIds") or {}).get("ArXiv")
            if x in idx and x != i and idx[x] not in refs:
                refs.append(idx[x])
        au = []
        for nm in arx[i]["authors"]:
            k = aidx[canon[nm]]
            if k not in au:
                au.append(k)
        papers.append({
            "id": i, "t": arx[i]["title"], "ab": arx[i]["abstract"], "d": arx[i]["published"],
            "y": int(arx[i]["published"][:4]), "au": au,
            "c": ads[i]["citations"] if i in ads else (p.get("citationCount") or 0), "cs": 0 if i in ads else 1,
            "bib": ads[i]["bibcode"] if i in ads else "",
            "v": p.get("venue") or "", "doi": (p.get("externalIds") or {}).get("DOI") or "",
            "x": round(float(pos2[i][0]), 4), "yy": round(float(pos2[i][1]), 4),
            "p3": [round(float(v), 4) for v in pos3[i]], "k": topic[i], "r": refs,
        })

    log(f"\n{len(papers)} papers, {len(authors)} authors, {sum(len(p['r']) for p in papers)} citation links")
    for k, (name, _) in enumerate(TOPICS):
        mem = sorted((p for p in papers if p["k"] == k), key=lambda p: -p["c"])
        log(f"\n[{k}] {name}  ({len(mem)} papers)")
        for p in mem[:4]:
            log(f"      - {p['t'][:90]}")
    return {"topics": [n for n, _ in TOPICS], "authors": authors, "papers": papers,
            "fetched": date.today().isoformat(), "nads": len(ads)}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="camels_atlas.html", help="output HTML file (default: camels_atlas.html)")
    ap.add_argument("--cache", default="atlas_cache", help="folder for downloaded data (default: atlas_cache)")
    ap.add_argument("--offline", action="store_true", help="rebuild from the cached downloads, no network")
    ap.add_argument("--artifact", action="store_true",
                    help="write the page body only (no <html>/<head>), for publishing as a claude.ai artifact")
    args = ap.parse_args()
    os.makedirs(args.cache, exist_ok=True)

    ids = load_or_fetch(args.cache, "ids.json", fetch_arxiv_ids, args.offline)
    arx = load_or_fetch(args.cache, "arxiv.json", lambda: fetch_arxiv(ids), args.offline)
    s2 = load_or_fetch(args.cache, "s2.json", lambda: fetch_semantic_scholar(ids), args.offline)
    ads = load_or_fetch(args.cache, "ads.json", lambda: fetch_ads(ids), args.offline)

    data = build_data(arx, s2, ads)
    blob = json.dumps(data, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    html = TEMPLATE.replace("__DATA__", blob)
    if not args.artifact:  # a complete document that opens directly in a browser or on GitHub Pages
        head, body = html.split("<header>", 1)
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
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    log(f"\nwrote {args.out} ({len(html) / 1e3:.0f} kB). Open it in a browser, or publish it as an artifact.")


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
  --t0: #2a78d6; --t1: #eb6834; --t2: #1baf7a; --t3: #eda100; --t4: #e87ba4; --t5: #008300; --t6: #6250d6; --t7: #e34948; --t8: #8c6d1f; --t9: #00a5b5;
  --f-display: "Space Grotesk", "Helvetica Neue", Arial, sans-serif;
  --f-body: Inter, system-ui, -apple-system, "Segoe UI", sans-serif;
  --f-mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #0e1116; --surface: #161a21; --ink: #eef0f4; --ink-2: #b6bcc8; --muted: #858c99;
  --line: #2a303b; --edge: rgba(238,240,244,0.12); --edge-hi: #eef0f4; --accent: #3987e5; --wash: rgba(57,135,229,0.14);
  --t0: #3987e5; --t1: #d95926; --t2: #199e70; --t3: #c98500; --t4: #d55181; --t5: #008300; --t6: #9085e9; --t7: #e66767; --t8: #9a7b26; --t9: #0f9fae; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #0e1116; --surface: #161a21; --ink: #eef0f4; --ink-2: #b6bcc8; --muted: #858c99;
  --line: #2a303b; --edge: rgba(238,240,244,0.12); --edge-hi: #eef0f4; --accent: #3987e5; --wash: rgba(57,135,229,0.14);
  --t0: #3987e5; --t1: #d95926; --t2: #199e70; --t3: #c98500; --t4: #d55181; --t5: #008300; --t6: #9085e9; --t7: #e66767; --t8: #9a7b26; --t9: #0f9fae; color-scheme: dark; }

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
}
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
  </div>
</header>

<div class="toolbar">
  <label class="search"><svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><circle cx="7" cy="7" r="4.5"/><path d="m10.5 10.5 3.5 3.5"/></svg>
    <input id="q" type="search" placeholder="Search titles, authors or arXiv IDs" aria-label="Search papers"></label>
  <span class="field"><label for="y0">Years</label> <select id="y0"></select> <span aria-hidden="true">–</span> <select id="y1" aria-label="To year"></select></span>
  <label class="chk" id="minwrap" hidden><input type="checkbox" id="min2" checked> Only authors with 2+ papers</label>
  <button class="ghost" id="reset">Clear filters</button>
</div>
<div class="legend" id="legend" aria-label="Topics"></div>

<main>
  <div class="stage" id="stage">
    <canvas id="cv" aria-label="Interactive map of CAMELS publications" role="img"></canvas>
    <div class="zoom"><button id="zin" aria-label="Zoom in">+</button><button id="zout" aria-label="Zoom out">−</button><button id="zfit" aria-label="Fit to view">⤢</button><button id="zspin" aria-label="Pause rotation" aria-pressed="true" hidden>⏸</button></div>
    <div class="stage-note" id="note"></div>
    <div class="tip" id="tip" hidden></div>
  </div>
  <aside id="panel" aria-live="polite"></aside>
</main>
<footer id="foot"></footer>

<script type="application/json" id="atlas-data">__DATA__</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
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
const CE = []; P.forEach(p => p.r.forEach(j => CE.push({ s: p.i, t: j })));
const years = [...new Set(P.map(p => p.y))].sort();

// ---------- state ----------
const S = { view: 'topics', sel: null, hover: null, q: '', y0: years[0], y1: years[years.length - 1], on: new Set(TOP.map((_, k) => k)), min2: true, nTop: { topics: 25, citations: 25, coauthors: 25 } };
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
let C = {};
const readColors = () => { C = { surface: css('--surface'), ink: css('--ink'), ink2: css('--ink-2'), muted: css('--muted'), edge: css('--edge'), edgeHi: css('--edge-hi'), accent: css('--accent'), t: TOP.map((_, k) => css('--t' + k)) }; };

// ---------- filters ----------
let pOn = [], aOn = [], nShown = 0;
function applyFilters() {
  const q = S.q.trim().toLowerCase();
  pOn = P.map(p => p.y >= S.y0 && p.y <= S.y1 && S.on.has(p.k) && (!q || p.hay.includes(q)));
  const nameHit = q ? AU.filter(a => a.lc.includes(q)) : [];
  aOn = AU.map(a => nameHit.length ? a.lc.includes(q) : a.papers.some(j => pOn[j]));
  nShown = pOn.filter(Boolean).length;
}

// ---------- layouts (world coords) ----------
const L = {};
// Shared 3D force layout (d3-force style: many-body repulsion, springs, centering, collision), run once, deterministic.
function force3d(nodes, links, o) {
  const N = nodes.length;
  nodes.forEach((n, i) => { // deterministic spherical spiral start
    const r = 12 * Math.cbrt(0.5 + i), th = i * Math.PI * (3 - Math.sqrt(5)), ph = Math.acos(1 - 2 * (i + 0.5) / N);
    n.x3 = r * Math.sin(ph) * Math.cos(th); n.y3 = r * Math.sin(ph) * Math.sin(th); n.z3 = r * Math.cos(ph); n.vx = n.vy = n.vz = 0;
  });
  const deg = new Map(); links.forEach(l => { deg.set(l.s, (deg.get(l.s) || 0) + 1); deg.set(l.t, (deg.get(l.t) || 0) + 1); });
  links.forEach(l => { l.bias = deg.get(l.s) / (deg.get(l.s) + deg.get(l.t)); });
  const decay = 1 - Math.pow(0.001, 1 / 300);
  let alpha = 1;
  for (let it = 0; it < 300; it++) {
    for (const l of links) {
      const a = l.source, b = l.target;
      let dx = b.x3 + b.vx - a.x3 - a.vx, dy = b.y3 + b.vy - a.y3 - a.vy, dz = b.z3 + b.vz - a.z3 - a.vz;
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1e-6, f = (d - l.len) / d * alpha * l.k;
      dx *= f; dy *= f; dz *= f;
      b.vx -= dx * l.bias; b.vy -= dy * l.bias; b.vz -= dz * l.bias;
      a.vx += dx * (1 - l.bias); a.vy += dy * (1 - l.bias); a.vz += dz * (1 - l.bias);
    }
    for (let i = 0; i < N; i++) { const a = nodes[i];
      for (let j = i + 1; j < N; j++) { const b = nodes[j];
        const dx = b.x3 - a.x3, dy = b.y3 - a.y3, dz = b.z3 - a.z3, d2 = Math.max(dx * dx + dy * dy + dz * dz, 1);
        const f = o.charge * alpha / d2; a.vx -= dx * f; a.vy -= dy * f; a.vz -= dz * f; b.vx += dx * f; b.vy += dy * f; b.vz += dz * f;
        const m = a.r + b.r + 1.5; if (d2 < m * m) { const d = Math.sqrt(d2), p = (m - d) / d * 0.35; a.vx -= dx * p; a.vy -= dy * p; a.vz -= dz * p; b.vx += dx * p; b.vy += dy * p; b.vz += dz * p; }
      } }
    for (const n of nodes) {
      n.vx -= n.x3 * o.gravity * alpha; n.vy -= n.y3 * o.gravity * alpha; n.vz -= n.z3 * o.gravity * alpha;
      n.vx *= 0.6; n.vy *= 0.6; n.vz *= 0.6; n.x3 += n.vx; n.y3 += n.vy; n.z3 += n.vz;
    }
    alpha += (0 - alpha) * decay;
  }
  return centre3d(nodes);
}
// centre on the mean and return a framing radius from the bulk (90th percentile), so far islands don't shrink the view
function centre3d(nodes) {
  const N = nodes.length, c = ['x3', 'y3', 'z3'].map(k => nodes.reduce((s, n) => s + n[k], 0) / N);
  nodes.forEach(n => { n.x3 -= c[0]; n.y3 -= c[1]; n.z3 -= c[2]; });
  const ds = nodes.map(n => Math.hypot(n.x3, n.y3, n.z3)).sort((a, b) => a - b);
  return Math.max(1, ds[Math.floor(0.9 * (N - 1))]);
}
function topicLayout() {
  const nodes = P.map(p => ({ id: p.i, kind: 'p', x3: p.p3[0] * 420, y3: -p.p3[1] * 420, z3: p.p3[2] * 420, r: Math.min(16, 3 + Math.sqrt(p.c) * 0.9) }));
  const R = centre3d(nodes), med = a => { const s = a.slice().sort((u, v) => u - v); return s[s.length >> 1]; };
  const labels = TOP.map((name, k) => {
    const pts = nodes.filter(n => P[n.id].k === k);
    return { k, name, x3: med(pts.map(n => n.x3)), y3: med(pts.map(n => n.y3)), z3: med(pts.map(n => n.z3)) };
  });
  return { nodes, links: [], labels, R, is3d: true };
}
function citationLayout() {
  const nodes = P.map(p => ({ id: p.i, kind: 'p', r: 3 + Math.sqrt(p.citers.length) * 2.2 }));
  const links = CE.map(e => ({ source: nodes[e.s], target: nodes[e.t], s: e.s, t: e.t, w: 0.45, len: 40, k: 0.35 }));
  const R = force3d(nodes, links, { charge: 90, gravity: 0.18 });
  return { nodes, links, labels: [], R, is3d: true };
}
function coauthorLayout(min2) {
  const keep = AU.filter(a => !min2 || a.papers.length >= 2);
  const nodes = keep.map(a => ({ id: a.i, kind: 'a', r: 2.5 + Math.sqrt(a.papers.length) * 2 }));
  const byId = new Map(nodes.map(n => [n.id, n]));
  const links = [...EM.values()].filter(e => byId.has(e.s) && byId.has(e.t)).map(e => ({
    source: byId.get(e.s), target: byId.get(e.t), s: e.s, t: e.t, papers: e.papers,
    w: e.papers.reduce((s, j) => s + 1 / (P[j].au.length - 1), 0),  // Newman collaboration strength
    len: 60 / Math.sqrt(e.papers.length), k: Math.min(0.6, 0.04 * e.papers.length) }));
  const R = force3d(nodes, links, { charge: min2 ? 240 : 70, gravity: 0.12 });
  return { nodes, links, labels: [], R, is3d: true };
}
// camera for the 3D view
const CAM = { yaw: 0.6, pitch: -0.25, spin: !matchMedia('(prefers-reduced-motion: reduce)').matches };
function project(lay) {
  const cy = Math.cos(CAM.yaw), sy_ = Math.sin(CAM.yaw), cp = Math.cos(CAM.pitch), sp = Math.sin(CAM.pitch), Dc = lay.R * 3;
  for (const n of lay.nodes.concat(lay.labels)) {
    const x1 = n.x3 * cy + n.z3 * sy_, z1 = -n.x3 * sy_ + n.z3 * cy;
    const y2 = n.y3 * cp - z1 * sp, z2 = n.y3 * sp + z1 * cp, f = Dc / Math.max(Dc + z2, Dc * 0.3);
    n.x = x1 * f; n.y = y2 * f; n.s = Math.min(f, 2); n.dz = z2; n.fog = Math.max(0.35, Math.min(1, 0.45 + 0.55 * (1 - (z2 + lay.R) / (2 * lay.R))));
  }
}
const layout = () => {
  const key = S.view === 'coauthors' ? 'co' + (S.min2 ? 2 : 1) : S.view;
  if (!L[key]) L[key] = S.view === 'topics' ? topicLayout() : S.view === 'citations' ? citationLayout() : coauthorLayout(S.min2);
  return L[key];
};

// ---------- canvas ----------
const cv = $('cv'), ctx = cv.getContext('2d'), stage = $('stage');
let W = 0, H = 0, T = d3.zoomIdentity, raf = 0;
const zoom = d3.zoom().scaleExtent([0.2, 14])
  .filter(e => (e.type === 'wheel' || (e.touches && e.touches.length > 1)) && (!e.ctrlKey || e.type === 'wheel') && !e.button).on('zoom', e => { T = e.transform; hideTip(); draw(); });
d3.select(cv).call(zoom).on('dblclick.zoom', null);
function resize() {
  const r = stage.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  W = r.width; H = r.height; cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); draw();
}
function fit(animate) {
  const lay = layout(), { nodes } = lay; if (!nodes.length || !W) return;
  const reduce0 = matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (lay.is3d) {
    const t = d3.zoomIdentity.translate(W / 2, H / 2).scale(Math.min(W - 40, H - 40) / (2 * lay.R * 1.15));
    (animate && !reduce0 ? d3.select(cv).transition().duration(450) : d3.select(cv)).call(zoom.transform, t); return;
  }
  const xs = nodes.map(n => n.x), ys = nodes.map(n => n.y);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const k = Math.min((W - 60) / (x1 - x0 || 1), (H - 60) / (y1 - y0 || 1));
  const t = d3.zoomIdentity.translate(W / 2, H / 2).scale(k).translate(-(x0 + x1) / 2, -(y0 + y1) / 2);
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  (animate && !reduce ? d3.select(cv).transition().duration(450) : d3.select(cv)).call(zoom.transform, t);
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
  if (links.length && lay.is3d) {
    // thin hairlines; opacity/width follow collaboration strength and depth, drawn in 10 buckets
    const NB = 10, paths = Array.from({ length: NB }, () => new Path2D());
    for (const l of links) { const a = l.source, b = l.target; if (!on(a) || !on(b) || (l.papers && !l.papers.some(j => pOn[j]))) continue;
      const v = (0.05 + 0.55 * Math.min(1, l.w)) * (a.fog + b.fog) / 2, q = Math.min(NB - 1, Math.floor(v / 0.06));
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
        if (!lit) continue;
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
    tip.innerHTML = `${esc(p.t)}<small>${esc(TOP[p.k])}</small><small>${p.y} · ${esc(a[0] || '')}${a.length > 1 ? ' et al.' : ''} · ${p.c} citations${S.view === 'citations' ? ` · cited by ${p.citers.length} CAMELS papers` : ''}</small>`; }
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
// 3D view: drag rotates (d3.zoom keeps wheel/pinch zoom), arrow keys rotate, gentle auto-spin
let rot = null, dragged = false;
cv.addEventListener('pointerdown', e => {
  dragged = false; if (e.button) return;
  rot = { x: e.clientX, y: e.clientY, x0: e.clientX, y0: e.clientY }; cv.setPointerCapture(e.pointerId);
});
cv.addEventListener('pointermove', e => {
  if (!rot || !e.buttons) return;
  CAM.yaw += (e.clientX - rot.x) * 0.008; CAM.pitch = Math.max(-1.45, Math.min(1.45, CAM.pitch + (e.clientY - rot.y) * 0.008));
  rot.x = e.clientX; rot.y = e.clientY; if (Math.hypot(e.clientX - rot.x0, e.clientY - rot.y0) > 4) dragged = true;
  hideTip(); draw();
});
const endRot = () => { rot = null; };
cv.addEventListener('pointerup', endRot); cv.addEventListener('pointercancel', endRot);
cv.addEventListener('keydown', e => {
  const m = { ArrowLeft: [-0.12, 0], ArrowRight: [0.12, 0], ArrowUp: [0, -0.12], ArrowDown: [0, 0.12] }[e.key]; if (!m) return;
  e.preventDefault(); CAM.yaw += m[0]; CAM.pitch = Math.max(-1.45, Math.min(1.45, CAM.pitch + m[1])); draw();
});
(function spin() {
  if (CAM.spin && !rot && !S.hover && !document.hidden) { CAM.yaw += 0.0025; render(); }
  requestAnimationFrame(spin);
})();
cv.addEventListener('click', e => { if (dragged) { dragged = false; return; } const r = cv.getBoundingClientRect(); select(pick(e.clientX - r.left, e.clientY - r.top)); });

// ---------- panel ----------
const panel = $('panel');
const dot = k => `<i style="background:var(--t${k})"></i>`;
const pItem = j => `<li><button data-p="${j}">${dot(P[j].k)}<span class="tt">${esc(P[j].t)}</span><span class="num">${P[j].y}</span></button></li>`;
const VIEWINFO = {
  topics: ['Each dot is a paper, laid out in 3D. Papers with similar abstracts sit close together. Nearer papers look larger and brighter.', (D.nads ? 'Total citations (NASA ADS)' : 'Total citations (Semantic Scholar)'), 'Topic cluster'],
  citations: ['Each dot is a paper, laid out in 3D. A line means one CAMELS paper cites another. Select a paper to see arrows pointing to the work it cites. Nearer papers look larger and brighter.', 'Citations from other CAMELS papers', 'Topic cluster'],
  coauthors: ['Each dot is an author, laid out in 3D. A line joins people who wrote a CAMELS paper together. Darker lines mean a closer collaboration: each shared paper counts more when it has fewer authors, so links from 40-author papers stay faint. Nearer authors look larger and brighter.', 'Number of CAMELS papers', 'Author’s most frequent topic'],
};
const USE_TXT = 'Drag to rotate (or use the arrow keys), scroll or pinch to zoom. Rotation pauses while you hover. Click a dot for details; your selection stays highlighted when you switch views.';
const USE = { topics: USE_TXT, citations: USE_TXT, coauthors: USE_TXT };
function overview() {
  const [what, size, color] = VIEWINFO[S.view];
  let list;
  if (S.view === 'coauthors') {
    const all = AU.filter(a => aOn[a.i]).sort((a, b) => b.papers.length - a.papers.length || a.name.localeCompare(b.name));
    const top = all.slice(0, S.nTop[S.view]), left = all.length - top.length;
    list = `<div><h3>Most frequent authors <span style="text-transform:none;letter-spacing:0;font-weight:400">· ${top.length} of ${all.length}</span></h3><ul class="list">${top.map((a, r) => `<li><button data-a="${a.i}">${dot(a.topic)}<span class="tt">${r + 1}. ${esc(a.name)}</span><span class="num">${a.papers.length} paper${a.papers.length > 1 ? 's' : ''}</span></button></li>`).join('')}</ul>
      ${left > 0 ? `<div style="display:flex;gap:8px;margin-top:8px"><button class="ghost" data-more="25">Show ${Math.min(25, left)} more</button><button class="ghost" data-more="all">Show all ${all.length}</button></div>` : ''}</div>`;
  } else {
    const key = S.view === 'citations' ? p => p.citers.length : p => p.c;
    const all = P.filter(p => pOn[p.i]).sort((a, b) => key(b) - key(a) || b.d.localeCompare(a.d));
    const top = all.slice(0, S.nTop[S.view]), left = all.length - top.length;
    list = `<div><h3>${S.view === 'citations' ? 'Most cited within CAMELS' : (D.nads ? 'Most cited overall (ADS)' : 'Most cited overall')} <span style="text-transform:none;letter-spacing:0;font-weight:400">· ${top.length} of ${all.length}</span></h3><ul class="list">${top.map((p, r) =>
      `<li><button data-p="${p.i}">${dot(p.k)}<span class="tt">${r + 1}. ${esc(p.t)}</span><span class="num">${key(p)}</span></button></li>`).join('')}</ul>
      ${left > 0 ? `<div style="display:flex;gap:8px;margin-top:8px"><button class="ghost" data-more="25">Show ${Math.min(25, left)} more</button><button class="ghost" data-more="all">Show all ${all.length}</button></div>` : ''}</div>`;
  }
  panel.innerHTML = `<div><h3>How to read this view</h3><p>${what}</p></div>
    <div class="how"><div><b>Size</b><span>${size}</span></div><div><b>Color</b><span>${color}</span></div>
    <div><b>Use</b><span>${USE[S.view]}</span></div></div>${list}`;
}
function paperPanel(i) {
  const p = P[i];
  const venue = p.v ? `<span>${esc(p.v)}</span>` : '';
  panel.innerHTML = `<button class="ghost back" data-back>← Overview</button>
    <div class="eyebrow">${dot(p.k)}${esc(TOP[p.k])}</div>
    <h2>${esc(p.t)}</h2>
    <div class="meta"><a href="https://arxiv.org/abs/${p.id}" target="_blank" rel="noopener">arXiv:${p.id}</a><span>${p.d}</span>${venue}${p.bib ? `<a href="https://ui.adsabs.harvard.edu/abs/${encodeURIComponent(p.bib)}/abstract" target="_blank" rel="noopener">${p.c} citations (ADS)</a>` : `<span>${p.c} citations (Semantic Scholar)</span>`}</div>
    <div><h3>${p.au.length} author${p.au.length > 1 ? 's' : ''}</h3><div class="names">${p.au.map(a => `<button data-a="${a}">${esc(AU[a].name)}</button>`).join('')}</div></div>
    <div><h3>Abstract</h3><p class="abstract">${esc(p.ab)}</p></div>
    <div><h3>Cites ${p.r.length} CAMELS paper${p.r.length === 1 ? '' : 's'}</h3>${p.r.length ? `<ul class="list">${p.r.slice().sort((a, b) => P[b].d.localeCompare(P[a].d)).map(pItem).join('')}</ul>` : '<p>None found.</p>'}</div>
    <div><h3>Cited by ${p.citers.length} CAMELS paper${p.citers.length === 1 ? '' : 's'}</h3>${p.citers.length ? `<ul class="list">${p.citers.slice().sort((a, b) => P[b].d.localeCompare(P[a].d)).map(pItem).join('')}</ul>` : '<p>None yet.</p>'}</div>`;
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
function select(s) { S.sel = s; renderPanel(); draw(); }

// ---------- controls ----------
function setView(v, animate) {
  S.view = v; ['topics', 'citations', 'coauthors'].forEach(x => $('m-' + x).setAttribute('aria-pressed', x === v));
  $('minwrap').hidden = v !== 'coauthors'; $('zspin').hidden = false; spinBtn();
  cv.tabIndex = 0;
  $('note').textContent = v === 'topics' ? 'Layout: 3D t-SNE of SPECTER2 abstract embeddings · axes have no units · drag to rotate'
    : v === 'citations' ? `${CE.length} citation links between CAMELS papers · drag to rotate` : `${AU.length} authors · ${EM.size} co-author pairs · drag to rotate`;
  try { if (location.hash !== '#' + v) history.replaceState(null, '', '#' + v); } catch (e) {}
  S.hover = null; hideTip(); fit(animate); if (!S.sel) overview(); draw(); updateStats();
}
document.querySelectorAll('.modes button').forEach(b => b.addEventListener('click', () => setView(b.dataset.view, true)));
window.addEventListener('hashchange', () => { const v = location.hash.slice(1); if (VIEWINFO[v] && v !== S.view) setView(v, true); });
const legend = $('legend');
function renderLegend() {
  legend.innerHTML = TOP.map((t, k) => `<button class="chip" data-k="${k}" aria-pressed="${S.on.has(k)}"><i style="background:var(--t${k})"></i>${esc(t)} <span class="n">${P.filter(p => p.k === k).length}</span></button>`).join('');
}
legend.addEventListener('click', e => { const b = e.target.closest('.chip'); if (!b) return; const k = +b.dataset.k;
  if (e.altKey || e.metaKey) { S.on = new Set([k]); } else if (S.on.has(k)) S.on.delete(k); else S.on.add(k);
  renderLegend(); refresh(); });
[$('y0'), $('y1')].forEach((sel, n) => { sel.innerHTML = years.map(y => `<option ${y === (n ? S.y1 : S.y0) ? 'selected' : ''}>${y}</option>`).join('');
  sel.addEventListener('change', () => { S.y0 = +$('y0').value; S.y1 = +$('y1').value; if (S.y0 > S.y1) { [S.y0, S.y1] = [S.y1, S.y0]; $('y0').value = S.y0; $('y1').value = S.y1; } refresh(); }); });
let qt; $('q').addEventListener('input', e => { clearTimeout(qt); qt = setTimeout(() => { S.q = e.target.value; refresh(); }, 120); });
$('min2').addEventListener('change', e => { S.min2 = e.target.checked; fit(false); draw(); });
$('reset').addEventListener('click', () => { S.q = ''; $('q').value = ''; S.y0 = years[0]; S.y1 = years[years.length - 1]; $('y0').value = S.y0; $('y1').value = S.y1;
  S.on = new Set(TOP.map((_, k) => k)); renderLegend(); refresh(); });
$('zin').addEventListener('click', () => d3.select(cv).transition().duration(200).call(zoom.scaleBy, 1.5));
$('zout').addEventListener('click', () => d3.select(cv).transition().duration(200).call(zoom.scaleBy, 1 / 1.5));
$('zfit').addEventListener('click', () => fit(true));
const spinBtn = () => { const b = $('zspin'); b.textContent = CAM.spin ? '⏸' : '⟳'; b.setAttribute('aria-label', CAM.spin ? 'Pause rotation' : 'Start rotation'); b.setAttribute('aria-pressed', CAM.spin); };
$('zspin').addEventListener('click', () => { CAM.spin = !CAM.spin; spinBtn(); });
function updateStats() {
  const filtered = nShown < P.length;
  $('stats').innerHTML = `<b>${filtered ? nShown + ' of ' : ''}${P.length}</b> papers · <b>${AU.length}</b> authors · <b>${CE.length}</b> citation links · ${years[0]}–${years[years.length - 1]}`;
}
function refresh() { applyFilters(); updateStats(); if (!S.sel) overview(); draw(); }
$('foot').innerHTML = `Papers listed on the <a href="https://camels.readthedocs.io/en/latest/publications.html" target="_blank" rel="noopener">CAMELS publications page</a> (${P.length} with arXiv IDs). Abstracts and authors from arXiv; ${D.nads ? 'citation counts from NASA ADS (Semantic Scholar where ADS has no record)' : 'citation counts from Semantic Scholar'}; SPECTER2 embeddings and the citation links between papers from Semantic Scholar, fetched ${D.fetched}. Topics are k-means clusters of the embeddings, named by hand.`;

// ---------- boot ----------
renderLegend(); applyFilters(); renderPanel();
new ResizeObserver(() => { const first = !W; resize(); if (first) fit(false); }).observe(stage);
const mq = matchMedia('(prefers-color-scheme: dark)'); mq.addEventListener && mq.addEventListener('change', draw);
new MutationObserver(draw).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
const h = location.hash.slice(1); setView(VIEWINFO[h] ? h : 'topics', false);
document.fonts && document.fonts.ready.then(draw);
})();
</script>
'''


if __name__ == '__main__':
    main()
