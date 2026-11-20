# CAMELS Literature Atlas

An interactive 3D map of the papers that use data from the
[CAMELS project](https://camels.readthedocs.io) (Cosmology and Astrophysics with MachinE Learning Simulations).

**Live site:** https://franciscovillaescusa.github.io/camels-atlas/

The site has four views. A button switches between them:

| View | Each dot is | Lines | Dot size |
|---|---|---|---|
| **Topic map** | a paper, placed by abstract similarity (3D t-SNE of SPECTER2 embeddings) | none | total citations |
| **Citations** | a paper | one CAMELS paper cites another | citations from other CAMELS papers (a toggle hides self-citations, i.e. links between papers sharing an author: 68% of all links) |
| **Co-authors** | an author | the two authors share a paper; darker lines mean closer collaboration | number of CAMELS papers |
| **Table** | a paper (one row) | — | sortable columns: title, first author, year, topic, methods, citations, citations within CAMELS; keyboard accessible |

Colors show 10 topics, found by k-means clustering of the paper embeddings and named by hand.
Search, the year range and the topic buttons filter every view. A selected paper or author
stays highlighted when you switch views. Drag to rotate, scroll or pinch to zoom. A 2D/3D button next to
the zoom controls switches the three map views to flat layouts, where dragging pans.

## Data sources

- Paper list: every entry on the [CAMELS publications page](https://camels.readthedocs.io/en/latest/publications.html)
- Titles, abstracts, authors, dates: [arXiv API](https://info.arxiv.org/help/api/); for entries without an arXiv ID, NASA ADS or the publications page itself (theses without a known date get a year estimated from their position in the list, marked as estimated on the site)
- Embeddings: [Semantic Scholar API](https://www.semanticscholar.org/product/api)
- Citation links between CAMELS papers: Semantic Scholar and NASA ADS reference lists combined (a link counts if either source has it)
- Total citation counts: [NASA ADS](https://ui.adsabs.harvard.edu) (Semantic Scholar counts if no ADS token is available)

## Rebuilding

Everything is produced by one script:

```bash
pip install -r requirements.txt
python build_camels_atlas.py --out index.html      # fetch fresh data and rebuild
python build_camels_atlas.py --offline --out index.html   # rebuild from ./atlas_cache, no network
```

Layouts are computed once and stored in `atlas_positions.json`. Later builds keep every known paper
and author where it is and place only new ones next to their most similar or most connected neighbours,
so the maps keep their shape as the list grows. Commit this file together with `index.html`.
`--relayout` recomputes all layouts from scratch. Package versions are pinned in `requirements.txt`
so local and automated builds give the same result.

Before writing anything, the script checks that it found at least 90% as many papers as the current
`index.html`, that arXiv returned at least 95% of the requested records and that at least 90% of the
entries have embeddings. If a check fails it stops without changing the page or the cache, so a broken
fetch is never published; use `--force` when a drop is real.

For ADS citation counts, put your [ADS API token](https://ui.adsabs.harvard.edu/user/settings/token)
in `~/.ads/dev_key` or the `ADS_TOKEN` environment variable.

### Automatic weekly updates

`.github/workflows/update-atlas.yml` reruns the script every Monday and commits the new `index.html`.
You can also start it by hand from the **Actions** tab ("Update atlas" → "Run workflow").
To use ADS citation counts there, add the token as a repository secret named `ADS_TOKEN`
(Settings → Secrets and variables → Actions → New repository secret).

### Hand corrections

`atlas_overrides.json` corrects individual entries without touching the code. Key an entry by arXiv ID,
ADS bibcode or title, and set any of: `topic`, `date`, `abstract`, `url`, `link`, `title`, `authors`,
`hide` (leave the entry out) or `comment`. The build reports each correction it applies and warns about
keys that match no paper, unknown fields and unknown topic names.

### Tags

`atlas_tags.json` gives every paper tags on three axes, assigned with AI assistance from its abstract:
the CAMELS suite used, the data product (maps, catalogs, power spectra...) and the method (CNN, GNN,
simulation-based inference, ...). Tags marked in `"uncertain"` were not clearly stated in the abstract
and show dashed with a "?" on the site; theses without an abstract carry a note. The site has a filter
for each axis. Edit the file to correct a tag (allowed names are listed in `TAG_AXES` in the script);
new papers are reported as untagged by the build until you add them.

### Topics

The 10 topics come from k-means clustering of the paper embeddings, named by hand (`TOPICS` at the
top of `build_camels_atlas.py`). The topic of every paper checked at that time is stored in
`TOPIC_SEED`, so the names keep matching their papers when the site is rebuilt. New papers join the
topic whose centre is nearest (or, without an embedding, the topic most common among the papers with
the most similar words). Every run prints the papers in each topic and the topic given to each new paper.
A new paper that is less similar to every topic centre than any existing paper is to its own topic goes
into a gray "Unassigned (new papers)" group instead, and the build reports it, so new research directions
stay visible; assign it by adding its arXiv ID to `TOPIC_SEED`, or recluster.
`--recluster` redoes the clustering from scratch; after that, check the printed titles and edit `TOPICS`.
