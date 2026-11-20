# CAMELS Literature Atlas

An interactive 3D map of the papers that use data from the
[CAMELS project](https://camels.readthedocs.io) (Cosmology and Astrophysics with MachinE Learning Simulations).

**Live site:** https://franciscovillaescusa.github.io/camels-atlas/

The site has three views. A button switches between them:

| View | Each dot is | Lines | Dot size |
|---|---|---|---|
| **Topic map** | a paper, placed by abstract similarity (3D t-SNE of SPECTER2 embeddings) | none | total citations |
| **Citations** | a paper | one CAMELS paper cites another | citations from other CAMELS papers |
| **Co-authors** | an author | the two authors share a paper; darker lines mean closer collaboration | number of CAMELS papers |

Colors show 10 topics, found by k-means clustering of the paper embeddings and named by hand.
Search, the year range and the topic buttons filter every view. A selected paper or author
stays highlighted when you switch views. Drag to rotate, scroll or pinch to zoom.

## Data sources

- Paper list: the [CAMELS publications page](https://camels.readthedocs.io/en/latest/publications.html) (every arXiv ID listed there)
- Titles, abstracts, authors, dates: [arXiv API](https://info.arxiv.org/help/api/)
- Embeddings and the citation links between CAMELS papers: [Semantic Scholar API](https://www.semanticscholar.org/product/api)
- Total citation counts: [NASA ADS](https://ui.adsabs.harvard.edu) (Semantic Scholar counts if no ADS token is available)

## Rebuilding

Everything is produced by one script:

```bash
pip install -r requirements.txt
python build_camels_atlas.py --out index.html      # fetch fresh data and rebuild
python build_camels_atlas.py --offline --out index.html   # rebuild from ./atlas_cache, no network
```

For ADS citation counts, put your [ADS API token](https://ui.adsabs.harvard.edu/user/settings/token)
in `~/.ads/dev_key` or the `ADS_TOKEN` environment variable.

### Automatic weekly updates

`.github/workflows/update-atlas.yml` reruns the script every Monday and commits the new `index.html`.
You can also start it by hand from the **Actions** tab ("Update atlas" → "Run workflow").
To use ADS citation counts there, add the token as a repository secret named `ADS_TOKEN`
(Settings → Secrets and variables → Actions → New repository secret).

### Topic names

The 10 topic names are written by hand in the `TOPICS` list at the top of `build_camels_atlas.py`,
and each is matched to a cluster by a few keywords. Every run prints the top papers in each topic.
If many new papers shift the clusters, check that the names still fit and edit the list.
