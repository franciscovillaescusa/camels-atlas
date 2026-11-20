# Literature Atlases

Interactive maps of the papers that use a simulation suite. One codebase builds an atlas per project:

| Project | Papers from | Live site |
|---|---|---|
| **CAMELS** (Cosmology and Astrophysics with MachinE Learning Simulations) | [publications page](https://camels.readthedocs.io/en/latest/publications.html) | https://franciscovillaescusa.github.io/literature-atlas/camels/ |
| **Quijote** (N-body simulations for cosmology and machine learning) | [publications page](https://quijote-simulations.readthedocs.io/en/latest/publications.html) | https://franciscovillaescusa.github.io/literature-atlas/quijote/ |

The old CAMELS address, https://franciscovillaescusa.github.io/camels-atlas/, redirects to the new one
(a small page in the `franciscovillaescusa.github.io` repository), keeping shared `#…` links.

Each site has five views. A button switches between them:

| View | Each dot is | Lines | Dot size |
|---|---|---|---|
| **Topic map** | a paper, placed by abstract similarity, with each topic drawn as a shaded 'island' (2D: t-SNE of SPECTER2 embeddings; 3D: each paper pulled toward its 4 most similar papers) | none | total citations |
| **Citations** | a paper | one listed paper cites another | citations from other papers of the same project (a toggle hides self-citations, i.e. links between papers sharing an author) |
| **Co-authors** | an author | the two authors share a paper; darker lines mean closer collaboration | number of papers in the atlas |
| **Table** | a paper (one row) | — | sortable columns: title, first author, year, topic, methods, citations, citations within the project; keyboard accessible |
| **Trends** | papers per year | — | stacked bars split by topic, method or suite; click a segment to list those papers |

Colors show the topics (10 per project), found by k-means clustering of the paper embeddings and named by hand.
The **▶** button (next to the years) animates the maps: papers appear one by one in date order (about 20 s in all).
The **Display** menu in the map corner holds the 2D/3D layout, *Color by* (topic, year, method or suite), the
self-citation switch and the 3D rotation; its button shows the current settings.
Each paper's panel lists its 5 most similar papers by embedding similarity, and selecting a paper on
the topic map highlights them. Topic, suite, data, method and "New" filters live in the **Filters** panel; active filters show as removable
chips next to it. Search (press `/`) covers titles, authors, IDs, abstracts and tags, ignores accents, tolerates small typos
and ranks title/author matches first. Search, the year range and the topic buttons filter every view. A selected paper or author
stays highlighted when you switch views. The maps open in 2D (drag to pan, scroll or pinch to zoom); a
2D/3D button next to the zoom controls switches them to rotatable 3D layouts (drag to rotate).

### Corrections from visitors

Each paper's panel has a **Report a problem with this paper** link, and the footer has **Suggest a missing
paper**. Both open a pre-filled issue in this repository (a GitHub account is needed). Fix reports by editing
`projects/<name>/overrides.json` or `projects/<name>/tags.json`; missing papers usually need adding to the
project's publications page first, since the atlas reads its list from there.

### Downloading papers

**Download ▾** (at the top of the *Table* view) saves the papers currently shown, after search and filters, as
BibTeX (`.bib`, with DOI, arXiv ID and ADS link where available) or as a spreadsheet (`.csv`).

### New papers

The build records when each paper first appeared on the site (in `projects/<name>/positions.json`). Papers
added in the last 30 days get a **New** badge, and a **New** option in the *Filters* panel filters to them.

### Sharing a view

The address bar keeps the current view, the selected paper or author, the filters and 2D/3D, so you can
share exactly what you see, e.g.
`https://franciscovillaescusa.github.io/literature-atlas/camels/#citations&paper=2201.02202` or
`#coauthors&author=Shy Genel`. Paper and author panels have a **Copy link** button, and the browser's
back button steps through your selections.

### Analytics

A site loads Google Analytics 4 when its project's `config.json` sets `ga_id` (CAMELS uses the same property
as franciscovillaescusa.github.io; filter reports by page path `/literature-atlas/camels/`). Besides page
views it sends anonymous events: `atlas_view`, `atlas_dim`, `atlas_color_by`, `atlas_play`, `atlas_download`,
`atlas_copy_link`, `atlas_report_problem` and `atlas_suggest_paper`. Search terms are not sent.
Set `"ga_id": ""` to disable it (Quijote has it off).

## Data sources

- Paper list: every entry on the project's publications page (`pubs_url` in its `config.json`)
- Titles, abstracts, authors, dates: [arXiv API](https://info.arxiv.org/help/api/); for entries without an arXiv ID, NASA ADS or the publications page itself (theses without a known date get a year estimated from their position in the list, marked as estimated on the site)
- Embeddings: [Semantic Scholar API](https://www.semanticscholar.org/product/api)
- Citation links between the listed papers: Semantic Scholar and NASA ADS reference lists combined (a link counts if either source has it)
- Total citation counts: [NASA ADS](https://ui.adsabs.harvard.edu) (Semantic Scholar counts if no ADS token is available)

## Files

| File | What it is |
|---|---|
| `build_atlas.py` | fetches the data and builds one project's site (`--project <name>`) |
| `atlas.html` | page structure; the build inserts `atlas.css`, `atlas.js`, the project name and the data into it |
| `atlas.css` | page styles |
| `atlas.js` | page behaviour (views, filters, maps, table, trends) |
| `projects/<name>/config.json` | what differs per project: name, publications page, site address, analytics ID, topics, hand-checked topic of every paper, allowed tag names, and the paper/author used by the page check |
| `projects/<name>/tags.json` | suite / data / method tags per paper (hand-editable) |
| `projects/<name>/overrides.json` | hand corrections per paper |
| `projects/<name>/positions.json` | stored layout positions and first-seen dates (generated; commit it) |
| `<name>/index.html`, `<name>/preview.png` | the project's site and its link-preview image (generated; do not edit by hand) |
| `index.html` | landing page linking to every atlas (hand-written) |
| `validate_atlas.py` | data validation against the sources (run by the update job) |
| `check_atlas.py` | headless-browser check of a built page (run by the update job) |
| `tagging.py` | lists papers that need tags, for tagging with Claude (see `CLAUDE.md`) |
| `CLAUDE.md` | step-by-step instructions Claude follows to tag new papers and to set up a new project |

## Rebuilding

```bash
pip install -r requirements.txt
python build_atlas.py --project camels             # fetch fresh data and rebuild camels/index.html
python build_atlas.py --project camels --offline   # rebuild from ./atlas_cache/camels, no network
```

Layouts are computed once and stored in `projects/<name>/positions.json`. Later builds keep every known paper
and author where it is and place only new ones next to their most similar or most connected neighbours,
so the maps keep their shape as the list grows. Commit this file together with the site.
`--relayout` recomputes all layouts from scratch. Package versions are pinned in `requirements.txt`
so local and automated builds give the same result.

Before writing anything, the script checks that it found at least 90% as many papers as the current
site, that arXiv returned at least 95% of the requested records and that at least 90% of the
arXiv papers have embeddings (theses rarely have one). After building, it also checks that the citation
links, the links found by ADS and the papers with ADS citation counts did not drop by more than 10% from the
current site, since an API that fails part-way makes the build fall back silently. If a check fails it stops without changing the page or the cache, so a broken
fetch is never published; use `--force` when a drop is real.

For ADS citation counts, put your [ADS API token](https://ui.adsabs.harvard.edu/user/settings/token)
in `~/.ads/dev_key` or the `ADS_TOKEN` environment variable.

### Validating the data

`python validate_atlas.py --project <name>` checks the data in the site against the downloaded sources and
against itself (about 55 checks). It recomputes key results independently: every entry on the publications
page is present, titles, abstracts, dates and ADS citation counts match their sources, the citation links
reproduce exactly from the raw reference lists, self-citations, co-author pairs and "most similar papers" are
recomputed, hand-checked topics and overrides are kept, the page matches the project's config, tags use the
allowed vocabulary, and every stored layout is complete. `--online` also compares with the live publications
page and live ADS counts. The update job runs it before publishing.

### Checking the page

`python check_atlas.py --project <name>` opens the built site in a headless browser, goes through all five
views, the 2D/3D switch, filters, selection and search, and checks that what the page shows is right: the
filtered table, BibTeX/CSV downloads, trend totals, the self-citation switch and shared links. It fails on
any JavaScript error. Locally: `pip install playwright`, then add `--chrome` to use your installed Chrome.

### Automatic updates (1st and 15th of each month)

`.github/workflows/update-atlas.yml` rebuilds every project on the 1st and 15th of each month, validates and
checks each one, and commits the sites that pass. A project that fails is left as it was (the job is then
marked failed), so one broken source never blocks the others. New papers appear right away with their topic,
citations and links; their suite/data/method tags are added by hand with Claude. When papers need tags, the
job opens (or updates) a GitHub issue **"New papers need tags (<name>)"**, which notifies you by email. Then
open Claude Code in your local `literature-atlas` folder and say *"pull, then tag the new <name> papers"*:
it follows `CLAUDE.md`, rebuilds, runs the tests, pushes and closes the issue.
You can also start it by hand from the **Actions** tab ("Update atlases" → "Run workflow").
To use ADS citation counts there, add the token as a repository secret named `ADS_TOKEN`
(Settings → Secrets and variables → Actions → New repository secret).

### Hand corrections

`projects/<name>/overrides.json` corrects individual entries without touching the code. Key an entry by
arXiv ID, ADS bibcode or title, and set any of: `topic`, `date`, `abstract`, `url`, `link`, `title`,
`authors`, `hide` (leave the entry out) or `comment`. The build reports each correction it applies and
warns about keys that match no paper, unknown fields and unknown topic names.

### Tags

`projects/<name>/tags.json` gives every paper tags on three axes, assigned with AI assistance from its
abstract: the simulation suite or set used, the data product (maps, catalogs, power spectra...) and the
method (CNN, GNN, simulation-based inference, ...). Tags marked in `"uncertain"` were not clearly stated in
the abstract and show dashed with a "?" on the site; theses without an abstract carry a note. The site has a
filter for each axis. Edit the file to correct a tag (allowed names are `tag_axes` in the project's
`config.json`); new papers are reported as untagged by the build until you add them.

### Topics

Each project's topics come from k-means clustering of the paper embeddings, named by hand (`topics` in its
`config.json`). The topic of every paper checked at that time is stored in `topic_seed`, so the names keep
matching their papers when the site is rebuilt. New papers join the topic whose centre is nearest (or,
without an embedding, the topic most common among the papers with the most similar words). Every run prints
the papers in each topic and the topic given to each new paper. A new paper that is less similar to every
topic centre than any existing paper is to its own topic goes into a gray "Unassigned (new papers)" group
instead, and the build reports it, so new research directions stay visible; assign it by adding its arXiv ID
to `topic_seed`, or recluster. `--recluster` redoes the clustering from scratch; after that, check the
printed titles and edit `topics`.

## Adding a project

1. Create `projects/<name>/config.json` (copy another project's and change `name`, `homepage`, `pubs_url`,
   `site_url`, `ga_id`, `first_year`, `check`, `tag_axes`; set `topics` to placeholder names and
   `topic_seed` to `{}`), plus `tags.json` and `overrides.json` containing `{}`.
2. `python build_atlas.py --project <name>`: with an empty `topic_seed` it clusters the papers from scratch.
3. Name the topics from `python tagging.py <name> topics`, fill `topic_seed`, tag the papers, rebuild, run the
   two checks, add a card to the landing page `index.html` and commit. The update job picks the project up
   automatically. The publications page must use the same Sphinx format (`<li><div class="line-block">`).
