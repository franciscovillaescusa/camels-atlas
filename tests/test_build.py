"""Unit tests for build_atlas.py: the pure functions behind names, overrides and the safety checks."""
import pytest
import requests

from conftest import record


# ----------------------------------------------------------------------------- names and titles
@pytest.mark.parametrize("names, expect", [
    # hyphenated given names written as one word elsewhere are one person (found in Quijote)
    (["Jun-Qing Xia", "Junqing Xia"], {"Jun-Qing Xia": "Jun-Qing Xia", "Junqing Xia": "Jun-Qing Xia"}),
    # ... and the opposite case stays merged too (CAMELS: Boon Kiat Oh)
    (["Boon Kiat Oh", "Boon Kiat Oh", "Boon-Kiat Oh"], {"Boon Kiat Oh": "Boon Kiat Oh", "Boon-Kiat Oh": "Boon Kiat Oh"}),
    # different full first names are different people; an initial joins the most frequent full form
    (["Bitao Wang", "Bitao Wang", "Bonny Wang", "B. Wang"],
     {"Bitao Wang": "Bitao Wang", "B. Wang": "Bitao Wang", "Bonny Wang": "Bonny Wang"}),
    (["S. Genel", "Shy Genel"], {"S. Genel": "Shy Genel", "Shy Genel": "Shy Genel"}),
    # accents and middle initials do not split a person; the most frequent spelling is shown
    (["José Bermejo", "Jose Bermejo", "Jose Bermejo"], {"José Bermejo": "Jose Bermejo", "Jose Bermejo": "Jose Bermejo"}),
    (["Benjamin D. Wandelt", "Benjamin Wandelt", "Benjamin Wandelt"],
     {"Benjamin D. Wandelt": "Benjamin Wandelt", "Benjamin Wandelt": "Benjamin Wandelt"}),
    (["J. Smith", "J. Smith"], {"J. Smith": "J. Smith"}),
])
def test_canonical_authors(B, names, expect):
    assert B.canonical_authors(names) == expect


def test_norm_name_drops_names_without_letters(B):
    # arXiv records occasionally list " :" or " ." as an author; the build skips names that normalise to nothing
    assert B.norm_name(" :") == [] and B.norm_name(" .") == []
    assert B.norm_name("Jun-Qing Xia") == ["jun", "qing", "xia"]


def test_norm_title_and_similar_title(B):
    assert B.norm_title("Ωm from  $σ_8$: a Test!") == B.norm_title("ωm from σ8 a test")
    assert B.similar_title("Precision cosmology using voids", "Precision cosmology using cosmic voids") > 0.8
    assert B.similar_title("Precision cosmology using voids", "Galaxy rotation curves") < 0.6


def test_first_sentence(B):
    text = "We study the clustering of <b>dark matter haloes</b> in simulations. Then we do more. A third one."
    assert B.first_sentence(text) == "We study the clustering of dark matter haloes in simulations."
    # markup right before punctuation leaves no space ("haloes ." was shown in ADS excerpts)
    assert B.first_sentence("We measure the abundance of <i>dark matter haloes</i>. Then more.") == \
        "We measure the abundance of dark matter haloes."
    assert B.first_sentence("Short. Text") == "Short. Text"           # no sentence of 40+ characters: up to 300 chars
    assert B.first_sentence("") == ""


def test_ads_author(B):
    assert B.ads_author("Villaescusa-Navarro, Francisco") == "Francisco Villaescusa-Navarro"
    assert B.ads_author("CAMELS Collaboration") == "CAMELS Collaboration"


# ----------------------------------------------------------------------------- overrides
def test_overrides_by_id_and_by_title(B):
    recs = [record("2101.00001", "First paper"), record("title:thesisone", "Thesis one", link="PhD thesis")]
    keep, topic_of, kept = B.apply_overrides(recs, {
        "_help": ["ignored"],
        "2101.00001": {"title": "First paper (corrected)", "date": "2023-05", "topic": B.TOPIC_NAMES[3]},
        "Thesis one": {"authors": ["Ana Lopez", "Bo Li"], "comment": "from the PDF"},
    })
    assert [r["title"] for r in keep] == ["First paper (corrected)", "Thesis one"]
    assert keep[0]["date"] == "2023-05" and keep[0]["dq"] is False
    assert topic_of == {0: 3} and kept == {}
    assert keep[1]["authors"] == ["Ana Lopez", "Bo Li"]


def test_overrides_hide_and_bad_values(B, capsys):
    recs = [record("2101.00001", "First"), record("2101.00002", "Second")]
    keep, topic_of, _ = B.apply_overrides(recs, {
        "2101.00002": {"hide": True},
        "2101.00001": {"date": "May 2023", "topic": "No such topic", "colour": "red"},
        "9999.99999": {"title": "matches nothing"},
    })
    assert [r["uid"] for r in keep] == ["2101.00001"] and topic_of == {}
    assert keep[0]["date"] == "2024-01-02"   # bad date ignored
    log = capsys.readouterr().out
    for warning in ("must look like 2025", "unknown topic", "unknown field(s) ['colour']", "no paper matches '9999.99999'"):
        assert warning in log


def test_override_abstract_is_published_in_full(B):
    keep, _, _ = B.apply_overrides([record("bib1", "Journal paper", excerpt=True)], {"bib1": {"abstract": "Mine."}})
    assert keep[0]["abstract"] == "Mine." and keep[0]["excerpt"] is False


def test_id_override_keeps_the_old_id(B):
    recs = [record("title:precisioncosmologyusingcosmicvoids", "Precision cosmology using cosmic voids")]
    keep, _, kept = B.apply_overrides(recs, {
        "title:precisioncosmologyusingcosmicvoids": {"id": "title:precisioncosmologyusingvoids"},
        "title:precisioncosmologyusingvoids": {"date": "2024"},   # other fields can use the kept ID
    })
    assert keep[0]["uid"] == "title:precisioncosmologyusingvoids" and keep[0]["date"] == "2024"
    assert kept == {"title:precisioncosmologyusingcosmicvoids": "title:precisioncosmologyusingvoids"}


def test_id_override_cannot_take_another_papers_id(B, capsys):
    recs = [record("2101.00001", "First"), record("2101.00002", "Second")]
    keep, _, kept = B.apply_overrides(recs, {"2101.00002": {"id": "2101.00001"}})
    assert [r["uid"] for r in keep] == ["2101.00001", "2101.00002"] and kept == {}
    assert "already the ID of another paper" in capsys.readouterr().out


# ----------------------------------------------------------------------------- publications-page problems
def entry(n, title, arxiv=None, authors=("Ana Lopez",), bibcode=None):
    return {"n": n, "title": title, "authors": list(authors), "label": "", "url": "", "arxiv": arxiv, "bibcode": bibcode}


def test_entry_problems(B):
    arx = {"2101.00001": {"title": "Neutrino mass from voids", "authors": ["Ana Lopez", "Bo Li"]},
           "2101.00002": {"title": "A different title on arXiv", "authors": ["Ana Lopez"]},
           "2101.00003": {"title": "Galaxy rotation curves", "authors": ["Zed Zulu"]}}
    entries = [entry(1, "Neutrino mass from voids", "2101.00001"),
               entry(2, "Old title on the publications page", "2101.00002"),   # retitled on arXiv, same authors: fine
               entry(3, "Weak lensing peaks", "2101.00003"),                  # different title AND authors
               entry(4, "Missing record", "2101.09999"),
               entry(5, "Neutrino mass from voids", "2101.00001")]            # duplicate
    out = "\n".join(B.entry_problems(entries, arx))
    assert "entry 2" not in out and "entry 1 " not in out
    assert "entry 3" in out and "wrong link?" in out
    assert "entry 4" in out and "no record for 2101.09999" in out
    assert "entry 5" in out and "duplicate of entry 1" in out


@pytest.mark.parametrize("url, expect, problem", [
    ("https://arxiv.org/abs/2101.00001", "https://arxiv.org/abs/2101.00001", False),
    ("HTTP://example.org/x", "HTTP://example.org/x", False),
    ("", "", False),
    ("_static/thesis.pdf", "https://camels.readthedocs.io/en/latest/_static/thesis.pdf", False),
    ("javascript:alert(1)", "", True),
    ("data:text/html,hi", "", True),
    ("mailto:someone@example.org", "", True),
])
def test_safe_url(B, url, expect, problem):
    out, bad = B.safe_url(url)
    assert out == expect and bool(bad) == problem


# ----------------------------------------------------------------------------- safety checks
def test_sanity_check_counts_embeddings_over_arxiv_papers(B):
    emb = {"embedding": {"vector": [1.0]}}
    entries = [entry(i, f"Paper {i}", f"2101.{i:05d}") for i in range(20)] + [entry(20 + i, f"Thesis {i}") for i in range(10)]
    arx = {e["arxiv"]: {} for e in entries if e["arxiv"]}
    s2 = [emb] * 20 + [None] * 10          # no thesis has an embedding: 67% of all entries, 100% of arXiv papers
    assert B.sanity_check(entries, arx, s2, 30) == []
    s2_down = [None] * 30                  # Semantic Scholar failed
    assert any("embeddings for only 0 of 20 arXiv papers" in p for p in B.sanity_check(entries, arx, s2_down, 30))
    assert any("more than 10% fewer" in p for p in B.sanity_check(entries[:20], arx, s2[:20], 30))
    assert any("no entries" in p for p in B.sanity_check([], {}, [], 30))


def page_data(n_links, ads_links, nads):
    papers = [{"r": [0] * n_links}]
    return {"papers": papers, "refsrc": {"both": ads_links, "s2": n_links - ads_links, "ads": 0}, "nads": nads}


def test_compare_with_previous(B):
    prev = page_data(1884, 1798, 276)
    assert B.compare_with_previous(page_data(1900, 1800, 280), prev) == []
    assert B.compare_with_previous(page_data(1700, 1650, 270), prev) == []       # 9.8% fewer links: still fine
    problems = B.compare_with_previous(page_data(1041, 0, 0), prev)             # ADS outage (measured on Quijote)
    assert len(problems) == 3 and "1041 now vs 1884" in problems[0]
    assert B.compare_with_previous(page_data(5, 0, 0), None) == []              # first build: nothing to compare
    assert B.compare_with_previous(page_data(5, 0, 0), page_data(10, 10, 10)) == []   # tiny projects are not judged


class FakeResponse:
    def __init__(self, code):
        self.status_code, self.text = code, "message"

    def json(self):
        return {"response": {"docs": ["doc"]}}


@pytest.mark.parametrize("sequence, result, calls", [
    ([503, requests.Timeout("t"), 200], ["doc"], 3),        # transient problems are retried
    ([429, 429, 429, 429], "ADS HTTP 429", 4),              # gives up after 4 attempts
    ([401, 200], "ADS HTTP 401", 1),                        # a bad token fails at once
    ([requests.ConnectionError("c")] * 4, "ADS request failed", 4),
])
def test_ads_query_retries(B, monkeypatch, sequence, result, calls):
    seq, made = list(sequence), []

    def get(*a, **k):
        x = seq.pop(0)
        made.append(x)
        if isinstance(x, Exception):
            raise x
        return FakeResponse(x)
    monkeypatch.setattr(B.requests, "get", get)
    monkeypatch.setattr(B.time, "sleep", lambda s: None)
    monkeypatch.setattr(B, "ads_token", lambda: "token")
    if isinstance(result, list):
        assert B.ads_query("q", "f") == result
    else:
        with pytest.raises(RuntimeError, match=result):
            B.ads_query("q", "f")
    assert len(made) == calls


# ----------------------------------------------------------------------------- project configs
def test_both_project_configs_load(B):
    from conftest import project_config
    for name in ("camels", "quijote"):
        cfg = project_config(name)
        B.configure(cfg)
        assert B.NAME == cfg["name"] and len(B.TOPIC_NAMES) == len(cfg["topics"]) + 1
        assert all(0 <= k < len(cfg["topics"]) for k in B.TOPIC_SEED.values())
        assert len(B.TOPIC_SHORT) == len(B.TOPIC_NAMES)


@pytest.mark.parametrize("change, message", [
    (lambda c: c["topics"].append({"name": "Topic 11", "probes": []}), "11 topics"),
    (lambda c: c.update(name="CAMELS<script>"), "may use only letters"),
    (lambda c: c.update(homepage="javascript:alert(1)"), "homepage must be an http"),
    (lambda c: c.update(pubs_url="ftp://example.org"), "pubs_url must be an http"),
])
def test_configure_refuses_bad_configs(B, cfg, change, message):
    change(cfg)
    with pytest.raises(SystemExit, match=message):
        B.configure(cfg)


# ----------------------------------------------------------------------------- Semantic Scholar title matches
class S2Response:
    def __init__(self, code, data=None):
        self.status_code, self._data = code, data

    def json(self):
        return self._data

    def raise_for_status(self):
        pass


def test_s2_title_matches_are_kept_between_runs(B, monkeypatch):
    from datetime import date, timedelta
    today = date.today()
    old = (today - timedelta(days=60)).isoformat()            # older than any recheck_days (30-59)
    entries = [entry(1, "Paper on arXiv", "2101.00001"),
               entry(2, "Thesis found before"), entry(3, "Thesis missed recently"),
               entry(4, "Thesis missed long ago"), entry(5, "Brand new thesis"), entry(6, "Thesis while rate limited")]
    nt = B.norm_title
    matches = {nt("Thesis found before"): {"id": "S2-found", "checked": old},
               nt("Thesis missed recently"): {"id": None, "checked": today.isoformat()},
               nt("Thesis missed long ago"): {"id": None, "checked": old}}
    searched = []

    def s2_get(url, method="GET", **kw):
        if url.endswith("/search/match"):
            q = kw["params"]["query"]
            searched.append(q)
            if q == "Thesis while rate limited":
                return S2Response(429)
            if q == "Thesis missed long ago":
                return S2Response(200, {"data": [{"paperId": "S2-late", "title": q}]})   # now known to Semantic Scholar
            return S2Response(404, {"error": "no match"})
        return S2Response(200, [{"paperId": k} for k in kw["json"]["ids"]])              # the batch request
    monkeypatch.setattr(B, "s2_get", s2_get)
    monkeypatch.setattr(B.time, "sleep", lambda s: None)
    recs = B.fetch_semantic_scholar(entries, {}, matches)
    # only the stale miss, the new title and the rate-limited one are searched
    assert searched == ["Thesis missed long ago", "Brand new thesis", "Thesis while rate limited"]
    assert [r["paperId"] if r else None for r in recs] == ["ARXIV:2101.00001", "S2-found", None, "S2-late", None, None]
    assert matches[nt("Thesis missed long ago")] == {"id": "S2-late", "checked": today.isoformat()}
    assert matches[nt("Brand new thesis")] == {"id": None, "checked": today.isoformat()}
    assert nt("Thesis while rate limited") not in matches          # a failed search is tried again next time
    assert matches[nt("Thesis found before")]["checked"] == old    # known IDs are not searched again


def test_recheck_days_spread(B):
    days = {B.recheck_days(f"title{i}") for i in range(200)}
    assert min(days) >= 30 and max(days) <= 59 and len(days) > 20
