#!/usr/bin/env python3
"""
Smoke test for the built site: opens index.html in a headless browser, exercises every view and the
main controls, and fails if the page throws a JavaScript error or something essential is missing.

Usage
  python3 check_atlas.py                   # checks ./index.html
  python3 check_atlas.py --page other.html
  python3 check_atlas.py --chrome          # use the installed Google Chrome instead of Playwright's Chromium

Requires: pip install playwright   (then, unless using --chrome: python -m playwright install chromium)
"""
import argparse
import pathlib
import sys

from playwright.sync_api import sync_playwright

VIEWS = ["topics", "citations", "coauthors", "table", "trends"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", default="index.html")
    ap.add_argument("--chrome", action="store_true", help="use the locally installed Google Chrome")
    args = ap.parse_args()
    url = pathlib.Path(args.page).resolve().as_uri()
    errors, checks = [], []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome" if args.chrome else None)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.on("pageerror", lambda e: errors.append(str(e)))
        # outside requests (fonts, analytics) are irrelevant here and must not slow or break the test
        page.route("**/*", lambda r: r.continue_() if r.request.url.startswith("file:") else r.abort())
        page.goto(url + "#topics")
        page.wait_for_timeout(800)

        stats = page.inner_text("#stats")
        n_papers = int(stats.split()[0])
        check("page shows papers", n_papers > 50, stats)

        for v in VIEWS:
            page.click(f"#m-{v}")
            page.wait_for_timeout(300)
            check(f"view '{v}' activates", page.get_attribute(f"#m-{v}", "aria-pressed") == "true")
        page.click("#m-table")
        page.wait_for_timeout(300)
        rows = page.locator("#tablewrap tbody tr").count()
        check("table lists every paper", rows == n_papers, f"{rows} rows")
        page.click("#m-trends")
        page.wait_for_timeout(300)
        check("trends chart drawn", page.locator("#trendswrap rect.tseg").count() > 0)

        for v in ("topics", "citations", "coauthors"):
            page.click(f"#m-{v}")
            page.click("#dispbtn")          # open the Display menu (it stays open while choosing)
            page.click("#d3")
            page.wait_for_timeout(300)
            page.click("#d2")
            page.wait_for_timeout(200)
            page.keyboard.press("Escape")    # close it
        check("2D/3D toggles in all map views", page.get_attribute("#d2", "aria-pressed") == "true")

        page.click("#fbtn")
        check("filters panel opens", page.is_visible("#fpanel"))
        page.select_option("#f-method", index=1)
        page.wait_for_timeout(200)
        check("filter narrows the papers", "of" in page.inner_text("#stats"), page.inner_text("#stats"))
        check("active-filter chip shown", page.locator("#active .achip").count() >= 1)
        page.click("#reset")
        page.wait_for_timeout(200)

        page.click("#m-topics")
        page.wait_for_timeout(300)
        page.locator("#panel [data-p]").first.click()
        page.wait_for_timeout(300)
        check("selecting a paper opens its panel", page.locator("#panel h2").count() == 1)
        check("selection is in the address", "paper=" in page.evaluate("location.hash"))

        page.fill("#q", "diffusion")
        page.wait_for_timeout(400)
        check("search finds papers", "match" in page.inner_text("#qcount"), page.inner_text("#qcount"))
        browser.close()

    width = max(len(n) for n, _, _ in checks)
    for name, ok, detail in checks:
        print(f"{'ok  ' if ok else 'FAIL'}  {name.ljust(width)}  {detail}")
    for e in errors:
        print(f"JS ERROR  {e}")
    failed = [n for n, ok, _ in checks if not ok]
    if failed or errors:
        sys.exit(f"\ncheck failed: {len(failed)} failed check(s), {len(errors)} JavaScript error(s)")
    print(f"\nall {len(checks)} checks passed, no JavaScript errors")


if __name__ == "__main__":
    main()
