#!/usr/bin/env python3
"""
Helper for tagging new papers (see CLAUDE.md).

  python3 tagging.py <project> list         # untagged papers with abstracts, plus the allowed tag names (for Claude)
  python3 tagging.py <project> unassigned   # papers in the "Unassigned" topic, with the topic names (for review)
  python3 tagging.py <project> issue        # markdown body for the "New papers need tags" GitHub issue
  python3 tagging.py <project> count        # number of untagged papers (used by the GitHub job)
  python3 tagging.py <project> topics       # every topic with all its papers (for naming topics)

<project> is a folder in projects/ (e.g. camels, quijote). Reads the data embedded in <project>/index.html,
so run it after a build.
"""
import difflib
import json
import os
import re
import sys
import unicodedata


def data(path):
    h = open(path, encoding="utf-8").read()
    return json.loads(h.split('id="atlas-data">', 1)[1].split("</script>", 1)[0].replace("<\\/", "</"))


def norm_title(t):
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", t).lower())


def renamed(proj, P):
    """Untagged papers that are probably a tagged paper under a new ID (its title was edited on the
    publications page, or its link changed): [(paper, old ID, old title)]. The fix is an "id" override."""
    tags = json.load(open(os.path.join("projects", proj, "tags.json"), encoding="utf-8"))
    ids = {p["id"] for p in P}
    old = {k: t.get("title", "") for k, t in tags.items() if k not in ids}
    out = []
    for p in P:
        if p["tg"] or not old:
            continue
        score, k = max((difflib.SequenceMatcher(None, norm_title(p["t"]), norm_title(t)).ratio(), k) for k, t in old.items())
        if score >= 0.8:
            out.append((p, k, old[k]))
    return out


def rename_lines(proj, ren):
    if not ren:
        return []
    lines = [f"**Possibly renamed:** these look like papers that were already tagged under another ID. Instead of "
             f"tagging them again, keep their old ID by adding to `projects/{proj}/overrides.json`:", ""]
    for p, k, t in ren:
        lines.append(f"- {p['t'][:90]} (was: {t[:90]})")
        lines.append(f"  `{json.dumps(p['id'])}: {{\"id\": {json.dumps(k)}}}`")
    return lines + [""]


def main():
    if len(sys.argv) < 2 or not os.path.isdir(os.path.join("projects", sys.argv[1])):
        sys.exit(__doc__)
    proj = sys.argv[1]
    cmd = sys.argv[2] if len(sys.argv) > 2 else "list"
    D = data(os.path.join(proj, "index.html"))
    P, TOP, name = D["papers"], D["topics"], D["proj"]["name"]
    untagged = [p for p in P if not p["tg"]]
    if cmd == "count":
        print(len(untagged))
    elif cmd == "list":
        print(f"Allowed tag names (projects/{proj}/tags.json uses exactly these):")
        for axis, names in D["tagaxes"].items():
            print(f"  {axis}: " + " | ".join(names))
        if D.get("notes"):
            print("\nProblems on the publications page (fix them there or in overrides.json):")
            for n_ in D["notes"]:
                print(f"  - {n_}")
        ren = renamed(proj, P)
        if ren:
            print("\n" + "\n".join(rename_lines(proj, ren)))
        print(f"\n{len(untagged)} untagged paper(s):\n")
        for p in untagged:
            print(f"### {p['id']}\nTitle: {p['t']}\nDate: {p['d']}   Topic: {TOP[p['k']]}\n"
                  f"Abstract: {p['ab'] or '(no abstract available; tag from the title and mark the tags uncertain)'}\n")
    elif cmd == "unassigned":
        print("Topics: " + " | ".join(f"{k}: {t}" for k, t in enumerate(TOP)) + "\n")
        for p in P:
            if TOP[p["k"]].startswith("Unassigned"):
                print(f"### {p['id']}\nTitle: {p['t']}\nAbstract: {(p['ab'] or '(none)')[:600]}\n")
    elif cmd == "topics":
        for k, t in enumerate(TOP):
            mem = sorted((p for p in P if p["k"] == k), key=lambda p: -p["c"])
            if mem:
                print(f"\n[{k}] {t}  ({len(mem)} papers)")
                for p in mem:
                    print(f"  {p['id']:<22} {p['c']:>5}  {p['t'][:110]}")
    elif cmd == "issue":
        lines = [f"The {D['fetched']} update of the **{name}** atlas added {len(untagged)} paper(s) that are not "
                 "tagged yet (they are already on the site, shown as \"Not tagged yet\"):", ""]
        lines += [f"- [{p['t']}]({p['url'] or ''}) ({p['id']})" for p in untagged]
        lines += [""] + rename_lines(proj, renamed(proj, P))
        if D.get("notes"):
            lines += ["**Problems on the publications page** (fix them there, or in "
                      f"`projects/{proj}/overrides.json`):", ""] + [f"- {n_}" for n_ in D["notes"]] + [""]
        lines += ["", "**To tag them:** open Claude Code in your local `literature-atlas` folder and say "
                  f"*\"pull, then tag the new {name} papers\"*. It follows `CLAUDE.md` and closes this issue when done.",
                  "", "@franciscovillaescusa"]
        print("\n".join(lines))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
