"""Download the Dresden manuscript text of the Book of Dede Korkut from Turkish Wikisource and clean it.

Source: https://tr.wikisource.org/wiki/Kitâb-ı_Dedem_Korkut_Alâ_Lisan-ı_Tâife-i_Oğuzân (CC BY-SA 4.0).
Writes data/boys/<nn>.txt (one file per story), data/korkut.txt (all of it) and data/SOURCE.json
(page revisions, so the exact text can be traced back).

    uv run python prepare.py            # download (cached in data/pages/) and clean
    uv run python prepare.py --offline  # clean from the cache only
"""

import argparse
import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://tr.wikisource.org/w/api.php"
UA = {"User-Agent": "korkut-gpt/0.1 (eqlx.ai@gmail.com)"}
BOOK = "Kitâb-ı Dedem Korkut Alâ Lisan-ı Tâife-i Oğuzân"
INDEX = "Kitab-i Dedem Korkut - 16th century manuscript in Old Turkish.pdf"
DATA = Path(__file__).parent / "data"

# "Son sayfa" is a later poem on the last leaf, not part of the book.
SKIP = {"Son sayfa"}


def api(**params) -> dict:
    params["format"] = "json"
    url = API + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA)) as r:
        return json.load(r)


def raw(title: str) -> tuple[str, int]:
    """Wikitext and revision id of a page."""
    pages = api(
        action="query", prop="revisions", titles=title, rvprop="content|ids", rvslots="main"
    )["query"]["pages"]
    rev = next(iter(pages.values()))["revisions"][0]
    return rev["slots"]["main"]["*"], rev["revid"]


def page_range(story: str) -> dict:
    """First and last manuscript page of a story, and the sections it takes on shared pages."""
    text, _ = raw(f"{BOOK}/{story}")
    tag = re.search(r"<pages\s[^>]*>", text)
    if not tag:
        sys.exit(f"no <pages> tag in {story}")
    attrs = dict(re.findall(r'(\w+)="?([^"\s/>]+)"?', tag.group(0)))
    return {
        "pages": [int(attrs["from"]), int(attrs["to"])],
        "fromsection": attrs.get("fromsection"),
        "tosection": attrs.get("tosection"),
    }


def section(wikitext: str, name: str | None) -> str:
    """The part of a page between <section begin=name/> and <section end=name/>; the whole page
    when no name is given or the page has no such section (Wikisource does the same)."""
    m = name and re.search(
        rf'<section begin="?{name}"?\s*/>(.*?)<section end="?{name}"?\s*/>', wikitext, re.DOTALL
    )
    return m.group(1) if m else wikitext


def without(wikitext: str, name: str | None) -> str:
    """The page with one section taken out: the part a neighbouring story claims."""
    if not name:
        return wikitext
    return re.sub(
        rf'<section begin="?{name}"?\s*/>.*?<section end="?{name}"?\s*/>',
        "",
        wikitext,
        flags=re.DOTALL,
    )


def clean(wikitext: str) -> str:
    """Strip wiki markup from one manuscript page; keep the words and line breaks."""
    t = re.sub(r"<noinclude>.*?</noinclude>", "", wikitext, flags=re.DOTALL)
    t = re.sub(r"<section[^>]*/>", "", t)
    t = re.sub(r"</?(poem|font|span|div|center|small|big|sup|ref)[^>]*>", "", t)
    t = re.sub(r"<br\s*/?>", "\n", t)
    t = re.sub(r"\{\{[^{}]*\}\}", "", t)  # templates
    t = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", t)  # links -> label
    t = t.replace("'''", "").replace("''", "")
    t = t.replace("­", "")  # soft hyphens
    t = unicodedata.normalize("NFC", t)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in t.split("\n")]
    return "\n".join(line for line in lines if line)


def lower_tr(text: str) -> str:
    """Turkish lower case: I -> ı and İ -> i (str.lower() gets both wrong)."""
    return text.replace("I", "ı").replace("İ", "i").lower()


def normalize(text: str) -> str:
    """One spelling for the model: Turkish lower case, no digits (a few stray typos like "vez3iri")."""
    return re.sub(r"[0-9]", "", lower_tr(text))


def download() -> None:
    pages_dir = DATA / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    source = {"book": BOOK, "index": INDEX, "license": "CC BY-SA 4.0", "stories": []}
    titles = api(action="query", list="allpages", apprefix=BOOK + "/", aplimit=50)["query"][
        "allpages"
    ]
    stories = [t["title"].split("/", 1)[1] for t in titles]
    # Manuscript order = page order: the introduction, then the twelve stories.
    ranges = sorted(
        ((page_range(s), s) for s in stories if s not in SKIP), key=lambda x: x[0]["pages"]
    )
    for r, story in ranges:
        first, last = r["pages"]
        revs = {}
        for n in range(first, last + 1):
            cache = pages_dir / f"{n:03d}.txt"
            if not cache.exists():
                text, revid = raw(f"Sayfa:{INDEX}/{n}")
                cache.write_text(text)
                revs[n] = revid
                time.sleep(0.3)
        source["stories"].append({"title": story, **r, "revisions": revs})
        print(f"pages {first}-{last}: {story}")
    old = DATA / "SOURCE.json"
    if old.exists():  # keep revision ids recorded on an earlier run
        prev = {s["title"]: s["revisions"] for s in json.loads(old.read_text())["stories"]}
        for s in source["stories"]:
            s["revisions"] = prev.get(s["title"], {}) | s["revisions"]
    old.write_text(json.dumps(source, ensure_ascii=False, indent=1))


def build() -> None:
    source = json.loads((DATA / "SOURCE.json").read_text())
    boys = DATA / "boys"
    boys.mkdir(exist_ok=True)
    parts = []
    stories = source["stories"]
    for i, s in enumerate(stories):
        first, last = s["pages"]
        prev = stories[i - 1] if i else None
        nxt = stories[i + 1] if i + 1 < len(stories) else None
        pages = []
        for n in range(first, last + 1):
            text = (DATA / "pages" / f"{n:03d}.txt").read_text()
            # A page shared with a neighbour: take our section, or else everything but theirs.
            if n == first:
                text = section(text, s["fromsection"])
                if not s["fromsection"] and prev and prev["pages"][1] == n:
                    text = without(text, prev["tosection"])
            if n == last:
                text = section(text, s["tosection"])
                if not s["tosection"] and nxt and nxt["pages"][0] == n:
                    text = without(text, nxt["fromsection"])
            pages.append(normalize(clean(text)))
        # A page may end mid-sentence; pages are joined with a space, never mid-word.
        text = re.sub(r"[ ]*\n[ ]*", "\n", " ".join(pages)).strip() + "\n"
        (boys / f"{i:02d}.txt").write_text(text)
        parts.append(text)
        print(f"{i:02d} {len(text):6d} chars  {s['title']}")
    full = "\n".join(parts)
    (DATA / "korkut.txt").write_text(full)
    print(
        f"total {len(full)} chars, {len(full.split())} words, {len(set(full))} distinct characters"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--offline", action="store_true", help="clean from data/pages/ without downloading"
    )
    if not ap.parse_args().offline:
        download()
    build()
