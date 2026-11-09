"""Check local HTML destinations and anchors without requesting private remote URLs."""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import sys


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.ids = set()
        self.links = []
        self.feed(text)

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        if attributes.get("id"):
            self.ids.add(attributes["id"])
        if tag == "a" and attributes.get("href"):
            self.links.append(attributes["href"])


def main():
    root = Path(sys.argv[1]).resolve()
    pages = {path: Page(path.read_text(encoding="utf-8")) for path in root.rglob("*.html")}
    failures = []
    for path, page in pages.items():
        for href in page.links:
            parts = urlsplit(href)
            if parts.scheme or parts.netloc or href.startswith("/"):
                continue
            target = (path.parent / unquote(parts.path)).resolve() if parts.path else path
            if target.is_dir():
                target /= "index.html"
            if not target.is_relative_to(root) or not target.is_file():
                failures.append(f"{path.relative_to(root)}: missing {href}")
            elif parts.fragment and target in pages and unquote(parts.fragment) not in pages[target].ids:
                failures.append(f"{path.relative_to(root)}: missing anchor {href}")
    for failure in failures:
        print(failure)
    if failures:
        raise SystemExit(1)
    print(f"Checked {len(pages)} HTML pages: all local links and anchors resolve.")


if __name__ == "__main__":
    main()
