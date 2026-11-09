"""Resolve repository file links without copying implementation files into the site."""
from pathlib import Path
import re
from urllib.parse import quote, unquote, urlsplit
from sphinx.util import logging

LOGGER = logging.getLogger(__name__)
LINK = re.compile(r"(?<!!)(\[[^\]\n]+\]\()([^\s)]+)([^)\n]*\))")
INLINE_CODE = re.compile(r"(\x60+[^\x60]*\x60+)")
FENCE = re.compile(r"^\s{0,3}(\x60{3,}|~{3,})")


def rewrite_links(app, docname, source):
    root = Path(app.srcdir).resolve()
    directory = (root / docname).parent

    def replace(match):
        target = match[2]
        parts = urlsplit(target)
        if parts.scheme or parts.netloc or not parts.path:
            return match[0]
        path = (directory / unquote(parts.path)).resolve()
        if not path.is_relative_to(root) or not path.exists():
            LOGGER.warning("Repository link does not exist: %s", target,
                           location=(docname, None), type="docs", subtype="missing-link")
            return match[0]
        relative = path.relative_to(root)
        if path.suffix in (".md", ".rst") and relative.with_suffix("").as_posix() in app.env.found_docs:
            return match[0]
        url = app.config.repository_url + ("/tree/main/" if path.is_dir() else "/blob/main/")
        url += quote(relative.as_posix(), safe="/")
        if parts.fragment:
            url += "#" + parts.fragment
        return match[1] + url + match[3]

    lines = []
    fence = None
    for line in source[0].splitlines(keepends=True):
        marker = FENCE.match(line)
        if marker:
            value = marker[1]
            if fence is None:
                fence = value
            elif value[0] == fence[0] and len(value) >= len(fence):
                fence = None
            lines.append(line)
            continue
        if fence:
            lines.append(line)
            continue
        chunks = INLINE_CODE.split(line)
        lines.append("".join(chunk if index % 2 else LINK.sub(replace, chunk)
                             for index, chunk in enumerate(chunks)))
    source[0] = "".join(lines)


def setup(app):
    app.add_config_value("repository_url", "", "html")
    app.connect("source-read", rewrite_links)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
