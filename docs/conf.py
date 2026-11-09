"""Build the project guides and live API declarations from their original files."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent / "_ext"))
project = "O_o_Solver"
author = "O_o contributors"
release = re.search(r"\bVERSION\s+([0-9.]+)", (ROOT / "CMakeLists.txt").read_text(encoding="utf-8"))[1]
version = release
extensions = ["myst_parser", "repository_links"]
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
language = "en"
exclude_patterns = ["build", "dist", "node_modules", ".git", ".github", ".venv*", ".vscode", ".idea",
                    "include", "CLAUDE.md", "**/__pycache__"]
myst_heading_anchors = 6
myst_enable_extensions = ["colon_fence", "deflist", "fieldlist"]
html_theme = "furo"
html_title = project + " documentation"
html_show_sphinx = False
html_show_copyright = False
html_theme_options = {
    "source_repository": "https://github.com/jcdulas/O_o_Solver",
    "source_branch": "main",
    "source_directory": "",
    "light_css_variables": {"color-brand-primary": "#245d78", "color-brand-content": "#245d78"},
}
repository_url = "https://github.com/jcdulas/O_o_Solver"

html_baseurl = "https://jcdulas.github.io/O_o_Solver/"
