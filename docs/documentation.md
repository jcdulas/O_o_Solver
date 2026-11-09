# Build and download the documentation

## Read it on GitHub

The [documentation index](../index.md) and its linked Markdown guides are available directly in this repository.
The API reference in the HTML site includes the current declaration files.

## Download the private HTML site

1. Open the [Documentation workflow](https://github.com/jcdulas/O_o_Solver/actions/workflows/documentation.yml).
2. Open its latest successful run on `main`.
3. Download the `O_o_Solver-documentation` artifact from **Artifacts** or the link in the run summary.
4. Extract the ZIP archive and open `index.html`.

The artifact requires repository access. Builds are not published to GitHub Pages. Artifacts are retained for 90 days;
if an older artifact has expired, use **Run workflow** to create another one from `main`.

For browser features such as search that need HTTP access, serve the extracted directory locally:

~~~bash
python -m http.server 8000 --bind 127.0.0.1
~~~

Open `http://127.0.0.1:8000/`. The server is bound to the local machine only.

## Build locally

The documentation toolchain uses Python 3.13 or later. It does not require compiling the C++ libraries or WebAssembly.
From the repository root:

~~~bash
python -m venv .venv-docs
# Linux/macOS: source .venv-docs/bin/activate
# PowerShell:  .\.venv-docs\Scripts\Activate.ps1
python -m pip install -r docs/requirements.txt
python -m sphinx -W --keep-going -b html -c docs -d build/docs/doctrees . build/docs/html
python docs/check_html.py build/docs/html
python -m http.server 8000 --bind 127.0.0.1 --directory build/docs/html
~~~

Open `http://127.0.0.1:8000/`. The generated files stay under the ignored `build/` directory.

## Maintain the pages

- Edit the original Markdown guides or module `README.md` files. Sphinx reads them in place.
- Edit API declarations in their original header or TypeScript file. The reference includes those files directly.
- Update the navigation in `index.md` and `docs/modules.md` when adding a page.
- Keep headings unique within each page so that stable fragment links can be generated.
- Build with warnings treated as errors and run the local-link checker before pushing.

The workflow runs on documentation/API changes and on pull requests, and can also be started manually. Each successful
build archives the complete HTML site with its navigation, stylesheet and search index. Source-code links point to
the corresponding private repository; they do not copy the implementation into the site.
