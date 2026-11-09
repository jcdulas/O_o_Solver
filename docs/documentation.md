# Read, download and build the documentation

## Read online

The public site is **[O_o_Solver documentation](https://jcdulas.github.io/O_o_Solver/)**.
The [architecture guide](architecture.md) explains the modules and data flow with diagrams.
The [source index](../index.md) and the Markdown guides can also be read directly on GitHub.

The Documentation workflow builds and publishes `main`. Pull requests build the site and validate its local links,
images and diagrams; publication follows a successful build on `main` or a manual run.

## Download HTML

1. Open the [Documentation workflow](https://github.com/jcdulas/O_o_Solver/actions/workflows/documentation.yml).
2. Open its latest successful run on `main`.
3. Download the `O_o_Solver-documentation` artifact from **Artifacts** or the run summary.
4. Extract the ZIP archive and open `index.html`.

Archives are retained for 90 days. Use **Run workflow** to create a fresh archive if an older one has expired.
The published site remains the stable reading address.

For local browser search, serve the extracted directory:

~~~bash
python -m http.server 8000 --bind 127.0.0.1
~~~

Open `http://127.0.0.1:8000/`.

## Build locally

The documentation toolchain uses Python 3.13 or later. It does not require compiling the C++ libraries or WebAssembly.
From the repository root:

~~~bash
python -m venv .venv-docs
# Linux/macOS: source .venv-docs/bin/activate
# PowerShell:  .\.venv-docs\Scripts\Activate.ps1
python -m pip install -r docs/requirements.txt
python docs/diagrams/render.py --check
python -m sphinx -W --keep-going -b html -c docs -d build/docs/doctrees . build/docs/html
python docs/check_html.py build/docs/html
python -m http.server 8000 --bind 127.0.0.1 --directory build/docs/html
~~~

Generated HTML stays under the ignored `build/` directory.

## Architecture images

The architecture figures are standalone SVG files in [images](images/). They retain sharp text when enlarged and
render directly in both the repository and the HTML site. Each embedded image has alternative text and an explanatory
caption in its guide.

Edit the nodes, descriptions and connecting arrows in [figures.json](diagrams/figures.json), then regenerate:

~~~bash
python docs/diagrams/render.py
~~~

The [renderer](diagrams/render.py) uses the Python standard library to generate the SVGs. Its `--check` mode
verifies that the committed images match their specifications; the documentation workflow runs that check before publication.

The reader/solver boundary figure is shared by the two public projects. Keep its two copies aligned when changing
the inter-module contract. Update an image's alternative text and surrounding explanation when its meaning changes.

## Maintain the pages

- Edit the original Markdown guides or module `README.md` files. Sphinx reads them in place.
- Edit API declarations in their original header or TypeScript file. The API reference includes these files directly.
- Update the navigation in `index.md` and `docs/modules.md` when adding a page.
- Keep headings unique within each page so fragment links remain stable.
- Build with warnings treated as errors and run the local-resource checker before pushing.

The downloadable archive includes navigation, styles, images and the search index. Source-code links open the public
repository at `main`.
