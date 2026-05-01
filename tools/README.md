# tools/

Maintainer-only Python tooling for basin. Not part of the published crate
(see `Cargo.toml` `exclude`).

Currently scoped to **paper ingestion** for solver implementation work.

## Setup

The devenv shell auto-syncs this project (`languages.python.directory = "./tools"`
in `devenv.nix`), so entering the shell installs everything. Manually:

```sh
cd tools && uv sync
```

## Paper ingestion: two-stage pipeline

We use two parsers with very different speed/quality trade-offs, run from
recipes in the repo-root `Taskfile.yml`.

### Stage 1 — fast pass with `pymupdf4llm`

```sh
task ingest-paper PDF=~/Downloads/paper.pdf NAME=lbfgs
```

Outputs `references/lbfgs/source.{pdf,md}`. Takes ~1s. Good for prose,
section structure, bibliography, and pseudocode. **Mangles equations**
(sub/superscripts become bracket noise) and **drops figures** (replaces
with placeholders).

For most solver papers, this is enough — the algorithm pseudocode is what
gets translated, and equations are usually rederived inline in code comments.

### Stage 2 — selective marker pass

If specific pages have math or figures you actually need:

```sh
task ingest-paper-pages NAME=lbfgs PAGES="3-4,7"
```

Outputs `references/lbfgs/source.marker.md`. Pages are 0-indexed (marker
convention). Slow on CPU (~minutes per page) — first run also downloads
~2-3 GB of models into `~/.cache/`.

The two outputs are kept side-by-side rather than spliced. When translating
from the paper, read `source.md` for structure and prose, switch to
`source.marker.md` for the specific equations or figures that need fidelity.

## Why these tools

- **`pymupdf4llm`** — pure-Python (via PyMuPDF C++ binding), no ML, fast.
  Strictly better than `pypdf` (which gives raw text with no markdown
  structure and worse hyphen/ligature handling).
- **`marker-pdf`** — ML pipeline, LaTeX equation rendering, figure handling.
  AMD GPU acceleration is possible via PyTorch ROCm but not configured here
  (would force a heavy ROCm-built torch wheel on every contributor). Run on
  a separate ROCm-enabled env if you want GPU and drop the resulting
  `source.marker.md` into the right `references/` directory.

## Adding a new tool

Add to `pyproject.toml` under `[project.dependencies]`, run `uv lock` from
this directory, and commit the updated `uv.lock`. The lockfile is the
authoritative pin — version constraints in `pyproject.toml` stay loose.
