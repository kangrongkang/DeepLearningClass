# Paper

LaTeX source for the project write-up. **IEEE single-column journal style**
(`\documentclass[journal,onecolumn,11pt]{IEEEtran}`).

## Layout

```
paper/
├── main.tex          # paper body (sections inline)
├── references.bib    # bibliography (BibTeX, used by natbib + plainnat)
├── figures/          # all figures referenced from main.tex (PNG)
├── sanity.py         # static check — sections / cites / figures / braces
└── README.md
```

## Build

Requires a TeX distribution with `pdflatex`, `bibtex`, and the `IEEEtran`
package (TeX Live and MiKTeX both ship it; `tlmgr install IEEEtran` if not
already present).

```bash
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

Or with `latexmk`:

```bash
latexmk -pdf main
```

Overleaf works out of the box (it has IEEEtran preinstalled).

## Fallback to article class

If you don't have IEEEtran available, change the first non-comment line of
`main.tex` from

```latex
\documentclass[journal,onecolumn,11pt]{IEEEtran}
```

to

```latex
\documentclass[11pt,letterpaper]{article}
\usepackage[margin=1in]{geometry}
```

The body of the paper requires no other changes.

## Notes

- Figures live in `figures/` and are kept in sync with `../outputs/figures/`
  by hand. The values they show come from
  `../outputs/predictions/<model>_clean_metrics.json` (clean test set,
  pHash-deduplicated split).
- The bibliography `references.bib` was hand-curated and validated by the
  citation-check agent (Task F7). Every one of the 34 entries was verified
  against arXiv, DOI, JMLR, Kaggle, GitHub, or first-party sources.

## Verifying the source

Run `python sanity.py` from this directory to check:

- All sections present
- No undefined cite keys
- No unused bib keys
- Brace balance OK
- Every `\includegraphics{...}` has a matching file in `figures/`
