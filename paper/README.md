# FathomNet CLEF 2026 Detector Benchmark — Paper

LaTeX paper directory. Edit `v1.tex`.

Copy results out of the Kaggle notebook output dirs into `json/` and
`images/`:
```
.\sync_results.cmd
```

Build once:
```
.\build.cmd
```

Build and auto-recompile on change:
```
.\watch.cmd
```

The PowerShell scripts (`build.ps1`, `watch.ps1`) provide the same commands
for shells that allow local script execution.

Directory layout (mirrors the CMS2015DarkMatterSearch paper scaffold):

```
paper/
  v1.tex                  - main source
  references.bib          - bibliography database
  paper_structure_and_references.md - section outline + citation plan
  README.md               - this file
  .latexmkrc              - latexmk config (pdf mode, nonstopmode)
  build.cmd / build.ps1   - one-shot build
  watch.cmd / watch.ps1   - continuous preview (pvc mode)
  sync_results.cmd/.ps1   - pull notebook outputs into json/ and images/
  images/                 - figures placed here, \includegraphics from here
  json/                   - result JSONs the tables are built from
  assets/                 - any other static assets
```

## Template

`v1.tex` reproduces the preamble and document conventions of
`DarkMatter/CMS2015DarkMatterSearch(HTMHT)/paper/v1.tex` so both papers stay
consistent:

- `\documentclass[11pt,a4paper,twocolumn]{article}` with 1 in margins
- `authblk` author/affiliation block, `orcidlink` for ORCIDs, first-author
  footnote convention
- `booktabs`, `array`, `makecell`, `longtable` (with the infinite-glue patch
  for page-splitting tables), `microtype`, `xurl`, `float`
- `hyperref` with blue links plus `cleveref` for cross-references
- `unsrt` bibliography style via `\bibliographystyle{unsrt}` and
  `\nocite{*}` before it, so every entry in `references.bib` is listed
- a `\section*{Declarations}` block (funding, ethics, conflicts, data and
  code availability) before `\clearpage\onecolumn\raggedbottom\appendix`
- `images/` as the figure directory, referenced as
  `\includegraphics[width=0.48\textwidth]{images/name.png}`

The only addition is a `\todo{...}` macro (red text) marking unwritten prose.
It is defined in a clearly marked block in the preamble — delete that block
and every `\todo` use before submission.

## State

Built and verified: `v1.pdf` (7 pages) compiles with zero errors, zero
undefined references, zero undefined citations, and no overfull boxes. All
prose is stubbed with `\todo{...}` (renders red) until the benchmark runs
finish. Only the Faster R-CNN parameter count is filled in so far.

## Troubleshooting

If MiKTeX tries to pop up an install dialog (fails with
"GUI framework cannot be initialized" when run non-interactively), enable
silent auto-install once:

```powershell
initexmf --set-config-value=[MPM]AutoInstall=1
```
