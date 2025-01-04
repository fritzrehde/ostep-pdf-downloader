# OSTEP Book PDF Scraper

A program that scrapes the individual chapters of the OSTEP book from the [official website](https://pages.cs.wisc.edu/~remzi/OSTEP/), merges them, and adds chapters/outline items (so you can jump to specific chapters in the Table of Content section of your PDF viewer), including sub-chapters (e.g. sub-chapter 19.1).

Install [uv](https://docs.astral.sh/uv/) and run
```sh
uv run main OSTEP.pdf
```
to have the the generated PDF saved to `./OSTEP.pdf`.
