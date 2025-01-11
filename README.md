# OSTEP Book PDF Downloader

A program that scrapes the individual chapters of the OSTEP book from the [official website](https://pages.cs.wisc.edu/~remzi/OSTEP/), merges them, and adds an embedded table of contents (so you can jump to specific chapters in the table of contents section of your PDF viewer), including sub-chapters (e.g. sub-chapter 19.1).

Install [uv](https://docs.astral.sh/uv/) and run
```sh
uv run main OSTEP.pdf
```
to have the the generated PDF saved to `./OSTEP.pdf`.

Personally, I found the book a little too tall to read comfortably on a tablet without wasting space, so I added an option to crop the book to a 4:3 aspect ratio, which you can enable with:
```sh
uv run main --crop OSTEP-cropped.pdf
```

You can some basic tests with
```sh
uv run pytest
```
