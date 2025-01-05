from dataclasses import dataclass
from io import BytesIO
import logging
import os
from pprint import pprint
from typing import List, Tuple
from urllib.parse import urljoin
from bs4 import BeautifulSoup, Tag
import pymupdf
import requests
import requests_cache


OSTEP_URL = "https://pages.cs.wisc.edu/~remzi/OSTEP/"


@dataclass
class SubChapter:
    title: str
    page_num: int


@dataclass
class Chapter:
    title: str
    page_num: int
    subchapters: List[SubChapter]


@dataclass
class Part:
    title: str
    page_num: int
    chapters: List[Chapter]


@dataclass
class Book:
    title: str
    author: str
    description: str
    parts: List[Part]


@dataclass
class PartStart:
    title: str


@dataclass
class ChapterPdf:
    in_memory_file: BytesIO


@dataclass
class ChapterPdfUrl:
    pdf_url: str

    def download(self) -> ChapterPdf:
        """Download pdf to in-memory file."""
        logging.info(f"Downloading {self.pdf_url}")
        response = requests.get(self.pdf_url)
        in_memory_file = BytesIO(response.content)
        return ChapterPdf(in_memory_file)


def parse_chapter(
    chapter_pdf: ChapterPdf, starting_page_num: int
) -> Tuple[Chapter, int]:
    """
    Return the parsed chapter and the number of pages in this chapter.
    """
    doc = pymupdf.open(stream=chapter_pdf.in_memory_file, filetype="pdf")
    page_count = len(doc)

    chapter_idx = None
    chapter_title = None
    chapter_page_num = None
    subchapters: List[SubChapter] = []

    for page_num in range(0, page_count):
        page = doc.load_page(page_num)
        page_dict = page.get_text("dict")

        for block in page_dict["blocks"]:
            for line in block["lines"]:
                for span in line["spans"]:
                    font_size: float = span["size"]
                    text: str = span["text"].strip()

                    # We identify headings by their (large) font sizes.
                    if 20 < font_size and text.isdigit():
                        chapter_idx = int(text)
                    elif 12 < font_size < 15:
                        chapter_title = text
                        chapter_page_num = starting_page_num + page_num
                    elif 10 < font_size < 12:
                        subchapter = SubChapter(
                            text, starting_page_num + page_num
                        )
                        subchapters.append(subchapter)

    assert chapter_title is not None and chapter_page_num is not None

    if chapter_idx is not None:
        chapter_title = f"{chapter_idx} {chapter_title}"

    chapter = Chapter(chapter_title, chapter_page_num, subchapters)

    return chapter, page_count


Cell = PartStart | ChapterPdfUrl | None
NonEmptyCell = PartStart | ChapterPdfUrl


def parse_cell(td_tag: Tag) -> Cell:
    if (a_tag := td_tag.find("a")) is not None:
        pdf_url = urljoin(OSTEP_URL, a_tag["href"])
        return ChapterPdfUrl(pdf_url)
    elif (bold_tag := td_tag.find("b")) is not None and (
        title := bold_tag.get_text(strip=True)
    ) != "":
        return PartStart(title)
    else:
        return None


def scrape_chapters_table() -> Tuple[int, int, List[List[Cell]]]:
    """
    Return row count, col count and chapters table.
    """
    response = requests.get(OSTEP_URL)
    soup = BeautifulSoup(response.text, "html.parser")

    chapters_table = soup.find_all("table")[3]
    chapters_table_raw_rows = [
        list(tr.find_all("td")) for tr in chapters_table.find_all("tr")
    ]

    row_count = len(chapters_table_raw_rows)
    col_count = len(chapters_table_raw_rows[0])
    chapters_table_rows = [
        [
            parse_cell(chapters_table_raw_rows[row][col])
            for col in range(col_count)
        ]
        for row in range(row_count)
    ]
    return row_count, col_count, chapters_table_rows


def scrape_parts_chapter_urls() -> List[Tuple[PartStart, List[ChapterPdfUrl]]]:
    """
    Return a tuples of the part start and the pdf urls to all chapters in that
    part.
    """
    row_count, col_count, chapters_table_rows = scrape_chapters_table()

    # Iterate through columns one by one, filtering out None cells.
    it = iter(
        cell
        for col in range(col_count)
        for row in range(row_count)
        if (cell := chapters_table_rows[row][col]) is not None
    )

    parts: List[Tuple[PartStart, List[ChapterPdfUrl]]] = []
    curr_part_chapters: List[ChapterPdfUrl] = []

    match next(it):
        case PartStart(title) as part_start:
            curr_part_start = part_start
        case _:
            raise Exception("first table entry must be a part start")

    for i in it:
        match i:
            case ChapterPdfUrl(_) as chapter_pdf:
                curr_part_chapters.append(chapter_pdf)
            case PartStart(_) as part_start:
                # This ends the previous part.
                parts.append((curr_part_start, curr_part_chapters))

                # Prepare for next part.
                curr_part_start = part_start
                curr_part_chapters = []

    return parts


def scrape_parts_chapter_pdfs() -> List[Tuple[PartStart, List[ChapterPdf]]]:
    # TODO: download in parallel across multiple threads
    return [
        (part, list(map(ChapterPdfUrl.download, chapter_urls)))
        for (part, chapter_urls) in scrape_parts_chapter_urls()
    ]


def parse_book() -> Book:
    page_num = 0

    parts: List[Part] = []
    # with tempfile.TemporaryDirectory(delete=True) as tmp_dir:
    for part_start, chapter_pdf_paths in scrape_parts_chapter_pdfs():
        # Assume the part starts at the first page of its first chapter.
        part_page_num = page_num
        part_title = part_start.title

        chapters: List[Chapter] = []
        for path in chapter_pdf_paths:
            chapter, page_count = parse_chapter(path, page_num)
            chapters.append(chapter)
            page_num += page_count

        part = Part(part_title, part_page_num, chapters)
        parts.append(part)

    # TODO: scrape these too
    title = ""
    author = ""
    description = ""
    book = Book(title, author, description, parts)

    return book


def setup_logging():
    logging.basicConfig(level=logging.INFO)


def setup_requests_cache():
    CACHE_DIR = "/tmp/ostep-downloader"
    os.makedirs(CACHE_DIR, exist_ok=True)
    requests_cache.install_cache(
        cache_name=CACHE_DIR,
        backend="filesystem",
        expire_after=None,
    )


def main():
    setup_logging()
    setup_requests_cache()

    # TODO: take as arg where to save output pdf

    # print(parse_chapter("/home/fritz/Downloads/cpu-intro.pdf", 0))
    # print(parse_chapter("/home/fritz/Downloads/dialogue-virtualization.pdf", 0))
    # print(parse_chapter("/home/fritz/Downloads/toc.pdf", 0))

    book = parse_book()
    pprint(book)


if __name__ == "__main__":
    main()
