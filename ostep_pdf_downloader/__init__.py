import asyncio
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from itertools import groupby
import logging
import os
from pprint import pprint
import sys
from typing import List, Tuple
from urllib.parse import urljoin
import aiohttp
from bs4 import BeautifulSoup, Tag
import pymupdf
import requests
import requests_cache


OSTEP_URL = "https://pages.cs.wisc.edu/~remzi/OSTEP/"


class Indexing(Enum):
    Zero = 0
    One = 1


@dataclass
class Offset:
    offset: int

    def apply(self, value: int) -> int:
        return value + self.offset


@dataclass
class PageNum:
    page_num: int
    actual_indexing: Indexing
    offset: Offset

    def get(self, desired_indexing: Indexing) -> int:
        page_num = self.offset.apply(self.page_num)
        match (self.actual_indexing, desired_indexing):
            case (Indexing.Zero, Indexing.Zero) | (Indexing.One, Indexing.One):
                # Actual and desired indexing is the same.
                return page_num
            case (Indexing.Zero, Indexing.One):
                # Convert 0-indexed to 1-indexed.
                return page_num + 1
            case (Indexing.One, Indexing.Zero):
                # Convert 1-indexed to 0-indexed.
                return page_num - 1
            case _:
                raise Exception("unreachable")

    def add(self, pages: int):
        self.page_num += pages


@dataclass
class TocEntry:
    # 1-based.
    lvl: int
    title: str
    page_num: PageNum


Toc = List[TocEntry]


@dataclass
class SubChapter:
    title: str
    page_num: PageNum

    def toc_entry(self) -> TocEntry:
        return TocEntry(lvl=3, title=self.title, page_num=self.page_num)


@dataclass
class Chapter:
    title: str
    page_num: PageNum
    subchapters: List[SubChapter]

    def toc_entry(self) -> TocEntry:
        return TocEntry(lvl=2, title=self.title, page_num=self.page_num)


@dataclass
class Pdf:
    in_mem_file: BytesIO

    def write_to_file(self, dst_file_path: str):
        with open(dst_file_path, "wb") as f:
            f.write(self.in_mem_file.getbuffer())


@dataclass
class ChapterPdfUrl:
    pdf_url: str

    async def download(self, session: aiohttp.ClientSession) -> Pdf:
        """Download pdf to in-memory file."""
        async with session.get(self.pdf_url) as response:
            logging.info(f"Downloading {self.pdf_url}")
            content = await response.read()
            in_mem_file = BytesIO(content)
            return Pdf(in_mem_file)


@dataclass
class Part:
    title: str
    page_num: PageNum
    chapters: List[Chapter]

    def toc_entry(self) -> TocEntry:
        return TocEntry(lvl=1, title=self.title, page_num=self.page_num)


@dataclass
class PartStart:
    title: str


# TODO: am i closing all pymupdf.open()'s correctly?


def doc_to_pdf(doc) -> Pdf:
    pdf = BytesIO()
    doc.save(pdf)
    return Pdf(pdf)


@dataclass
class Book:
    title: str
    author: str
    description: str
    parts: List[Part]
    ordered_pdf_chapters: List[Pdf]

    def build_toc(self) -> Toc:
        toc: Toc = []

        for part in self.parts:
            toc.append(part.toc_entry())
            for chapter in part.chapters:
                toc.append(chapter.toc_entry())
                for subchapter in chapter.subchapters:
                    toc.append(subchapter.toc_entry())

        return toc

    def generate_merged_pdf(self) -> Pdf:
        logging.info("Generating merged pdf")

        with pymupdf.open() as merged_doc:
            # Add all chapter pdf files.
            for pdf in self.ordered_pdf_chapters:
                with pymupdf.open(
                    stream=pdf.in_mem_file, filetype="pdf"
                ) as src_doc:
                    merged_doc.insert_pdf(src_doc)

            # Add TOC.
            toc = self.build_toc()
            toc_formatted: List[Tuple[int, str, int]] = [
                (x.lvl, x.title, x.page_num.get(desired_indexing=Indexing.One))
                for x in toc
            ]
            merged_doc.set_toc(toc_formatted)

            return doc_to_pdf(merged_doc)


def parse_chapter(
    chapter_pdf: Pdf, starting_page_num: int, offset: Offset
) -> Tuple[Chapter, int]:
    """
    Return the parsed chapter and the number of pages in this chapter.
    """
    doc = pymupdf.open(stream=chapter_pdf.in_mem_file, filetype="pdf")
    page_count = len(doc)

    chapter_idx = None
    chapter_title = None
    chapter_page_num = None
    subchapters: List[SubChapter] = []

    for page_num in range(0, page_count):
        page = doc.load_page(page_num)
        page_dict = page.get_text("dict")

        for block in page_dict["blocks"]:
            for font_size, lines in groupby(
                block["lines"], key=lambda line: line["spans"][0]["size"]
            ):
                # We define a "line block" as consecutive lines as long as they
                # have the same font size, to enable parsing titles that span
                # over multiple lines.

                # The lines in a "line block" are joined with a space as separator.
                text: str = " ".join(
                    # Each line is joined together, ignoring differing fonts.
                    ("".join(span["text"] for span in line["spans"]))
                    for line in lines
                )

                # We identify headings by their (large) font sizes.
                if 20.662 < font_size and text.isdigit():
                    chapter_idx = int(text)
                elif 12.575 < font_size < 12.576 or 14.346 < font_size < 14.347:
                    chapter_title = text
                    chapter_page_num = PageNum(
                        starting_page_num + page_num, Indexing.Zero, offset
                    )
                elif 10.909 < font_size < 10.910:
                    subchapter = SubChapter(
                        text,
                        PageNum(
                            starting_page_num + page_num,
                            Indexing.Zero,
                            offset,
                        ),
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
    logging.info(f"Scraping chapters table from {OSTEP_URL}")

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
        case PartStart(_) as part_start:
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


async def scrape_parts_chapter_pdfs() -> List[Tuple[PartStart, List[Pdf]]]:
    """
    For each part (start), also return its chapter pdfs.
    """
    # Download pdf files concurrently.
    async with aiohttp.ClientSession() as session:

        async def download_chapters_for_part(
            part: PartStart, chapter_urls: List[ChapterPdfUrl]
        ) -> Tuple[PartStart, List[Pdf]]:
            futures = [
                chapter_url.download(session) for chapter_url in chapter_urls
            ]
            downloaded_pdfs = await asyncio.gather(*futures)
            return part, downloaded_pdfs

        futures = [
            download_chapters_for_part(part, chapter_urls)
            for (part, chapter_urls) in scrape_parts_chapter_urls()
        ]
        return await asyncio.gather(*futures)


async def parse_book() -> Book:
    # Add the same mutable offset to every page number, so we can later add a
    # cover page (offset = 1).
    offset = Offset(0)

    page_num = 0

    parts: List[Part] = []
    ordered_chapter_pdfs: List[Pdf] = []
    for part_start, chapter_pdfs in await scrape_parts_chapter_pdfs():
        # Assume the part starts at the first page of its first chapter.
        part_page_num = PageNum(page_num, Indexing.Zero, offset)
        part_title = part_start.title

        chapters: List[Chapter] = []
        for chapter_pdf in chapter_pdfs:
            chapter, page_count = parse_chapter(chapter_pdf, page_num, offset)
            logging.info(f"Parsing chapter: {chapter.title}")
            chapters.append(chapter)
            ordered_chapter_pdfs.append(chapter_pdf)
            page_num += page_count

        part = Part(part_title, part_page_num, chapters)
        parts.append(part)

    # TODO: scrape these too
    title = ""
    author = ""
    description = ""
    book = Book(title, author, description, parts, ordered_chapter_pdfs)

    # TODO: add cover page

    return book


def setup_logging():
    logging.basicConfig(level=logging.INFO)


# TODO: remove once done, only here to speed up debugging
def setup_requests_cache():
    CACHE_DIR = "/tmp/ostep-downloader"
    os.makedirs(CACHE_DIR, exist_ok=True)
    requests_cache.install_cache(
        cache_name=CACHE_DIR,
        backend="filesystem",
        expire_after=None,
    )


async def _main():
    setup_logging()
    # setup_requests_cache()

    dst_file_path = sys.argv[1]

    # TODO: add option on whether to crop to A4

    book = await parse_book()

    pdf = book.generate_merged_pdf()
    pdf.write_to_file(dst_file_path)


def main():
    asyncio.run(_main())


if __name__ == "__main__":
    main()
