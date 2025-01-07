import asyncio
from dataclasses import dataclass
from enum import Enum
import functools
from io import BytesIO
from itertools import groupby
import logging
import multiprocessing
from pprint import pprint
import sys
from typing import List, Tuple
from urllib.parse import urljoin
import aiohttp
from bs4 import BeautifulSoup, Tag
import pymupdf
import requests
from PIL import Image


OSTEP_URL = "https://pages.cs.wisc.edu/~remzi/OSTEP/"


def get_soup(url: str) -> BeautifulSoup:
    response = requests.get(url)
    return BeautifulSoup(response.text, "html.parser")


@functools.cache
def ostep_soup() -> BeautifulSoup:
    return get_soup(OSTEP_URL)


class Indexing(Enum):
    Zero = 0
    One = 1


@dataclass
class Offset:
    offset: int

    def change(self, delta: int):
        self.offset += delta

    def apply(self, value: int) -> int:
        return value + self.offset


@dataclass
class PageNum:
    page_num: int
    actual_indexing: Indexing
    # A pointer to an offset shared by all pages.
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

    def add_infront(self, page_count: int):
        """
        A given number of pages are added infront of this page.
        """
        self.page_num += page_count


@dataclass
class PendingPageNum:
    """A PageNum missing the offset field."""

    page_num: int
    actual_indexing: Indexing

    def to_page_num(self, offset: Offset) -> PageNum:
        return PageNum(self.page_num, self.actual_indexing, offset)


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
class PendingSubChapter:
    """A SubChapter where the PageNum is missing the offset field."""

    title: str
    pending_page_num: PendingPageNum

    def to_subchapter(self, offset: Offset) -> SubChapter:
        return SubChapter(self.title, self.pending_page_num.to_page_num(offset))


@dataclass
class Chapter:
    title: str
    page_num: PageNum
    subchapters: List[SubChapter]

    def toc_entry(self) -> TocEntry:
        return TocEntry(lvl=2, title=self.title, page_num=self.page_num)


@dataclass
class PendingChapter:
    """A SubChapter where the PageNum is missing the offset field."""

    title: str
    pending_page_num: PendingPageNum
    pending_subchapters: List[PendingSubChapter]

    def to_chapter(self, offset: Offset) -> Chapter:
        return Chapter(
            self.title,
            self.pending_page_num.to_page_num(offset),
            [
                pending_subchapter.to_subchapter(offset)
                for pending_subchapter in self.pending_subchapters
            ],
        )


@dataclass
class Pdf:
    in_mem_file: BytesIO

    def height_width(self) -> Tuple[int, int]:
        with pymupdf.open(stream=self.in_mem_file, filetype="pdf") as doc:
            first_page = doc[0].rect
            return int(first_page.height), int(first_page.width)

    def write_to_file(self, dst_file_path: str):
        with open(dst_file_path, "wb") as f:
            f.write(self.in_mem_file.getbuffer())


@dataclass
class Jpg:
    in_mem_file: BytesIO

    def fit_in_pdf(self, pdf_height: int, pdf_width: int) -> Pdf:
        with Image.open(self.in_mem_file) as img:
            scaled_img = img.resize((pdf_width, pdf_height))
            in_mem_pdf = BytesIO()
            scaled_img.save(in_mem_pdf, format="PDF")
            return Pdf(in_mem_pdf)


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


def doc_to_pdf(doc) -> Pdf:
    pdf = BytesIO()
    doc.save(pdf)
    return Pdf(pdf)


@dataclass
class Book:
    title: str
    author: str
    cover_image: Jpg
    parts: List[Part]
    ordered_pdf_chapters: List[Pdf]
    # Every page number has this offset.
    offset: Offset

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

        with pymupdf.open() as new_doc:
            # Add cover image.
            height, width = self.ordered_pdf_chapters[0].height_width()
            cover_pdf = self.cover_image.fit_in_pdf(height, width)
            with pymupdf.open(
                stream=cover_pdf.in_mem_file, filetype="pdf"
            ) as cover_doc:
                new_doc.insert_pdf(cover_doc)
            # Increment every page number.
            self.offset.change(delta=+1)

            # Add all chapter pdf files.
            for pdf in self.ordered_pdf_chapters:
                with pymupdf.open(
                    stream=pdf.in_mem_file, filetype="pdf"
                ) as chapter_doc:
                    new_doc.insert_pdf(chapter_doc)

            # Add TOC.
            toc = self.build_toc()
            toc_formatted: List[Tuple[int, str, int]] = [
                (x.lvl, x.title, x.page_num.get(desired_indexing=Indexing.One))
                for x in toc
            ]
            new_doc.set_toc(toc_formatted)

            # Add metadata.
            metadata = {"title": self.title, "author": self.author}
            new_doc.set_metadata(metadata)

            return doc_to_pdf(new_doc)


def parse_chapter(chapter_pdf: Pdf) -> Tuple[PendingChapter, int]:
    """
    Return the parsed chapter and the number of pages in this chapter. The page
    numbers of the chapters and subchapters will be relative to the page number
    of start of the this chapter pdf.
    """
    with pymupdf.open(stream=chapter_pdf.in_mem_file, filetype="pdf") as doc:
        page_count = len(doc)

        chapter_idx = None
        chapter_title = None
        chapter_page_num = None
        subchapters: List[PendingSubChapter] = []

        for page_num in range(0, page_count):
            page = doc.load_page(page_num)
            page_dict = page.get_text("dict")

            for block in page_dict["blocks"]:
                for font_size, lines in groupby(
                    block["lines"], key=lambda line: line["spans"][0]["size"]
                ):
                    # We define a "line block" as consecutive lines as long as
                    # they have the same font size, to enable parsing titles
                    # that span over multiple lines.

                    # The lines in a "line block" are joined with a space as
                    # separator.
                    text: str = " ".join(
                        # Each line is joined together, ignoring differing fonts.
                        ("".join(span["text"] for span in line["spans"]))
                        for line in lines
                    )

                    # We identify headings by their (large) font sizes.
                    if 20.662 < font_size and text.isdigit():
                        chapter_idx = int(text)
                    elif (
                        12.575 < font_size < 12.576
                        or 14.346 < font_size < 14.347
                    ):
                        chapter_title = text
                        chapter_page_num = PendingPageNum(
                            page_num, Indexing.Zero
                        )
                    elif 10.909 < font_size < 10.910:
                        subchapter = PendingSubChapter(
                            text,
                            PendingPageNum(page_num, Indexing.Zero),
                        )
                        subchapters.append(subchapter)

        assert chapter_title is not None and chapter_page_num is not None

        if chapter_idx is not None:
            chapter_title = f"{chapter_idx} {chapter_title}"

        chapter = PendingChapter(chapter_title, chapter_page_num, subchapters)

        logging.info(f"Parsing chapter: {chapter.title}")

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

    soup = ostep_soup()

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


async def download_parts_chapter_pdfs() -> List[Tuple[PartStart, List[Pdf]]]:
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


# NOTE: The function executed in parellel across processes by multiprocessing
# must be a top-level function so it can be pickled. The function's arguments
# and return values are also pickled.
def exec_task(input) -> Tuple[str, Tuple[PendingChapter, int]]:
    part_title, chapter_pdf = input
    return part_title, parse_chapter(chapter_pdf)


def parse_all_chapters(
    parts_chapter_pdfs: List[Tuple[PartStart, List[Pdf]]], offset: Offset
) -> List[Tuple[str, List[Tuple[Chapter, int]]]]:
    """
    Returns [(part title, [(chapter, chapter page count)])].
    """
    # `parse_chapter` is the CPU-bound function, so need to create a flat task
    # list to parallelise.
    tasks = [
        (part_start.title, chapter_pdf)
        for (part_start, chapter_pdfs) in parts_chapter_pdfs
        for chapter_pdf in chapter_pdfs
    ]

    # NOTE: To share memory across processes, each of the tasks is pickled
    # (serialized) and sent to a worker process. The task is then unpickled,
    # executed, and the result is pickled again before being sent back to the
    # main process. This implementation detail is crucial because it means each
    # process receives and returns deep copies instead of references to task
    # arguments. In my case, I have one global Offset that every PageNum should
    # reference. However, during pickling, new Offset deep copies are created,
    # which is not the intended behaviour. Fixing this bug required not
    # including the Offsets as arguments and instead adding them in later on the
    # main process.

    # This is parallelised across multiple processes.
    with multiprocessing.Pool() as pool:
        results = pool.map(exec_task, tasks)

    def get_part_title(
        results_item: Tuple[str, Tuple[PendingChapter, int]]
    ) -> str:
        (part_title, (_pending_chapter, _page_count)) = results_item
        return part_title

    # Join back together by grouping by part title.
    return [
        (
            part_title,
            [
                (pending_chapter.to_chapter(offset), page_count)
                for (_, (pending_chapter, page_count)) in group
            ],
        )
        for part_title, group in groupby(results, key=get_part_title)
    ]


def scrape_metadata_title_author() -> Tuple[str, str]:
    logging.info(f"Scraping metadata (title and author) from {OSTEP_URL}")

    soup = ostep_soup()
    blockquote = soup.find("blockquote")
    lines = [
        stripped
        for line in blockquote.get_text().split("\n")
        if (stripped := line.strip()) != ""
    ]
    title = lines[0]
    author = lines[1]

    return title, author


def scrape_cover_image() -> Jpg:
    logging.info(f"Scraping cover image from {OSTEP_URL}")

    soup = ostep_soup()
    img_tag = soup.find(
        "img",
        src=lambda text: text
        and "book" in text.lower()
        and ".jpg" in text.lower(),
    )
    cover_url = urljoin(OSTEP_URL, img_tag["src"])

    response = requests.get(cover_url)
    in_mem_file = BytesIO(response.content)
    return Jpg(in_mem_file)


async def parse_book() -> Book:
    # Add the same mutable offset to every page number, so we can later add a
    # cover page (offset = 1).
    offset = Offset(0)

    parts_chapter_pdfs = await download_parts_chapter_pdfs()

    ordered_chapter_pdfs: List[Pdf] = [
        pdf for _, pdfs in parts_chapter_pdfs for pdf in pdfs
    ]

    page_num = 0
    parts: List[Part] = []
    for part_title, chapters_page_counts in parse_all_chapters(
        parts_chapter_pdfs, offset
    ):
        # Assume the part starts at the first page of its first chapter.
        part_page_num = PageNum(page_num, Indexing.Zero, offset)

        # Convert relative (within chapters) into absolute (within whole book)
        # page numbers.
        for chapter, page_count in chapters_page_counts:
            chapter.page_num.add_infront(page_num)

            for subchapter in chapter.subchapters:
                subchapter.page_num.add_infront(page_num)

            page_num += page_count

        chapters: List[Chapter] = [
            chapter for (chapter, _) in chapters_page_counts
        ]

        part = Part(part_title, part_page_num, chapters)
        parts.append(part)

    cover_image = scrape_cover_image()
    title, author = scrape_metadata_title_author()

    book = Book(
        title,
        author,
        cover_image,
        parts,
        ordered_chapter_pdfs,
        offset,
    )

    return book


def setup_logging():
    logging.basicConfig(level=logging.INFO)


async def _main():
    setup_logging()

    dst_file_path = sys.argv[1]

    # TODO: add option on whether to crop to A4

    book = await parse_book()

    pdf = book.generate_merged_pdf()
    pdf.write_to_file(dst_file_path)


def main():
    asyncio.run(_main())


if __name__ == "__main__":
    main()
