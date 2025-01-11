from argparse import ArgumentParser
import asyncio
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from itertools import groupby
import logging
import multiprocessing
from typing import List, Tuple
from urllib.parse import urljoin
from bs4 import BeautifulSoup, Tag
import pymupdf
from PIL import Image
from aiohttp import ClientSession


OSTEP_URL = "https://pages.cs.wisc.edu/~remzi/OSTEP/"


async def get_soup(url: str, session: ClientSession) -> BeautifulSoup:
    response = await session.get(url)
    bytes = await response.read()
    logging.info(f"Finished getting bs4 soup for {OSTEP_URL}")
    return BeautifulSoup(bytes, "html.parser")


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
    # A pointer to an offset shared by all page numbers.
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
    """A Chapter where the PageNum is missing the offset field."""

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

    def page_rect(self) -> pymupdf.Rect:
        with pymupdf.open(stream=self.in_mem_file, filetype="pdf") as doc:
            return doc[0].rect

    def write_to_file(self, dst_file_path: str):
        with open(dst_file_path, "wb") as f:
            f.write(self.in_mem_file.getbuffer())


def doc_to_pdf(doc) -> Pdf:
    pdf = BytesIO()
    doc.save(pdf)
    return Pdf(pdf)


@dataclass
class Img:
    in_mem_file: BytesIO

    def height_width(self) -> Tuple[float, float]:
        with Image.open(self.in_mem_file) as img:
            return img.height, img.width

    def scale_to_fit_onto_page(self, page: pymupdf.Page) -> pymupdf.Rect:
        """
        We assume this image is taller than the page. Scale it down, mainting
        height-width ratio. This only returns a Rect with scaled values, pymupdf
        does the actual scaling of the image.
        """
        img_h, img_w = self.height_width()
        page_h, page_w = (page.rect.height, page.rect.width)

        # Our cover image is expected to be "taller" (ratio-wise) than
        # the rect.
        assert (img_h / img_w) > (page_h / page_w)

        # Because the image is taller, it can be scaled down such that
        # the scaled-down height matches the page's height.
        img_scaled_h = page_h

        # The height and width should scale down the same to maintain
        # their ratio:
        # ratio = (scaled height / height) = (scaled width / width)
        scale_down_ratio = img_scaled_h / img_h
        img_scaled_w = img_w * scale_down_ratio

        # Since we're squeezing (by scaling, not cropping) a taller
        # image into a smaller frame, there will be gaps on the left and
        # right of the image in the frame.
        width_gap = (page_w - img_scaled_w) / 2

        # Center scaled image on page.
        x0, y0 = width_gap, 0
        x1, y1 = width_gap + img_scaled_w, img_scaled_h
        return pymupdf.Rect(x0, y0, x1, y1)


def crop_page_to_fit_height(
    page_rect: pymupdf.Rect,
    desired_height: float,
    top_bottom_crop_ratio: float,
) -> pymupdf.Rect:
    """
    Crop the height of a page to a desired height. Keep the width of the
    page.
    """
    # We expect to have to make the page shorter or leave it the same.
    assert page_rect.height >= desired_height

    to_crop = page_rect.height - desired_height

    # If we have to crop some vertical space, ratio how much to take
    # from top vs bottom?
    crop_top_perc = top_bottom_crop_ratio
    crop_bottom_perc = 1 - crop_top_perc
    crop_top = to_crop * crop_top_perc
    crop_bottom = to_crop * crop_bottom_perc

    x0, y0 = page_rect.x0, page_rect.y0 + crop_top
    x1, y1 = page_rect.x1, page_rect.y1 - crop_bottom
    return pymupdf.Rect(x0, y0, x1, y1)


@dataclass
class ChapterPdfUrl:
    pdf_url: str

    async def download(self, session: ClientSession) -> Pdf:
        """Download pdf to in-memory file."""
        async with session.get(self.pdf_url) as response:
            content = await response.read()
        in_mem_file = BytesIO(content)
        pdf = Pdf(in_mem_file)

        logging.info(f"Finished downloading {self.pdf_url}")
        return pdf


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


@dataclass
class Metadata:
    title: str
    author: str


@dataclass
class Book:
    metadata: Metadata
    cover_image: Img
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

    def generate_merged_pdf(self, should_crop: bool) -> Pdf:
        logging.info("Generating merged pdf")

        page_rect = self.ordered_pdf_chapters[0].page_rect()

        # We always maintain the original width.
        desired_width = page_rect.width
        if should_crop:
            # ratio = (desired height / desired width)
            ratio = 4 / 3
            desired_height = ratio * desired_width
        else:
            desired_height = page_rect.height

        with pymupdf.open() as new_doc:
            # Create a cover page and insert cover image by scaling it to
            # fit onto cover page. We colour the gaps left by shrinking the
            # cover image height-wise in black to match the cover's colour.
            cover_page: pymupdf.Page = new_doc.new_page(
                height=desired_height, width=desired_width
            )
            black = (0, 0, 0)
            cover_page.draw_rect(cover_page.rect, color=black, fill=black)
            cover_img_rect = self.cover_image.scale_to_fit_onto_page(cover_page)
            cover_page.insert_image(
                rect=cover_img_rect,
                stream=self.cover_image.in_mem_file,
                keep_proportion=True,
            )

            # Adding one cover page causes every subsequent page number to be
            # incremented.
            self.offset.change(delta=+1)

            # Add all chapter pdf files.
            for pdf in self.ordered_pdf_chapters:
                with pymupdf.open(
                    stream=pdf.in_mem_file, filetype="pdf"
                ) as chapter_doc:
                    new_doc.insert_pdf(chapter_doc)

            # Add table of contents.
            toc = self.build_toc()
            toc_formatted: List[Tuple[int, str, int]] = [
                (x.lvl, x.title, x.page_num.get(desired_indexing=Indexing.One))
                for x in toc
            ]
            new_doc.set_toc(toc_formatted)

            # Add metadata.
            metadata = {
                "title": self.metadata.title,
                "author": self.metadata.author,
            }
            new_doc.set_metadata(metadata)

            # Crop non-cover pages to desired height-to-width ratio.
            cropped_rect = crop_page_to_fit_height(
                page_rect, desired_height, top_bottom_crop_ratio=0.7
            )
            non_cover_pages = (new_doc[i] for i in range(1, len(new_doc)))
            for page in non_cover_pages:
                page.set_cropbox(cropped_rect)

            return doc_to_pdf(new_doc)


def parse_chapter(chapter_pdf: Pdf) -> Tuple[PendingChapter, int]:
    """
    Return the parsed chapter and the number of pages in this chapter. The page
    numbers of the chapters and subchapters will be relative to the page number
    of start of this chapter.
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
                    # To parse titles spanning multiple lines, define a "line
                    # block" as consecutive lines with the same font-size.

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

        logging.info(f"Finished parsing chapter: {chapter.title}")

        return chapter, page_count


Cell = PartStart | ChapterPdfUrl | None


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


def scrape_chapters_table(
    ostep_soup: BeautifulSoup,
) -> Tuple[int, int, List[List[Cell]]]:
    """
    Return row count, col count and chapters table.
    """
    chapters_table = ostep_soup.find_all("table")[3]
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

    logging.info(f"Finished scraping chapters table from {OSTEP_URL}")
    return row_count, col_count, chapters_table_rows


def group_parts_chapter_urls(
    cells: List[PartStart | ChapterPdfUrl],
) -> List[Tuple[PartStart, List[ChapterPdfUrl]]]:
    """
    Convert a flat list of cells to a list of parts where each part contains its
    chapters.
    """
    if cells == []:
        return []

    head, *tail = cells

    # The first cell must be a part start.
    match head:
        case PartStart(_) as part_start:
            curr_part_start = part_start
        case _:
            raise Exception("first cell must be a part start")

    # Collect chapters until the next part starts.
    curr_part_chapters = []
    i = 0
    while i < len(tail):
        match tail[i]:
            case PartStart(_) as part_start:
                # A new part start ends the current part.
                break
            case ChapterPdfUrl(_) as chapter_pdf:
                curr_part_chapters.append(chapter_pdf)
        i += 1
    next_part_start_idx = i

    return [(curr_part_start, curr_part_chapters)] + group_parts_chapter_urls(
        tail[next_part_start_idx:]
    )


def scrape_parts_chapter_urls(
    ostep_soup: BeautifulSoup,
) -> List[Tuple[PartStart, List[ChapterPdfUrl]]]:
    """
    Return tuples of the part start and the pdf urls to all chapters in that
    part.
    """
    row_count, col_count, chapters_table_rows = scrape_chapters_table(
        ostep_soup
    )

    # Iterate down columns one by one, filtering out None cells.
    it = iter(
        cell
        for col in range(col_count)
        for row in range(row_count)
        if (cell := chapters_table_rows[row][col]) is not None
    )
    cells = list(it)
    return group_parts_chapter_urls(cells)


async def download_parts_chapter_pdfs(
    ostep_session: ClientSession, ostep_soup: BeautifulSoup
) -> List[Tuple[PartStart, List[Pdf]]]:
    """
    For each part (start), also return its chapter pdfs.
    """

    async def download_chapters_for_part(
        part: PartStart, chapter_urls: List[ChapterPdfUrl]
    ) -> Tuple[PartStart, List[Pdf]]:
        tasks = [
            chapter_url.download(ostep_session) for chapter_url in chapter_urls
        ]
        downloaded_pdfs = await asyncio.gather(*tasks)
        return part, downloaded_pdfs

    def cpu_bound():
        return scrape_parts_chapter_urls(ostep_soup)

    # Download pdf files concurrently.
    tasks = [
        download_chapters_for_part(part, chapter_urls)
        for (part, chapter_urls) in await asyncio.to_thread(cpu_bound)
    ]
    return await asyncio.gather(*tasks)


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
                # Complete the chapter creation by adding ref to offset.
                (pending_chapter.to_chapter(offset), page_count)
                for (_, (pending_chapter, page_count)) in group
            ],
        )
        for part_title, group in groupby(results, key=get_part_title)
    ]


def scrape_metadata(ostep_soup: BeautifulSoup) -> Metadata:
    blockquote = ostep_soup.find("blockquote")
    lines = [
        stripped
        for line in blockquote.get_text().split("\n")
        if (stripped := line.strip()) != ""
    ]
    title = lines[0]
    author = lines[1]
    metadata = Metadata(title, author)

    logging.info(
        f"Finished scraping metadata (title and author) from {OSTEP_URL}"
    )

    return metadata


async def scrape_cover_image(ostep_soup: BeautifulSoup) -> Img:

    def cpu_bound():
        img_tag = ostep_soup.find(
            "img",
            src=lambda text: text
            and "book" in text.lower()
            and ".jpg" in text.lower(),
        )
        return urljoin(OSTEP_URL, img_tag["src"])

    # NOTE: Run a synchronous cpu-bound task without blocking the event loop.
    cover_url = await asyncio.to_thread(cpu_bound)

    async with ClientSession() as session:
        response = await session.get(cover_url)
        # NOTE: read() needs to be inside "with" statement, otherwise
        # use-after-free.
        bytes = await response.read()
    in_mem_file = BytesIO(bytes)
    jpg = Img(in_mem_file)

    logging.info(f"Finished scraping cover image from {OSTEP_URL}")
    return jpg


async def parse_book() -> Book:
    # Add the same mutable offset to every page number, so we can later add a
    # cover page (offset = 1).
    offset = Offset(0)

    async with ClientSession() as ostep_session:
        ostep_soup = await get_soup(OSTEP_URL, ostep_session)
        # These tasks are IO-bound, so execute asynchronously.
        tasks = (
            download_parts_chapter_pdfs(ostep_session, ostep_soup),
            scrape_cover_image(ostep_soup),
        )
        parts_chapter_pdfs, cover_image = await asyncio.gather(*tasks)
        metadata = scrape_metadata(ostep_soup)

    ordered_chapter_pdfs: List[Pdf] = [
        pdf for _, pdfs in parts_chapter_pdfs for pdf in pdfs
    ]

    def cpu_bound():
        return parse_all_chapters(parts_chapter_pdfs, offset)

    page_num = 0
    parts: List[Part] = []
    for part_title, chapters_page_counts in await asyncio.to_thread(cpu_bound):
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

    book = Book(
        metadata,
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

    parser = ArgumentParser()
    parser.add_argument("dst_file_path")
    parser.add_argument(
        "--crop",
        action="store_true",
        required=False,
        help="crop every page to a 4:3 aspect ratio",
    )
    args = parser.parse_args()
    dst_file_path = args.dst_file_path
    should_crop = args.crop

    book = await parse_book()

    pdf = book.generate_merged_pdf(should_crop)
    pdf.write_to_file(dst_file_path)


def main():
    asyncio.run(_main())


if __name__ == "__main__":
    main()
