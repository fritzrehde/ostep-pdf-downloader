from dataclasses import dataclass
from pprint import pprint
from typing import List, Optional, Tuple
from bs4 import BeautifulSoup
import pymupdf


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


def parse_chapter(
    chapter_pdf_path: str,
) -> Tuple[Chapter, int, Optional[Tuple[str, int]]]:
    """
    Return the parsed chapter, the number of pages in this chapter and, if this
    chapter starts a new part, the title and start page of the part.
    """
    doc = pymupdf.open(chapter_pdf_path)
    page_count = len(doc)

    new_part_title = None
    new_part_page_num = None
    chapter_idx = None
    chapter_title = None
    chapter_page_num = None
    subchapters: List[SubChapter] = []

    for page_num in range(0, page_count):
        page = doc.load_page(page_num)
        page_dict = page.get_text("dict")

        # pprint(page_dict)
        for block in page_dict["blocks"]:
            for line in block["lines"]:
                for span in line["spans"]:
                    font_size: float = span["size"]
                    text: str = span["text"].strip()

                    if font_size >= 10:
                        print(font_size, text)

                    # We identify headings by their (large) font sizes.
                    if 17 < font_size < 20:
                        new_part_title = text
                        new_part_page_num = page_num
                    if 20 < font_size and text.isdigit():
                        chapter_idx = int(text)
                    if 12 < font_size < 15:
                        chapter_title = text
                        chapter_page_num = page_num
                    elif 10 < font_size < 12:
                        subchapter = SubChapter(text, page_num)
                        subchapters.append(subchapter)

    assert chapter_title is not None and chapter_page_num is not None

    if chapter_idx is not None:
        chapter_title = f"{chapter_idx} {chapter_title}"

    chapter = Chapter(chapter_title, chapter_page_num, subchapters)
    maybe_new_part = (
        (new_part_title, new_part_page_num)
        if new_part_title is not None and new_part_page_num is not None
        else None
    )

    return chapter, page_count, maybe_new_part


def parse_chapters(chapter_pdf_paths: List[str]) -> List[Chapter]:
    page_num = 0

    chapters_iter = iter(map(parse_chapter, chapter_pdf_paths))

    parts: List[Part] = []

    def parse_part():
        pass

    while next(chapters_iter, None) is not None:
        curr_part_title = None
        curr_part_page_num = None

        chapters_in_part: List[Chapter] = []

        # TODO: adjust the page numbers

        for chapter, page_count, maybe_new_part in chapters_iter:
            match maybe_new_part:
                case (new_part_title, new_part_page_num):
                    pass
                case None:
                    pass

            page_num += page_count

    return []


def main():
    # print(parse_chapter("/home/fritz/Downloads/cpu-intro.pdf", 0))
    print(parse_chapter("/home/fritz/Downloads/dialogue-virtualization.pdf", 0))
    # print(parse_chapter("/home/fritz/Downloads/toc.pdf", 0))


if __name__ == "__main__":
    main()
