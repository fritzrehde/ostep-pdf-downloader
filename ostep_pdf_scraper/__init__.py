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
    chapter_pdf_path: str, starting_page_num: int
) -> Tuple[Chapter, int]:
    """
    Return the parsed chapter and the number of pages in this chapter.
    """
    doc = pymupdf.open(chapter_pdf_path)
    page_count = len(doc)

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

                    # if font_size >= 10:
                    #     print(font_size, text)

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


def scrape_parts_chapter_files() -> List[Tuple[str, List[str]]]:
    # TODO: scrapce https://pages.cs.wisc.edu/~remzi/OSTEP/ and create a temp dir with files and return (kinda) map from part title to the chapter pdf file paths

    return []


def parse_book() -> Book:
    page_num = 0

    parts: List[Part] = []
    # TODO: init tempfile here
    for part_title, chapter_pdf_paths in scrape_parts_chapter_files():
        # Assume the part starts at the first page of its first chapter.
        part_page_num = page_num

        chapters: List[Chapter] = []
        for path in chapter_pdf_paths:
            chapter, page_count = parse_chapter(path, page_num)
            chapters.append(chapter)
            page_num += page_count

        part = Part(part_title, part_page_num, chapters)
        parts.append(part)

    title = ""
    author = ""
    description = ""
    book = Book(title, author, description, parts)

    return book


def main():
    # print(parse_chapter("/home/fritz/Downloads/cpu-intro.pdf", 0))
    # print(parse_chapter("/home/fritz/Downloads/dialogue-virtualization.pdf", 0))
    # print(parse_chapter("/home/fritz/Downloads/toc.pdf", 0))

    book = parse_book()


if __name__ == "__main__":
    main()
