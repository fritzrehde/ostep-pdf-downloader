import asyncio
import re
from ostep_pdf_downloader import (
    Book,
    Chapter,
    Indexing,
    Offset,
    PageNum,
    SubChapter,
    parse_book,
)

book = asyncio.run(parse_book())


def test_title_parsing():
    # Test that titles (of parts, chapters and subchapters) are either one of
    # the hardcoded special titles, or adhere to a more general regex.
    # In my opinion, this approach involved too much "hardcoding" for the actual
    # parsing, but is great for testing, since my assumptions == the hardcoded
    # values, so if the hardcoded values no longer pass, I need to revisit my
    # assumptions.

    SPECIAL_PART_TITLES = {
        "Intro",
        "Virtualization",
        "Concurrency",
        "Persistence",
        "Security",
        "Appendices",
    }

    SPECIAL_CHAPTER_TITLES = {
        "Preface",
        "Contents",
        "Virtual Machine Monitors",
        "Monitors (Deprecated)",
    }
    CHAPTER_PATTERNS = [
        # Most chapters start with a number and then have at least one letter.
        re.compile(r"^\d{1,2} \w.+$"),
        re.compile(r"^A Dialogue on .+$"),
        re.compile(r"^Laboratory: .+$"),
    ]

    SPECIAL_SUBCHAPTER_TITLES = {
        "References",
        "Homework",
        "Homework (Simulation)",
        "Homework (Code)",
        "Homework (Measurement)",
    }
    # Subchapters start with a number.number and then have at least one letter, followed
    # by anything.
    SUBCHAPTER_PATTERN = re.compile(r"^(\d|\w){1,2}\.\d{1,2} \w.+$")

    for part in book.parts:
        assert part.title in SPECIAL_PART_TITLES

        for chapter in part.chapters:
            assert chapter.title in SPECIAL_CHAPTER_TITLES or any(
                pat.search(chapter.title) is not None
                for pat in CHAPTER_PATTERNS
            )

            for subchapter in chapter.subchapters:
                assert (
                    subchapter.title in SPECIAL_SUBCHAPTER_TITLES
                    or SUBCHAPTER_PATTERN.search(subchapter.title) is not None
                )


def get_all_chapters(book: Book):
    for part in book.parts:
        for chapter in part.chapters:
            yield chapter


def get_all_subchapters(book: Book):
    for part in book.parts:
        for chapter in part.chapters:
            for subchapter in chapter.subchapters:
                yield subchapter


def test_page_num_parsing():
    all_chapters = list(get_all_chapters(book))
    all_subchapters = list(get_all_subchapters(book))

    chapter = Chapter(
        "Contents", PageNum(12, Indexing.Zero, Offset(0)), subchapters=[]
    )
    assert chapter in all_chapters

    chapter = Chapter(
        "11 Summary Dialogue on CPU Virtualization",
        PageNum(145, Indexing.Zero, Offset(0)),
        subchapters=[],
    )
    assert chapter in all_chapters

    subchapter = SubChapter(
        "7.7 Round Robin", PageNum(99, Indexing.Zero, Offset(0))
    )
    assert subchapter in all_subchapters
