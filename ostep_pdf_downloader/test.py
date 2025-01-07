import asyncio
import re
from ostep_pdf_downloader import parse_book


def test_parse_book():
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
    }

    SPECIAL_CHAPTER_TITLES = {"Preface", "Contents"}
    # Chapters start with a number and then have at least one letter, followed
    # by anything.
    CHAPTER_PATTERN = re.compile(r"^\d{1,2} \w.+$")

    SPECIAL_SUBCHAPTER_TITLES = {
        "References",
        "Homework",
        "Homework (Simulation)",
        "Homework (Code)",
        "Homework (Measurement)",
    }
    # Subchapters start with a number.number and then have at least one letter, followed
    # by anything.
    SUBCHAPTER_PATTERN = re.compile(r"^\d{1,2}\.\d{1,2} \w.+$")

    book = asyncio.run(parse_book())
    for part in book.parts:
        assert part.title in SPECIAL_PART_TITLES

        for chapter in part.chapters:
            assert (
                chapter.title in SPECIAL_CHAPTER_TITLES
                or CHAPTER_PATTERN.search(chapter.title) is not None
            )

            for subchapter in chapter.subchapters:
                assert (
                    subchapter.title in SPECIAL_SUBCHAPTER_TITLES
                    or SUBCHAPTER_PATTERN.search(subchapter.title) is not None
                )


# TODO: hardcode test to check page numbers are correct (just choose one chapter and subchapter and hardcode their pagenums)
