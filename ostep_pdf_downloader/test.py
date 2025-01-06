import re
from ostep_pdf_downloader import parse_book, setup_requests_cache


def test_parse_book():
    # TODO: remove
    setup_requests_cache()

    # TODO: write a test that asserts that chapters adhere to a regex (too "hardcoded" for parsing, but great for testing, since if my assumptions were wrong (now due to changes), i want to know)
    PART_TITLES = {
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

    book = parse_book()
    for part in book.parts:
        assert part.title in PART_TITLES

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
