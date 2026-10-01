"""Public community links; opening a page always requires a user action."""

REPOSITORY_URL = "https://github.com/ken-jo/qev"
SUPPORT_URL = "https://github.com/sponsors/ken-jo"


def community_markdown():
    return (
        "[Star QEV on GitHub](" + REPOSITORY_URL + ") · [GitHub Sponsors](" + SUPPORT_URL + ")"
    )
