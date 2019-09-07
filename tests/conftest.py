"""
Define fixtures.

Author: Nikolay Lysenko
"""


from tempfile import NamedTemporaryFile

import pytest


@pytest.fixture()
def path_to_tmp_file() -> str:
    """Get path to empty temporary file."""
    with NamedTemporaryFile() as tmp_file:
        yield tmp_file.name


path_to_another_tmp_file = path_to_tmp_file
