# Copyright 2024 Facundo Batista
# Licensed under the GPL v3 License
# For further info, check https://github.com/facundobatista/substool

"""Infrastructure tests."""

import itertools
import os
import re
import subprocess

import pydocstyle
import pytest
from flake8.api.legacy import get_style_guide


def get_python_filepaths(*, roots=None, python_paths=None):
    """Retrieve paths of Python files."""
    python_paths = ["setup.py"]
    for root in ["substool", "tests"]:
        for dirpath, dirnames, filenames in os.walk(root):
            for filename in filenames:
                if filename.endswith(".py"):
                    python_paths.append(os.path.join(dirpath, filename))
    return python_paths


def test_codespell():
    """Verify all words are correctly spelled."""
    cmd = ["codespell", "substool", "tests", "docs"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    report = [x.strip() for x in proc.stdout.split("\n")]
    indented_issues = [f" - {issue}" for issue in report if issue]
    if indented_issues:
        msg = "Please fix the following codespell issues!\n" + "\n".join(indented_issues)
        pytest.fail(msg, pytrace=False)


def test_pep8(capsys):
    """Verify all files are nicely styled."""
    python_filepaths = get_python_filepaths()
    style_guide = get_style_guide()
    report = style_guide.check_files(python_filepaths)

    # if flake8 didn't report anything, we're done
    if report.total_errors == 0:
        return

    # grab on which files we have issues
    out, _ = capsys.readouterr()
    pytest.fail(f"Please fix {report.total_errors} issue(s):\n{''.join(out)}", pytrace=False)


def test_pep257():
    """Verify all files have nice docstrings."""
    python_filepaths = get_python_filepaths()
    to_ignore = {
        "D105",  # Missing docstring in magic method
        "D107",  # Missing docstring in __init__
    }
    to_include = pydocstyle.violations.conventions.pep257 - to_ignore
    errors = list(pydocstyle.check(python_filepaths, select=to_include))

    if errors:
        report = ["Please fix files as suggested by pydocstyle ({:d} issues):".format(len(errors))]
        report.extend(str(e) for e in errors)
        msg = "\n".join(report)
        pytest.fail(msg, pytrace=False)


def test_ensure_copyright():
    """Check that all non-empty Python files have copyright somewhere in the first 5 lines."""
    issues = []
    regex = re.compile(r"# Copyright \d{4}(-\d{4})? Facundo Batista$")
    for filepath in get_python_filepaths():
        if os.stat(filepath).st_size == 0:
            continue

        with open(filepath, "rt", encoding="utf8") as fh:
            for line in itertools.islice(fh, 5):
                if regex.match(line):
                    break
            else:
                issues.append(filepath)
    if issues:
        msg = "Please add copyright headers to the following files:\n" + "\n".join(
            issues
        )
        pytest.fail(msg, pytrace=False)
