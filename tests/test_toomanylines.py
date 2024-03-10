# Copyright 2014-2024 Facundo Batista
# Licensed under the Apache v2 License
# For further info, check https://github.com/facundobatista/subtitles

"""Tests for the _fix_toomanylines function."""

import textwrap

import pytest

from subtitles.helpers import SubItem
from subtitles.command_check import _fix_toomanylines


def test_toomanylines_just_1():
    """One line is fine."""
    s1 = SubItem(tfrom=1, tto=2, text="foobar")
    src_items = [s1]
    new_items = _fix_toomanylines(src_items)
    assert src_items == new_items


def test_toomanylines_just_2():
    """One line is fine."""
    s1 = SubItem(tfrom=1, tto=2, text=textwrap.dedent("""\
        foobar
        otherline
    """))
    src_items = [s1]
    new_items = _fix_toomanylines(src_items)
    assert src_items == new_items


def test_toomanylines_excess_3():
    """Too many lines."""
    s1 = SubItem(tfrom=1, tto=2, text=textwrap.dedent("""\
        Some test in line 1,
        then more stuff and
        finally this here."""))
    src_items = [s1]
    new_items = _fix_toomanylines(src_items)
    should_rewrapped = textwrap.dedent("""\
        Some test in line 1, then more
        stuff and finally this here.""")
    assert new_items[0].text == should_rewrapped


def test_toomanylines_excess_4():
    """Too many lines."""
    s1 = SubItem(tfrom=1, tto=2, text=textwrap.dedent("""\
        Some test in line
        and then more
        stuff, eventually
        other thing here."""))
    src_items = [s1]
    new_items = _fix_toomanylines(src_items)
    should_rewrapped = textwrap.dedent("""\
        Some test in line and then more
        stuff, eventually other thing here.""")
    assert new_items[0].text == should_rewrapped


@pytest.mark.parametrize("dialog_symbol", ["*", "-", "♪"])
def test_toomanylines_excess_dialog_1(dialog_symbol):
    """Respect dialogs, case 1."""
    s1 = SubItem(tfrom=1, tto=2, text=textwrap.dedent("""\
        {0} Some test in line 1
        {0} More stuff and
        finally this here.""".format(dialog_symbol)))
    src_items = [s1]
    new_items = _fix_toomanylines(src_items)
    should_rewrapped = textwrap.dedent("""\
        {0} Some test in line 1
        {0} More stuff and finally this here.""".format(dialog_symbol))
    assert new_items[0].text == should_rewrapped


@pytest.mark.parametrize("dialog_symbol", ["*", "-", "♪"])
def test_toomanylines_excess_dialog_2(dialog_symbol):
    """Respect dialogs, case 1."""
    s1 = SubItem(tfrom=1, tto=2, text=textwrap.dedent("""\
        {0} Some test in line 1
        and more stuff
        {0} finally this here""".format(dialog_symbol)))
    src_items = [s1]
    new_items = _fix_toomanylines(src_items)
    should_rewrapped = textwrap.dedent("""\
        {0} Some test in line 1 and more stuff
        {0} finally this here""".format(dialog_symbol))
    assert new_items[0].text == should_rewrapped
