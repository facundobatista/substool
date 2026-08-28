# Copyright 2026 Facundo Batista
# Licensed under the Apache v2 License
# For further info, check https://github.com/facundobatista/substool

"""Tests for the _fix_times function."""

import pytest

from substool.helpers import SubItem
from substool.command_check import _fix_times


def test_times_nothing_to_fix():
    """Times are all fine, nothing changes."""
    s1 = SubItem(tfrom=0, tto=1, text="foo")
    s2 = SubItem(tfrom=2, tto=3, text="bar")
    src_items = [s1, s2]
    new_items = _fix_times(src_items)
    assert new_items == src_items


def test_times_same_from_to():
    """Same from/to times get fixed using the text length."""
    text = "x" * 40  # 40 * .07 = 2.8s
    s1 = SubItem(tfrom=5, tto=5, text=text)
    new_items = _fix_times([s1])
    assert len(new_items) == 1
    assert new_items[0].tfrom == 5
    assert new_items[0].tto == pytest.approx(7.8)
    assert new_items[0].text == text


def test_times_inverted():
    """Inverted (tto before tfrom) times get fixed using the text length."""
    text = "x" * 40  # 40 * .07 = 2.8s
    s1 = SubItem(tfrom=10, tto=5, text=text)
    new_items = _fix_times([s1])
    assert new_items == [SubItem(tfrom=10, tto=12.8, text=text)]


def test_times_minimum_one_second():
    """A very short text still gets at least one second of duration."""
    s1 = SubItem(tfrom=5, tto=5, text="hi")  # 2 * .07 = .14, below the 1s floor
    new_items = _fix_times([s1])
    assert new_items == [SubItem(tfrom=5, tto=6, text="hi")]


def test_times_cross_timing_between_subs():
    """Individually-valid times that overlap the next sub get fixed."""
    s1 = SubItem(tfrom=0, tto=3, text="foo")  # tto(3) overlaps s2's tfrom(2)
    s2 = SubItem(tfrom=2, tto=4, text="bar")
    s3 = SubItem(tfrom=5, tto=6, text="baz")
    new_items = _fix_times([s1, s2, s3])
    # "foo" is short, so the recalculated length (1s min) doesn't even
    # need clamping to fit before s2's tfrom
    assert new_items[0] == SubItem(tfrom=0, tto=1, text="foo")
    assert new_items[1] == s2
    assert new_items[2] == s3


def test_times_fixed_length_clamped_to_next_sub():
    """The recalculated length must not invade the next sub's time."""
    text = "x" * 50  # 50 * .07 = 3.5s, more than the available gap
    s1 = SubItem(tfrom=0, tto=3, text=text)  # 3 > next's tfrom (2): needs fixing
    s2 = SubItem(tfrom=2, tto=4, text="bar")
    s3 = SubItem(tfrom=5, tto=6, text="baz")
    new_items = _fix_times([s1, s2, s3])
    assert new_items[0].tto == pytest.approx(2)
    assert new_items[0].tto <= s2.tfrom
    assert new_items[1] == s2
    assert new_items[2] == s3


def test_times_bug_second_to_last_item_not_clamped():
    """A fixed sub that is second-to-last must still be clamped to the last one."""
    text = "x" * 50  # 50 * .07 = 3.5s, more than the available gap
    s1 = SubItem(tfrom=0, tto=1, text="foo")
    s2 = SubItem(tfrom=10, tto=9, text=text)  # inverted, needs fixing
    s3 = SubItem(tfrom=10.5, tto=12, text="baz")
    new_items = _fix_times([s1, s2, s3])
    assert new_items[1].tto <= s3.tfrom
