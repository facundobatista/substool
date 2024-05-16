# Copyright 2014-2024 Facundo Batista
# Licensed under the Apache v2 License
# For further info, check https://github.com/facundobatista/substool

"""The generic small commands."""

import itertools
import textwrap

from craft_cli import BaseCommand, emit, ArgumentParsingError

from substool.helpers import save_srt, SubItem, valid_filepath, time_sub2stamp, time_stamp2sub
from substool.parsers import parse_subtitle


def _subtime(value):
    """Return a valid timestamp; else ArgumentParsingError is raised."""
    try:
        return time_sub2stamp(value)
    except ValueError as exc:
        raise ArgumentParsingError(str(exc))


def _friendly_float(value):
    """Float conversion accepting comma as decimal point."""
    return float(value.replace(",", "."))


def _build_fixed_filename(filename):
    return filename.parent / (filename.stem + "-fixed" + filename.suffix)


def _rescale(subitems, inpfile, delta, speed):
    """Real rescaling."""
    newitems = []
    for item in subitems:
        newfrom = item.tfrom * speed + delta
        newto = item.tto * speed + delta
        newitems.append(SubItem(newfrom, newto, item.text))

    outfile = _build_fixed_filename(inpfile)
    save_srt(newitems, outfile)
    emit.message("Done")


class RescalePointsCommand(BaseCommand):
    """Rescale subs using two points displacing them to the new times."""

    name = "rescale-points"
    help_msg = "Rescale using two points in time."
    overview = textwrap.dedent("""
        Rescale a subtitles file using two points (id1 and id2) displacing
        them to the new times (time1 and time2).

          e.g.:  substool rescale-points Movie.srt 4 43,5 168 1:02:15

        If id1 is 0, time1 is ignored and all is calculated against beginning.
    """)

    def fill_parser(self, parser):
        """Add own parameters to the general parser."""
        parser.add_argument("filepath", type=valid_filepath, help="The subtitle to process")
        parser.add_argument("id1", type=int, help="The id of subtitle item 1")
        parser.add_argument("tstamp1", type=_subtime, help="The time for subtitle item 1")
        parser.add_argument("id2", type=int, help="The id of subtitle item 2")
        parser.add_argument("tstamp2", type=_subtime, help="The time for subtitle item 2")

    def run(self, parsed_args):
        """Run the command."""
        with open(parsed_args.filepath, 'rt', encoding='utf8') as fh:
            subitems = parse_subtitle(fh.read())

        if parsed_args.id1 == "0":
            emit.debug("Calculating from zero")
            real1 = should1 = 0
        else:
            item1 = subitems[int(parsed_args.id1) - 1]
            emit.debug(
                f"Found 1st item {parsed_args.id1} @ {time_stamp2sub(item1.tfrom)}, "
                f"should be {time_stamp2sub(parsed_args.tstamp1)} ({item1.text!r})")
            real1 = item1.tfrom
            should1 = parsed_args.tstamp1

        item2 = subitems[int(parsed_args.id2) - 1]
        emit.debug(
            f"Found 2nd item {parsed_args.id2} @ {time_stamp2sub(item2.tfrom)}, "
            f"should be {time_stamp2sub(parsed_args.tstamp2)} ({item2.text!r})")
        real2 = item2.tfrom

        should2 = parsed_args.tstamp2

        speed = (should2 - should1) / (real2 - real1)
        delta = (real2 * should1 - real1 * should2) / (real2 - real1)

        emit.message("Rescaling with delta={:3f} and speed={:3f}".format(delta, speed))
        _rescale(subitems, parsed_args.filepath, delta, speed)


class RescaleParamsCommand(BaseCommand):
    """Rescale a subs file using parameters delta and speed."""

    name = "rescale-params"
    help_msg = "Rescale using parameters delta and speed."
    overview = textwrap.dedent("""
        Rescale a subtitles file using parameters delta and speed.

          e.g.: substool rescale-params Movie.srt 2.3 1.0014

        These params are normally retrieved from rescale-points applied to a similar file.
    """)

    def fill_parser(self, parser):
        """Add own parameters to the general parser."""
        parser.add_argument("filepath", type=valid_filepath, help="The subtitle to process")
        parser.add_argument("delta", type=float, help="The delta to apply to times")
        parser.add_argument("speed", type=float, help="The speed to rescale times")

    def run(self, parsed_args):
        """Run the command."""
        with open(parsed_args.filepath, 'rt', encoding='utf8') as fh:
            subitems = parse_subtitle(fh.read())
        _rescale(subitems, parsed_args.filepath, parsed_args.delta, parsed_args.speed)


class RescaleMimicCommand(BaseCommand):
    """Do several checks on the subtitle file(s); decompress and extract if needed."""

    name = "rescale-mimic"
    help_msg = "Rescale using a source file."
    overview = textwrap.dedent("""
        Rescale a subtitles file using initial and final points from 'source' subtitles file.

          e.g.: substool rescale-mimic subs-to-fix.srt source-subs.srt
    """)

    def fill_parser(self, parser):
        """Add own parameters to the general parser."""
        parser.add_argument("tofix-filepath", type=valid_filepath, help="The subtitle to fix")
        parser.add_argument("source-filepath", type=valid_filepath, help="The source subtitle")

    def run(self, parsed_args):
        """Run the command."""
        with open(parsed_args.source_filepath, 'rt', encoding='utf8') as fh:
            subitems = parse_subtitle(fh.read())
        should1 = subitems[0].tfrom
        should2 = subitems[-1].tfrom

        with open(parsed_args.tofix_filepath, 'rt', encoding='utf8') as fh:
            subitems = parse_subtitle(fh.read())
        real1 = subitems[0].tfrom
        real2 = subitems[-1].tfrom

        speed = (should2 - should1) / (real2 - real1)
        delta = (real2 * should1 - real1 * should2) / (real2 - real1)

        emit.message(f"Rescaling with delta={delta:3f} and speed={speed:3f}")
        _rescale(subitems, parsed_args.tofix_filepath, delta, speed)


class ShiftCommand(BaseCommand):
    """Shift the times in a subtitles file the specified seconds."""

    name = "shift"
    help_msg = "Shift times in a subtitles file."
    overview = textwrap.dedent("""
        Shift the times in a subtitles file the specified seconds.

          example: substool shift Movie.srt 3.22
                   substool shift Movie.srt -2,1
    """)

    def fill_parser(self, parser):
        """Add own parameters to the general parser."""
        parser.add_argument("filepath", type=valid_filepath, help="The subtitle to process")
        parser.add_argument("delta", type=_friendly_float, help="Delta seconds to shift all times")

    def run(self, parsed_args):
        """Run the command."""
        inpfile = parsed_args.filepath
        with open(inpfile, 'rt', encoding='utf8') as fh:
            subitems = parse_subtitle(fh.read())

        delta = parsed_args.delta
        emit.debug(f"Delta: {delta}")

        newitems = []
        for item in subitems:
            newfrom = item.tfrom + delta
            newto = item.tto + delta
            newitems.append(SubItem(newfrom, newto, item.text))

        outfile = _build_fixed_filename(inpfile)
        save_srt(newitems, outfile)
        emit.message("Done")


class AdjustCommand(BaseCommand):
    """Adjust the .srt phrase times using the timepoints from the .idx one."""

    name = "adjust"
    help_msg = "Adjust all .srt times to what .idx presents."
    overview = textwrap.dedent("""
        Adjust the .srt phrase times, using the timepoints from the .idx one.
    """)

    # how much time is ok between items divergence in subtitle adjustment
    MAX_SUB_SEPARATION = .5

    def fill_parser(self, parser):
        """Add own parameters to the general parser."""
        parser.add_argument("srt-filepath", type=valid_filepath, help="The subtitle to fix")
        parser.add_argument("idx-filepath", type=valid_filepath, help="The source subtitle")

    def run(self, parsed_args):
        """Run the command."""
        inpfile = parsed_args.srt_filepath
        with open(inpfile, 'rt', encoding='utf8') as fh:
            srt_items = parse_subtitle(fh.read())

        with open(parsed_args.idx_filepath, 'rt', encoding='ascii') as fh:
            idx_tstamps = []
            for line in fh:
                if line.startswith('timestamp'):
                    time_string = line[11:23]
                    tstamp = time_sub2stamp(time_string)
                    idx_tstamps.append(tstamp)

        newitems = []
        srt_pos = idx_pos = 0
        while srt_pos < len(srt_items) and idx_pos < len(idx_tstamps):
            srt_item = srt_items[srt_pos]
            idx_tstamp = idx_tstamps[idx_pos]

            sub_len = srt_item.tto - srt_item.tfrom
            delta = abs(idx_tstamp - srt_item.tfrom)
            if delta > self.MAX_SUB_SEPARATION:
                # too much of a difference, let's find a better match
                new_srt_pos, new_idx_pos = self.find_matching_pair(
                    srt_items, idx_tstamps, srt_pos, idx_pos)
                if new_srt_pos != srt_pos or new_idx_pos != idx_pos:
                    for i in range(srt_pos, new_srt_pos):
                        newitems.append(srt_items[i])
                    srt_pos = new_srt_pos
                    idx_pos = new_idx_pos
                    continue
                else:
                    emit.message(
                        f"WARNING: big delta: {delta:.3f} (srt={time_stamp2sub(srt_item.tfrom)} "
                        f"idx={time_stamp2sub(idx_tstamp)}) {srt_item.text!r}")

            new_from = idx_tstamp
            new_to = idx_tstamp + sub_len

            # check that it doesn't overlap to the next one
            if srt_pos + 1 < len(srt_items):
                next_from = srt_items[srt_pos + 1].tfrom
                if new_to > next_from:
                    new_to = next_from - 0.01

            newitems.append(SubItem(new_from, new_to, srt_item.text))
            srt_pos += 1
            idx_pos += 1

        # check outliers
        if idx_pos < len(idx_tstamps):
            emit.message(
                f"WARNING: timestamps missing at the end! {idx_pos} {len(idx_tstamps)}")
        for i in range(srt_pos, len(srt_items)):
            emit.message(f"WARNING: missing outlier sub: {srt_items[i]}")

        outfile = _build_fixed_filename(inpfile)
        save_srt(newitems, outfile)
        emit.debug("Done")

    def find_matching_pair(self, srt_items, idx_tstamps, srt_pos, idx_pos):
        """Find a match between next five items."""
        min_delta = 999999999999999999
        delta_items = None
        pairs = itertools.chain(zip(range(5), [0] * 5),
                                zip([0] * 4, range(1, 5)))
        for si, ii in pairs:
            new_si = srt_pos + si
            new_ii = idx_pos + ii
            if new_si >= len(srt_items) or new_ii >= len(idx_tstamps):
                continue
            srt_t = srt_items[new_si].tfrom
            idx_t = idx_tstamps[new_ii]
            delta = abs(srt_t - idx_t)
            if delta < min_delta:
                min_delta = delta
                delta_items = new_si, new_ii
        return delta_items
