# Copyright 2014-2024 Facundo Batista
# Licensed under the Apache v2 License
# For further info, check https://github.com/facundobatista/substool

"""The Check command."""

import textwrap

from craft_cli import BaseCommand, emit

from substool.helpers import open_rar, open_zip, save_srt, SubItem, valid_filepath
from substool.parsers import parse_subtitle

# lines longer than this (in chars) do not fit nicely in the screen
MAX_TEXT_LENGTH = 90


# subs with this text will be removed
SPAM_STRINGS = [
    "OpenSubtitles",
    "Poker en Línea",
    "Subtitles MKV Player",
    "Subtitles downloaded from Podnapisi",
    "Subtítulos por aRGENTeaM",
    "TUSUBTITULO",
    "TaMaBin",
    "WWW.MY-SUBS.CO",
    "califica este sub",
    "subdivx.com",
    "www.SUBTITULOS.es",
    "www.magicgelnuru.es",
    "www.subtitulamos.tv",
    "www.tvsubtitles.net",
]


def _open_multiple_encodings(inpfile):
    """Open the text file trying different encodings."""
    emit.debug("Test encoding...")
    try:
        with open(inpfile, 'rt', encoding='utf8') as fh:
            content = fh.read()
    except UnicodeDecodeError:
        pass
    else:
        emit.debug("Encoding was ok")
        return content

    try:
        with open(inpfile, 'rt', encoding='utf16') as fh:
            content = fh.read()
    except UnicodeError:
        pass
    else:
        with open(inpfile, 'wt', encoding='utf8') as fh:
            fh.write(content)
        emit.verbose("Fixed encoding (was utf16)")
        return content

    # default!
    with open(inpfile, 'rt', encoding='latin1') as fh:
        content = fh.read()
    with open(inpfile, 'wt', encoding='utf8') as fh:
        fh.write(content)
    emit.verbose("Fixed encoding (was latin1)")
    return content


def _fix_times(subitems):
    """Fix subitems times."""
    newitems = []
    for i, item in enumerate(subitems, 1):
        # check if something needs to be fixed
        if item.tfrom >= item.tto:
            emit.verbose(f"Times: fixing sub {i} (same or inverted times)")
        elif i < len(subitems) and item.tto > subitems[i].tfrom:
            emit.verbose(f"Times: fixing cross timings between {i} and {i + 1}")
        else:
            newitems.append(item)
            continue

        # fix it! a priori, the length should be 70ms per char, with 1s min
        fixed_len = len(item.text) * .07
        if fixed_len < 1:
            fixed_len = 1

        # check that it doesn't overlap to the next one
        if i + 1 < len(subitems):
            next_from = subitems[i].tfrom
            if item.tfrom + fixed_len > next_from:
                fixed_len = next_from - item.tfrom

        new_to = item.tfrom + fixed_len
        newitems.append(SubItem(item.tfrom, new_to, item.text))
    return newitems


def _balanced_wrap(text, q_parts):
    """Wrap text in a balanced fashion."""
    limit = len(text) // q_parts

    while True:
        parts = textwrap.wrap(text, limit)
        if len(parts) <= q_parts:
            return parts
        limit += 1


def _fix_toomanylines(subitems):
    """Handle texts that are across too many lines."""
    newitems = []
    for item in subitems:
        parts = item.text.strip().split("\n")
        if len(parts) < 3:
            newitems.append(item)
            continue

        # same subtitle may have un-joinable parts (e.g. dialogs)
        chunks = []
        for part in parts:
            if not chunks or part.startswith(("*", "-", "♪")):
                chunks.append([part])
            else:
                chunks[-1].append(part)

        # if one chunk, make it use two lines; else just put one chunk on each line
        if len(chunks) == 1:
            lines = _balanced_wrap(" ".join(chunks[0]), 2)
        else:
            lines = [" ".join(chunk) for chunk in chunks]

        newtext = "\n".join(lines)
        newitem = SubItem(item.tfrom, item.tto, newtext)
        newitems.append(newitem)
    return newitems


def _fix_toolong(subitems):
    """Handle texts that are too long."""
    newitems = []
    for i, item in enumerate(subitems, 1):
        if len(item.text) <= MAX_TEXT_LENGTH:
            newitems.append(item)
            continue

        # decide how many parts and split original text into that many (results are not
        # equal, as we keep words complete)
        total_chars = len(item.text)
        q_parts = (total_chars // MAX_TEXT_LENGTH) + 1

        # textwrap returns lines with at most the limit, which tends to produce too many parts
        # (as each part is shorter than the limit), so we increment slightly the limit until
        # we have the desired parts quantity
        parts = _balanced_wrap(item.text, q_parts)

        # calculate how much each part should stay on screen (as they have different lengths)
        total_time = item.tto - item.tfrom
        durations = [(total_time * len(p) / total_chars) for p in parts]

        tfrom = item.tfrom
        for duration, text in zip(durations, parts):
            newitem = SubItem(tfrom, tfrom + duration, text)
            newitems.append(newitem)
            tfrom += duration
    return newitems


class CheckCommand(BaseCommand):
    """Do several checks on the subtitle file(s); decompress and extract if needed."""

    name = "check"
    help_msg = "Check subtitle files."
    overview = textwrap.dedent("""
        Do several checks on the subtitle file(s); decompress and extract if needed.
    """)

    def fill_parser(self, parser):
        """Add own parameters to the general parser."""
        parser.add_argument(
            "filepath", type=valid_filepath, nargs="+",
            help="The file to be checked (can be indicated several times)")

    def run(self, parsed_args):
        """Run the command."""
        for filepath in parsed_args.filepath:
            self._check(filepath)

    def _check(self, inpfile):
        """Check subtitles sanity."""
        emit.debug(f"Processing {inpfile}")
        to_process = [inpfile]
        while True:
            new_to_process = []
            dirty = False
            for toproc in to_process:
                ext = toproc.suffix.lower()[1:]
                if ext == 'zip':
                    results = open_zip(toproc)
                    dirty = True
                    new_to_process.extend(results)
                elif ext == 'rar':
                    results = open_rar(toproc)
                    dirty = True
                    new_to_process.extend(results)
                elif ext in ("srt", "ssa", "vtt", "tt", "sub", "xml"):
                    new_to_process.append(toproc)
                else:
                    emit.message(f"Ignoring filename: {str(toproc)!r}")
            to_process = new_to_process
            if not dirty:
                break

        for inpfile in to_process:
            emit.message(f"Found: {str(inpfile)}")

            # encoding, fix it if needed
            content = _open_multiple_encodings(inpfile)
            subitems = parse_subtitle(content)

            # times sanity
            emit.verbose("Test times sanity...")
            newitems = _fix_times(subitems)
            if newitems == subitems:
                emit.verbose("Times were sane")
            subitems = newitems

            # split texts with too many lines
            emit.verbose("Check texts with too many lines...")
            newitems = _fix_toomanylines(subitems)
            if newitems == subitems:
                emit.verbose("Items were ok")
            subitems = newitems

            # split too-long texts
            emit.verbose("Check too-long texts...")
            newitems = _fix_toolong(subitems)
            if newitems == subitems:
                emit.verbose("Lengths were ok")
            subitems = newitems

            # clean spam
            emit.verbose("Checking for spam...")
            for item in subitems[:]:
                if any(x in item.text for x in SPAM_STRINGS):
                    emit.verbose(f"Removing spam: {item.text!r}")
                    subitems.remove(item)

            save_srt(subitems, inpfile)
            emit.verbose("Done")
