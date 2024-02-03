# Copyright 2014-2024 Facundo Batista
# Licensed under the Apache v2 License
# For further info, check https://github.com/facundobatista/subtitles

import collections
import itertools
import logging
import os
import re
import subprocess
import sys
import textwrap
import zipfile
from xml.etree import ElementTree

logger = logging.getLogger()
_h = logging.StreamHandler()
_h.setLevel(logging.DEBUG)
_h.setFormatter(logging.Formatter("%(asctime)s  %(message)s"))
logger.addHandler(_h)
logger.setLevel(logging.INFO)

SubItem = collections.namedtuple("SubItem", "tfrom tto text")

# how much time is ok between items divergence in subtitle adjustment
MAX_SUB_SEPARATION = .5

# lines longer than this (in chars) do not fit nicely in the screen
MAX_TEXT_LENGTH = 90

# to clean some tags
RE_TAGS = re.compile("<[^>]*>")

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


def time_sub2stamp(subinfo):
    """Convert time from sub style to timestamp."""
    if "," in subinfo:
        hms, msec = subinfo.split(",")
    elif "." in subinfo:
        hms, msec = subinfo.split(".")
    else:
        hms = subinfo
        msec = "0"

    parts = hms.split(":")
    if len(parts) == 1:
        s = int(parts[0])
        h, m = 0, 0
    elif len(parts) == 2:
        m, s = int(parts[0]), int(parts[1])
        h = 0
    elif len(parts) == 3:
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    elif len(parts) == 4:
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
        msec = parts[3]
    else:
        raise ValueError("Time not understood: {!r}".format(subinfo))
    tstamp = h * 3600 + m * 60 + s + int(msec.ljust(3, '0')) / 1000
    return tstamp


def _time_stamp2sub(tstamp):
    """Convert time from timestamp to sub style."""
    msec = int(round(1000 * (tstamp % 1)))
    x, s = divmod(int(tstamp), 60)
    h, m = divmod(x, 60)
    subinfo = "{:02}:{:02}:{:02},{:03}".format(h, m, s, msec)
    return subinfo


def _build_srt_item(pack):
    """Build an item from the lines."""
    times = pack[1].split()
    assert times[1] == '-->', "Bad separation in timestamp {}".format(times)
    tfrom = time_sub2stamp(times[0])
    tto = time_sub2stamp(times[2])
    text = '\n'.join(pack[2:])
    if not text.strip():
        # empty block! ignore
        return
    return SubItem(tfrom=tfrom, tto=tto, text=text)


def load_srt(content):
    """Parse the subtitle file in a SRT format."""
    results = []
    pack = []
    errors = False
    prevempty = False
    for i, line in enumerate(content.splitlines(), 1):
        # clean the tags
        line = RE_TAGS.sub("", line)

        line = line.strip()
        if not line:
            prevempty = True
            continue

        if prevempty and line.isdigit() and pack:
            try:
                results.append(_build_srt_item(pack))
            except Exception as err:
                errors = True
                logger.error("ERROR parsing the subtitle: %r", err)
                logger.error("The problem is in this block (line=%s): %r", i, pack)
            pack = []
        prevempty = False
        pack.append(line)

    if pack:
        try:
            results.append(_build_srt_item(pack))
        except Exception as err:
            errors = True
            logger.error("ERROR parsing the subtitle: %r", err)
            logger.error("The problem is in this block: %r", pack)

    results = [r for r in results if r is not None]

    if errors:
        exit()
    else:
        logger.debug("File parsed ok")
    return results


def _build_sub_item(pack):
    """Build an item from the lines."""
    times = pack[0].split(',')
    tfrom = time_sub2stamp(times[0])
    tto = time_sub2stamp(times[1])

    # grab the text lines, splitting them
    text = []
    for line in pack[1:]:
        text.extend(x.strip() for x in line.split('[br]'))
    text = '\n'.join(text)

    return SubItem(tfrom=tfrom, tto=tto, text=text)


def _load_sub(content):
    """Parse the subtitle file in a SUB format."""
    results = []
    pack = []
    errors = False
    prevempty = False
    started = False
    for i, line in enumerate(content.splitlines(), 1):
        line = line.strip()

        # consume the header
        if not started:
            if line[0] == '[':
                continue
            else:
                started = True

        # flag the end of the block
        if not line:
            prevempty = True
            continue

        if prevempty and pack:
            try:
                results.append(_build_sub_item(pack))
            except Exception as err:
                errors = True
                logger.error("ERROR parsing the subtitle: %r", err)
                logger.error("The problem is in this block (line=%s): %r", i, pack)
            pack = []
        prevempty = False
        pack.append(line)

    if pack:
        try:
            results.append(_build_sub_item(pack))
        except Exception as err:
            errors = True
            logger.error("ERROR parsing the subtitle: %r", err)
            logger.error("The problem is in this block: %r", pack)

    if errors:
        exit()
    else:
        logger.debug("File parsed ok")
    return results


def _load_sub2(content):
    """Parse the subtitle file in another SUB format (not sure how named)."""
    # remove the possible BOM
    if content.startswith("\ufeff"):
        content = content[1:]

    results = []
    errors = False
    for i, line in enumerate(content.splitlines(), 1):
        line = line.strip()
        m = re.match(r"{(\d+)}{(\d+)}(.*)", line)
        if not m:
            errors = True
            logger.error("ERROR parsing the subtitle: unknown line format for %r", line)
            continue

        tfrom_ticks, tto_ticks, raw_text = m.groups()

        # time is in ticks, almost 24fps
        tfrom = int(tfrom_ticks) / 23.976
        tto = int(tto_ticks) / 23.976

        # text lines are separated by pipe
        text = '\n'.join(raw_text.split('|'))

        sub_item = SubItem(tfrom=tfrom, tto=tto, text=text)
        results.append(sub_item)

    if errors:
        exit()
    else:
        logger.debug("File parsed ok")
    return results


def _load_tt(content):
    """Parse the subtitle file in a TimedText format."""
    xml = ElementTree.fromstring(content)
    body = xml[0]
    assert body.tag == 'body'

    results = []
    for item in body:
        assert item.tag == 'p'
        tfrom = int(item.get('t')) / 1000
        tto = tfrom + int(item.get('d')) / 1000
        text = item.text
        sub = SubItem(tfrom=tfrom, tto=tto, text=text)
        results.append(sub)

    logger.debug("File parsed ok")
    return results


def _load_vtt(content):
    """Parse the subtitle file in a VTT format.

    It just remove any header, add a subtitle number to each block,
    and send the rest to SRT loader.
    """
    content = content.splitlines()
    for i, line in enumerate(content):
        if not line:
            # header finished
            break
    new_content = []
    chunk_number = 1
    for line in content[i + 1:]:
        # detect block headers
        m = re.match(r"(\d\d:\d\d:\d\d.\d+ --> \d\d:\d\d:\d\d.\d+).*", line)
        if m:
            (header,) = m.groups()
            new_content.append(str(chunk_number))
            chunk_number += 1
            new_content.append(header)
            continue

        # clean the tags and store
        line = RE_TAGS.sub("", line).strip()
        new_content.append(line)

    return load_srt("\n".join(new_content))


def _load_xml(content):
    """Parse the subtitle file from a XML.

    This is typically what we found in Ñuflex.
    """
    xml = ElementTree.fromstring(content)
    (body,) = [node for node in xml if node.tag.endswith('body')]

    (div,) = body
    assert div.tag.endswith('div')

    def _parse_time(time_point):
        assert time_point[-1] == 't'
        tstamp = int(time_point[:-1]) / 10000000
        return tstamp

    results = []
    for item in div:
        assert item.tag.endswith('p')
        tfrom = _parse_time(item.get('begin'))
        tto = _parse_time(item.get('end'))

        lines = []
        for span in item:
            if span.text is not None:
                lines.append(span.text.strip())
        text = '\n'.join(lines)

        sub = SubItem(tfrom=tfrom, tto=tto, text=text)
        results.append(sub)

    logger.debug("File parsed ok")
    return results


def load_ssa(content):
    """Parse the subtitle file in a SSA format."""
    fields_names = None
    items = []
    for i, line in enumerate(content.splitlines(), 1):
        line = line.strip()

        if line.startswith('Format:'):
            # store the format to use in Dialogue lines
            fields_names = [x.strip().lower() for x in line[7:].split(',')]

        if line.startswith('Dialogue:'):
            if fields_names is None:
                raise ValueError("Found a Dialogue line before having Format")
            parts = [x.strip() for x in line[9:].split(',', maxsplit=len(fields_names) - 1)]
            fields = dict(zip(fields_names, parts))

            tfrom = time_sub2stamp(fields['start'])
            tto = time_sub2stamp(fields['end'])
            text = fields['text'].replace('\\N', '\n')
            si = SubItem(tfrom=tfrom, tto=tto, text=text)
            items.append(si)

    return items


def _load_subtitle(content):
    """Load the subtitle in any of the supported formats."""
    if content[0] == "\ufeff":
        content = content[1:]
    first_line = content.split("\n")[0]
    if content.startswith('[Script Info]'):
        loader = load_ssa
    elif content.startswith('WEBVTT'):
        loader = _load_vtt
    elif '<timedtext format="3">' in content:
        loader = _load_tt
    elif content.startswith('[INFORMATION]'):
        loader = _load_sub
    elif first_line.count("{") == 2 and first_line.count("}") == 2:
        loader = _load_sub2
    elif first_line.startswith("<?xml"):
        loader = _load_xml
    else:
        loader = load_srt
    return loader(content)


def save_srt(subitems, outfile):
    """Save the items to a srt file."""
    with open(outfile, 'wt', encoding='utf8') as fh:
        for i, item in enumerate(subitems, 1):
            tfrom = _time_stamp2sub(item.tfrom)
            tto = _time_stamp2sub(item.tto)
            tline = "{} --> {}".format(tfrom, tto)
            fh.write('\n'.join((str(i), tline, item.text)) + '\n\n')

    base, ext = os.path.splitext(outfile)
    if ext != ".srt":
        new_outfile = base + ".srt"
        logger.debug("Renaming subtitle file from %r to %r", outfile, new_outfile)
        os.rename(outfile, new_outfile)


def _rescale(subitems, inpfile, delta, speed):
    """Real rescaling."""
    newitems = []
    for item in subitems:
        newfrom = item.tfrom * speed + delta
        newto = item.tto * speed + delta
        newitems.append(SubItem(newfrom, newto, item.text))

    outfile = inpfile[:-4] + "-fixed" + inpfile[-4:]
    save_srt(newitems, outfile)
    logger.debug("Done")


def rescale_params(inpfile, delta, speed):
    """Rescaling main process, based in params."""
    with open(inpfile, 'rt', encoding='utf8') as fh:
        subitems = _load_subtitle(fh.read())

    _rescale(subitems, inpfile, float(delta), float(speed))


def rescale_mimic(inpfile, sourcefile):
    """Rescaling main process, based in other sub."""
    with open(sourcefile, 'rt', encoding='utf8') as fh:
        subitems = _load_subtitle(fh.read())
    should1 = subitems[0].tfrom
    should2 = subitems[-1].tfrom

    with open(inpfile, 'rt', encoding='utf8') as fh:
        subitems = _load_subtitle(fh.read())
    real1 = subitems[0].tfrom
    real2 = subitems[-1].tfrom

    speed = (should2 - should1) / (real2 - real1)
    delta = (real2 * should1 - real1 * should2) / (real2 - real1)

    logger.info("Rescaling with delta={:3f} and speed={:3f}".format(delta, speed))
    _rescale(subitems, inpfile, delta, speed)


def rescale_points(inpfile, num1, time1, num2, time2):
    """Rescaling main process, based in points."""
    with open(inpfile, 'rt', encoding='utf8') as fh:
        subitems = _load_subtitle(fh.read())

    if num1 == "0":
        logger.debug("Calculating from zero")
        real1 = should1 = 0
    else:
        item1 = subitems[int(num1) - 1]
        tstamp1 = time_sub2stamp(time1)
        logger.debug("Found 1st item %-4s with time %-4s, should be %s (%r)",
                     num1, _time_stamp2sub(item1.tfrom), _time_stamp2sub(tstamp1), item1.text)
        real1 = item1.tfrom
        should1 = tstamp1

    item2 = subitems[int(num2) - 1]
    tstamp2 = time_sub2stamp(time2)
    logger.debug("Found 2nd item %-4s with time %-4s, should be %s (%r)",
                 num2, _time_stamp2sub(item2.tfrom), _time_stamp2sub(tstamp2), item2.text)
    real2 = item2.tfrom

    should2 = tstamp2

    speed = (should2 - should1) / (real2 - real1)
    delta = (real2 * should1 - real1 * should2) / (real2 - real1)

    logger.info("Rescaling with delta={:3f} and speed={:3f}".format(delta, speed))
    _rescale(subitems, inpfile, delta, speed)


def shift(inpfile, delta_str):
    """Rescaling main process."""
    with open(inpfile, 'rt', encoding='utf8') as fh:
        subitems = _load_subtitle(fh.read())

    delta = float(delta_str.replace(",", "."))
    logger.debug("Delta: %f", delta)

    newitems = []
    for item in subitems:
        newfrom = item.tfrom + delta
        newto = item.tto + delta
        newitems.append(SubItem(newfrom, newto, item.text))

    outfile = inpfile[:-4] + "-fixed" + inpfile[-4:]
    save_srt(newitems, outfile)
    logger.debug("Done")


def _fix_times(subitems):
    """Really fix subitems times."""
    newitems = []
    for i, item in enumerate(subitems, 1):
        # check if something needs to be fixed
        if item.tfrom == item.tto:
            logger.info("Times: fixing sub {} (same times)".format(i))
        elif i < len(subitems) and item.tto > subitems[i].tfrom:
            logger.info("Times: fixing cross timings between {} and {}".format(i, i + 1))
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


def fix_times(inpfile):
    """Fix permanency in screen of each text."""
    with open(inpfile, 'rt', encoding='utf8') as fh:
        subitems = _load_subtitle(fh.read())

    newitems = _fix_times(subitems)

    outfile = inpfile[:-4] + "-fixed" + inpfile[-4:]
    save_srt(newitems, outfile)
    logger.debug("Done")


def adjust(inpfile_srt, inpfile_idx):
    """Fix permanency in screen of each text."""
    with open(inpfile_srt, 'rt', encoding='utf8') as fh:
        srt_items = _load_subtitle(fh.read())

    with open(inpfile_idx, 'rt', encoding='ascii') as fh:
        idx_tstamps = []
        for line in fh:
            if line.startswith('timestamp'):
                time_string = line[11:23]
                tstamp = time_sub2stamp(time_string)
                idx_tstamps.append(tstamp)

    def _find_matching_pair(srt_pos, idx_pos):
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

    newitems = []
    srt_pos = idx_pos = 0
    while srt_pos < len(srt_items) and idx_pos < len(idx_tstamps):
        srt_item = srt_items[srt_pos]
        idx_tstamp = idx_tstamps[idx_pos]

        sub_len = srt_item.tto - srt_item.tfrom
        delta = abs(idx_tstamp - srt_item.tfrom)
        if delta > MAX_SUB_SEPARATION:
            # too much of a difference, let's find a better match
            new_srt_pos, new_idx_pos = _find_matching_pair(srt_pos, idx_pos)
            if new_srt_pos != srt_pos or new_idx_pos != idx_pos:
                for i in range(srt_pos, new_srt_pos):
                    newitems.append(srt_items[i])
                srt_pos = new_srt_pos
                idx_pos = new_idx_pos
                continue
            else:
                logger.warning("WARNING: big delta: %.3f (srt=%s idx=%s) %r",
                               delta, _time_stamp2sub(srt_item.tfrom),
                               _time_stamp2sub(idx_tstamp), srt_item.text)

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
        logger.warning("WARNING: timestamps missing at the end! %s %s", idx_pos, len(idx_tstamps))
    for i in range(srt_pos, len(srt_items)):
        logger.warning("WARNING: missing outlier sub: %s", srt_items[i])

    outfile = inpfile_srt[:-4] + "-fixed" + inpfile_srt[-4:]
    save_srt(newitems, outfile)
    logger.debug("Done")


def _open_zip(inpfile):
    """Open zip files."""
    logger.info("Opening the ZIP file: %r", inpfile)
    zf = zipfile.ZipFile(inpfile)
    to_process = zf.namelist()
    for fname in to_process:
        zf.extract(fname)
    os.remove(inpfile)
    return to_process


def _open_rar(inpfile):
    """Open RAR files."""
    logger.info("Opening the RAR file: %r", inpfile)
    cmd = ["/usr/bin/unrar", "-y", "x", inpfile]
    out = subprocess.check_output(cmd)
    lines = out.decode("utf8").split("\n")
    inside = [line.split('\x08') for line in lines if line.startswith("Extracting  ")]
    if not all(x[-1].strip() == "OK" for x in inside):
        logger.error("    ERROR opening the .rar:\n %s", out)
        exit()
    os.remove(inpfile)
    to_process = [x[0].split(maxsplit=1)[1].strip() for x in inside]
    return to_process


def _open_multiple_encodings(inpfile):
    """Open the text file trying different encodings."""
    logger.debug("Test encoding...")
    try:
        with open(inpfile, 'rt', encoding='utf8') as fh:
            content = fh.read()
    except UnicodeDecodeError:
        pass
    else:
        logger.debug("Encoding was ok")
        return content

    try:
        with open(inpfile, 'rt', encoding='utf16') as fh:
            content = fh.read()
    except UnicodeError:
        pass
    else:
        with open(inpfile, 'wt', encoding='utf8') as fh:
            fh.write(content)
        logger.info("Fixed encoding (was utf16)")
        return content

    # default!
    with open(inpfile, 'rt', encoding='latin1') as fh:
        content = fh.read()
    with open(inpfile, 'wt', encoding='utf8') as fh:
        fh.write(content)
    logger.info("Fixed encoding (was latin1)")
    return content


def check(inpfile):
    """Check subtitles sanity."""
    to_process = [inpfile]
    while True:
        new_to_process = []
        dirty = False
        for toproc in to_process:
            ext = toproc.split(".")[-1]
            if ext.lower() == 'zip':
                results = _open_zip(toproc)
                dirty = True
                new_to_process.extend(results)
            elif ext.lower() == 'rar':
                results = _open_rar(toproc)
                dirty = True
                new_to_process.extend(results)
            elif ext.lower() in ("srt", "ssa", "vtt", "tt", "sub", "xml"):
                new_to_process.append(toproc)
            else:
                logger.warning("WARNING: ignoring filename: %r", toproc)
        to_process = new_to_process
        if not dirty:
            break

    for inpfile in to_process:
        logger.info("Found: %r", inpfile)

        # encoding, fix it if needed
        content = _open_multiple_encodings(inpfile)
        subitems = _load_subtitle(content)

        # times sanity
        logger.debug("Test times sanity...")
        newitems = _fix_times(subitems)
        if newitems == subitems:
            logger.debug("Times were sane")
        subitems = newitems

        # split texts with too many lines
        logger.debug("Check texts with too many lines...")
        newitems = _fix_toomanylines(subitems)
        if newitems == subitems:
            logger.debug("Items were ok")
        subitems = newitems

        # split too-long texts
        logger.debug("Check too-long texts...")
        newitems = _fix_toolong(subitems)
        if newitems == subitems:
            logger.debug("Lengths were ok")
        subitems = newitems

        # clean spam
        logger.debug("Checking for spam...")
        for item in subitems[:]:
            if any(x in item.text for x in SPAM_STRINGS):
                logger.info("Removing spam: %r", item.text)
                subitems.remove(item)

        save_srt(subitems, inpfile)
        logger.debug("Done")


def die(error=None):
    if error:
        print("ERROR:", error)
    print("""
Usage: subtitles.py [-v|--verbose] cmd <options>

    rescale_points subfile.srt id1 time1 id2 time2
      Rescale the subtitles using two points (id1 and id2) traslating
      them to the new times (time1 and time2)
        example: subtitles.py rescale_points Movie.srt 4 43,5 168 1:02:15
      If id1 is 0, time1 is ignored and all is calculated against beginning

    rescale_params subfile.srt delta speed
      Rescale the subtitles using parameters delta & speed (normally received from rescale_points)
        example: subtitles.py rescale_params Movie.srt 2.3 1.0014

    rescale_mimic subfile.srt source.srt
      Rescale the subtitles using initial and final points from 'source' subtitle
        example: subtitles.py rescale_mimic movie_es.srt movie_en.srt

    shift subfile.srt delta_seconds
      Move the subtitles in time the specified seconds
        example: subtitles.py shift Movie.srt 3.22
                 subtitles.py shift Movie.srt -2,1

    fix-times subfile.srt
      Fix the times of each phrase in the subtitle, using arbitrarious rules

    check subfile [subfile2 [...]]
      Do several checks on the subfile; decompress and extract if needed

    adjust subfile.srt subfile.idx
      Adjust the .srt phrase times, using the timepoints from the .idx one
""")
    exit()


def main():
    if "-v" in sys.argv:
        verbose = True
        sys.argv.remove("-v")
    elif "--verbose" in sys.argv:
        verbose = True
        sys.argv.remove("--verbose")
    else:
        verbose = False
    if verbose:
        logger.setLevel(logging.DEBUG)

    if len(sys.argv) < 2:
        die()
    cmd = sys.argv[1]
    params = sys.argv[2:]
    if cmd == 'rescale_points':
        if len(params) != 5:
            die("Need 5 parameters for rescale_points, got %d" % len(params))
        rescale_points(*params)
    elif cmd == 'rescale_mimic':
        if len(params) != 2:
            die("Need 2 parameters for rescale_params, got %d" % len(params))
        rescale_mimic(*params)
    elif cmd == 'rescale_params':
        if len(params) != 3:
            die("Need 5 parameters for rescale_params, got %d" % len(params))
        rescale_params(*params)
    elif cmd == 'fix-times':
        if len(params) != 1:
            die()
        fix_times(*params)
    elif cmd == 'adjust':
        if len(params) != 2:
            die()
        adjust(*params)
    elif cmd == 'check':
        if len(params) < 1:
            die()
        for inpfile in params:
            check(inpfile)
    elif cmd == 'shift':
        if len(params) != 2:
            die()
        shift(*params)
    else:
        die()
