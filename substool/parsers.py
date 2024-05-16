# Copyright 2014-2024 Facundo Batista
# Licensed under the Apache v2 License
# For further info, check https://github.com/facundobatista/substool

"""Subtitles parsers."""

import re
from xml.etree import ElementTree

from craft_cli import emit, CraftError

from substool.helpers import time_sub2stamp, SubItem


# to clean some tags
RE_TAGS = re.compile("<[^>]*>")


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


def _parse_sub(content):
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
                emit.error(f"ERROR parsing the subtitle: {err!r}")
                emit.error(f"The problem is in this block (line={i}): {pack!r}")
            pack = []
        prevempty = False
        pack.append(line)

    if pack:
        try:
            results.append(_build_sub_item(pack))
        except Exception as err:
            errors = True
            emit.error(f"ERROR parsing the subtitle: {err!r}")
            emit.error(f"The problem is in this block: {pack!r}")

    if not errors:
        return results


def _parse_sub2(content):
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
            emit.error(f"ERROR parsing the subtitle: unknown line format for {line!r}")
            continue

        tfrom_ticks, tto_ticks, raw_text = m.groups()

        # time is in ticks, almost 24fps
        tfrom = int(tfrom_ticks) / 23.976
        tto = int(tto_ticks) / 23.976

        # text lines are separated by pipe
        text = '\n'.join(raw_text.split('|'))

        sub_item = SubItem(tfrom=tfrom, tto=tto, text=text)
        results.append(sub_item)

    if not errors:
        return results


def _parse_tt(content):
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

    return results


def _parse_vtt(content):
    """Parse the subtitle file in a VTT format.

    It just remove any header, add a subtitle number to each block,
    and send the rest to SRT parser.
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

    return _parse_srt("\n".join(new_content))


def _parse_xml(content):
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

    return results


def _parse_ssa(content):
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


def _parse_srt(content):
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
                emit.error(f"ERROR parsing the subtitle: {err!r}")
                emit.error(f"The problem is in this block (line={i}): {pack!r}")
            pack = []
        prevempty = False
        pack.append(line)

    if pack:
        try:
            results.append(_build_srt_item(pack))
        except Exception as err:
            errors = True
            emit.error(f"ERROR parsing the subtitle: {err!r}")
            emit.error(f"The problem is in this block: {pack!r}")

    results = [r for r in results if r is not None]

    if not errors:
        return results


def parse_subtitle(content):
    """Prase the subtitle in any of the supported formats."""
    if content[0] == "\ufeff":
        content = content[1:]
    first_line = content.split("\n")[0]
    if content.startswith('[Script Info]'):
        parser = _parse_ssa
    elif content.startswith('WEBVTT'):
        parser = _parse_vtt
    elif '<timedtext format="3">' in content:
        parser = _parse_tt
    elif content.startswith('[INFORMATION]'):
        parser = _parse_sub
    elif first_line.count("{") == 2 and first_line.count("}") == 2:
        parser = _parse_sub2
    elif first_line.startswith("<?xml"):
        parser = _parse_xml
    else:
        parser = _parse_srt

    results = parser(content)
    if results is None:
        raise CraftError("Problems parsing the content")
    emit.verbose("File parsed ok")
    return results
