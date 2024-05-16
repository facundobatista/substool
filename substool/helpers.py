# Copyright 2014-2024 Facundo Batista
# Licensed under the Apache v2 License
# For further info, check https://github.com/facundobatista/substool

"""Helpers for the different commands."""

import collections
import os
import pathlib
import subprocess
import zipfile

from craft_cli import emit, CraftError, ArgumentParsingError

SubItem = collections.namedtuple("SubItem", "tfrom tto text")


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


def time_stamp2sub(tstamp):
    """Convert time from timestamp to sub style."""
    msec = int(round(1000 * (tstamp % 1)))
    x, s = divmod(int(tstamp), 60)
    h, m = divmod(x, 60)
    subinfo = "{:02}:{:02}:{:02},{:03}".format(h, m, s, msec)
    return subinfo


def open_zip(inpfile):
    """Open zip files."""
    emit.message(f"Opening the ZIP file: {str(inpfile)!r}")
    zf = zipfile.ZipFile(inpfile)
    to_process = zf.namelist()
    for fname in to_process:
        zf.extract(fname)
    inpfile.unlink()
    return [pathlib.Path(fname) for fname in to_process]


def open_rar(inpfile):
    """Open RAR files."""
    emit.message(f"Opening the RAR file: {str(inpfile)!r}")
    cmd = ["/usr/bin/unrar", "-y", "x", inpfile]
    out = subprocess.check_output(cmd)
    lines = out.decode("utf8").split("\n")
    inside = [line.split('\x08') for line in lines if line.startswith("Extracting  ")]
    if not all(x[-1].strip() == "OK" for x in inside):
        raise CraftError(f"ERROR opening the .rar:\n {out}")

    inpfile.unlink()
    to_process = [x[0].split(maxsplit=1)[1].strip() for x in inside]
    return [pathlib.Path(fname) for fname in to_process]


def save_srt(subitems, outfile):
    """Save the items to a srt file."""
    with open(outfile, 'wt', encoding='utf8') as fh:
        for i, item in enumerate(subitems, 1):
            tfrom = time_stamp2sub(item.tfrom)
            tto = time_stamp2sub(item.tto)
            tline = "{} --> {}".format(tfrom, tto)
            fh.write('\n'.join((str(i), tline, item.text)) + '\n\n')

    if outfile.suffix != ".srt":
        new_outfile = outfile.with_suffix(".srt")
        emit.verbose(f"Renaming subtitle file from {outfile} to {new_outfile}")
        outfile.rename(new_outfile)


def valid_filepath(filepath):
    """Return a valid Path with user name expansion for filepath.

    ArgumentParsingError is raised if filepath is not a valid file or is not readable.
    """
    filepath = pathlib.Path(filepath).expanduser()
    if not os.access(filepath, os.R_OK):
        raise ArgumentParsingError("Cannot access {!r}.".format(str(filepath)))
    if not filepath.is_file():
        raise ArgumentParsingError("{!r} is not a file.".format(str(filepath)))
    return filepath
