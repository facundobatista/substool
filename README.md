# substool

Tool to handle and fix subtitles general issues.

Main command is `check`, it does several checks on the subtitle file(s).

- it decompress the given files and extract internal files if needed; otherwise just process the subtitle in any format

- convert any encoding to utf-8

- fixes phrases timings if they are clearly wrong (e.g. starting and finishing at the same time, or finishing before starting)

- split parts if lines are too long or there are too many lines in the phrase

- remove some spam

Other commands:

- `shift`: Shift times in a subtitles file

- `rescale-params`: Rescale using parameters delta and speed

- `rescale-points`: Rescale using two points in time

- `rescale-mimic`: Rescale using a source file

- `adjust`: Adjust all .srt times to what .idx presents
