# Copyright 2026 Facundo Batista
# Licensed under the Apache v2 License
# For further info, check https://github.com/facundobatista/substool

"""Shared test fixtures."""

import os
import tempfile
from pathlib import Path

import pytest
from craft_cli import messages, printer


@pytest.fixture(autouse=True)
def init_emitter(monkeypatch):
    """Override craft_cli's own `init_emitter` fixture.

    craft_cli.pytest_plugin.init_emitter keeps a NamedTemporaryFile handle open while
    craft-cli re-opens that same path internally; that works on POSIX (multiple handles to
    one file are fine) but raises PermissionError on Windows, where the first handle holds
    an exclusive lock. Here the temp file's descriptor is closed before craft-cli touches
    the file, which is safe on every platform.

    A local conftest.py fixture takes precedence over a same-named one provided by a
    plugin, so this replaces craft_cli's version everywhere, without needing any
    platform check.
    """
    fd, raw_path = tempfile.mkstemp(prefix="emitter-logs")
    os.close(fd)
    log_filepath = Path(raw_path)
    monkeypatch.setattr(messages, "TESTMODE", True)
    monkeypatch.setattr(printer, "TESTMODE", True)
    try:
        messages.emit.init(
            messages.EmitterMode.QUIET,
            "test-emitter",
            "Hello world",
            log_filepath=log_filepath,
        )
        yield
    finally:
        # end machinery (just in case it was not ended before; note it's ok to "double end")
        messages.emit.ended_ok()
        log_filepath.unlink(missing_ok=True)
