# Copyright 2024 Facundo Batista
# Licensed under the Apache v2 License
# For further info, check https://github.com/facundobatista/subtitles

"""Main entry point of the modules."""

import sys

from craft_cli import (
    ArgumentParsingError,
    CommandGroup,
    CraftError,
    Dispatcher,
    EmitterMode,
    ProvideHelpException,
    emit,
)

from subtitles import command_check, commands_rest

commands = [
    command_check.CheckCommand,
    commands_rest.RescalePointsCommand,
    commands_rest.RescaleParamsCommand,
    commands_rest.RescaleMimicCommand,
    commands_rest.ShiftCommand,
    commands_rest.AdjustCommand,
]


def main():
    """Handle the main entry point."""
    emit.init(EmitterMode.BRIEF, "subtitles", "Starting the subtitles app.")
    command_groups = [CommandGroup("Basic", commands)]
    summary = "Tool to handle and fix subtitles general issues."

    try:
        dispatcher = Dispatcher("subtitles", command_groups, summary=summary)
        dispatcher.pre_parse_args(sys.argv[1:])
        dispatcher.load_command(None)
        dispatcher.run()
    except (ArgumentParsingError, ProvideHelpException) as err:
        print(err, file=sys.stderr)  # to stderr, as argparse normally does
        emit.ended_ok()
    except CraftError as err:
        emit.error(err)
    except KeyboardInterrupt as exc:
        error = CraftError("Interrupted.")
        error.__cause__ = exc
        emit.error(error)
    except Exception as exc:
        error = CraftError(f"Application internal error: {exc!r}")
        error.__cause__ = exc
        emit.error(error)
    else:
        emit.ended_ok()
