import sys

from . import watcher, core

match sys.argv[1]:
    case "scratch":
        watcher.watch()
    case "paper":
        core.generate_elcc_paper_output()
    case "export":
        core.export_xferbench_output()
    case _:
        raise ValueError(f"Unrecognized command: {sys.argv[1]}.")
