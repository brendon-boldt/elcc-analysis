"""Watch and rerun a file when it updates."""

import time
from typing import Callable, Any
import importlib
import traceback
import os

from watchdog.events import FileSystemEvent, FileSystemEventHandler  # type: ignore
from watchdog.observers import Observer  # type: ignore

DEBOUNCE_TIME = 1
scratch_mod = None


def exec_module() -> None:
    global scratch_mod
    os.system("clear")
    try:
        if scratch_mod is None:
            scratch_mod = importlib.import_module("src.scratch")
        else:
            importlib.reload(scratch_mod)
    except Exception as e:
        print(traceback.format_exc())


def watch() -> None:
    class Handler(FileSystemEventHandler):

        last_ran = 0.0

        def on_any_event(self, e: FileSystemEvent) -> None:
            is_py_src = e.src_path[-3:] == ".py"
            is_modified = e.event_type == "modified"
            too_soon = time.time() - self.last_ran < DEBOUNCE_TIME
            if is_py_src and is_modified and not too_soon:
                exec_module()
                self.last_ran = time.time()

    event_handler = Handler()
    observer = Observer()
    observer.schedule(event_handler, "./src", recursive=True)
    observer.start()

    try:
        while observer.is_alive():
            time.sleep(1)
    finally:
        observer.stop()
        observer.join()
