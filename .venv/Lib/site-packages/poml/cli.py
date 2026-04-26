from __future__ import annotations

import os
import sys
import subprocess
import nodejs_wheel
from typing import Any, NoReturn


def node(*args: str, **kwargs: Any) -> subprocess.CompletedProcess[bytes | str]:
    return nodejs_wheel.node(args, return_completed_process=True, **kwargs)


def run(*args: str, **kwargs: Any) -> subprocess.CompletedProcess[bytes | str]:
    script = os.path.join(os.path.dirname(__file__), 'js', 'cli.js')
    if not os.path.exists(script):
        raise RuntimeError(f'Expected CLI entrypoint: {script} to exist')

    return node(script, *args, **kwargs)


def entrypoint() -> NoReturn:
    sys.exit(run(*sys.argv[1:]).returncode)
