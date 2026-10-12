"""Run the full red-team experiment matrix and write reports/.

Thin wrapper around ``aegis.report.generate_report`` so the experiments can be
run either as ``aegis report`` or ``python scripts/red_team.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from aegis.config import load_config
from aegis.report import generate_report


def main() -> int:
    cfg = load_config()
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else cfg.path("reports_dir")
    generate_report(cfg, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
