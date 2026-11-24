"""兼容入口：默认只跑 MOSS；三 codec 请用 run_codecs_multi_roundtrip_feat_n1500。"""

from __future__ import annotations

import sys

from base.npu_parallel import ensure_spawn


def main() -> None:
    # Force moss-only unless user already set --codecs
    if not any(a.startswith("--codecs") for a in sys.argv[1:]):
        sys.argv.extend(["--codecs", "moss"])
    from P1.experiments.run_codecs_multi_roundtrip_feat_n1500 import main as _main

    _main()


if __name__ == "__main__":
    ensure_spawn()
    main()
