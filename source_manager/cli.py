from __future__ import annotations

import argparse

from .cleanup import cleanup_source
from .download import download_source


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="benchmark-source")
    parser.add_argument("action", choices=("download", "cleanup"))
    parser.add_argument("--config", default="source_manager/config.json")
    args = parser.parse_args(argv)
    if args.action == "download":
        result = download_source(args.config)
        print(f"source={result.path}\ncommit={result.commit}")
    else:
        result = cleanup_source(args.config)
        print(f"source={result.path}\nremoved={str(result.removed).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
