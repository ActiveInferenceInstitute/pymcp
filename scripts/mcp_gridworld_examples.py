#!/usr/bin/env python3
"""Three actual grid scenarios; no synthetic histories, metrics or fallback plots."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.example_support import example_cli, grid_case


async def grid_3x3(client, directory):
    return await grid_case(client, directory, "Grid3x3", [3, 3], [[2, 2]], 20)


async def grid_4x4(client, directory):
    return await grid_case(client, directory, "Grid4x4", [4, 4], [[3, 3]], 30)


async def multiple_rewards(client, directory):
    return await grid_case(client, directory, "MultiReward", [5, 5],
                           [[1, 1], [1, 3], [3, 1], [3, 3]], 40)


CASES = (("grid_3x3", grid_3x3), ("grid_4x4", grid_4x4), ("multiple_rewards", multiple_rewards))


def main(argv=None):
    return example_cli(__doc__, CASES, argv)


if __name__ == "__main__":
    raise SystemExit(main())
