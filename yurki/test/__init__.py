import os

import yurki


def copy(data: list[str], jobs: int = 1) -> list[str]:
    return yurki.internal.copy_string_list(data, jobs=jobs)


def copy_v2(data: list[str], jobs: int = 1) -> list[str]:
    return yurki.internal.copy_string_list_v2(data, jobs=jobs)


__all__ = ["copy", "copy_v2"]
