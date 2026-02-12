# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import functools
import glob
import os
from dataclasses import dataclass

from cuda.pathfinder._dynamic_libs.load_dl_common import DynamicLibNotFoundError as DynamicLibNotFoundError
from cuda.pathfinder._dynamic_libs.supported_nvidia_libs import IS_WINDOWS
from cuda.pathfinder._utils.env_vars import get_cuda_home_or_path
from cuda.pathfinder._utils.find_sub_dirs import find_sub_dirs_all_sitepackages

# Alias for libdevice-specific API (see __init__.py)
LibdeviceNotFoundError = DynamicLibNotFoundError


@dataclass
class LocatedLibdevice:
    """Path to libdevice bitcode and how it was found."""

    abs_path: str
    found_via: str


# Site-package paths for libdevice (following SITE_PACKAGES_LIBDIRS pattern)
SITE_PACKAGES_LIBDEVICE_DIRS = (
    "nvidia/cuda_nvvm/nvvm/libdevice",  # CTK 13+
    "nvidia/cuda_nvcc/nvvm/libdevice",  # CTK <13
)

FILENAME = "libdevice.10.bc"
if IS_WINDOWS:
    bases = [r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA", r"C:\CUDA"]
else:
    bases = ["/usr/local/cuda", "/opt/cuda"]


def _no_such_file_in_dir(dir_path: str, filename: str, error_messages: list[str], attachments: list[str]) -> None:
    error_messages.append(f"No such file: {os.path.join(dir_path, filename)}")
    if os.path.isdir(dir_path):
        attachments.append(f'  listdir("{dir_path}"):')
        for node in sorted(os.listdir(dir_path)):
            attachments.append(f"    {node}")
    else:
        attachments.append(f'  Directory does not exist: "{dir_path}"')


class _FindLibdevice:
    REL_PATH = os.path.join("nvvm", "libdevice")

    def __init__(self) -> None:
        self.error_messages: list[str] = []
        self.attachments: list[str] = []
        self.abs_path: str | None = None

    def try_site_packages(self) -> str | None:
        for rel_dir in SITE_PACKAGES_LIBDEVICE_DIRS:
            sub_dir = tuple(rel_dir.split("/"))
            for abs_dir in find_sub_dirs_all_sitepackages(sub_dir):
                file_path = os.path.join(abs_dir, FILENAME)
                if os.path.isfile(file_path):
                    return file_path
        return None

    def try_with_conda_prefix(self) -> str | None:
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if not conda_prefix:
            return None

        anchor = os.path.join(conda_prefix, "Library") if IS_WINDOWS else conda_prefix
        file_path = os.path.join(anchor, self.REL_PATH, FILENAME)
        if os.path.isfile(file_path):
            return file_path
        return None

    def try_with_cuda_home(self) -> str | None:
        cuda_home = get_cuda_home_or_path()
        if cuda_home is None:
            self.error_messages.append("CUDA_HOME/CUDA_PATH not set")
            return None

        file_path = os.path.join(cuda_home, self.REL_PATH, FILENAME)
        if os.path.isfile(file_path):
            return file_path

        _no_such_file_in_dir(
            os.path.join(cuda_home, self.REL_PATH),
            FILENAME,
            self.error_messages,
            self.attachments,
        )
        return None

    def try_common_paths(self) -> str | None:

        for base in bases:
            # Direct path
            file_path = os.path.join(base, self.REL_PATH, FILENAME)
            if os.path.isfile(file_path):
                return file_path
            # Versioned paths (e.g., /usr/local/cuda-13.0)
            for versioned in sorted(glob.glob(base + "*"), reverse=True):
                if os.path.isdir(versioned):
                    file_path = os.path.join(versioned, self.REL_PATH, FILENAME)
                    if os.path.isfile(file_path):
                        return file_path
        return None

    def raise_not_found_error(self) -> None:
        err = ", ".join(self.error_messages) if self.error_messages else "No search paths available"
        att = "\n".join(self.attachments) if self.attachments else ""
        raise DynamicLibNotFoundError(f'Failure finding "{FILENAME}": {err}\n{att}')


@functools.cache
def locate_libdevice() -> LocatedLibdevice | None:
    """Locate libdevice*.bc and how it was found.

    Returns:
        LocatedLibdevice with abs_path and found_via, or None if not found.
        found_via is one of: "site-packages", "conda", "CUDA_HOME", "supported_install_dir".
    """
    finder = _FindLibdevice()

    abs_path = finder.try_site_packages()
    if abs_path is not None:
        return LocatedLibdevice(abs_path=abs_path, found_via="site-packages")

    abs_path = finder.try_with_conda_prefix()
    if abs_path is not None:
        return LocatedLibdevice(abs_path=abs_path, found_via="conda")

    abs_path = finder.try_with_cuda_home()
    if abs_path is not None:
        return LocatedLibdevice(abs_path=abs_path, found_via="CUDA_HOME")

    abs_path = finder.try_common_paths()
    if abs_path is not None:
        return LocatedLibdevice(abs_path=abs_path, found_via="supported_install_dir")

    return None


def get_libdevice_path() -> str | None:
    """Get the path to libdevice*.bc, or None if not found."""
    loc = locate_libdevice()
    return loc.abs_path if loc else None


def find_libdevice() -> str:
    """Find the path to libdevice*.bc.

    Raises:
        DynamicLibNotFoundError: If libdevice.10.bc cannot be found
    """
    loc = locate_libdevice()
    if loc is None:
        raise DynamicLibNotFoundError(f"{FILENAME} not found")
    return loc.abs_path
