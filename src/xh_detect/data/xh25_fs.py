from __future__ import annotations

import ctypes
import os
import stat
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path


def _windows_error(code: int, message: str, path: Path | None = None) -> OSError:
    win_error = getattr(ctypes, "WinError", None)
    if win_error is not None:
        error = win_error(code)
        error.strerror = f"{message}: {error.strerror}"
        if path is not None:
            error.filename = str(path)
        return error
    error = OSError(0, message, str(path) if path is not None else None)
    error.winerror = code  # type: ignore[attr-defined]
    return error


def _open_windows_directory(path: Path) -> int:
    from ctypes import wintypes

    class ByHandleFileInformation(ctypes.Structure):
        _fields_ = [
            ("file_attributes", wintypes.DWORD),
            ("creation_time", wintypes.FILETIME),
            ("last_access_time", wintypes.FILETIME),
            ("last_write_time", wintypes.FILETIME),
            ("volume_serial_number", wintypes.DWORD),
            ("file_size_high", wintypes.DWORD),
            ("file_size_low", wintypes.DWORD),
            ("number_of_links", wintypes.DWORD),
            ("file_index_high", wintypes.DWORD),
            ("file_index_low", wintypes.DWORD),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    get_information = kernel32.GetFileInformationByHandle
    get_information.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ByHandleFileInformation),
    ]
    get_information.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL

    file_read_attributes = 0x0080
    delete_access = 0x00010000
    file_share_read = 0x0001
    file_share_write = 0x0002
    open_existing = 3
    file_flag_backup_semantics = 0x02000000
    file_flag_open_reparse_point = 0x00200000
    invalid_handle = ctypes.c_void_p(-1).value
    handle = create_file(
        str(path),
        file_read_attributes | delete_access,
        file_share_read | file_share_write,
        None,
        open_existing,
        file_flag_backup_semantics | file_flag_open_reparse_point,
        None,
    )
    if handle == invalid_handle:
        error = ctypes.get_last_error()
        raise _windows_error(error, "cannot lock directory", path)

    information = ByHandleFileInformation()
    if not get_information(handle, ctypes.byref(information)):
        error = ctypes.get_last_error()
        close_handle(handle)
        raise _windows_error(error, "cannot inspect locked directory", path)
    file_attribute_directory = 0x0010
    file_attribute_reparse_point = 0x0400
    if information.file_attributes & file_attribute_reparse_point:
        close_handle(handle)
        raise ValueError(f"refusing reparse point directory: {path}")
    if not information.file_attributes & file_attribute_directory:
        close_handle(handle)
        raise ValueError(f"output path is not a directory: {path}")
    return int(handle)


def _close_windows_handle(handle: int) -> None:
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL
    if not close_handle(handle):
        error = ctypes.get_last_error()
        raise _windows_error(error, "cannot close directory lock")


@contextmanager
def locked_directories(paths: Iterable[Path]) -> Iterator[None]:
    normalized = tuple(dict.fromkeys(Path(path) for path in paths))
    handles: list[int] = []
    try:
        if os.name == "nt":
            for path in normalized:
                handles.append(_open_windows_directory(path))
        else:
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
            for path in normalized:
                if stat.S_ISLNK(os.lstat(path).st_mode):
                    raise ValueError(f"refusing symlink directory: {path}")
                descriptor = os.open(path, flags)
                if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
                    os.close(descriptor)
                    raise ValueError(f"output path is not a directory: {path}")
                handles.append(descriptor)
        yield
    finally:
        errors: list[OSError] = []
        for handle in reversed(handles):
            try:
                if os.name == "nt":
                    _close_windows_handle(handle)
                else:
                    os.close(handle)
            except OSError as error:
                errors.append(error)
        if errors:
            raise errors[0]
