"""Path handling shared by the entry points.

``pairs.csv`` is portable data. A manifest gets written on one machine and
read on another, so deciding whether a value in it already names a location
cannot be left to ``os.path.isabs``, which answers differently depending on
the platform and -- since 3.13 -- the Python version.
"""

from __future__ import annotations

import ntpath
import os

__all__ = ["is_rooted"]


def is_rooted(value: str) -> bool:
    r"""True if *value* already names a location and must not be joined.

    \oth entry points resolve a manifest's relative paths against the
    directory the manifest sits in, and must leave rooted ones alone. The
    obvious test for that, ``os.path.isabs``, is wrong in both directions.

    Python 3.13 narrowed ``ntpath.isabs``: on Windows ``"/data/x.png"`` is
    *rooted* but not *absolute*, because it carries no drive, so ``isabs``
    now returns False where 3.11 returned True. Falling through to
    ``os.path.join(base, value)`` then splices the current drive onto it, and
    ``"/data/x.png"`` silently becomes ``"C:/data/x.png"`` -- a path that
    resolves somewhere nobody named.

    The mirror case is a manifest written on Windows and read on POSIX:
    ``posixpath.isabs("C:/data/x.png")`` is False and the value starts with
    no separator, so it too would be joined to the manifest's directory.

    This function deliberately diverges from ``os.path.isabs`` for exactly
    those Windows-authored forms. What it does not change is either kind of
    value a graded run carries: a POSIX-absolute path and a relative one are
    classified identically to before, on every platform and both Python
    versions. The Windows forms now pass through and fail at open time if
    they are genuinely unreachable, which is the honest outcome.
    """
    if os.path.isabs(value):
        return True
    # Rooted on the current drive ("/data/x", "\data\x"), and UNC roots.
    if value.startswith(("/", "\\")):
        return True
    # A drive letter, however the separators lean: "C:/data/x", "C:\data\x".
    # ntpath rather than os.path so this holds when the manifest is read on
    # POSIX.
    return bool(ntpath.splitdrive(value)[0])
