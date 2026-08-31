#!/usr/bin/env python3
"""Authenticated Drive access, using the generator notebook's own credential.

The public endpoint `drive.usercontent.google.com/download?id=...` enforces a
per-file *public sharing* abuse quota. Once tripped it answers HTTP 200 with a
~2 KB "Quota exceeded" HTML page for roughly a day, and no credential changes
that -- it is not an API quota at all.

The Drive API is a different system. Authenticated as the files' owner, the
relevant limits are request-rate ones (12,000 queries per 100 s per user,
effectively unlimited per day). Downloading the whole collection is ~51,000
range requests spread over hours, so under 1% of that ceiling.

The credential is a real OAuth secret for a personal Google account with full
drive scope. It is read out of the notebook at runtime and never written into
this repository -- do not copy it into a file here, and do not commit it.
"""
from __future__ import annotations

import io
import json
import os
import re

NOTEBOOK = os.environ.get(
    "DRIFTSENSE_NOTEBOOK",
    "/home/pranesh/Downloads/phase2driftsensedatasetgenerator.ipynb")
AUTH_CELL_MARKER = "GDRIVE_CLIENT_ID"


def _creds_from_notebook(path: str = NOTEBOOK):
    if not os.path.exists(path):
        raise SystemExit(f"notebook not found: {path}  (set DRIFTSENSE_NOTEBOOK)")
    nb = json.load(open(path))
    src = next((s for s in (''.join(c['source']) for c in nb['cells'])
                if AUTH_CELL_MARKER in s), None)
    if src is None:
        raise SystemExit("no cell in the notebook defines GDRIVE_CLIENT_ID")

    def grab(name):
        m = re.search(rf'^{name}\s*=\s*"([^"]+)"', src, re.M)
        if not m:
            raise SystemExit(f"{name} not found in the notebook's auth cell")
        return m.group(1)

    from google.oauth2.credentials import Credentials
    return Credentials(
        None,
        refresh_token=grab("GDRIVE_REFRESH_TOKEN"),
        client_id=grab("GDRIVE_CLIENT_ID"),
        client_secret=grab("GDRIVE_CLIENT_SECRET"),
        token_uri="https://oauth2.googleapis.com/token",
        scopes=["https://www.googleapis.com/auth/drive"],
    )


def service():
    """A Drive v3 client. Build one per thread -- httplib2 is not thread-safe."""
    from googleapiclient.discovery import build
    return build("drive", "v3", credentials=_creds_from_notebook(),
                 cache_discovery=False)


def download(svc, file_id: str, dest: str, chunk_mb: int = 16) -> str | None:
    """Stream one file to `dest`. Returns None on success, else a reason.

    Written to a .part file and renamed only on success, so an interrupted run
    can never leave a truncated tar that looks complete to the next pass.
    """
    from googleapiclient.http import MediaIoBaseDownload
    tmp = dest + ".part"
    try:
        with io.FileIO(tmp, "wb") as fh:
            dl = MediaIoBaseDownload(fh, svc.files().get_media(fileId=file_id),
                                     chunksize=chunk_mb * 1024 * 1024)
            done = False
            while not done:
                _, done = dl.next_chunk(num_retries=5)
        if os.path.getsize(tmp) < 100_000:
            os.remove(tmp)
            return "suspiciously small"
        os.replace(tmp, dest)
        return None
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        return f"{type(e).__name__}: {str(e)[:120]}"


if __name__ == "__main__":
    svc = service()
    ab = svc.about().get(fields="user(emailAddress),storageQuota").execute()
    q = ab["storageQuota"]
    print(f"authenticated as {ab['user']['emailAddress']}")
    print(f"drive usage {int(q.get('usage', 0))/1024**3:.1f} GB of "
          f"{int(q['limit'])/1024**3:.0f} GB" if q.get("limit") else "unlimited")
