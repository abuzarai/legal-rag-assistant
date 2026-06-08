import os
import io
from dataclasses import dataclass
from typing import List, Optional, Tuple, Set, Dict

import google.auth
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account

from src.common.logger import get_logger


SCOPES = ["https://www.googleapis.com/auth/drive"]
SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "service_account.json")
FOLDER_MIME = "application/vnd.google-apps.folder"

logger = get_logger(__name__)


@dataclass
class DriveFile:
    """Normalized metadata for a Drive file ready for ingestion."""
    id: str
    name: str
    mime_type: str
    md5_checksum: Optional[str]
    modified_time: Optional[str]
    category: str
    relative_path: str


# --------------------------- Auth / Service --------------------------- #

def get_drive_service():
    """Return an authenticated Drive API client (ADC first, SA fallback)."""
    creds = None
    try:
        creds, _ = google.auth.default(scopes=SCOPES)
        logger.info("[DRIVE] Using Application Default Credentials for Drive API")
    except Exception:
        pass

    if creds is None:
        logger.info("[DRIVE] Falling back to service account credentials for Drive API")
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES
        )

    logger.info("[DRIVE] Initializing Google Drive service client")
    return build("drive", "v3", credentials=creds, cache_discovery=False)


# --------------------------- Helpers --------------------------- #

def _list_children(service, folder_id, page_token=None, drive_id=None):
    """List immediate children of a folder, handling shared drives if applicable."""
    params = dict(
        q=f"'{folder_id}' in parents and trashed=false",
        fields=(
            "nextPageToken, files("
            "id, name, mimeType, md5Checksum, modifiedTime"
            ")"
        ),
        pageToken=page_token,
        includeItemsFromAllDrives=True,
        supportsAllDrives=True,
        pageSize=1000,
    )
    if drive_id:
        params["corpora"] = "drive"
        params["driveId"] = drive_id
    else:
        params["corpora"] = "allDrives"
    return service.files().list(**params).execute()


def _entry_from_meta(meta: dict, path: Tuple[str, ...]) -> DriveFile:
    """Convert raw Drive file metadata into a DriveFile dataclass."""
    category = path[0] if path else "unclassified"
    relative_path = "/".join(path + (meta.get("name") or "",))
    return DriveFile(
        id=meta["id"],
        name=meta.get("name") or "untitled",
        mime_type=meta.get("mimeType", ""),
        md5_checksum=meta.get("md5Checksum"),
        modified_time=meta.get("modifiedTime"),
        category=category,
        relative_path=relative_path,
    )


def _ext_and_mime_allowlists(allowed_exts: List[str]) -> tuple[Set[str], Set[str]]:
    """
    Build allowlists for extensions and common MIME types that correspond to those
    extensions, so files without a proper suffix still pass if their mimeType matches.
    """
    ext_set = {e.lower().lstrip(".") for e in allowed_exts if e}
    # Minimal, safe mapping. Extend if you support more types.
    mime_map: Dict[str, Set[str]] = {
        "pdf": {"application/pdf"},
        "txt": {"text/plain"},
        # add more as needed
    }
    mime_set: Set[str] = set()
    for ext in ext_set:
        mime_set |= mime_map.get(ext, set())
    return ext_set, mime_set


def _maybe_add_file(meta: dict, path: Tuple[str, ...], allowed_exts: Set[str], allowed_mimes: Set[str], results: List[DriveFile]):
    """
    Add a file if it matches by extension OR by mimeType.
    This fixes the case where Drive filenames lack a .pdf suffix but have application/pdf mime.
    """
    name = meta.get("name") or ""
    mime = meta.get("mimeType", "")
    ext = os.path.splitext(name)[1].lower().lstrip(".")

    # If an allowlist is provided, we admit the file if either:
    #  1) it has an extension and that extension is allowed, OR
    #  2) its mime type is in our allowed mime set
    if allowed_exts:
        if (ext and ext in allowed_exts) or (mime and mime in allowed_mimes):
            results.append(_entry_from_meta(meta, path))
        return

    # No allowlist: accept anything with any (or no) extension.
    results.append(_entry_from_meta(meta, path))


# --------------------------- Recursive Listing --------------------------- #

def list_files_recursive(
    service,
    root_folder_id: str,
    allowed_exts: List[str],
    drive_id: Optional[str] = None,
) -> List[dict]:
    """
    Return normalized metadata for all matching files under the root folder.
    Traverses all nested subfolders recursively (BFS).
    """
    allowed_ext_set, allowed_mime_set = _ext_and_mime_allowlists(allowed_exts)
    from collections import deque

    dq = deque([(root_folder_id, tuple())])
    visited_folders = set()
    results: List[DriveFile] = []

    logger.info(f"[DRIVE] Starting recursive scan from root folder: {root_folder_id}")

    while dq:
        folder_id, path = dq.popleft()
        if folder_id in visited_folders:
            continue
        visited_folders.add(folder_id)

        page_token = None
        while True:
            try:
                resp = _list_children(service, folder_id, page_token, drive_id)
            except HttpError as exc:
                logger.warning(f"[DRIVE] Failed to list folder {folder_id}: {exc}")
                break

            for item in resp.get("files", []):
                name = item.get("name") or ""
                mime = item.get("mimeType", "")

                if mime == FOLDER_MIME:
                    dq.append((item["id"], path + (name,)))
                    continue

                _maybe_add_file(item, path, allowed_ext_set, allowed_mime_set, results)

            page_token = resp.get("nextPageToken")
            if not page_token:
                break

    logger.info(f"[DRIVE] Completed scan. Found {len(results)} eligible files.")
    return [entry.__dict__ for entry in results]


# --------------------------- Download --------------------------- #

def download_file(service, file_id, filename, save_dir="./data/raw_pdfs/"):
    """Download a file from Google Drive into a local directory."""
    os.makedirs(save_dir, exist_ok=True)
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    filepath = os.path.join(save_dir, filename)
    with io.FileIO(filepath, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    logger.info(f"[DRIVE] Downloaded {filename} to {filepath}")
    return filepath


# --------------------------- Diagnostic --------------------------- #

if __name__ == "__main__":
    root_id = os.environ.get("DRIVE_ROOT_FOLDER_ID")
    if not root_id:
        raise SystemExit("Set DRIVE_ROOT_FOLDER_ID before running this diagnostic script.")

    from src.common.config import get_drive_allowed_exts

    svc = get_drive_service()
    files = list_files_recursive(svc, root_id, get_drive_allowed_exts())
    print(f"Found {len(files)} eligible files under root {root_id}.")
    for f in files[:20]:
        print(" -", f.get("category", "unclassified"), "|", f.get("relative_path"))
