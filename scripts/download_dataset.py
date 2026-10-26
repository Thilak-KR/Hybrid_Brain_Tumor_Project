
"""Download the BraTS-GLI training archive from Synapse without extracting it."""

from __future__ import annotations

import hashlib
import sys
from getpass import getpass
from pathlib import Path

from synapseclient import Synapse
from synapseclient.core.exceptions import (
    SynapseAuthenticationError,
    SynapseAuthorizationError,
    SynapseError,
)
from synapseclient.models import File
from synapseclient.operations import FileOptions, get as get_entity


SYNAPSE_ID = "syn51514132"
DESTINATION = Path(r"D:\BrainTumorDataset\raw")
HASH_CHUNK_SIZE = 8 * 1024 * 1024



def format_size(size_bytes: int) -> str:
    """Format a byte count using binary units."""
    size = float(size_bytes)

    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:,.2f} {unit}"
        size /= 1024

    return f"{size_bytes:,} B"


def calculate_md5(path: Path) -> str:
    """Calculate the MD5 checksum of a file in chunks."""
    digest = hashlib.md5()

    with path.open("rb") as archive:
        for chunk in iter(
            lambda: archive.read(HASH_CHUNK_SIZE),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def verify_archive(
    path: Path,
    expected_size: int,
    expected_md5: str,
) -> int:
    """Verify the archive's size and checksum."""
    actual_size = path.stat().st_size

    if actual_size != expected_size:
        raise RuntimeError(
            "Archive size does not match Synapse metadata: "
            f"expected {expected_size:,} bytes, "
            f"found {actual_size:,} bytes. "
            "The existing file was left untouched."
        )

    actual_md5 = calculate_md5(path)

    if actual_md5.lower() != expected_md5.lower():
        raise RuntimeError(
            "Archive MD5 does not match Synapse metadata. "
            "The existing file was left untouched."
        )

    return actual_size


def main() -> int:
    """Authenticate, download, and verify the training archive."""
    drive = Path("D:\\")

    if not drive.is_dir():
        print(
            "Destination drive D: is not available.",
            file=sys.stderr,
        )
        return 1

    try:
        DESTINATION.mkdir(parents=True, exist_ok=True)
        destination = DESTINATION.resolve()

        pat = getpass(
            "Enter your Synapse PAT (input is hidden): "
        ).strip()

        if not pat:
            raise ValueError("No Synapse PAT was provided.")

        syn = Synapse()
        syn.login(authToken=pat)

        # Retrieve metadata without downloading the archive.
        remote_file = get_entity(
            synapse_id=SYNAPSE_ID,
            file_options=FileOptions(download_file=False),
            synapse_client=syn,
        )

        if not isinstance(remote_file, File):
            raise RuntimeError(
                f"{SYNAPSE_ID} is not a Synapse file entity."
            )

        if (
            not remote_file.name
            or Path(remote_file.name).name != remote_file.name
        ):
            raise RuntimeError(
                "Synapse returned an invalid archive filename."
            )

        # Validate the file-handle metadata before downloading.
        file_handle = remote_file.file_handle

        expected_size = (
            file_handle.content_size if file_handle else None
        )
        expected_md5 = (
            file_handle.content_md5 if file_handle else None
        )

        if (
            not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size <= 0
        ):
            raise RuntimeError(
                "Synapse file-handle metadata did not include "
                "a valid archive size."
            )

        if (
            not isinstance(expected_md5, str)
            or len(expected_md5) != 32
            or any(
                char not in "0123456789abcdefABCDEF"
                for char in expected_md5
            )
        ):
            raise RuntimeError(
                "Synapse file-handle metadata did not include "
                "a valid MD5."
            )

        target = destination / remote_file.name

        print(f"Dataset: {remote_file.name} ({SYNAPSE_ID})")
        print(f"Download destination: {target}")
        print(f"Expected archive size: {format_size(expected_size)}")

        # Never overwrite or delete an existing file automatically.
        if target.exists():
            if not target.is_file():
                raise RuntimeError(
                    "The destination already exists and is not "
                    f"a file: {target}"
                )

            size = verify_archive(
                target,
                expected_size,
                expected_md5,
            )

            print(
                "A complete, verified archive already exists; "
                "no download started."
            )
            print(
                f"Final file size: {size:,} bytes "
                f"({format_size(size)})"
            )
            return 0

        print(
            "Starting download. The archive will not be extracted."
        )

        downloaded_file = get_entity(
            synapse_id=SYNAPSE_ID,
            file_options=FileOptions(
                download_file=True,
                download_location=str(destination),
                if_collision="keep.local",
            ),
            synapse_client=syn,
        )

        downloaded_path = Path(
            downloaded_file.path or target
        )

        if downloaded_path.resolve() != target.resolve():
            raise RuntimeError(
                "Synapse returned an unexpected download path: "
                f"{downloaded_path}"
            )

        if not target.is_file():
            raise RuntimeError(
                "Synapse returned without creating the expected "
                f"archive: {target}"
            )

        size = verify_archive(
            target,
            expected_size,
            expected_md5,
        )

        print("\nDownload completed and integrity checks passed.")
        print(f"Dataset: {remote_file.name}")
        print(f"Download destination: {target}")
        print(
            f"Final file size: {size:,} bytes "
            f"({format_size(size)})"
        )

        return 0

    except (
        SynapseAuthenticationError,
        SynapseAuthorizationError,
    ) as exc:
        print(
            "Synapse authentication or access was denied. "
            f"Check your PAT and dataset permissions: {exc}",
            file=sys.stderr,
        )

    except SynapseError as exc:
        print(
            f"Synapse download or API request failed: {exc}",
            file=sys.stderr,
        )

    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Download stopped: {exc}", file=sys.stderr)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
