from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

from .config import PipelineConfig
from .utils import file_sha256


OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
ZIP_MAGIC = b"PK\x03\x04"
OOXML_EXTENSIONS = {".docx", ".xlsx", ".xlsm", ".xltx", ".xltm"}
IGNORED_NAMES = {".ds_store", "thumbs.db"}


class IngestionError(Exception):
    def __init__(self, error_code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.error_code = error_code
        self.retryable = retryable


def discover_input_files(
    sources: list[str | Path], config: PipelineConfig, staging_dir: Path | None = None
) -> tuple[list[Path], dict]:
    """Resolve files, folders and ZIP archives into a deterministic document list."""
    staging_dir = staging_dir or config.work_dir.parent / "ingestion_staging"
    accepted: list[Path] = []
    skipped: list[dict[str, str]] = []
    seen: set[Path] = set()

    def add_file(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        if path.name.lower() in IGNORED_NAMES or path.name.startswith("~$"):
            skipped.append({"path": str(path), "reason": "temporary_or_system_file"})
        elif path.suffix.lower() in config.ingestion.allowed_extensions:
            accepted.append(path)
        else:
            skipped.append({"path": str(path), "reason": "unsupported_extension"})

    for raw_source in sources:
        source = Path(raw_source)
        if not source.exists():
            skipped.append({"path": str(source), "reason": "not_found"})
            continue
        if source.is_dir():
            for child in sorted(source.rglob("*")):
                if child.is_file():
                    add_file(child)
            continue
        if source.suffix.lower() != ".zip":
            add_file(source)
            continue

        digest = file_sha256(source)[:12]
        target_root = staging_dir / f"{source.stem}__{digest}"
        target_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(source) as bundle:
            members = bundle.infolist()
            if len(members) > config.ingestion.max_archive_files:
                raise IngestionError(
                    "archive_too_many_files",
                    f"ZIP contains {len(members)} entries; limit is {config.ingestion.max_archive_files}",
                )
            uncompressed_mb = sum(member.file_size for member in members) / (1024 * 1024)
            if uncompressed_mb > config.ingestion.max_archive_mb:
                raise IngestionError(
                    "archive_too_large",
                    f"ZIP expands to {uncompressed_mb:.1f} MB; limit is {config.ingestion.max_archive_mb} MB",
                )
            for member in members:
                target = (target_root / member.filename).resolve()
                if not target.is_relative_to(target_root.resolve()):
                    raise IngestionError("unsafe_zip", f"Unsafe path in ZIP: {member.filename}")
            bundle.extractall(target_root)
        for child in sorted(target_root.rglob("*")):
            if child.is_file():
                add_file(child)

    accepted.sort(key=lambda path: str(path).lower())
    return accepted, {
        "sources": [str(Path(source)) for source in sources],
        "accepted": [str(path) for path in accepted],
        "skipped": skipped,
    }


def validate_file(path: Path, config: PipelineConfig) -> dict:
    """Validate extension, size and file signature before parsing.

    Returns facts that parsers and the manifest need (for example macro presence).
    Raises ``IngestionError`` so failures are reported instead of silently skipped.
    """
    if not path.exists() or not path.is_file():
        raise IngestionError("file_not_found", f"File not found: {path}")
    extension = path.suffix.lower()
    if extension not in config.ingestion.allowed_extensions:
        raise IngestionError("unsupported_type", f"Unsupported file type: {extension or '(none)'}")
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > config.ingestion.max_file_mb:
        raise IngestionError(
            "file_too_large",
            f"{path.name} is {size_mb:.1f} MB, above the {config.ingestion.max_file_mb} MB limit",
        )
    if path.stat().st_size == 0:
        raise IngestionError("empty_file", f"{path.name} is empty")

    with path.open("rb") as stream:
        head = stream.read(8)

    facts = {"extension": extension, "size_mb": round(size_mb, 3), "has_macros": False}
    if extension == ".pdf":
        if not head.startswith(b"%PDF"):
            raise IngestionError("invalid_signature", f"{path.name} is not a valid PDF")
    elif extension in OOXML_EXTENSIONS:
        if head.startswith(OLE_MAGIC):
            raise IngestionError(
                "encrypted_file",
                f"{path.name} is password protected or encrypted; remove the password before ingestion",
            )
        if not head.startswith(ZIP_MAGIC):
            raise IngestionError("invalid_signature", f"{path.name} is not a valid Office Open XML file")
        try:
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
        except zipfile.BadZipFile as exc:
            raise IngestionError("corrupted_file", f"{path.name} is corrupted: {exc}") from exc
        # Macros are never executed; their presence is recorded for audit only.
        facts["has_macros"] = any(name.lower().endswith("vbaproject.bin") for name in names)
    elif extension == ".xls":
        if not head.startswith(OLE_MAGIC):
            raise IngestionError("invalid_signature", f"{path.name} is not a valid legacy Excel file")
    return facts


def find_libreoffice(configured: str | None = None) -> str | None:
    if configured:
        return configured if Path(configured).exists() or shutil.which(configured) else None
    return shutil.which("soffice") or shutil.which("libreoffice")


def convert_with_libreoffice(source: Path, target_format: str, output_dir: Path, binary: str) -> Path:
    """Convert with LibreOffice headless; used for XLS -> XLSX when available."""
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [binary, "--headless", "--convert-to", target_format, "--outdir", str(output_dir), str(source)]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=300, check=False)
    target = output_dir / f"{source.stem}.{target_format}"
    if completed.returncode != 0 or not target.exists():
        raise IngestionError(
            "conversion_failed",
            f"LibreOffice could not convert {source.name}: {completed.stderr.strip()[:300]}",
            retryable=True,
        )
    return target
