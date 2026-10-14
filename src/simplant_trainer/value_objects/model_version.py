"""Model version value object for the registry."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .env_metadata import EnvMetadata
from .metrics import EvaluationReport

_SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")


@dataclass(frozen=True, slots=True)
class ModelVersion:
    """A registered version of an exported model (immutable).

    Links an ONNX artifact to its environment metadata, evaluation
    metrics, and integrity hash.
    """

    version_id: str
    env_metadata: EnvMetadata
    evaluation: EvaluationReport
    artifact_path: Path
    sha256: str
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    def __post_init__(self) -> None:
        if not isinstance(self.version_id, str) or not self.version_id.strip():
            raise ValueError("version_id must be a non-empty string")

        if not isinstance(self.env_metadata, EnvMetadata):
            raise TypeError(
                f"env_metadata must be EnvMetadata, "
                f"got {type(self.env_metadata).__name__}"
            )

        if not isinstance(self.evaluation, EvaluationReport):
            raise TypeError(
                f"evaluation must be EvaluationReport, "
                f"got {type(self.evaluation).__name__}"
            )

        # Accept str or Path for artifact_path, normalise to Path.
        if isinstance(self.artifact_path, str):
            object.__setattr__(self, "artifact_path", Path(self.artifact_path))
        elif not isinstance(self.artifact_path, Path):
            raise TypeError(
                f"artifact_path must be a Path or str, "
                f"got {type(self.artifact_path).__name__}"
            )

        # Normalise sha256 to lowercase, validate format.
        if not isinstance(self.sha256, str):
            raise TypeError(f"sha256 must be a str, got {type(self.sha256).__name__}")
        sha = self.sha256.lower()
        if not _SHA256_PATTERN.match(sha):
            raise ValueError(
                f"sha256 must be a 64-character lowercase hex string, "
                f"got '{self.sha256}'"
            )
        object.__setattr__(self, "sha256", sha)

        if not isinstance(self.created_at, datetime):
            raise TypeError(
                f"created_at must be a datetime, got {type(self.created_at).__name__}"
            )
