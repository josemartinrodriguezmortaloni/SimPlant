"""Versioned model registry backed by the local filesystem."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from simplant_trainer.value_objects.env_metadata import EnvMetadata
from simplant_trainer.value_objects.metrics import EvaluationReport
from simplant_trainer.value_objects.model_version import ModelVersion
from simplant_trainer.value_objects.spaces import ActionSpace, ObservationSpace

_INDEX_FILE = "index.json"


class FileModelRegistry:
    """Versioned model registry on the local filesystem.

    Directory layout::

        registry_dir/
        ├── index.json              ← atomic-written manifest
        ├── v1_20260309T120000/
        │   └── model.onnx
        └── v2_20260310T143000/
            └── model.onnx

    The index is the single source of truth: all ``ModelVersion`` entries
    are serialised to ``index.json`` (including numpy-based spaces).
    Writes use a *.tmp + rename* strategy for atomicity on POSIX.

    Thread-safety: **NOT** thread-safe.  Caller must synchronise if
    concurrent access is needed.
    """

    def __init__(self, registry_dir: Path) -> None:
        self._dir = Path(registry_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._versions: dict[str, ModelVersion] = {}
        self._load_index()

    # --- IModelRegistry protocol ------------------------------------------

    def register(
        self,
        onnx_path: Path,
        metadata: EnvMetadata,
        evaluation: EvaluationReport,
    ) -> ModelVersion:
        """Register an exported ONNX model.  Returns the created version.

        Steps:
          1. Compute SHA-256 of the ONNX artifact.
          2. Generate a ``v{n}_{timestamp}`` version id.
          3. Copy the ONNX file into ``registry_dir/version_id/model.onnx``.
          4. Create a ``ModelVersion`` and update the in-memory index.
          5. Persist ``index.json`` atomically.

        Raises:
            FileNotFoundError: If *onnx_path* does not exist.
        """
        onnx_path = Path(onnx_path)
        if not onnx_path.is_file():
            raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

        sha256 = _compute_sha256(onnx_path)

        seq = len(self._versions) + 1
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        version_id = f"v{seq}_{ts}"

        version_dir = self._dir / version_id
        version_dir.mkdir(parents=True, exist_ok=True)
        dest = version_dir / "model.onnx"
        shutil.copy2(onnx_path, dest)

        version = ModelVersion(
            version_id=version_id,
            env_metadata=metadata,
            evaluation=evaluation,
            artifact_path=dest,
            sha256=sha256,
        )

        self._versions[version_id] = version
        self._persist_index()

        return version

    def get(self, version_id: str) -> ModelVersion:
        """Return a version by ID.  Raises ``KeyError`` if not found."""
        try:
            return self._versions[version_id]
        except KeyError:
            raise KeyError(f"Version '{version_id}' not found in registry") from None

    def list_versions(self) -> list[ModelVersion]:
        """All versions ordered by creation date (oldest first)."""
        return sorted(self._versions.values(), key=lambda v: v.created_at)

    def latest(self) -> ModelVersion | None:
        """Most recent version, or ``None`` if the registry is empty."""
        if not self._versions:
            return None
        return max(self._versions.values(), key=lambda v: v.created_at)

    # --- persistence (internal) -------------------------------------------

    def _load_index(self) -> None:
        index_path = self._dir / _INDEX_FILE
        if not index_path.exists():
            return

        raw: list[dict[str, Any]] = json.loads(index_path.read_text())
        for entry in raw:
            version = _deserialize_version(entry)
            self._versions[version.version_id] = version

    def _persist_index(self) -> None:
        """Atomically write the index to disk (.tmp → rename)."""
        payload = [_serialize_version(v) for v in self._versions.values()]
        content = json.dumps(payload, indent=2)

        tmp_path = self._dir / f"{_INDEX_FILE}.tmp"
        index_path = self._dir / _INDEX_FILE

        tmp_path.write_text(content)
        tmp_path.replace(index_path)  # atomic on POSIX


# --- pure helpers (module-private) ----------------------------------------


def _compute_sha256(path: Path) -> str:
    """Return lowercase hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _serialize_version(v: ModelVersion) -> dict[str, Any]:
    obs = v.env_metadata.observation_space
    act = v.env_metadata.action_space

    return {
        "version_id": v.version_id,
        "artifact_path": str(v.artifact_path),
        "sha256": v.sha256,
        "created_at": v.created_at.isoformat(),
        "env_metadata": {
            "env_id": v.env_metadata.env_id,
            "observation_space": {
                "shape": list(obs.shape),
                "low": obs.low.tolist(),
                "high": obs.high.tolist(),
            },
            "action_space": {
                "shape": list(act.shape),
                "low": act.low.tolist(),
                "high": act.high.tolist(),
            },
        },
        "evaluation": {
            "total_reward": v.evaluation.total_reward,
            "mean_reward": v.evaluation.mean_reward,
            "std_reward": v.evaluation.std_reward,
            "episodes": v.evaluation.episodes,
            "constraint_violations": v.evaluation.constraint_violations,
        },
    }


def _deserialize_version(raw: dict[str, Any]) -> ModelVersion:
    meta_raw = raw["env_metadata"]
    obs_raw = meta_raw["observation_space"]
    act_raw = meta_raw["action_space"]

    env_metadata = EnvMetadata(
        env_id=meta_raw["env_id"],
        observation_space=ObservationSpace(
            shape=tuple(obs_raw["shape"]),
            low=np.array(obs_raw["low"]),
            high=np.array(obs_raw["high"]),
        ),
        action_space=ActionSpace(
            shape=tuple(act_raw["shape"]),
            low=np.array(act_raw["low"]),
            high=np.array(act_raw["high"]),
        ),
    )

    evaluation = EvaluationReport(
        total_reward=raw["evaluation"]["total_reward"],
        mean_reward=raw["evaluation"]["mean_reward"],
        std_reward=raw["evaluation"]["std_reward"],
        episodes=raw["evaluation"]["episodes"],
        constraint_violations=raw["evaluation"]["constraint_violations"],
    )

    return ModelVersion(
        version_id=raw["version_id"],
        artifact_path=Path(raw["artifact_path"]),
        sha256=raw["sha256"],
        created_at=datetime.fromisoformat(raw["created_at"]),
        env_metadata=env_metadata,
        evaluation=evaluation,
    )
