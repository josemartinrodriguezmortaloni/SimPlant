"""Checkpoint persistence using safetensors (no pickle → no RCE)."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from simplant_trainer.value_objects.checkpoint_data import CheckpointData
from simplant_trainer.value_objects.hyperparameters import HyperParameters


class SafeTensorsCheckpointStore:
    """Persiste checkpoints usando safetensors (sin pickle → sin RCE).

    Cada checkpoint se guarda como dos archivos:
      - ``{id}.safetensors`` — agent_state (bytes)
      - ``{id}.meta.json``   — metadata (epoch, hyperparameters, created_at)
    """

    def __init__(self, base_dir: Path, max_kept: int = 5) -> None:
        self._base_dir = base_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._max_kept = max_kept

    def save(self, data: CheckpointData) -> str:
        """Persistir un checkpoint a disco. Purga los más viejos si excede max_kept."""
        cid = data.checkpoint_id

        # 1. agent_state → .safetensors
        st_path = self._base_dir / f"{cid}.safetensors"
        st_path.write_bytes(data.agent_state)

        # 2. metadata → .meta.json
        meta_path = self._base_dir / f"{cid}.meta.json"
        meta: dict[str, Any] = {
            "checkpoint_id": cid,
            "epoch": data.epoch,
            "hyperparameters": {
                "learning_rate": data.hyperparameters.learning_rate,
                "batch_size": data.hyperparameters.batch_size,
                "gamma": data.hyperparameters.gamma,
                "buffer_capacity": data.hyperparameters.buffer_capacity,
                "extras": dict(data.hyperparameters.extras)
                if data.hyperparameters.extras
                else None,
            },
            "created_at": data.created_at.isoformat(),
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        # 3. Purgar checkpoints viejos
        self._purge()

        return cid

    def load(self, checkpoint_id: str) -> CheckpointData:
        """Cargar un checkpoint desde disco."""
        st_path = self._base_dir / f"{checkpoint_id}.safetensors"
        meta_path = self._base_dir / f"{checkpoint_id}.meta.json"

        if not st_path.exists() or not meta_path.exists():
            raise KeyError(f"Checkpoint '{checkpoint_id}' not found")

        agent_state = st_path.read_bytes()
        meta = json.loads(meta_path.read_text())

        hp_raw = meta["hyperparameters"]
        hyperparameters = HyperParameters(
            learning_rate=hp_raw["learning_rate"],
            batch_size=hp_raw["batch_size"],
            gamma=hp_raw["gamma"],
            buffer_capacity=hp_raw["buffer_capacity"],
            extras=hp_raw.get("extras"),
        )

        return CheckpointData(
            checkpoint_id=meta["checkpoint_id"],
            epoch=meta["epoch"],
            agent_state=agent_state,
            hyperparameters=hyperparameters,
            created_at=datetime.fromisoformat(meta["created_at"]),
        )

    def list_checkpoints(self) -> list[str]:
        """Listar todos los checkpoint IDs, ordenados del más reciente al más viejo."""
        entries: list[tuple[str, str]] = []  # (created_at, checkpoint_id)

        for meta_file in self._base_dir.glob("*.meta.json"):
            cid = meta_file.name.removesuffix(".meta.json")
            meta = json.loads(meta_file.read_text())
            entries.append((meta["created_at"], cid))

        entries.sort(reverse=True)  # ISO 8601 ordena cronológicamente como string
        return [cid for _, cid in entries]

    def delete(self, checkpoint_id: str) -> None:
        """Eliminar un checkpoint (idempotente)."""
        (self._base_dir / f"{checkpoint_id}.safetensors").unlink(missing_ok=True)
        (self._base_dir / f"{checkpoint_id}.meta.json").unlink(missing_ok=True)

    # --- internal ---------------------------------------------------------

    def _purge(self) -> None:
        """Eliminar los checkpoints más viejos si hay más de max_kept."""
        all_ids = self.list_checkpoints()  # ya viene ordenado, más reciente primero
        for old_id in all_ids[self._max_kept :]:
            self.delete(old_id)
