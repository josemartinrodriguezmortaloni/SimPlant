from pathlib import Path
from typing import Protocol

from simplant_trainer.value_objects.env_metadata import EnvMetadata
from simplant_trainer.value_objects.metrics import EvaluationReport
from simplant_trainer.value_objects.model_version import ModelVersion


class IModelRegistry(Protocol):
    """Contrato para el registro versionado de modelos exportados."""

    def register(
        self,
        onnx_path: Path,
        metadata: EnvMetadata,
        evaluation: EvaluationReport,
    ) -> ModelVersion:
        """Registrar un modelo exportado. RETORNAR la versión creada."""
        ...

    def get(self, version_id: str) -> ModelVersion:
        """Obtener una versión por ID. KeyError si no existe."""
        ...

    def list_versions(self) -> list[ModelVersion]:
        """Listar todas las versiones, ordenadas por fecha de creación."""
        ...

    def latest(self) -> ModelVersion | None:
        """Retornar la versión más reciente, o None si no hay ninguna."""
        ...
