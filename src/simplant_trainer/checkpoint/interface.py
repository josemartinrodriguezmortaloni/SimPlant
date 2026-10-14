from typing import Protocol

from simplant_trainer.value_objects.checkpoint_data import CheckpointData


class ICheckpointStore(Protocol):
    """Contrato para persistencia de checkpoints de entrenamiento."""

    def save(self, data: CheckpointData) -> str:
        """Persistir un checkpoint. RETORNAR el checkpoint_id."""
        ...

    def load(self, checkpoint_id: str) -> CheckpointData:
        """Cargar un checkpoint por ID. KeyError si no existe."""
        ...

    def list_checkpoints(self) -> list[str]:
        """Listar todos los checkpoint_ids disponibles, ordenados por fecha."""
        ...

    def delete(self, checkpoint_id: str) -> None:
        """Eliminar un checkpoint (idempotente)."""
        ...
