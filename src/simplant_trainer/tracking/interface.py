from typing import Protocol

from simplant_trainer.value_objects.metrics import EpochMetrics


class IMetricsTracker(Protocol):
    """Contrato para trackers de métricas de entrenamiento."""

    def record(self, metrics: EpochMetrics) -> None:
        """Registrar métricas de una época."""
        ...

    def history(self, last_n: int = 0) -> list[EpochMetrics]:
        """Retornar historial de métricas.
        Si last_n <= 0, retornar todo. Sino, las últimas N."""
        ...

    def flush(self) -> None:
        """Flush de datos pendientes (ej. a disco o red)."""
        ...
