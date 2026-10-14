from pathlib import Path
from typing import Protocol

from simplant_trainer.agents.interface import IAgent
from simplant_trainer.value_objects.env_metadata import EnvMetadata


class IOnnxExporter(Protocol):
    """Contrato para exportación de políticas a formato ONNX."""

    def export(self, agent: IAgent, metadata: EnvMetadata, path: Path) -> Path:
        """Exportar la policy del agente a ONNX. RETORNAR path del archivo."""
        ...

    def validate(self, onnx_path: Path) -> bool:
        """Validar que un archivo ONNX es correcto y cargable."""
        ...
