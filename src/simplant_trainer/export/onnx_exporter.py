"""ONNX export for trained policies."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn

from simplant_trainer.agents.interface import IAgent
from simplant_trainer.value_objects.env_metadata import EnvMetadata

logger = logging.getLogger(__name__)


class OnnxExporter:
    """Export a trained policy to portable ONNX format.

    The resulting file is framework-independent and can be loaded by
    the SimPlant Runtime (or any ONNX-compatible runtime) without
    PyTorch installed.

    Environment metadata (``env_id``, observation/action shapes) is
    embedded as ONNX model properties so the Runtime can validate
    compatibility at load time.
    """

    def export(
        self,
        agent: IAgent,
        metadata: EnvMetadata,
        path: Path,
    ) -> Path:
        """Export the agent's policy network to an ONNX file.

        Steps:
          1. Extract the policy ``nn.Module`` from *agent*.
          2. Create a dummy input matching *metadata.observation_space*.
          3. ``torch.onnx.export`` with a dynamic batch axis.
          4. Embed environment metadata as ONNX model properties.
          5. Validate the resulting file.

        Returns the *path* of the written ONNX file.

        Raises
        ------
        TypeError
            If the policy network cannot be extracted from *agent*.
        RuntimeError
            If ONNX validation fails after export.
        FileNotFoundError
            If *path*'s parent directory cannot be created.
        """
        import onnx

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        policy = _extract_policy(agent)

        # eval mode: disables dropout / uses running BN stats
        was_training = policy.training
        policy.eval()

        try:
            dummy = torch.zeros(1, *metadata.observation_space.shape)
            batch_dim = torch.export.Dim("batch_size")

            torch.onnx.export(
                policy,
                (dummy,),
                str(path),
                input_names=["observation"],
                output_names=["action"],
                dynamic_shapes=({0: batch_dim},),
                opset_version=18,
            )
        finally:
            policy.train(was_training)

        # embed metadata so the Runtime can validate at load-time
        model = onnx.load(str(path))
        _embed_metadata(model, metadata)
        onnx.save(model, str(path))

        if not self.validate(path):
            raise RuntimeError(f"ONNX validation failed for {path}")

        logger.info("Exported ONNX model to %s", path)
        return path

    def validate(self, onnx_path: Path) -> bool:
        """Check that an ONNX file has a valid graph and is loadable.

        Returns ``True`` when both ``onnx.checker.check_model`` and an
        ``onnxruntime.InferenceSession`` load succeed.
        """
        import onnx
        import onnxruntime as ort

        try:
            model = onnx.load(str(onnx_path))
            onnx.checker.check_model(model)
        except Exception:
            logger.exception("ONNX graph validation failed for %s", onnx_path)
            return False

        try:
            ort.InferenceSession(
                str(onnx_path),
                providers=["CPUExecutionProvider"],
            )
        except Exception:
            logger.exception("ONNX runtime load failed for %s", onnx_path)
            return False

        return True


# --- module-private helpers -----------------------------------------------


def _extract_policy(agent: IAgent) -> nn.Module:
    """Get the policy network from *agent* via duck typing.

    Tries, in order:

    1. ``agent.export_policy()``  — explicit export hook
    2. ``agent.policy``           — public attribute
    3. ``agent._policy``          — convention for internal storage

    Raises :class:`TypeError` if none of the above yields an
    ``nn.Module``.
    """
    # 1. explicit method
    export_fn = getattr(agent, "export_policy", None)
    if callable(export_fn):
        result = export_fn()
        if isinstance(result, nn.Module):
            return result

    # 2. public attribute
    policy = getattr(agent, "policy", None)
    if isinstance(policy, nn.Module):
        return policy

    # 3. private convention
    policy = getattr(agent, "_policy", None)
    if isinstance(policy, nn.Module):
        return policy

    raise TypeError(
        "Cannot extract policy network from agent.  "
        "The agent must provide one of: export_policy() -> nn.Module, "
        "a .policy attribute, or a ._policy attribute."
    )


def _embed_metadata(model: Any, metadata: EnvMetadata) -> None:
    """Write environment metadata into the ONNX model's property bag."""
    props = {
        "env_id": metadata.env_id,
        "observation_shape": str(metadata.observation_space.shape),
        "action_shape": str(metadata.action_space.shape),
        "observation_low": str(metadata.observation_space.low.tolist()),
        "observation_high": str(metadata.observation_space.high.tolist()),
        "action_low": str(metadata.action_space.low.tolist()),
        "action_high": str(metadata.action_space.high.tolist()),
    }
    for key, value in props.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
