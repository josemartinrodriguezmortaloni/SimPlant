"""Tests for networks.MLP — Phase 3."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from simplant_trainer.networks import INetwork, MLP


# ---------------------------------------------------------------------------
# Construction — happy path
# ---------------------------------------------------------------------------


class TestMLPConstruction:
    """Verify MLP builds the correct topology."""

    def test_minimal_two_layers(self) -> None:
        mlp = MLP(layers=[4, 2])
        out = mlp.forward(torch.randn(1, 4))
        assert out.shape == (1, 2)

    def test_single_hidden_layer(self) -> None:
        mlp = MLP(layers=[8, 64, 3])
        out = mlp.forward(torch.randn(1, 8))
        assert out.shape == (1, 3)

    def test_two_hidden_layers(self) -> None:
        mlp = MLP(layers=[4, 256, 256, 2])
        out = mlp.forward(torch.randn(1, 4))
        assert out.shape == (1, 2)

    def test_deep_network(self) -> None:
        mlp = MLP(layers=[10, 128, 64, 32, 16, 1])
        out = mlp.forward(torch.randn(1, 10))
        assert out.shape == (1, 1)

    def test_batch_forward(self) -> None:
        mlp = MLP(layers=[4, 64, 2])
        out = mlp.forward(torch.randn(32, 4))
        assert out.shape == (32, 2)


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------


class TestMLPActivations:
    """Verify activation selection works correctly."""

    @pytest.mark.parametrize("act", ["relu", "tanh", "sigmoid", "elu"])
    def test_valid_activation(self, act: str) -> None:
        mlp = MLP(layers=[4, 64, 2], activation=act)
        out = mlp.forward(torch.randn(1, 4))
        assert out.shape == (1, 2)

    @pytest.mark.parametrize("act", ["relu", "tanh", "sigmoid", "elu"])
    def test_valid_output_activation(self, act: str) -> None:
        mlp = MLP(layers=[4, 64, 2], output_activation=act)
        out = mlp.forward(torch.randn(1, 4))
        assert out.shape == (1, 2)

    def test_output_activation_none_is_linear(self) -> None:
        """With output_activation=None the last layer has NO activation."""
        mlp = MLP(layers=[4, 64, 2], output_activation=None)
        # Last module should be nn.Linear, not an activation
        last = list(mlp._net.children())[-1]
        assert isinstance(last, nn.Linear)

    def test_output_activation_applied(self) -> None:
        """With output_activation set, the last module IS an activation."""
        mlp = MLP(layers=[4, 64, 2], output_activation="tanh")
        last = list(mlp._net.children())[-1]
        assert isinstance(last, nn.Tanh)

    def test_hidden_layers_have_activation(self) -> None:
        """Hidden layers should have activation between them."""
        mlp = MLP(layers=[4, 64, 32, 2], activation="relu")
        children = list(mlp._net.children())
        # Expected: Linear, ReLU, Linear, ReLU, Linear
        # (3 Linear layers, 2 ReLU between hidden layers)
        assert isinstance(children[0], nn.Linear)
        assert isinstance(children[1], nn.ReLU)
        assert isinstance(children[2], nn.Linear)
        assert isinstance(children[3], nn.ReLU)
        assert isinstance(children[4], nn.Linear)
        assert len(children) == 5

    def test_hidden_with_output_activation(self) -> None:
        """Hidden activation + output activation both present."""
        mlp = MLP(layers=[4, 64, 32, 2], activation="relu", output_activation="tanh")
        children = list(mlp._net.children())
        # Expected: Linear, ReLU, Linear, ReLU, Linear, Tanh
        assert isinstance(children[0], nn.Linear)
        assert isinstance(children[1], nn.ReLU)
        assert isinstance(children[2], nn.Linear)
        assert isinstance(children[3], nn.ReLU)
        assert isinstance(children[4], nn.Linear)
        assert isinstance(children[5], nn.Tanh)
        assert len(children) == 6


# ---------------------------------------------------------------------------
# Validation — error cases
# ---------------------------------------------------------------------------


class TestMLPValidation:
    """Construction must fail fast on invalid inputs."""

    def test_empty_layers(self) -> None:
        with pytest.raises(ValueError, match="at least 2 elements"):
            MLP(layers=[])

    def test_single_layer(self) -> None:
        with pytest.raises(ValueError, match="at least 2 elements"):
            MLP(layers=[64])

    def test_layers_not_a_list(self) -> None:
        with pytest.raises(ValueError, match="at least 2 elements"):
            MLP(layers=(4, 2))  # type: ignore[arg-type]

    def test_zero_in_layers(self) -> None:
        with pytest.raises(TypeError, match="positive int"):
            MLP(layers=[4, 0, 2])

    def test_negative_in_layers(self) -> None:
        with pytest.raises(TypeError, match="positive int"):
            MLP(layers=[4, -1, 2])

    def test_float_in_layers(self) -> None:
        with pytest.raises(TypeError, match="positive int"):
            MLP(layers=[4, 64.0, 2])  # type: ignore[list-item]

    def test_bool_in_layers(self) -> None:
        """bool is subclass of int — must be explicitly rejected."""
        with pytest.raises(TypeError, match="positive int"):
            MLP(layers=[4, True, 2])

    def test_invalid_activation(self) -> None:
        with pytest.raises(ValueError, match="Unknown activation"):
            MLP(layers=[4, 2], activation="gelu")

    def test_invalid_output_activation(self) -> None:
        with pytest.raises(ValueError, match="Unknown output_activation"):
            MLP(layers=[4, 2], output_activation="softmax")


# ---------------------------------------------------------------------------
# INetwork protocol conformance
# ---------------------------------------------------------------------------


class TestMLPProtocol:
    """MLP must structurally satisfy INetwork."""

    def test_satisfies_inetwork(self) -> None:
        """MLP is a structural subtype of INetwork (duck typing)."""
        mlp = MLP(layers=[4, 64, 2])
        # Check all protocol methods exist and are callable
        assert callable(getattr(mlp, "forward", None))
        assert callable(getattr(mlp, "parameters", None))
        assert callable(getattr(mlp, "state_dict", None))
        assert callable(getattr(mlp, "load_state_dict", None))

    def test_parameters_returns_iterator(self) -> None:
        mlp = MLP(layers=[4, 64, 2])
        params = list(mlp.parameters())
        assert len(params) > 0
        assert all(isinstance(p, torch.nn.Parameter) for p in params)

    def test_state_dict_is_dict(self) -> None:
        mlp = MLP(layers=[4, 64, 2])
        state = mlp.state_dict()
        assert isinstance(state, dict)
        assert len(state) > 0

    def test_load_state_dict_round_trip(self) -> None:
        """Save → load → forward produces identical output."""
        mlp = MLP(layers=[4, 64, 2])
        x = torch.randn(1, 4)

        # save state
        state = mlp.state_dict()
        out_before = mlp.forward(x)

        # create fresh MLP, load state
        mlp2 = MLP(layers=[4, 64, 2])
        mlp2.load_state_dict(state)
        out_after = mlp2.forward(x)

        assert torch.allclose(out_before, out_after)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestMLPGradients:
    """Verify gradients flow correctly for training."""

    def test_gradients_flow_through_all_layers(self) -> None:
        mlp = MLP(layers=[4, 64, 32, 2])
        x = torch.randn(8, 4)
        out = mlp.forward(x)
        loss = out.sum()
        loss.backward()

        for param in mlp.parameters():
            assert param.grad is not None
            assert param.grad.abs().sum() > 0

    def test_no_nan_in_forward(self) -> None:
        mlp = MLP(layers=[4, 256, 256, 2])
        x = torch.randn(32, 4)
        out = mlp.forward(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
