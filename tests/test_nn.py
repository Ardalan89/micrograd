import torch

from micrograd.engine import Value
from micrograd.nn import Neuron


def _run_neuron_test(activation):
    # Micrograd neuron
    n = Neuron(3, activation=activation)

    # Fixed parameters
    n.w[0].data = 0.1
    n.w[1].data = -0.2
    n.w[2].data = 0.3
    n.b.data = 0.4

    # Inputs
    x = [Value(2.0), Value(-1.0), Value(0.5)]

    # Forward
    out = n(x)

    # Equivalent PyTorch computation
    tx = torch.tensor(
        [2.0, -1.0, 0.5],
        dtype=torch.float64,
        requires_grad=True,
    )

    tw = torch.tensor(
        [0.1, -0.2, 0.3],
        dtype=torch.float64,
        requires_grad=True,
    )

    tb = torch.tensor(
        0.4,
        dtype=torch.float64,
        requires_grad=True,
    )

    preact = (tx * tw).sum() + tb

    if activation == "linear":
        tout = preact
    elif activation == "tanh":
        tout = torch.tanh(preact)
    elif activation == "relu":
        tout = torch.relu(preact)
    else:
        raise ValueError(f"Unknown activation: {activation}")

    # Backward
    out.backward()
    tout.backward()

    # Forward value
    assert abs(out.data - tout.item()) < 1e-6

    # Weight gradients
    for mw, tg in zip(n.w, tw.grad):
        assert abs(mw.grad - tg.item()) < 1e-6

    # Bias gradient
    assert abs(n.b.grad - tb.grad.item()) < 1e-6

    # Input gradients
    for mx, tg in zip(x, tx.grad):
        assert abs(mx.grad - tg.item()) < 1e-6


def test_linear_neuron_matches_pytorch():
    _run_neuron_test("linear")


def test_tanh_neuron_matches_pytorch():
    _run_neuron_test("tanh")


def test_relu_neuron_matches_pytorch():
    _run_neuron_test("relu")