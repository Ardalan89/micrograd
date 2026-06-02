from micrograd.engine import Value
import torch 

def test_value_matches_pytorch():
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(0.5)

    out = ((a * b) + c).tanh() + a**2
    out.backward()

    ta = torch.tensor(2.0, requires_grad=True)
    tb = torch.tensor(-3.0, requires_grad=True)
    tc = torch.tensor(0.5, requires_grad=True)

    tout = torch.tanh((ta * tb) + tc) + ta**2
    tout.backward()

    assert abs(out.data - tout.item()) < 1e-6
    assert abs(a.grad - ta.grad.item()) < 1e-6
    assert abs(b.grad - tb.grad.item()) < 1e-6
    assert abs(c.grad - tc.grad.item()) < 1e-6

    
