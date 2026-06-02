import random
from micrograd.engine import Value

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []


class Neuron(Module):
    VALID_ACTIVATIONS = {"linear", "tanh", "relu"}
    def __init__(self, nin, activation="linear"):
        
        if activation not in self.VALID_ACTIVATIONS:
            raise ValueError(
                f"Unknown activation: {activation}"
            )
        # initialize weights and bias
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.activation = activation
        
    def __call__(self, x):
        # forward path w*x+b
        act = sum((xi*wi for xi,wi in zip(x,self.w)),self.b)
        
        if self.activation == "linear":
            return act
        elif self.activation == "tanh":
            return act.tanh()
        elif self.activation == "relu":
            return act.relu()

        raise ValueError(f"Unknown activation: {self.activation}")
    
    def parameters(self):
        return self.w + [self.b]

class Layer(Module):
    def __init__(self, nin, nout, activation="linear"):
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP(Module):
    def __init__(self, nin, nouts, activations=None):
        
        sizes = [nin] + nouts
        if activations is None:
            activations = ["linear"] * len(nouts)
        
        elif isinstance(activations, str):
            activations = [activations] * len(nouts)
        
        if len(activations) != len(nouts):
            raise ValueError(
                f"Expected {len(nouts)} activations, got {len(activations)}"
    )

        self.layers = [Layer(sizes[i],sizes[i+1], activations[i]) for i in range(len(nouts))]
    
    def __call__(self, x):
        for layer in self.layers:
             x = layer(x) # the output of one layer becomes the input to the next layer
        return x
        
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters() ]
