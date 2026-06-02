# micrograd

A small educational autograd and neural network project built as part of my learning process.

This repository is inspired by Andrej Karpathy's `micrograd` YouTube video and was created to better understand how automatic differentiation, backpropagation, and simple neural network building blocks work under the hood.

## What is included

- A scalar-based autograd engine in `micrograd/engine.py`
- Simple neural network modules in `micrograd/nn.py`
- Tests comparing selected behavior against PyTorch in `tests/`
- A notebook for experimentation in `notebook/micrograd.ipynb`

## Purpose

The goal of this project is learning, not building a production deep learning framework. It is meant as a hands-on exercise to explore:

- computational graphs
- forward passes
- gradient propagation
- basic neurons, layers, and MLP structure

## Running locally

Install dependencies:

```bash
uv sync
```

Run the test suite:

```bash
uv run pytest -v
```

## Acknowledgement

Credit and inspiration go to Andrej Karpathy for the original `micrograd` teaching material and YouTube walkthrough that motivated this implementation.
