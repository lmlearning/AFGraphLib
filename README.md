# AFGraphLib: Graph Learning for Abstract Argumentation

Graph learning tools and research materials for **abstract argumentation**: representing arguments and attacks as graphs, then learning to predict argument acceptance.

## Explore the repository

| Path | Purpose |
| --- | --- |
| [GraphLib](GraphLib/) | Graph utilities, models and inference code. |
| [AFs](AFs/) | Example argumentation frameworks and associated solutions. |
| [AFGCNv2](AFGCNv2/) | Solver implementation, checkpoints and accompanying research material. |
| [AFGCN_new.py](AFGCN_new.py) | GCN experiment implementation. |
| [pyg_train.py](pyg_train.py) | Graph-learning training script. |

Start with the [model code](GraphLib/model.py) to inspect the architecture or the [solver documentation](AFGCNv2/README) to explore solving. Training scripts are research entry points; review their imports, data paths and configuration before running them.

## Related projects

- [AFGCN](https://github.com/lmlearning/AFGCN): dedicated solver and training repository.
- [FastAFGCN](https://github.com/lmlearning/FastAFGCN): quantized ONNX inference.
- [AFSubsample](https://github.com/lmlearning/AFSubsample): framework subsampling and analysis.

## Component tests

```bash
python -m pip install pytest
python -m pytest tests
```

The device-transfer helper is isolated in `GraphLib/device.py` and remains available
through `GraphLib.dglutil`. It uses the requested device and keeps the original
attribute if its transfer fails. The graph is modified in place; earlier successful
transfers are not rolled back. These tests use graph/tensor protocol doubles and do
not require DGL or validate GPU execution or model training.

## License

See [LICENSE](LICENSE).
