# HPC Benchmarking Code- Framework
Code for the 2026 HPC benchmarking project framework\

The framework is structured as follows:

<pre>
benchmark_framework/
├── configs/                # Hyperparameters for different models
│   ├── resnet.yaml
│   └── bert.yaml
|
├── models/                 # Model-specific PyTorch logic
│   ├── resnet_benchmark.py
│   └── other_benchmark.py
|
├── scripts/                # Utility scripts (e.g., GPU warming)
├── run_benchmark.sh        # The main entry point
└── engine.py               # Common PyTorch benchmarking logic
</pre>

To add a new model:

-Add the model's unique logic to the engine.py script
-Add the .yaml or .json to the configs folder (if needed)
-

To run a benchmarking command:
./run_benchmark --nodes=1 --gpus=8 --partition=l40s --model=resnet