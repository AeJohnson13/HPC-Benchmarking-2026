```
docker run --gpus all -it -v ${PWD}:/workspace -w /workspace pytorch/pytorch:2.4.0-cuda12.1-cudnn8-runtime python train.py
  ```