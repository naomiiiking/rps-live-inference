---
license: mit
tags:
- pytorch
- object-detection
---
## Paper, scissors, rock classifier

A TorchVision model trained on images of hands being paper, scissors or rock. Part of my PyTorch Fundamentals (LFD2000) course. Educational use.

To run:

```bash
git clone https://github.com/naomiiiking/paper-scissors-rock-classifier.git
cd paper-scissors-rock-classifier
git clone https://github.com/lftraining/LFD2000-resources.git

uv sync
uv run pipeline.py
```

Example output:
```bash
Test accuracy: 94.62%
Confusion matrix:
tensor([[104,  13,   7],
        [  0, 124,   0],
        [  0,   0, 124]])
     paper: precision = 1.00, recall = 0.84
      rock: precision = 0.91, recall = 1.00
  scissors: precision = 0.95, recall = 1.00
```