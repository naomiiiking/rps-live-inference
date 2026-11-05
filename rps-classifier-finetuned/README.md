---
license: mit
tags:
- pytorch
- image-classification
pipeline_tag: image-classification
---
## Paper, scissors, rock classifier

A small PyTorch CNN that classifies a hand as paper, rock or scissors. Part of my PyTorch Fundamentals (LFD2000) course. Educational use.

- Architecture: `RPSClassifier` (3 conv blocks, then AdaptiveAvgPool, Dropout and Linear), defined in the [training repo](https://github.com/naomiiiking/paper-scissors-rock-classifier).
- Input: `(N, 3, 128, 128)` RGB, float in `[0, 1]`, with no `Normalize` step. Classes (in `config.json`): `paper`, `rock`, `scissors`.

### Versions

- **Current (fine-tuned for webcams):** the CGI model fine-tuned on a mix of the CGI training set and about 300 real webcam crops of a hand in front of a light wall.
- **Original (CGI only):** trained only on the CGI images (300×300, white background). It is available in this repo's commit history.

### Results

| Test set | CGI only | Fine-tuned |
|---|---|---|
| CGI test (372 images) | 94.1% | 97.3% |
| Webcam crops held out (48) | 37.5% | 100%* |

\* The held-out webcam crops come from the same sessions as the training crops (same person, wall and lighting). Expect lower accuracy in other conditions.

### Preprocessing (important)

A webcam sees a white wall as grey, but the model was trained on pure white backgrounds. Before the usual transform, rescale the crop so the background maps to white, as was done in fine-tuning:

```python
import numpy as np, torch
from torchvision.transforms import v2

def whiten(rgb):  # uint8 HWC RGB
    ref = np.percentile(rgb.reshape(-1, 3), 90, axis=0)
    return np.clip(rgb.astype(np.float32) * (255.0 / np.maximum(ref, 1.0)), 0, 255).astype(np.uint8)

transform = v2.Compose([v2.ToImage(), v2.Resize((128, 128)), v2.ToDtype(torch.float32, scale=True)])
x = transform(whiten(square_crop_rgb)).unsqueeze(0)
```

Use a square crop where the hand fills about a third of the frame, against a plain light background.

### Limitations

- The model has no "no hand" class. An empty white image is classified as rock with high confidence, so check that a hand is present first (for example, that at least 5% of pixels are darker than 200 after whitening).
- The webcam data comes from one person, one background and one room.
