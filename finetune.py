"""Phase 2: fine-tune the CGI-trained RPSClassifier on CGI images + webcam captures.

    uv run finetune.py

Reads   rps-classifier/ (starting weights), the CGI dataset, captures/<class>/*.png
Writes  rps-classifier-finetuned/ (model.safetensors + config.json)
Then    uv run main.py rps-classifier-finetuned
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors.torch import save_file
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import v2

from main import load_model, whiten

HERE = Path(__file__).parent
CGI_DIR = HERE.parent / "Rock paper scissors classifier/LFD2000-resources/datasets/RPS"
CAPTURE_DIR = HERE / "captures"
OUT_DIR = HERE / "rps-classifier-finetuned"

SEED = 52
EPOCHS = 15
LR = 3e-4
BATCH_SIZE = 32
CACHE_SIZE = 160          # images are cached at this size, then augmented down to 128
CAPTURE_SHARE = 0.4       # fraction of each epoch drawn from captures (they are ~10x fewer)
RUN_LEN, TEST_EVERY = 10, 5  # captures: runs of 10 consecutive frames, every 5th run is test


def load_image(path):
    """Read as RGB and whiten the background, just like main.py does live."""
    rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    rgb, _ = whiten(rgb)
    return cv2.resize(rgb, (CACHE_SIZE, CACHE_SIZE), interpolation=cv2.INTER_AREA)


def split_captures(class_names):
    """Hold out whole runs of consecutive frames, so near-identical neighbours of a
    test frame are less likely to be in the training set."""
    train, test = [], []
    for label, name in enumerate(class_names):
        files = sorted((CAPTURE_DIR / name).glob("*.png"))  # names are timestamps
        for i, f in enumerate(files):
            (test if (i // RUN_LEN) % TEST_EVERY == TEST_EVERY - 1 else train).append((f, label))
    return train, test


def cgi_items(split, class_names):
    return [(f, label) for label, name in enumerate(class_names)
            for f in sorted((CGI_DIR / split / name).glob("*.png"))]


class Images(Dataset):
    def __init__(self, items, transform):
        self.images = [load_image(f) for f, _ in items]
        self.labels = [label for _, label in items]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.transform(self.images[i]), self.labels[i]


def evaluate(model, dataset, device, class_names, title):
    model.eval()
    loader = DataLoader(dataset, batch_size=128)
    preds, labels = [], []
    with torch.inference_mode():
        for x, y in loader:
            preds.append(model(x.to(device)).argmax(1).cpu())
            labels.append(y)
    preds, labels = torch.cat(preds), torch.cat(labels)
    n = len(class_names)
    cm = torch.zeros(n, n, dtype=torch.int64)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    acc = (preds == labels).float().mean().item()
    print(f"\n{title}: accuracy {acc:.1%}  (n={len(labels)})")
    print("            pred " + " ".join(f"{c[:5]:>6}" for c in class_names))
    for i, c in enumerate(class_names):
        recall = cm[i, i].item() / max(cm[i].sum().item(), 1)
        print(f"  {c:>9}      " + " ".join(f"{v:6d}" for v in cm[i].tolist()) + f"   recall {recall:.2f}")
    return acc


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, class_names, image_size = load_model(device)

    train_tf = v2.Compose([
        v2.ToImage(),
        # Zoom out as well as in, so the model sees hands of different sizes in the box.
        v2.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.75, 1.15), fill=255),
        v2.RandomHorizontalFlip(),
        v2.Resize(image_size),
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03),
        v2.ToDtype(torch.float32, scale=True),
    ])
    test_tf = v2.Compose([v2.ToImage(), v2.Resize(image_size), v2.ToDtype(torch.float32, scale=True)])

    cap_train, cap_test = split_captures(class_names)
    print(f"captures: {len(cap_train)} train, {len(cap_test)} test; loading images...")
    cgi_train = Images(cgi_items("train", class_names), train_tf)
    cap_train_ds = Images(cap_train, train_tf)
    cgi_test = Images(cgi_items("test", class_names), test_tf)
    cap_test_ds = Images(cap_test, test_tf)

    train_ds = torch.utils.data.ConcatDataset([cgi_train, cap_train_ds])
    weights = ([(1 - CAPTURE_SHARE) / len(cgi_train)] * len(cgi_train)
               + [CAPTURE_SHARE / len(cap_train_ds)] * len(cap_train_ds))
    sampler = WeightedRandomSampler(weights, num_samples=len(cgi_train), replacement=True,
                                    generator=torch.Generator().manual_seed(SEED))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)

    print("\n===== BEFORE fine-tuning =====")
    before = (evaluate(model, cgi_test, device, class_names, "CGI test"),
              evaluate(model, cap_test_ds, device, class_names, "Webcam test"))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"epoch {epoch + 1:2d}/{EPOCHS}: loss {total / len(train_loader):.4f}")

    print("\n===== AFTER fine-tuning =====")
    after = (evaluate(model, cgi_test, device, class_names, "CGI test"),
             evaluate(model, cap_test_ds, device, class_names, "Webcam test"))
    print(f"\nCGI test    {before[0]:.1%} -> {after[0]:.1%}")
    print(f"Webcam test {before[1]:.1%} -> {after[1]:.1%}")

    OUT_DIR.mkdir(exist_ok=True)
    save_file(model.state_dict(), OUT_DIR / "model.safetensors")
    (OUT_DIR / "config.json").write_text(json.dumps({
        "num_classes": len(class_names), "class_names": class_names, "image_size": list(image_size),
        "preprocessing": "whiten() background to white before the test transform",
    }, indent=2))
    print(f"saved {OUT_DIR.relative_to(HERE)}/")


if __name__ == "__main__":
    main()
