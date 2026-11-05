import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors.torch import load_file
from torchvision.transforms import v2

from model import RPSClassifier

MODEL_DIR = Path(__file__).parent / "rps-classifier"
ROI_SIZE = 300          
EMA_KEEP = 0.7          
CONF_THRESHOLD = 0.6
PREVIEW_SCALE = 2       
WHITEN_PERCENTILE = 90  
DARK_LEVEL = 200
MIN_HAND_FRACTION = 0.05
CAPTURE_DIR = Path(__file__).parent / "captures"
CAPTURE_KEYS = {ord("p"): "paper", ord("r"): "rock", ord("s"): "scissors"}


def whiten(rgb):
    ref = np.percentile(rgb.reshape(-1, 3), WHITEN_PERCENTILE, axis=0)
    out = rgb.astype(np.float32) * (255.0 / np.maximum(ref, 1.0))
    return np.clip(out, 0, 255).astype(np.uint8), ref


def load_model(device, model_dir=MODEL_DIR):
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    model = RPSClassifier(num_classes=config["num_classes"])
    model.load_state_dict(load_file(model_dir / "model.safetensors"))
    model.to(device).eval()
    return model, config["class_names"], tuple(config["image_size"])


def roi_box(frame_h, frame_w, size):
    side = min(size, frame_h, frame_w)
    x0 = (frame_w - side) // 2
    y0 = (frame_h - side) // 2
    return x0, y0, side


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else MODEL_DIR
    model, class_names, image_size = load_model(device, model_dir)
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(image_size),
        v2.ToDtype(torch.float32, scale=True),
    ])
    print(f"Loaded {model_dir} on {device}; classes = {class_names}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit(
            "Could not open the webcam. On macOS, give your terminal/IDE camera "
            "permission in System Settings → Privacy & Security → Camera."
        )

    avg = None
    use_whiten = True
    roi_size = ROI_SIZE
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame")
                break
            frame = cv2.flip(frame, 1) 

            h, w = frame.shape[:2]
            x0, y0, side = roi_box(h, w, roi_size)
            crop_bgr = frame[y0:y0 + side, x0:x0 + side]
            raw_crop_bgr = crop_bgr.copy()  # Saving captures for fine tuning
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            crop_rgb, bg_level = whiten(crop_rgb) if use_whiten else (crop_rgb, None)
            hand_fraction = (crop_rgb.min(axis=-1) < DARK_LEVEL).mean()

            x = transform(crop_rgb).unsqueeze(0).to(device)
            if hand_fraction >= MIN_HAND_FRACTION:
                with torch.inference_mode():
                    probs = torch.softmax(model(x), dim=1)[0].cpu()
                avg = probs if avg is None else EMA_KEEP * avg + (1 - EMA_KEEP) * probs
                conf, idx = avg.max(dim=0)
                label = class_names[idx] if conf >= CONF_THRESHOLD else "?"
            else:
                avg = None  
                conf, label = torch.tensor(0.0), "no hand"

            # --- overlay ---
            colour = {"?": (0, 165, 255), "no hand": (160, 160, 160)}.get(label, (0, 200, 0))
            cv2.rectangle(frame, (x0, y0), (x0 + side, y0 + side), colour, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x0, max(y0 - 10, 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, colour, 2)
            for i, (name, p) in enumerate(zip(class_names, avg.tolist() if avg is not None else [0.0] * 3)):
                y = 30 + i * 25
                cv2.putText(frame, f"{name:>8}: {p:.2f}", (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.rectangle(frame, (150, y - 12), (150 + int(p * 150), y), colour, -1)

            bg_text = f"bg {bg_level.mean():.0f}/255" if bg_level is not None else "bg -"
            cv2.putText(frame, f"whiten {'ON' if use_whiten else 'OFF'} (w)  {bg_text}  box {side}px ([ ])  hand {hand_fraction:.2f}",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Preview of exactly what the model sees (tensor -> uint8 BGR).
            seen = (x[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            seen = cv2.cvtColor(seen, cv2.COLOR_RGB2BGR)
            ph, pw = image_size[0] * PREVIEW_SCALE, image_size[1] * PREVIEW_SCALE
            seen = cv2.resize(seen, (pw, ph), interpolation=cv2.INTER_NEAREST)
            frame[h - ph - 10:h - 10, w - pw - 10:w - 10] = seen

            cv2.imshow("RPS live", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key in CAPTURE_KEYS:
                out = CAPTURE_DIR / CAPTURE_KEYS[key]
                out.mkdir(parents=True, exist_ok=True)
                path = out / f"{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}.png"
                cv2.imwrite(str(path), raw_crop_bgr)
                print(f"saved {path.relative_to(Path(__file__).parent)}  (model said: {label})")
            elif key == ord("w"):
                use_whiten = not use_whiten
            elif key == ord("["):
                roi_size = max(128, roi_size - 20)
            elif key == ord("]"):
                roi_size = min(min(h, w), roi_size + 20)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
