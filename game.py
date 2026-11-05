"""Best-of-3 rock-paper-scissors against the computer, using the webcam classifier.

    uv run game.py [model_dir]      (default: rps-classifier-finetuned/)

Each round: 3, 2, 1, SHOOT! Your hand in the box is read during SHOOT, and only
then is the computer's choice revealed (and spoken), so it can't be reacted to.
Ties and unreadable hands replay the round. First to 2 round wins takes the match.

Keys: space = start round / new match, q / Esc = quit.
"""

import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import torch
from torchvision.transforms import v2

from main import (CONF_THRESHOLD, DARK_LEVEL, EMA_KEEP, MIN_HAND_FRACTION, ROI_SIZE,
                  load_model, roi_box, whiten)

DEFAULT_MODEL = Path(__file__).parent / "rps-classifier-finetuned"
WINS_NEEDED = 2           # best of 3
COUNT_SECONDS = 1.0       # per number in the countdown
SHOOT_SECONDS = 0.5       # window in which your hand is read
RESULT_SECONDS = 2.5      # how long the round result stays up before the next prompt
BEATS = {"rock": "scissors", "scissors": "paper", "paper": "rock"}

WHITE, GREEN, RED, ORANGE, GREY = (255, 255, 255), (0, 200, 0), (0, 0, 230), (0, 165, 255), (160, 160, 160)


class Voice:
    """Speak with macOS `say`, without blocking the video loop."""

    def __init__(self):
        self.available = shutil.which("say") is not None
        self.proc = None

    def say(self, text):
        if not self.available:
            return
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
        self.proc = subprocess.Popen(["say", "-r", "220", text])


def text(frame, s, org, scale=1.0, colour=WHITE, thickness=2, center=False):
    """Text with a black outline so it reads on any background."""
    font = cv2.FONT_HERSHEY_DUPLEX
    if center:
        (tw, th), _ = cv2.getTextSize(s, font, scale, thickness)
        org = (org[0] - tw // 2, org[1] + th // 2)
    cv2.putText(frame, s, org, font, scale, (0, 0, 0), thickness + max(2, int(2 * scale)), cv2.LINE_AA)
    cv2.putText(frame, s, org, font, scale, colour, thickness, cv2.LINE_AA)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MODEL
    model, class_names, image_size = load_model(device, model_dir)
    transform = v2.Compose([v2.ToImage(), v2.Resize(image_size), v2.ToDtype(torch.float32, scale=True)])
    print(f"Loaded {model_dir} on {device}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam (check camera permission for your terminal).")

    voice = Voice()
    state, t0 = "ready", time.monotonic()
    score = {"you": 0, "cpu": 0}
    round_no = 1
    avg = None
    last_count = None
    cpu_choice = player = outcome = None
    shoot_probs = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            now = time.monotonic()
            elapsed = now - t0

            # --- classify the box (same pipeline as main.py) ---
            x0, y0, side = roi_box(h, w, ROI_SIZE)
            crop = cv2.cvtColor(frame[y0:y0 + side, x0:x0 + side], cv2.COLOR_BGR2RGB)
            crop, _ = whiten(crop)
            has_hand = (crop.min(axis=-1) < DARK_LEVEL).mean() >= MIN_HAND_FRACTION
            probs = None
            if has_hand:
                with torch.inference_mode():
                    probs = torch.softmax(model(transform(crop).unsqueeze(0).to(device)), 1)[0].cpu()
                avg = probs if avg is None else EMA_KEEP * avg + (1 - EMA_KEEP) * probs
            else:
                avg = None
            if avg is not None and avg.max() >= CONF_THRESHOLD:
                live_label = class_names[avg.argmax()]
            else:
                live_label = "?" if has_hand else "no hand"

            # --- game state machine ---
            if state == "countdown":
                n = 3 - int(elapsed // COUNT_SECONDS)
                if n != last_count and n >= 1:
                    voice.say(str(n))
                    last_count = n
                if n < 1:
                    state, t0, shoot_probs = "shoot", now, []
                    voice.say("shoot")
            elif state == "shoot":
                if probs is not None:
                    shoot_probs.append(probs)
                if elapsed >= SHOOT_SECONDS:
                    mean = torch.stack(shoot_probs).mean(0) if shoot_probs else None
                    if mean is None or mean.max() < CONF_THRESHOLD:
                        player, outcome = None, "unread"
                        voice.say(f"I chose {cpu_choice}. Couldn't read your hand, go again")
                    else:
                        player = class_names[mean.argmax()]
                        if player == cpu_choice:
                            outcome = "tie"
                        elif BEATS[player] == cpu_choice:
                            outcome = "win"
                            score["you"] += 1
                        else:
                            outcome = "lose"
                            score["cpu"] += 1
                        line = {"tie": "Tie", "win": "You win the round", "lose": "I win the round"}[outcome]
                        voice.say(f"{cpu_choice}. {line}")
                    state, t0 = "result", now
            elif state == "result" and elapsed >= RESULT_SECONDS:
                if max(score.values()) >= WINS_NEEDED:
                    state = "game_over"
                    voice.say("You win the match!" if score["you"] > score["cpu"] else "I win the match!")
                else:
                    if outcome == "win" or outcome == "lose":
                        round_no += 1
                    state = "ready"

            # --- drawing ---
            box_colour = {"?": ORANGE, "no hand": GREY}.get(live_label, GREEN)
            cv2.rectangle(frame, (x0, y0), (x0 + side, y0 + side), box_colour, 2)
            text(frame, live_label, (x0 + 5, y0 + side - 10), 0.8, box_colour)
            text(frame, f"YOU {score['you']}  -  {score['cpu']} CPU", (w // 2, 35), 1.2, WHITE, 2, center=True)
            text(frame, f"Round {round_no}  (first to {WINS_NEEDED})", (w // 2, 75), 0.7, WHITE, 1, center=True)

            mid = (w // 2, y0 + side // 2)
            if state == "ready":
                text(frame, "Press SPACE to play", (w // 2, y0 + side + 45), 1.2, WHITE, 2, center=True)
            elif state == "countdown":
                text(frame, str(max(1, 3 - int(elapsed // COUNT_SECONDS))), mid, 5, WHITE, 8, center=True)
            elif state == "shoot":
                text(frame, "SHOOT!", mid, 3, ORANGE, 6, center=True)
            elif state == "result":
                text(frame, f"CPU: {cpu_choice.upper()}", (w // 2, y0 - 50), 1.5, WHITE, 3, center=True)
                if outcome == "unread":
                    text(frame, "Couldn't read your hand - replay", mid, 1.1, ORANGE, 2, center=True)
                else:
                    text(frame, f"You: {player.upper()}", (w // 2, y0 + side + 45), 1.2, WHITE, 2, center=True)
                    msg, col = {"win": ("You win the round!", GREEN), "lose": ("CPU wins the round", RED),
                                "tie": ("Tie - replay", ORANGE)}[outcome]
                    text(frame, msg, mid, 1.5, col, 3, center=True)
            elif state == "game_over":
                won = score["you"] > score["cpu"]
                text(frame, "YOU WIN!" if won else "CPU WINS", mid, 3, GREEN if won else RED, 6, center=True)
                text(frame, "SPACE = new match   q = quit", (w // 2, y0 + side + 45), 0.9, WHITE, 2, center=True)

            cv2.imshow("Rock Paper Scissors", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                if state == "game_over":
                    score = {"you": 0, "cpu": 0}
                    round_no, state = 1, "ready"
                elif state == "ready":
                    cpu_choice = random.choice(class_names)
                    state, t0, last_count = "countdown", now, None
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
