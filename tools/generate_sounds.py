"""Generate all 8-bit audio for Blocks With Guns RL (no external assets).

Run:  python tools/generate_sounds.py
Writes .wav files into assets/sfx/ (22050 Hz, 16-bit mono).

Themes are simple square-wave leads over triangle-wave bass loops;
effects are short synthesized blips. Nothing sophisticated on purpose.
"""

from __future__ import annotations

import math
import os
import struct
import wave

import numpy as np

SR = 22050
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "sfx")


def note(freq: float, seconds: float) -> int:
    n = int(SR * seconds)
    return n


def square(freq: float, t: np.ndarray, duty: float = 0.5) -> np.ndarray:
    return np.where((t * freq) % 1.0 < duty, 1.0, -1.0)


def triangle(freq: float, t: np.ndarray) -> np.ndarray:
    return 2.0 * np.abs(2.0 * ((t * freq) % 1.0) - 1.0) - 1.0


def noise(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-1, 1, n)


def envelope(n: int, attack: float = 0.01, decay: float = 0.15, sustain: float = 0.7,
             release: float = 0.1) -> np.ndarray:
    a = max(1, int(n * attack))
    d = max(1, int(n * decay))
    r = max(1, int(n * release))
    s = max(0, n - a - d - r)
    env = np.concatenate([
        np.linspace(0, 1, a),
        np.linspace(1, sustain, d),
        np.full(s, sustain),
        np.linspace(sustain, 0, r),
    ])
    return env[:n] if len(env) >= n else np.pad(env, (0, n - len(env)))


def seq(notes: list[tuple[float, float]], wave_fn=square, vol: float = 0.5,
        duty: float = 0.5, gap: float = 0.02) -> np.ndarray:
    """notes: list of (freq, seconds). 0 freq = rest."""
    out = []
    for f, dur in notes:
        n = int(SR * dur)
        if f <= 0:
            out.append(np.zeros(n))
            continue
        t = np.arange(n) / SR
        w = wave_fn(f, t, duty) if wave_fn is square else wave_fn(f, t)
        out.append(w * envelope(n) * vol)
        gn = int(SR * gap)
        if gn:
            out.append(np.zeros(gn))
    return np.concatenate(out) if out else np.zeros(1)


def loop_to(sig: np.ndarray, seconds: float) -> np.ndarray:
    n = int(SR * seconds)
    if len(sig) >= n:
        return sig[:n]
    reps = int(np.ceil(n / len(sig)))
    return np.tile(sig, reps)[:n]


def save(name: str, sig: np.ndarray) -> str:
    os.makedirs(OUT, exist_ok=True)
    sig = np.clip(sig, -1.0, 1.0)
    path = os.path.join(OUT, name)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(b"".join(struct.pack("<h", int(v * 32767)) for v in sig))
    return path


# ---------------------------------------------------------------------- #
A2, C3, D3, E3, F3, G3, A3, B3 = 110.0, 130.81, 146.83, 164.81, 174.61, 196.0, 220.0, 246.94
C4, D4, E4, F4, G4, A4, B4 = 261.63, 293.66, 329.63, 349.23, 392.0, 440.0, 493.88
C5, D5, E5, F5, G5, A5, B5 = 523.25, 587.33, 659.25, 698.46, 783.99, 880.0, 987.77


def menu_theme() -> np.ndarray:
    lead = [
        (E4, .25), (G4, .25), (A4, .5), (G4, .25), (E4, .25), (C4, .5),
        (D4, .25), (E4, .25), (G4, .5), (E4, .25), (D4, .25), (C4, .5),
        (E4, .25), (G4, .25), (A4, .5), (C5, .25), (B4, .25), (A4, .5),
        (G4, .25), (E4, .25), (D4, .5), (C4, .25), (D4, .25), (C4, .5),
    ]
    bass = [
        (C3, .5), (G2 := 98.0, .5), (A2, .5), (E3, .5),
        (F3, .5), (C3, .5), (G3, .5), (G2, .5),
        (C3, .5), (G2, .5), (A2, .5), (E3, .5),
        (F3, .5), (G3, .5), (C3, .5), (C3, .5),
    ]
    lead_sig = seq(lead, square, vol=0.30, duty=0.5)
    bass_sig = loop_to(seq(bass, triangle, vol=0.35), len(lead_sig) / SR)
    return lead_sig + bass_sig


def battle_theme() -> np.ndarray:
    b = 0.16  # driving tempo
    lead = [
        (A4, b), (A4, b), (C5, b), (A4, b), (G4, b), (E4, b), (G4, b * 2),
        (A4, b), (A4, b), (C5, b), (D5, b), (C5, b), (A4, b), (G4, b * 2),
        (E4, b), (E4, b), (G4, b), (A4, b), (G4, b), (E4, b), (D4, b * 2),
        (E4, b), (G4, b), (A4, b), (C5, b), (B4, b), (A4, b), (G4, b * 2),
    ]
    bass_line = [(A2, b), (0, b)] * 8 + [(F3, b), (0, b)] * 4 + \
                [(G3, b), (0, b)] * 4 + [(A2, b), (0, b)] * 8 + \
                [(F3, b), (0, b)] * 4 + [(E3, b), (0, b)] * 4
    lead_sig = seq(lead, square, vol=0.26, duty=0.3, gap=0.01)
    bass_sig = loop_to(seq(bass_line, triangle, vol=0.38, gap=0.0), len(lead_sig) / SR)
    # simple hat: short noise ticks
    n = len(lead_sig)
    hats = np.zeros(n)
    tick = int(SR * b)
    for i in range(0, n, tick):
        m = min(int(SR * 0.02), n - i)
        hats[i:i + m] = np.random.default_rng(i).uniform(-1, 1, m) * 0.08
    return lead_sig + bass_sig + hats


def shot() -> np.ndarray:
    rng = np.random.default_rng(5)
    n = int(SR * 0.12)
    t = np.arange(n) / SR
    body = noise(n, rng) * np.exp(-t * 40)
    tone = square(180, t) * np.exp(-t * 55)
    return body * 0.5 + tone * 0.3


def hit() -> np.ndarray:
    n = int(SR * 0.15)
    t = np.arange(n) / SR
    return (triangle(90, t) * np.exp(-t * 30) * 0.7 +
            square(60, t) * np.exp(-t * 45) * 0.4)


def powerup() -> np.ndarray:
    return seq([(C5, .06), (E5, .06), (G5, .06), (C5 * 2, .12)], square, vol=0.4, gap=0.01)


def quickshot() -> np.ndarray:
    return seq([(G5, .04), (G5, .04), (G5, .04), (B5, .1)], square, vol=0.4, duty=0.25)


def swap() -> np.ndarray:
    n = int(SR * 0.28)
    t = np.arange(n) / SR
    sweep = np.linspace(800, 200, n)
    sig = np.sign(np.sin(2 * np.pi * np.cumsum(sweep) / SR))
    return sig * envelope(n) * 0.4


def victory() -> np.ndarray:
    return seq([(C4, .15), (E4, .15), (G4, .15), (C5, .3), (G4, .15), (C5, .5)],
               square, vol=0.4)


def defeat() -> np.ndarray:
    return seq([(E4, .25), (D4, .25), (C4, .25), (A3 * 0.5, .7)], triangle, vol=0.5)


def ui_hover() -> np.ndarray:
    return seq([(A5, .04)], square, vol=0.2, duty=0.25)


def heart_pickup() -> np.ndarray:
    return seq([(E5, .07), (G5, .07), (E5 * 2, .14)], triangle, vol=0.45)


ALL = {
    "menu_theme.wav": menu_theme,
    "battle_theme.wav": battle_theme,
    "shot.wav": shot,
    "hit.wav": hit,
    "powerup.wav": powerup,
    "quickshot.wav": quickshot,
    "swap.wav": swap,
    "victory.wav": victory,
    "defeat.wav": defeat,
    "ui_hover.wav": ui_hover,
    "heart_pickup.wav": heart_pickup,
}


def main() -> None:
    for name, fn in ALL.items():
        path = save(name, fn())
        print("wrote", path)


if __name__ == "__main__":
    main()
