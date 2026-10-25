"""Tube arrival (IMG_4659) → King's Cross platform (IMG_4242) with a chromatic silhouette
(reference: motion pulse "Chromatic Silhouette"): no cross-scene compositing — the arriving train
flashes as a white silhouette with a purple/cyan fringe on the beats, then a hard cut on the beat.

  beats 0-3  train pulls in, slow push
  beat 3     2-frame silhouette pulse
  beat 4     3-frame silhouette, hard cut to IMG_4242 with a short RGB split
  beats 4-7  platform

usage: python3 train_chroma.py <source_dir> <out.mp4>
"""
import sys, subprocess
import numpy as np, cv2
from vcam import OW, OH, FPS, BEAT, c01, lerp, ease_out
from vcam2 import Clip
from dome import view
from golden import film
from train_arrival import train_masks, BODY, A0, A1

PURPLE = np.float32([190, 70, 255])
CYAN = np.float32([60, 230, 255])


def silhouette(img, m, k):
    """White fill + chromatic fringe (purple outside-left, cyan outside-right); k = strength 0..1."""
    m = cv2.GaussianBlur(m, (0, 0), 2.0)
    sh = 34
    ml = np.zeros_like(m); ml[:, :-sh] = m[:, sh:]               # shifted copies, no wrap-around
    mr = np.zeros_like(m); mr[:, sh:] = m[:, :-sh]
    halo = cv2.GaussianBlur(m, (0, 0), 10)
    fp = np.clip(ml - m, 0, 1)[..., None]; fc = np.clip(mr - m, 0, 1)[..., None]
    glow = np.clip(halo - m, 0, 1)[..., None]
    bloom = np.clip(cv2.GaussianBlur(m, (0, 0), 45) * 1.5 - m, 0, 1)[..., None]   # wide soft light spilling past the edge
    mm = m[..., None]
    out = img * (1 - 0.35 * k) * (1 - k * mm) + 250 * k * mm         # dim the world so the light pops
    out = out * (1 - k * fp) + PURPLE * k * fp
    out = out * (1 - k * fc) + CYAN * k * fc
    return out + glow * k * PURPLE * 1.8 + bloom * k * (PURPLE * 0.7 + CYAN * 0.3) * 1.1


def rgb_split(img, d):
    if d < 1:
        return img
    o = img.copy()
    o[..., 0] = np.roll(img[..., 0], d, 1); o[..., 2] = np.roll(img[..., 2], -d, 1)
    return o


def main(src, out):
    A = Clip(f"{src}/IMG_4659.MOV", A0, A1 - A0); A.path = f"{src}/IMG_4659.MOV"
    B = Clip(f"{src}/IMG_4242.MOV", 0.0, 4.0)
    masks, _ = train_masks(A)
    BT = [k * BEAT for k in range(8)]
    n = round(BT[7] * FPS)
    cut = round(BT[4] * FPS)
    p3 = round(BT[3] * FPS)
    sA = lambda t: A0 + 0.1 + t
    for_idx = lambda t: min(max(round((sA(t) - A0) * FPS), 0), len(masks) - 1)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17",
                            "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    for i in range(n):
        t = i / FPS
        if i < cut:
            z = lerp(1.0, 1.12, t / BT[4])
            img = view(A.at(sA(t)), 0.5, 0.5, z, 0.5, 0.5, 0.0)
            k = {p3: 1.0, p3 + 1: 0.55, cut - 3: 0.8, cut - 2: 1.0, cut - 1: 1.0}.get(i, 0.0)
            if k:
                m = view(masks[for_idx(t)][..., None].repeat(3, 2), 0.5, 0.5, z, 0.5, 0.5, 0.0)[..., :1] * BODY
                img = silhouette(img, m[..., 0], k)
        else:
            j = i - cut
            p = ease_out(c01((t - BT[4]) / (BT[7] - BT[4])))
            img = view(B.at(0.4 + (t - BT[4])), 0.5, 0.5, lerp(1.16, 1.06, p), 0.5, 0.5, 0.0)
            img = rgb_split(img, [16, 9, 4][j] if j < 3 else 0)
        enc.stdin.write(film(img).tobytes())
    enc.stdin.close(); enc.wait()
    print("frames", n, "cut", cut)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
