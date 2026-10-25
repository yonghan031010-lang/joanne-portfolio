"""E: night street → red billboard → RUNWAY → stage, 8 beats, film grade.

  beat 0     black, a horizontal slit opens onto IMG_4196 (reference opener)
  beats 0-2  night street, white car drives right
  beat 2     whip pan right into IMG_4817
  beats 2-4  push into the red stiletto on the Devil Wears Prada billboard until red fills frame
  beat 4     red → red: out of the red RUNWAY letters (IMG_4823), pulling back
  beats 5-6  push into the glowing purple doors
  beat 6     purple → purple: into the stage (IMG_4834, landscape, centre-cropped)
  beats 6-8  stage settles

usage: python3 night.py <source_dir> <out.mp4>
"""
import sys, subprocess, math
import numpy as np, cv2
from vcam import OW, OH, FPS, BEAT, c01, lerp, ease_in, ease_out
from vcam2 import Clip
from dome import view
from golden import film

SHOE = (0.30, 0.33)          # red stiletto on the billboard (IMG_4817)
LETTER = (0.27, 0.33)        # red "U" of RUNWAY (IMG_4823)
DOORS = (0.50, 0.80)         # lit doors under RUNWAY
STAGE = (0.50, 0.55)         # stage centre (IMG_4834, landscape)
YY = np.arange(OH, dtype=np.float32)[:, None, None]


def smooth(p):
    p = c01(p); return p * p * (3 - 2 * p)


def main(src, out):
    S1 = Clip(f"{src}/IMG_4196.MOV", 0.6, 1.9)
    S2 = Clip(f"{src}/IMG_4817.MOV", 0.3, 1.9)
    S3 = Clip(f"{src}/IMG_4823.MOV", 0.1, 1.7)
    S4 = Clip(f"{src}/IMG_4834.MOV", 0.8, 2.4)
    Z4 = (OH / S4.h) * S4.w / OW                    # zoom that fills the 9:16 frame with the landscape stage
    BT = [k * BEAT for k in range(9)]
    n = round(BT[8] * FPS)
    WH = 0.16                                       # whip half-length (s)

    def frame(t):
        if t < BT[2]:
            s = 0.8 + t
            z = lerp(1.0, 1.08, t / BT[2])
            whip = ease_in(c01((t - (BT[2] - WH)) / WH))
            return view(S1.at(s), 0.5, 0.5, z, 0.5 - 0.9 * whip, 0.5, 0.0)
        if t < BT[4]:
            s = 0.5 + (t - BT[2])
            whip = 1 - ease_out(c01((t - BT[2]) / WH))
            p = ease_in(c01((t - BT[2]) / (BT[4] - BT[2])))
            z = lerp(1.0, 3.2, p)
            cx, cy = lerp(0.5, SHOE[0], p), lerp(0.5, SHOE[1], p)
            return view(S2.at(s), cx, cy, z, 0.5 + 0.9 * whip, 0.5, 0.0)
        if t < BT[6]:
            s = 0.2 + (t - BT[4])
            out_p = ease_out(c01((t - BT[4]) / BEAT))
            in_p = ease_in(c01((t - BT[5]) / BEAT))
            z = lerp(3.0, 1.0, out_p) + (2.8 - 1.0) * in_p
            cx = lerp(LETTER[0], 0.5, out_p); cy = lerp(LETTER[1], 0.5, out_p)
            cx = lerp(cx, DOORS[0], in_p); cy = lerp(cy, DOORS[1], in_p)
            return view(S3.at(s), cx, cy, z, 0.5, 0.5, 0.0)
        s = 1.0 + (t - BT[6])
        p = ease_out(c01((t - BT[6]) / BEAT))
        z = Z4 * (lerp(1.7, 1.0, p) + 0.04 * c01((t - BT[7]) / BEAT))
        return view(S4.at(s), STAGE[0], STAGE[1], z, 0.5, 0.5, 0.0)

    cuts = [BT[2], BT[4], BT[6]]
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17",
                            "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    for i in range(n):
        t = i / FPS
        d = min(abs(t - c) for c in cuts)
        near = d < 0.3
        K = 12 if near else 3
        sh = 0.5 + 1.4 * math.exp(-(d / 0.15) ** 2)
        acc = np.zeros((OH, OW, 3), np.float32)
        for j in range(K):
            tj = t + (j / max(K - 1, 1) - 0.5) * sh / FPS
            c = min(cuts, key=lambda c: abs(tj - c))
            w = smooth((tj - (c - 0.05)) / 0.10)                    # short hand-off inside the blur
            if 0 < w < 1:
                img = frame(min(tj, c - 1e-3)) * (1 - w) + frame(max(tj, c)) * w
            else:
                img = frame(tj)
            acc += img
        img = acc / K
        if t < BT[1]:                                              # horizontal slit opening from black
            half = ease_out(c01(t / BT[1])) * OH / 2
            m = np.clip((half - np.abs(YY - OH / 2)) / 3 + 0.5, 0, 1)
            img = img * m
        if abs(t - BT[2]) < WH * 1.5:                              # streak of the whip
            bw = int(120 * math.exp(-((t - BT[2]) / 0.08) ** 2))
            if bw > 3:
                img = cv2.blur(img, (bw, 1))
        enc.stdin.write(film(img).tobytes())
    enc.stdin.close(); enc.wait()
    print("frames", n, "stage zoom base %.2f" % Z4)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
