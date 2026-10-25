"""C: Oxford arch → Great Hall → rotating tilt-up (reference move), 8 beats, film grade.

  beats 0-2  IMG_4499 slow push into the stone arch
  beats 2-3  push accelerates through the opening (zoom + motion blur)
  beat 3     cut inside the blur to IMG_4500 walking through the crowd into the hall
  beats 3-5  camera settles as the hall opens up
  beats 5-8  tilt up with a slight roll onto the hammerbeam ceiling and the great window

usage: python3 hall.py <source_dir> <out.mp4>
"""
import sys, subprocess, math
import numpy as np, cv2
from vcam import OW, OH, FPS, BEAT, c01, lerp, ease_in, ease_out
from vcam2 import Clip
from dome import view
from golden import film

ARCH = (0.50, 0.40)          # centre of the arch opening (IMG_4499)
HALL_UP = (0.50, 0.24)       # ceiling / window target (IMG_4500)


def smooth(p):
    p = c01(p); return p * p * (3 - 2 * p)


def main(src, out):
    A = Clip(f"{src}/IMG_4499.MOV", 0.3, 2.4)
    B = Clip(f"{src}/IMG_4500.MOV", 0.6, 5.6)
    BT = [k * BEAT for k in range(9)]
    n = round(BT[8] * FPS)
    vB = 1.4

    def frame(t):
        if t < BT[3]:
            s = 0.3 + t
            p_slow = smooth(t / BT[2])
            p_fast = ease_in(c01((t - BT[2]) / BEAT))
            z = lerp(1.0, 1.18, p_slow) + (2.6 - 1.18) * p_fast
            return view(A.at(s), ARCH[0], ARCH[1], z, 0.5, ARCH[1], 0.0)
        s = 0.6 + (t - BT[3]) * vB
        settle = ease_out(c01((t - BT[3]) / (BT[5] - BT[3])))
        up = smooth((t - BT[5]) / (BT[8] - BT[5]))
        z = lerp(1.7, 1.18, settle) + 0.22 * up                  # arrive pushed-in, relax, then lean in while tilting
        cy = lerp(0.50, HALL_UP[1], up)
        roll = -9.0 * up
        return view(B.at(s), 0.5, cy, z, 0.5, lerp(0.5, 0.42, up), roll)

    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17",
                            "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    for i in range(n):
        t = i / FPS
        near = abs(t - BT[3]) < 0.35
        K = 12 if near else 3
        sh = 0.5 + 1.4 * math.exp(-((t - BT[3]) / 0.18) ** 2)
        acc = np.zeros((OH, OW, 3), np.float32)
        for j in range(K):
            tj = t + (j / max(K - 1, 1) - 0.5) * sh / FPS
            w = smooth((tj - (BT[3] - 0.05)) / 0.10)             # ~3-frame hand-off inside the blur
            if w <= 0:
                img = frame(min(tj, BT[3] - 1e-3))
            elif w >= 1:
                img = frame(max(tj, BT[3]))
            else:
                img = frame(min(tj, BT[3] - 1e-3)) * (1 - w) + frame(max(tj, BT[3])) * w
            acc += img
        enc.stdin.write(film(acc / K).tobytes())
    enc.stdin.close(); enc.wait()
    print("frames", n, "IMG_4500 to %.2fs" % (0.6 + (BT[8] - BT[3]) * vB))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
