"""Bus → tube seamless match: the side of the double-decker (IMG_4221) and the tube carriages
(IMG_4243) both slide right across a full frame at ~70 px/frame. One shared speed curve keeps
their on-screen velocity identical; a ~5-frame hand-off with a short whip and horizontal blur
streaks (same direction as both vehicles) turns the bus into the train. The tube then runs on,
she walks in, and the train's blur peak lands on the closing beat.

usage: python3 bus_train.py <source_dir> <out.mp4>
"""
import sys, subprocess, math
import numpy as np, cv2
from vcam import OW, OH, FPS, BEAT, grade, c01, lerp
from vcam2 import Clip, remap, punch, post

PX_PER_FRAME = 70.0         # measured screen speed of both vehicles at 1x (1080-wide frame)
A_MATCH, B_MATCH = 2.45, 0.47   # source times where bus and train both move at ~70 px/frame
B_END = 3.70                # train's motion-blur peak


def smooth(p):
    p = c01(p); return p * p * (3 - 2 * p)


def main(src, out):
    A = Clip(f"{src}/IMG_4221.MOV", 1.2, 1.8)
    B = Clip(f"{src}/IMG_4243.MOV", 0.0, 4.4)
    TM, END = BEAT, 4 * BEAT
    n = round(END * FPS)
    # one speed curve for both clips: ~1.3x through the hand-off, rising to ~1.7x at the end
    b, c = 1.3, 1.7
    for _ in range(30):                                         # solve the end speed so the blur peak hits END
        base = lambda t, c=c: b if t < TM else lerp(b, c, smooth((t - TM) / (END - TM)))
        v = lambda t, base=base: base(t) * (1 + 1.0 * math.exp(-((t - TM) / 0.09) ** 2))   # short whip at the hand-off
        s0 = remap(v, n)
        err = (s0(END) - s0(TM) + B_MATCH) - B_END
        c -= err / ((END - TM) / 2)
    
    sA = lambda t: s0(t) - s0(TM) + A_MATCH
    sB = lambda t: s0(t) - s0(TM) + B_MATCH
    print("bus from %.2fs, speed %.2f→%.2fx, hand-off at %.2fs, train ends %.2fs" % (sA(0), b, c, TM, sB(END)))

    # exposure of the tube pulled toward the sunny street at the hand-off, released after
    gA = A.at(A_MATCH).reshape(-1, 3).mean(0); gB = B.at(B_MATCH).reshape(-1, 3).mean(0)
    gain = (gA / gB).astype(np.float32)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17",
                            "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    for i in range(n):
        t = i / FPS
        K = 10 if abs(t - TM) < 0.3 else 6
        sh = 0.6 + 0.6 * math.exp(-((t - TM) / 0.2) ** 2)
        acc = np.zeros((OH, OW, 3), np.float32)
        for j in range(K):
            tj = t + (j / (K - 1) - 0.5) * sh / FPS
            w = smooth((tj - TM) / 0.17 + 0.5)                      # bus → train over ~5 frames
            g = 1 + (gain - 1) * math.exp(-((tj - TM) / 0.35) ** 2)
            if w <= 0:
                img = A.at(sA(tj))
            elif w >= 1:
                img = B.at(sB(tj)) * g
            else:
                img = A.at(sA(tj)) * (1 - w) + B.at(sB(tj)) * g * w
            z, d = punch(tj, END, amp=0.035, w=0.10, shake_px=10)
            acc += post(img, z, d)
        img = acc / K
        bw = int(150 * math.exp(-((t - TM) / 0.08) ** 2))          # horizontal streak, same direction as both vehicles
        if bw > 3:
            img = cv2.blur(img, (bw, 1))
        enc.stdin.write(grade(img).tobytes())
    enc.stdin.close(); enc.wait()
    print("frames", n)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
