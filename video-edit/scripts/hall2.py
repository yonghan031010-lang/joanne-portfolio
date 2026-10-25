"""C v2: the Oxford arch (IMG_4499) becomes a window onto the Great Hall (IMG_4500), 8 beats.

  beats 0-2  IMG_4499 slow push, the courtyard through the arch
  beat 2     bounce-in: a window in the shape of the arch springs open from its centre (soft edge, no rim), the Great Hall inside it; people standing in front of the arch stay on top
  beats 2-5  push through the arch; the hall, being "far", grows slower than the stone (parallax)
  beat 5     the arch leaves frame — we are in the hall
  beats 5-8  tilt up with a slight roll onto the hammerbeam ceiling and the great window

usage: python3 hall2.py <source_dir> <out.mp4>
"""
import sys, subprocess, math
import numpy as np, cv2, mediapipe as mp
from vcam import OW, OH, FPS, BEAT, c01, lerp, ease_in, ease_out
from vcam2 import Clip
from dome import view, segmenter
from golden import film

# inner edge of the arch opening in IMG_4499 (frame fractions), measured at 1.0s
ARCH = [(0.50, 0.146), (0.42, 0.155), (0.33, 0.178), (0.25, 0.208), (0.20, 0.245), (0.183, 0.29),
        (0.188, 0.45), (0.205, 0.60), (0.218, 0.682), (0.825, 0.682), (0.832, 0.55), (0.818, 0.30),
        (0.80, 0.25), (0.75, 0.208), (0.67, 0.176), (0.58, 0.152)]
NOT_FRONT = (0.27, 0.43, 0.585)   # backpack man standing on the threshold: x range, above this row he is not "front"
THRESH = 0.74                # only people clearly on our side of the threshold stay on top
CAM = (0.50, 0.42)           # we push toward the middle of the opening


def smooth(p):
    p = c01(p); return p * p * (3 - 2 * p)


def spring(p):
    p = c01(p); return 1 - math.exp(-6 * p) * math.cos(9 * p)


def arch_mask(w, h):
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.int32([(x * w, y * h) for x, y in ARCH])], 1)
    return cv2.GaussianBlur(m.astype(np.float32), (0, 0), 2.5)


def person_prob(frame, seg):
    r = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame)))
    return (np.squeeze(r.category_mask.numpy_view()) > 0).astype(np.float32)


def front_people(probs):
    """People clearly on our side of the threshold, temporally steadied, with clean edges."""
    out = []
    for i in range(len(probs)):
        m = (np.mean(probs[max(i - 1, 0):i + 2], 0) > 0.5).astype(np.uint8)    # 3-frame vote: no flicker
        h = m.shape[0]
        core = cv2.erode(m, np.ones((9, 9), np.uint8))                          # split people who touch
        n, lab, st, _ = cv2.connectedComponentsWithStats(core)
        keep = np.zeros_like(m)
        for k in range(1, n):
            bottom = st[k, cv2.CC_STAT_TOP] + st[k, cv2.CC_STAT_HEIGHT]
            if bottom > THRESH * h and st[k, cv2.CC_STAT_AREA] > 600:
                keep[lab == k] = 1
        keep = cv2.dilate(keep, np.ones((9, 9), np.uint8)) & m                  # back to the true outline
        keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        w = keep.shape[1]
        keep[:int(NOT_FRONT[2] * h), int(NOT_FRONT[0] * w):int(NOT_FRONT[1] * w)] = 0   # his feet touch her head
        out.append(cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 1.2))
    return out


def main(src, out):
    A = Clip(f"{src}/IMG_4499.MOV", 0.3, 2.6)
    B = Clip(f"{src}/IMG_4500.MOV", 2.0, 4.7)
    seg = segmenter()
    front = front_people([person_prob(f, seg) for f in A.f])
    am = arch_mask(A.w, A.h)
    BT = [k * BEAT for k in range(9)]
    n = round(BT[8] * FPS)
    sA = lambda t: 0.3 + t * 0.8
    iA = lambda t: min(max(round((sA(t) - A.ss) * FPS), 0), len(A.f) - 1)
    sB = lambda t: 2.0 + max(t - BT[2], 0)
    ZA_END = 3.0                                                   # arch fully outside frame by beat 5

    def zA(t):
        slow = lerp(1.0, 1.10, smooth(t / BT[2]))
        thru = ease_in(c01((t - BT[2]) / (BT[5] - BT[2])))
        return slow * lerp(1.0, ZA_END / 1.10, thru)

    AC = (0.51, 0.45)                                              # centre of the arch opening (source)

    def window(t):
        """Arch-shaped window springing open from its centre (overshoots, clipped by the real arch)."""
        k = spring((t - (BT[2] - 0.05)) / 0.55)
        sc = max(lerp(0.0, 1.0, k), 0.0)
        M = cv2.getRotationMatrix2D((AC[0] * A.w, AC[1] * A.h), 0, max(sc, 1e-3))
        return np.minimum(cv2.warpAffine(am, M, (A.w, A.h)), am * 1.0), cv2.warpAffine(am, M, (A.w, A.h))

    def hall_cam(t):
        """Hall framing inside the window → continues after the arch is gone."""
        k = spring((t - BT[2]) / 0.6)
        z = lerp(1.15, 1.0, k) * lerp(1.0, 1.12, smooth((t - BT[2]) / (BT[5] - BT[2])))
        up = smooth((t - BT[5]) / (BT[8] - BT[5]))
        z = z + 0.18 * up
        return view(B.at(sB(t)), 0.5, lerp(0.42, 0.24, up), z, 0.5, lerp(0.45, 0.40, up), -8.0 * up)

    def frame(t):
        if t >= BT[5]:
            return hall_cam(t)
        z = zA(t)
        a = view(A.at(sA(t)), CAM[0], CAM[1], z, 0.5, CAM[1], 0.0)
        if t < BT[2] - 0.05:
            return a
        wm, raw = window(t)
        V = lambda m: view(m[..., None].repeat(3, 2), CAM[0], CAM[1], z, 0.5, CAM[1], 0.0)[..., :1]
        # while it grows the window edge is feathered wide so it reads as light, not a cut-out
        grow = 1 - smooth((t - (BT[2] + 0.15)) / 0.40)
        if grow > 0.01:
            wm = np.minimum(cv2.GaussianBlur(wm, (0, 0), 1 + 18 * grow), am)
        win = V(wm)
        fp = V(front[iA(t)])
        w = win * (1 - fp)
        return a * (1 - w) + hall_cam(t) * w

    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17",
                            "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    for i in range(n):
        t = i / FPS
        thru = BT[4] < t < BT[5] + 0.2
        K = 8 if thru else 2
        sh = 1.2 if thru else 0.5
        acc = np.zeros((OH, OW, 3), np.float32)
        for j in range(K):
            acc += frame(t + (j / max(K - 1, 1) - 0.5) * sh / FPS)
        enc.stdin.write(film(acc / K).tobytes())
    enc.stdin.close(); enc.wait()
    print("frames", n)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
