"""Bus wipe: IMG_4221's double-decker passes; once it fills the frame, everything that is
not bus becomes IMG_4243, so the bus drives off to reveal the tube platform and the train
pulling out. Both clips speed-ramp up; the reveal and the train's departure sit on beats.

The bus is segmented per frame against a clean street plate (median of the frames before
it arrives, each frame registered to the plate by phase correlation).

usage: python3 bus_wipe.py <source_dir> <out.mp4> [debug_dir]
"""
import sys, subprocess, math
import numpy as np, cv2
from vcam import OW, OH, FPS, BEAT, grade, c01, lerp
from vcam2 import Clip, remap, punch, post

HW, HH = 540, 960
A_SS, A_DUR = 0.5, 3.1          # IMG_4221 range we load
WIPE = (2.78, 3.47)             # IMG_4221 times where the bus is composited over IMG_4243
PLATE = (0.6, 1.35)             # IMG_4221 times with an empty street


def half(img):
    return cv2.resize(img, (HW, HH), interpolation=cv2.INTER_AREA).astype(np.float32)


def bus_box(frame):
    """Bus as a box (x0, x1, top, bottom) in half-res px from its paint colour, or None."""
    f = cv2.resize(frame, (HW, HH), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    # sampled: bus H109-112 S107-134 V97-125 · sky V>220 · building S~67 · road shadow V~62
    m = ((h >= 106) & (h <= 120) & (s > 95) & (v > 78) & (v < 190)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1 or st[1:, cv2.CC_STAT_AREA].max() < 800:
        return None
    keep = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    m = (lab == keep).astype(np.uint8)
    cols = np.where(m.sum(0) > 6)[0]
    busish = ((m > 0) | ((v < 125) & (s > 45))).astype(np.uint8)
    busish = cv2.morphologyEx(busish, cv2.MORPH_CLOSE, np.ones((45, 1), np.uint8))
    tops, bots = [], []
    for x in cols:
        ys = np.where(m[:, x])[0]
        top = ys[0]
        while top > 0 and busish[top - 1, x]:
            top -= 1
        tops.append(top); bots.append(ys[-1])
    top = int(np.percentile(tops, 8))
    sky = (v > 185) & (s < 120)                                   # bright, washed-out: sky / sunlit stone
    while top < np.percentile(tops, 50) and sky[top, cols[0]:cols[-1] + 1].mean() > 0.25:
        top += 1
    return cols[0], cols[-1], float(top), float(np.percentile(bots, 85)) + 0.035 * HH


def boxes_for(frames_idx, A):
    raw = {i: bus_box(A.f[i]) for i in frames_idx}
    idx = [i for i in frames_idx if raw[i] is not None]
    arr = np.array([raw[i] for i in idx], float)
    k = np.array([1, 2, 3, 2, 1], float); k /= k.sum()
    sm = np.stack([np.convolve(np.pad(arr[:, j], 2, mode="edge"), k, "valid") for j in range(4)], 1)
    out = {}
    for i in frames_idx:
        out[i] = tuple(np.interp(i, idx, sm[:, j]) for j in range(4))
    return out


def bus_mask(box):
    x0, x1, top, bot = [c * 2 for c in box]                       # back to 1080x1920
    if x0 <= 12:                                                  # bus still runs off the left edge
        x0 = -40
    if x1 >= OW - 12:
        x1 = OW + 40
    xs = np.arange(OW, dtype=np.float32); ys = np.arange(OH, dtype=np.float32)
    fx = np.clip(np.minimum(xs - x0, x1 - xs) / 3 + 0.5, 0, 1)    # ~3px soft edge
    fy = np.clip(np.minimum(ys - top, bot - ys) / 3 + 0.5, 0, 1)
    return fy[:, None] * fx[None, :]


def main(src, out, debug=None):
    A = Clip(f"{src}/IMG_4221.MOV", A_SS, A_DUR)
    B = Clip(f"{src}/IMG_4243.MOV", 1.8, 2.75)
    plate = None

    rng = range(round((WIPE[0] - A_SS) * FPS), round((WIPE[1] - A_SS) * FPS) + 1)
    boxes = boxes_for(list(rng), A)
    masks = {i: bus_mask(boxes[i]) for i in rng}
    if debug:
        for i in list(masks)[::3]:
            v = A.f[i].copy(); v[masks[i] < 0.5] = (v[masks[i] < 0.5] * 0.25).astype(np.uint8)
            cv2.imwrite(f"{debug}/m_{i:03d}.jpg", cv2.cvtColor(cv2.resize(v, (270, 480)), cv2.COLOR_RGB2BGR))

    n = round(4 * BEAT * FPS)
    TM = 2 * BEAT                                           # rear of the bus crosses mid-frame
    vA = lambda t: 1.1 + 1.1 * c01(t / TM) ** 2             # 1.1x → 2.2x into the reveal
    sA0 = remap(vA, n)
    sA = lambda t: sA0(t) - sA0(TM) + 3.10
    vB = lambda t: lerp(1.4, 1.85, c01((t - TM) / (2 * BEAT)))   # tube keeps accelerating
    sB0 = remap(vB, n)
    END = 4 * BEAT
    sB = lambda t: sB0(t) - sB0(END) + 3.70                # train's blur peak lands on the closing beat
    print("IMG_4221 from %.2fs, IMG_4243 to %.2fs" % (sA(0), sB(n / FPS)))

    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17",
                            "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    for i in range(n):
        t = i / FPS
        K = 8 if abs(t - TM) < 0.5 else 4
        sh = 0.5 + 0.6 * math.exp(-((t - TM) / 0.25) ** 2)   # wider shutter at the peak speed
        acc = np.zeros((OH, OW, 3), np.float32)
        for j in range(K):
            tj = t + (j / (K - 1) - 0.5) * sh / FPS
            a = sA(tj)
            if a < WIPE[0]:
                img = A.at(a)
            elif a <= WIPE[1]:
                m = masks[min(max(round((a - A_SS) * FPS), min(masks)), max(masks))][..., None]
                img = A.at(a) * m + B.at(sB(tj)) * (1 - m)
            else:
                img = B.at(sB(tj))
            z, d = punch(tj, TM, amp=0.03, w=0.14, shake_px=8)
            acc += post(img, z, d)
        enc.stdin.write(grade(acc / K).tobytes())
    enc.stdin.close(); enc.wait()
    print("frames", n)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
