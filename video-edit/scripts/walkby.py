"""Walk-by wipe: the man with the black backpack in IMG_4710 crosses close to the lens right→left;
behind his trailing edge the world is already IMG_4712 (the radio tower). On the beat the camera
carries on its rightward pan, flicking him out of frame with motion blur. 6 beats.

  beats 0-2  IMG_4710 poster wall, he walks in from the right
  beat 2     he covers the most frame; 4712 is revealed behind him
  beats 2-3  pan flick: he leaves frame left, 4712 settles
  beats 3-6  IMG_4712

usage: python3 walkby.py <source_dir> <out.mp4>
"""
import sys, subprocess, math
import numpy as np, cv2, mediapipe as mp
from vcam import OW, OH, FPS, BEAT, c01, lerp, ease_in, ease_out
from vcam2 import Clip
from dome import view, segmenter
from golden import film

PEAK = 2.45                  # IMG_4710 time his trailing edge is well inside frame (on the beat)
XS = np.arange(OW, dtype=np.float32)[None, :]


def smooth(p):
    p = c01(p); return p * p * (3 - 2 * p)


def person_masks(frames, seg):
    """Him (the big close figure) per frame. A woman standing behind him sometimes touches his backpack;
    where she is a separate blob we remember her, and subtract that from frames where they merge."""
    raw, sep = [], []
    for f in frames:
        r = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(f)))
        m = (np.squeeze(r.category_mask.numpy_view()) > 0).astype(np.uint8)
        n, lab, st, cen = cv2.connectedComponentsWithStats(m)
        if n < 2:
            raw.append(np.zeros_like(m)); sep.append(None); continue
        k = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
        others = [j for j in range(1, n) if j != k and st[j, cv2.CC_STAT_AREA] > 50000 and cen[j][0] > cen[k][0]]
        raw.append((lab == k).astype(np.uint8))
        sep.append(np.isin(lab, others).astype(np.uint8) if others else None)
    # nearest frame (either direction) where she stood apart
    known = [i for i, x in enumerate(sep) if x is not None]
    out = []
    for i, m in enumerate(raw):
        if known and sep[i] is None:
            j = min(known, key=lambda j: abs(j - i))
            if abs(j - i) <= 30:
                m = m & (1 - cv2.dilate(sep[j], np.ones((25, 25), np.uint8)))
                n, lab, st, _ = cv2.connectedComponentsWithStats(m)
                if n > 1:
                    m = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        out.append(cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.5))
    return out


def main(src, out):
    BT = [k * BEAT for k in range(7)]
    n = round(BT[6] * FPS)
    s0 = PEAK - BT[2]
    A = Clip(f"{src}/IMG_4710.MOV", s0, BT[3] + 0.2)
    B = Clip(f"{src}/IMG_4712.MOV", 0.3, 4.6)
    seg = segmenter()
    masks = person_masks(A.f, seg)
    sA = lambda t: s0 + t
    iA = lambda t: min(max(round((sA(t) - A.ss) * FPS), 0), len(masks) - 1)
    sB = lambda t: 0.3 + max(t - (BT[2] - 0.3), 0)
    T_GATE = BT[2] - 0.20                                             # his trailing edge starts revealing 4712
    T_FLICK = BT[2] + 0.12
    FLICK = 0.32

    def flick(t):
        return ease_in(c01((t - T_FLICK) / FLICK))

    def frame(t):
        if t < T_GATE - 0.03:
            return view(A.at(sA(t)), 0.5, 0.5, lerp(1.0, 1.06, t / BT[2]), 0.5, 0.5, 0.0)
        z = 1.06
        f = flick(t)
        dx = -1.3 * f                                                 # camera keeps panning right: he flies out left
        a = view(A.at(min(sA(t), A.ss + A.f.__len__() / FPS - 0.05)), 0.5, 0.5, z, 0.5 + dx, 0.5, 0.0)
        m = view(masks[iA(t)][..., None].repeat(3, 2), 0.5, 0.5, z, 0.5 + dx, 0.5, 0.0)[..., 0]
        # 4712 arrives with the same pan and eases to rest
        settle = 1 - ease_out(c01((t - (T_FLICK + FLICK * 0.6)) / 0.5))
        bx = 0.5 + 0.18 * settle * c01((t - T_GATE) / 0.1)
        b = view(B.at(sB(t)), 0.5, 0.5, 1.08, bx, 0.5, 0.0)
        bw = int(45 * math.sin(math.pi * c01((t - T_FLICK) / (FLICK + 0.2))))
        if bw > 3:
            b = cv2.blur(b, (bw, 1))
        # per row: right of his trailing edge = new world; once he reaches the left edge, everything not him
        cols = m > 0.5
        has = cols.any(1)
        right = np.where(has, OW - 1 - np.argmax(cols[:, ::-1], 1), -1).astype(np.float32)
        if has.any():                                                 # rows above his head carry on his edge
            idx = np.arange(OH)
            right = np.interp(idx, idx[has], right[has]).astype(np.float32)
            right = cv2.GaussianBlur(right[:, None], (1, 0), sigmaX=0.1, sigmaY=12)[:, 0]
        rev = np.clip((XS - right[:, None]) / 10 + 0.5, 0, 1)
        allin = smooth((t - (T_FLICK + FLICK * 0.55)) / 0.08)         # leftovers swap inside the flick blur
        rev = np.maximum(rev, allin) * smooth((t - T_GATE) / 0.06)
        rev = rev * (1 - m)
        if f >= 1:
            return b
        return a * (1 - rev[..., None]) + b * rev[..., None]

    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17",
                            "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    for i in range(n):
        t = i / FPS
        busy = T_GATE - 0.1 < t < T_FLICK + FLICK + 0.15
        K = 8 if busy else 2
        sh = 1.0 if busy else 0.5
        acc = np.zeros((OH, OW, 3), np.float32)
        for j in range(K):
            acc += frame(t + (j / max(K - 1, 1) - 0.5) * sh / FPS)
        enc.stdin.write(film(acc / K).tobytes())
    enc.stdin.close(); enc.wait()
    print("frames", n)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
