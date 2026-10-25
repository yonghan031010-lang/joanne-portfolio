"""Tube arrival (IMG_4659) → King's Cross platform (IMG_4242).

The train is cut out per frame against an empty-platform plate (SIFT-registered homography,
since the camera pans with the train). Its nose is the wipe edge: behind it the world becomes
IMG_4242 while the train stays on top; then the train streaks out on the beat and the platform
remains.

usage: python3 train_arrival.py <source_dir> <out.mp4> [debug.jpg]
"""
import sys, subprocess, math
import numpy as np, cv2
from vcam import OW, OH, FPS, BEAT, c01, lerp, ease_in, ease_out
from vcam2 import Clip
from dome import view
from golden import film

HW, HH = 540, 960
# train body in the output frame, measured on the arrival frames: roof and skirt lines (fractions)
_y, _x = np.mgrid[0:OH, 0:OW].astype(np.float32)
_xf, _yf = _x / OW, _y / OH
BODY = (np.clip(((_yf - (0.230 - 0.150 * _xf)) * OH) / 6 + 0.5, 0, 1) *
        np.clip((((0.470 + 0.385 * _xf) - _yf) * OH) / 6 + 0.5, 0, 1))[..., None].astype(np.float32)
A0, A1 = 1.6, 4.6            # IMG_4659 range we use


def smooth(p):
    p = c01(p); return p * p * (3 - 2 * p)


def robust_line(x, y, iters=4):
    a, b = np.polyfit(x, y, 1)
    for _ in range(iters):
        r = np.abs(y - (a * x + b))
        keep = r < max(np.percentile(r, 60), 3)
        if keep.sum() < 10:
            break
        a, b = np.polyfit(x[keep], y[keep], 1)
    return a, b


def clip_to_body(mk):
    """Keep only the train body: between straight roof and skirt lines (perspective box)."""
    cols = np.where(mk.sum(0) > 20)[0]
    if len(cols) < 30:
        return mk
    tops = np.array([np.argmax(mk[:, x]) for x in cols], float)
    bots = np.array([HH - 1 - np.argmax(mk[::-1, x]) for x in cols], float)
    # roof: the lower envelope of the top edges (ceiling blobs sit above it); skirt: upper envelope of bottoms
    at, bt = robust_line(cols.astype(float), tops)
    ab, bb = robust_line(cols.astype(float), bots)
    yy, xx = np.mgrid[0:HH, 0:HW]
    body = (yy >= at * xx + bt - 4) & (yy <= ab * xx + bb + 4)
    return (mk & body.astype(np.uint8))


def train_masks(A):
    """Per-frame train matte (1080x1920 float) and nose x (frame fraction)."""
    fr = [cv2.resize(f, (HW, HH), interpolation=cv2.INTER_AREA).astype(np.float32) for f in A.f]
    plate_clip = Clip(A.path, 0.0, 0.45)
    plate = np.median(np.stack([cv2.resize(f, (HW, HH), interpolation=cv2.INTER_AREA).astype(np.float32) for f in plate_clip.f]), 0)
    sift = cv2.SIFT_create(nfeatures=1500); bf = cv2.BFMatcher()
    kp0, d0 = sift.detectAndCompute(cv2.cvtColor(plate.astype(np.uint8), cv2.COLOR_RGB2GRAY), None)
    masks, nose = [], []
    Hprev = np.eye(3)
    for f in fr:
        kp, d = sift.detectAndCompute(cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_RGB2GRAY), None)
        Hm = Hprev
        if d is not None:
            m = [a for a, b in (x for x in bf.knnMatch(d0, d, k=2) if len(x) == 2) if a.distance < 0.75 * b.distance]
            if len(m) >= 12:
                A_ = np.float32([kp0[x.queryIdx].pt for x in m]); B_ = np.float32([kp[x.trainIdx].pt for x in m])
                Hn, inl = cv2.findHomography(A_, B_, cv2.RANSAC, 3.0)
                if Hn is not None and inl.sum() >= 12:
                    Hm = Hn
        Hprev = Hm
        p = cv2.warpPerspective(plate, Hm, (HW, HH), borderMode=cv2.BORDER_REPLICATE)
        dd = np.abs(f - p).max(2)
        mk = (cv2.GaussianBlur(dd, (7, 7), 0) > 32).astype(np.uint8)
        mk = cv2.morphologyEx(mk, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mk = cv2.morphologyEx(mk, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(mk)
        if n > 1:
            keep = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
            mk = (lab == keep).astype(np.uint8)
            ff = mk.copy(); fm = np.zeros((HH + 2, HW + 2), np.uint8)
            cv2.floodFill(ff, fm, (0, HH - 1), 1)                  # fill holes inside the train
            mk = mk | (1 - ff)
            mk = clip_to_body(mk)
            xs = np.where(mk.any(0))[0]
            nose.append(xs[-1] / HW if len(xs) else 0.0)
        else:
            mk[:] = 0; nose.append(0.0)
        m = cv2.GaussianBlur(cv2.resize(mk.astype(np.float32), (OW, OH)), (9, 9), 0)
        masks.append(m)
    nose = np.maximum.accumulate(np.array(nose))                    # the nose only moves right
    return masks, nose


def main(src, out, debug=None):
    A = Clip(f"{src}/IMG_4659.MOV", A0, A1 - A0)
    A.path = f"{src}/IMG_4659.MOV"
    B = Clip(f"{src}/IMG_4242.MOV", 0.0, 4.0)
    masks, nose = train_masks(A)
    BT = [k * BEAT for k in range(7)]
    n = round(BT[6] * FPS)
    sA = lambda t: A0 + 0.1 + t                                    # 1x
    sB = lambda t: 0.4 + max(t - BT[2], 0) * 1.0
    gA = A.at(3.6).reshape(-1, 3).mean(0); gB = B.at(0.6).reshape(-1, 3).mean(0)
    gain = (gA / gB).astype(np.float32)                            # pull the platform toward tunnel light
    XS = np.arange(OW, dtype=np.float32)[None, :, None]
    T_OUT = BT[4]

    def idx(t):
        return min(max(round((sA(t) - A0) * FPS), 0), len(masks) - 1)

    def frame(t):
        if t < BT[2]:
            return view(A.at(sA(t)), 0.5, 0.5, lerp(1.0, 1.08, t / BT[2]), 0.5, 0.5, 0.0)
        i = idx(t)
        a = view(A.at(sA(t)), 0.5, 0.5, 1.08, 0.5, 0.5, 0.0)
        mk = view(masks[i][..., None].repeat(3, 2), 0.5, 0.5, 1.08, 0.5, 0.5, 0.0)[..., :1] * BODY
        g = 1 + (gain - 1) * (1 - ease_out(c01((t - T_OUT) / 0.6)))
        # background panned and streaked like the same camera following the train, easing out after it leaves
        follow = 1 - ease_out(c01((t - T_OUT) / 0.45))
        pan = -0.10 * ease_out(c01((t - BT[2]) / (T_OUT + 0.45 - BT[2])))       # camera tracks right → world slides left
        b = view(B.at(sB(t)), 0.5, 0.5, 1.12, 0.5 + pan, 0.5, 0.0) * g
        bw = int(85 * follow * c01((t - BT[2]) / 0.12))
        if bw > 3:
            b = cv2.blur(b, (bw, 1))
        # behind the nose the world is already the new platform
        nx = (0.5 + (nose[i] - 0.5) * 1.08) * OW
        behind = np.clip((nx - XS) / 40 + 0.5, 0, 1) * c01((t - BT[2]) / 0.08)
        bg = a * (1 - behind) + b * behind
        # train leaves on the beat: fades out of its matte as it streaks
        leave = ease_in(c01((t - (T_OUT - 0.18)) / 0.25))
        tm = cv2.GaussianBlur(mk, (0, 0), 6)[..., None] * (1 - leave)
        edge = np.clip(tm * (1 - tm) * 4, 0, 1)                    # band where train meets background
        wrap = cv2.GaussianBlur(bg, (0, 0), 18)                      # background light bleeding onto the train edge
        train = a * (1 - 0.35 * edge) + wrap * 0.35 * edge
        return bg * (1 - tm) + train * tm, leave

    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17",
                            "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    dbg = []
    for i in range(n):
        t = i / FPS
        busy = BT[2] - 0.1 < t < T_OUT + 0.4
        K = 8 if busy else 3
        sh = 0.6 + 0.8 * math.exp(-((t - T_OUT) / 0.2) ** 2)
        acc = np.zeros((OH, OW, 3), np.float32); lv = 0.0
        for j in range(K):
            tj = t + (j / max(K - 1, 1) - 0.5) * sh / FPS
            r = frame(tj)
            if isinstance(r, tuple):
                img, l = r; lv += l / K
            else:
                img = r
            acc += img
        img = acc / K
        streak = int(140 * math.exp(-((t - T_OUT) / 0.12) ** 2))   # train shoots off to the right
        if streak > 3:
            img = cv2.blur(img, (streak, 1))
        if debug and i % 6 == 0:
            dbg.append(cv2.resize(img, (120, 213)))
        enc.stdin.write(film(img).tobytes())
    enc.stdin.close(); enc.wait()
    if debug:
        cv2.imwrite(debug, cv2.cvtColor(np.hstack(dbg).astype(np.uint8), cv2.COLOR_RGB2BGR))
    print("frames", n, "nose path", np.round(nose[::10], 2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
