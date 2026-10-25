"""Mirror outfit-swap match cut.

The phone (same case in every clip) is tracked per frame by multi-scale template
matching on its camera module, then each frame is warped so the phone sits at the
exact same position and size. Outfits switch on every beat while the phone and face
stay pinned.

usage: python3 mirror_swap.py <source_dir> <out.mp4> [beats]
"""
import sys, subprocess
import numpy as np, cv2
from vcam import OW, OH, FPS, BEAT, grade
from detail import decode

# clip, segment start/end (s), seed time (s, absolute), camera-module centre at the seed (1080x1920 px),
# phone width at the seed (px) — measured by hand from crops; sets the cross-clip scale
CLIPS = [
    ("IMG_4052", 0.20, 4.30, 2.0, (600, 605), 125),
    ("IMG_4124", 5.30, 7.95, 6.5, (550, 625), 125),
    ("IMG_4648", 2.60, 6.70, 4.5, (570, 560), 117),
]
MOD = (96, 84)                                        # camera-module template size at 1080 px
HW, HH = 540, 960                                     # tracking resolution


def gray_half(f):
    return cv2.cvtColor(cv2.resize(f, (HW, HH), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY).astype(np.float32)


def find(g, tmpl, guess, scales, R):
    best = (-2, None, None)
    for s in scales:
        t = cv2.resize(tmpl, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        th, tw = t.shape
        if guess is None:
            x0, y0, x1, y1 = 0, 0, HW, HH // 2
        else:
            x0, y0 = int(max(guess[0] - tw / 2 - R, 0)), int(max(guess[1] - th / 2 - R, 0))
            x1, y1 = int(min(guess[0] + tw / 2 + R, HW)), int(min(guess[1] + th / 2 + R, HH))
        win = g[y0:y1, x0:x1]
        if win.shape[0] < th or win.shape[1] < tw:
            continue
        r = cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(r)
        if mx > best[0]:
            best = (mx, (x0 + loc[0] + tw / 2, y0 + loc[1] + th / 2), s)
    return best


SIFT = cv2.SIFT_create(nfeatures=400, contrastThreshold=0.02)
BF = cv2.BFMatcher(cv2.NORM_L2)


def phone_feats(g, c, r):
    x0, y0 = int(max(c[0] - r[0], 0)), int(max(c[1] - r[1], 0))
    roi = g[y0:int(c[1] + r[1]), x0:int(c[0] + r[0])]
    kp, des = SIFT.detectAndCompute(roi.astype(np.uint8), None)
    pts = np.float32([k.pt for k in kp]) + (x0, y0) if kp else np.zeros((0, 2), np.float32)
    return pts, des


def sim_fit(p_src, d_src, p_dst, d_dst):
    """Similarity (scale, rotation, shift) mapping src points onto dst; None if unreliable."""
    if d_src is None or d_dst is None or len(p_src) < 6 or len(p_dst) < 6:
        return None, 0
    m = BF.knnMatch(d_src, d_dst, k=2)
    good = [a for a, b in (x for x in m if len(x) == 2) if a.distance < 0.78 * b.distance]
    if len(good) < 6:
        return None, len(good)
    A = np.float32([p_src[g.queryIdx] for g in good]); B = np.float32([p_dst[g.trainIdx] for g in good])
    M, inl = cv2.estimateAffinePartial2D(A, B, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=4000)
    n = int(inl.sum()) if inl is not None else 0
    return (M if n >= 6 else None), n


def track(frames, i0, seed, width):
    """Per-frame similarity G_i: seed-frame coords -> frame i coords, from SIFT on the phone (half-res).
    Each frame is matched straight to the seed (no drift); misses fall back to chaining from the neighbour."""
    c = (seed[0] / 2, seed[1] / 2); R = (70, 95)                     # phone + a little margin, half-res
    g = [gray_half(f) for f in frames]
    ps, ds = phone_feats(g[i0], c, R)
    G = {i0: np.float32([[1, 0, 0], [0, 1, 0]])}
    to3 = lambda M: np.vstack([M, [0, 0, 1]])
    for step in (1, -1):
        i = i0 + step
        while 0 <= i < len(frames):
            prev = G[i - step]
            pc = to3(prev) @ np.array([c[0], c[1], 1.0])
            pt, dt = phone_feats(g[i], pc[:2], (R[0] * 1.6, R[1] * 1.4))
            M, n = sim_fit(ps, ds, pt, dt)
            if M is None:                                            # chain via the neighbour frame
                pp, dp = phone_feats(g[i - step], pc[:2], (R[0] * 1.4, R[1] * 1.3))
                Mn, _ = sim_fit(pp, dp, pt, dt)
                M = (to3(Mn) @ to3(prev))[:2] if Mn is not None else prev
            G[i] = M.astype(np.float32)
            i += step
    out = []
    for i in range(len(frames)):
        M = G[i]
        s = float(np.hypot(M[0, 0], M[1, 0])); th = float(np.arctan2(M[1, 0], M[0, 0]))
        p = M @ np.array([c[0], c[1], 1.0])
        out.append((p[0] * 2, p[1] * 2, s * width / 125.0, th))      # phone centre (1080 px), scale, angle
    a = np.array(out)
    k = np.array([1, 2, 3, 2, 1], float); k /= k.sum()
    for j in range(4):
        a[:, j] = np.convolve(np.pad(a[:, j], 2, mode="edge"), k, "valid")
    return a


def main(src, out, beats=9):
    data = {}
    for name, a, b, ts, seed, width in CLIPS:
        fr = list(decode(f"{src}/{name}.MOV", a, b - a))
        fr = [cv2.resize(f, (OW, OH), interpolation=cv2.INTER_AREA) for f in fr]
        tr = track(fr, round((ts - a) * FPS), seed, width)
        data[name] = (fr, tr)
        print(name, len(fr), "frames; phone x %.0f–%.0f y %.0f–%.0f scale %.3f–%.3f; angle %.1f–%.1f°"
              % (tr[:, 0].min(), tr[:, 0].max(), tr[:, 1].min(), tr[:, 1].max(), tr[:, 2].min(), tr[:, 2].max(),
                 np.degrees(tr[:, 3].min()), np.degrees(tr[:, 3].max())), flush=True)

    n = len(CLIPS)
    cut = [round(j * BEAT * FPS) for j in range(beats + 1)]
    rounds = -(-beats // n)
    L = max(cut[j + 1] - cut[j] for j in range(beats))

    def need_zoom(row, TX, TY, TS):
        z = TS / row[2]
        px, py = row[0], row[1]
        return max(1.0, TX / (z * px), (OW - TX) / (z * (OW - px)), TY / (z * py), (OH - TY) / (z * (OH - py)))

    def pick(tr, TX, TY, TS):
        """Best `rounds` ordered, non-overlapping beat windows (lowest worst-case zoom)."""
        starts = list(range(0, len(tr) - L + 1, 2))
        cost = {s0: max(need_zoom(tr[s0 + k], TX, TY, TS) for k in range(L)) for s0 in starts}
        best = None
        def rec(prev_end, chosen):
            nonlocal best
            if len(chosen) == rounds:
                c = max(cost[x] for x in chosen)
                if best is None or c < best[0]:
                    best = (c, list(chosen))
                return
            for s0 in starts:
                if s0 >= prev_end and (best is None or cost[s0] < best[0]):
                    rec(s0 + L, chosen + [s0])
        rec(0, [])
        return best

    # pinned target: start from the mean phone pose, then re-centre on the chosen windows
    alltr = np.concatenate([d[1] for d in data.values()])
    TX, TY, TS = alltr[:, 0].mean(), alltr[:, 1].mean(), alltr[:, 2].mean()
    for _ in range(3):
        wins = {nm: pick(data[nm][1], TX, TY, TS)[1] for nm, *_ in CLIPS}
        sel = np.concatenate([data[nm][1][s0:s0 + L] for nm, ss in wins.items() for s0 in ss])
        TX, TY, TS = sel[:, 0].mean(), sel[:, 1].mean(), sel[:, 2].mean()

    plan = []
    for j in range(beats):
        name = CLIPS[j % n][0]
        s0 = wins[name][j // n]
        for i in range(cut[j], cut[j + 1]):
            plan.append((name, s0 + (i - cut[j]), i))
    Z0 = max(need_zoom(data[nm][1][si], TX, TY, TS) for nm, si, _ in plan)
    print("windows", wins, "pinned phone at (%.0f, %.0f) scale %.3f; base zoom %.3f" % (TX, TY, TS, Z0))

    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "16",
                            "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    total = len(plan)
    for name, si, i in plan:
        fr, tr = data[name]
        si = min(si, len(fr) - 1)
        px, py, ps, th = tr[si]
        drift = 1 + 0.04 * i / total                     # slow push-in across the whole sequence
        z = TS / ps * Z0 * drift
        M = cv2.getRotationMatrix2D((float(px), float(py)), float(np.degrees(th)), float(z))   # undo the phone's tilt
        M[:, 2] += (TX - px, TY - py)
        img = cv2.warpAffine(fr[si], M, (OW, OH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        enc.stdin.write(grade(img.astype(np.float32)).tobytes())
    enc.stdin.close(); enc.wait()
    print("frames", total, "cuts at", cut)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 9)
