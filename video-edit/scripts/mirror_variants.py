"""Two follow-ups to the mirror outfit swap (round 1 stays the beat-locked full-body swap):

  push  – rounds 2 and 3 push in on the pinned phone (waist-up, then face), from the 4K sources
  mix   – rounds 2 and 3 split at the waist: top of one outfit over the bottom of another

usage: python3 mirror_variants.py push|mix <source_dir> <out.mp4>
"""
import sys, subprocess
import numpy as np, cv2
from vcam import OW, OH, FPS, BEAT, grade
from detail import decode, probe
from mirror_swap import CLIPS, track

ROUNDS = 3
PUSH = [1.0, 1.7, 2.8]                  # zoom on top of the base zoom, per round
SPLIT_Y = 0.79                          # waistline in the pinned frame
MIX = [("IMG_4052", "IMG_4124"), ("IMG_4124", "IMG_4648"), ("IMG_4648", "IMG_4052"),
       ("IMG_4052", "IMG_4648"), ("IMG_4124", "IMG_4052"), ("IMG_4648", "IMG_4124")]   # (top, bottom)


def prepare(src):
    data = {}
    for name, a, b, ts, seed, width in CLIPS:
        fr = [cv2.resize(f, (OW, OH), interpolation=cv2.INTER_AREA) for f in decode(f"{src}/{name}.MOV", a, b - a)]
        data[name] = track(fr, round((ts - a) * FPS), seed, width)
        print(name, "tracked", len(fr), flush=True)
    return data


def need_zoom(row, TX, TY, TS):
    z = TS / row[2]
    px, py = row[0], row[1]
    return max(1.0, TX / (z * px), (OW - TX) / (z * (OW - px)), TY / (z * py), (OH - TY) / (z * (OH - py)))


def choose(data, L):
    def pick(tr, TX, TY, TS):
        starts = list(range(0, len(tr) - L + 1, 2))
        cost = {s0: max(need_zoom(tr[s0 + k], TX, TY, TS) for k in range(L)) for s0 in starts}
        best = None
        def rec(prev_end, chosen):
            nonlocal best
            if len(chosen) == ROUNDS:
                c = max(cost[x] for x in chosen)
                if best is None or c < best[0]:
                    best = (c, list(chosen))
                return
            for s0 in starts:
                if s0 >= prev_end and (best is None or cost[s0] < best[0]):
                    rec(s0 + L, chosen + [s0])
        rec(0, [])
        return best[1]
    alltr = np.concatenate(list(data.values()))
    TX, TY, TS = alltr[:, 0].mean(), alltr[:, 1].mean(), alltr[:, 2].mean()
    for _ in range(3):
        wins = {nm: pick(tr, TX, TY, TS) for nm, tr in data.items()}
        sel = np.concatenate([data[nm][s0:s0 + L] for nm, ss in wins.items() for s0 in ss])
        TX, TY, TS = sel[:, 0].mean(), sel[:, 1].mean(), sel[:, 2].mean()
    return wins, (TX, TY, TS)


def hires(src, needed):
    """Native-resolution frames, only the indices that will be shown."""
    out = {}
    for name, a, b, *_ in CLIPS:
        keep = needed.get(name, set())
        w, h = probe(f"{src}/{name}.MOV")
        out[name] = ({i: f.copy() for i, f in enumerate(decode(f"{src}/{name}.MOV", a, b - a)) if i in keep}, w / OW)
    return out


def warp(img, f, row, TX, TY, TS, z_extra):
    px, py, ps, th = row
    z = TS / ps * z_extra
    M = cv2.getRotationMatrix2D((float(px * f), float(py * f)), float(np.degrees(th)), float(z / f))
    M[:, 2] += (TX - px * f, TY - py * f)
    return cv2.warpAffine(img, M, (OW, OH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def main(mode, src, out):
    data = prepare(src)
    n = len(CLIPS); beats = ROUNDS * n
    cut = [round(j * BEAT * FPS) for j in range(beats + 1)]
    L = max(cut[j + 1] - cut[j] for j in range(beats))
    wins, (TX, TY, TS) = choose(data, L)
    Z0 = max(need_zoom(data[nm][s0 + k], TX, TY, TS) for nm, ss in wins.items() for s0 in ss for k in range(L))
    order = [c[0] for c in CLIPS]

    # what each output frame shows: list of (clip, source index) layers
    plan = []
    for j in range(beats):
        r = j // n
        for i in range(cut[j], cut[j + 1]):
            k = i - cut[j]
            if mode == "mix" and r > 0:
                top, bot = MIX[j - n]
                plan.append((j, [(top, wins[top][r] + k), (bot, wins[bot][r] + k)]))
            else:
                nm = order[j % n]
                plan.append((j, [(nm, wins[nm][r] + k)]))
    needed = {}
    for _, layers in plan:
        for nm, si in layers:
            needed.setdefault(nm, set()).add(si)
    hr = hires(src, needed)
    print("pinned (%.0f, %.0f) scale %.3f base zoom %.3f; windows %s" % (TX, TY, TS, Z0, wins), flush=True)

    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "16",
                            "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    ys = int(SPLIT_Y * OH)
    for i, (j, layers) in enumerate(plan):
        r = j // n
        k = (i - cut[j]) / max(cut[j + 1] - cut[j], 1)
        if mode == "push":
            z = Z0 * PUSH[r] * (1 + 0.03 * k)                       # step in on the beat, drift inside it
        else:
            z = Z0 * (1 + 0.04 * i / len(plan))
        imgs = []
        for nm, si in layers:
            frames, f = hr[nm]
            imgs.append(warp(frames[si], f, data[nm][si], TX, TY, TS, z))
        img = imgs[0]
        if len(imgs) == 2:
            img = img.copy(); img[ys:] = imgs[1][ys:]
            cv2.line(img, (0, ys), (OW, ys), (255, 255, 255), 3, cv2.LINE_AA)
        enc.stdin.write(grade(img.astype(np.float32)).tobytes())
    enc.stdin.close(); enc.wait()
    print(mode, len(plan), "frames; cuts", cut)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
