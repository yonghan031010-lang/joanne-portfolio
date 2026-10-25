"""Seeded template tracking → keyframes in a call-out config.

For every item with "seed": [t, cx, cy] (source-relative seconds, frame fractions),
match a template cut at the seed forward and backward through the clip (normalised
cross-correlation, small scale search, local window), smooth the path and write it
back as "keys" sampled every 0.1s.

usage: python3 tmpltrack.py <source_dir> <callout.json>
"""
import sys, json
import numpy as np, cv2
from detail import decode, FPS
from vcam import BEAT

W, H = 360, 640


def follow(frames, i0, cx, cy, bw, bh, step):
    tw, th = max(int(bw * W), 12), max(int(bh * H), 12)
    g0 = frames[i0]
    x0, y0 = int(cx * W - tw / 2), int(cy * H - th / 2)
    tmpl = g0[max(y0, 0):y0 + th, max(x0, 0):x0 + tw].astype(np.float32)
    path = {i0: (cx, cy)}
    x, y = cx * W, cy * H
    i = i0 + step
    while 0 <= i < len(frames):
        g = frames[i].astype(np.float32)
        best = (-2, x, y, 1.0)
        R = int(0.10 * W)
        for s in (0.92, 1.0, 1.08):
            t = cv2.resize(tmpl, None, fx=s, fy=s)
            sx0, sy0 = int(max(x - t.shape[1] / 2 - R, 0)), int(max(y - t.shape[0] / 2 - R, 0))
            sx1, sy1 = int(min(x + t.shape[1] / 2 + R, W)), int(min(y + t.shape[0] / 2 + R, H))
            win = g[sy0:sy1, sx0:sx1]
            if win.shape[0] < t.shape[0] or win.shape[1] < t.shape[1]:
                continue
            r = cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(r)
            if mx > best[0]:
                best = (mx, sx0 + loc[0] + t.shape[1] / 2, sy0 + loc[1] + t.shape[0] / 2, s)
        if best[0] > 0.35:                     # keep last position when the item is lost
            x, y = best[1], best[2]
            # slow template refresh keeps up with turning / lighting without drifting off
            nx0, ny0 = int(x - tw / 2), int(y - th / 2)
            new = g[max(ny0, 0):ny0 + th, max(nx0, 0):nx0 + tw]
            if new.shape == tmpl.shape:
                tmpl = 0.85 * tmpl + 0.15 * new
        path[i] = (x / W, y / H)
        i += step
    return path


def main(src, cfg_path):
    c = json.load(open(cfg_path))
    dur = c["beats"] * BEAT * c["speed"] + 0.3
    frames = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in decode(f"{src}/{c['clip']}.MOV", c["start"], dur, (W, H))]
    for it in c["items"]:
        if "seed" not in it:
            continue
        t0, cx, cy = it["seed"]
        i0 = min(round(t0 * FPS), len(frames) - 1)
        p = follow(frames, i0, cx, cy, it["w"], it["h"], 1)
        p.update(follow(frames, i0, cx, cy, it["w"], it["h"], -1))
        a = np.array([p[i] for i in range(len(frames))])
        k = np.ones(5) / 5
        a = np.stack([np.convolve(np.pad(a[:, j], 2, mode="edge"), k, "valid") for j in range(2)], 1)
        it["keys"] = [[round(i / FPS, 3), round(float(a[i, 0]), 4), round(float(a[i, 1]), 4)] for i in range(0, len(frames), 3)]
        print(it["label"], "tracked", len(frames), "frames")
    open(cfg_path, "w").write(json.dumps(c, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
