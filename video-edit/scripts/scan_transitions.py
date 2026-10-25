"""Scan every clip for transition opportunities.

Per sampled frame (8 fps, 180x320 gray): camera pan from median optical flow,
a foreground "wipe" score (share of pixels moving much faster than the camera),
darkness (share of near-black pixels — push-through / black-fill cuts) and
frame-to-frame change. Prints the strongest moments per clip as JSON lines.
"""
import sys, glob, os, json, subprocess
import numpy as np, cv2

W, H, FS = 180, 320, 8


def frames(path):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-vf", f"fps={FS},scale={W}:{H},format=gray",
                          "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    while True:
        b = p.stdout.read(W * H)
        if len(b) < W * H:
            return
        yield np.frombuffer(b, np.uint8).reshape(H, W)


def scan(path):
    prev = None; rows = []
    for i, g in enumerate(frames(path)):
        if prev is not None:
            f = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            dx, dy = np.median(f[..., 0]), np.median(f[..., 1])
            rel = np.hypot(f[..., 0] - dx, f[..., 1] - dy)
            wipe = float((rel > 6).mean())                       # big independent motion
            cols = (rel > 6).mean(0)                             # where across the frame
            span = float((cols > 0.35).mean())                   # how wide the moving thing is
            rows.append(dict(t=round(i / FS, 2), pan=round(float(dx), 1), tilt=round(float(dy), 1),
                             wipe=round(wipe, 3), span=round(span, 2), dark=round(float((g < 35).mean()), 3),
                             change=round(float(np.abs(g.astype(int) - prev).mean()), 1)))
        prev = g
    return rows


out = {}
for path in sorted(glob.glob(os.path.join(sys.argv[1], "*.MOV")) + glob.glob(os.path.join(sys.argv[1], "*.mov"))):
    r = scan(path)
    if not r:
        continue
    name = os.path.basename(path).rsplit(".", 1)[0]
    pans = [abs(x["pan"]) for x in r]
    best_wipe = max(r, key=lambda x: x["wipe"] * (0.5 + x["span"]))
    best_pan = max(r, key=lambda x: abs(x["pan"]) + abs(x["tilt"]))
    best_dark = max(r, key=lambda x: x["dark"])
    out[name] = dict(dur=round(len(r) / FS, 1), mean_pan=round(float(np.mean([x["pan"] for x in r])), 2),
                     wipe=best_wipe, whip=best_pan, dark=best_dark)
    print(name, json.dumps(out[name]), flush=True)
json.dump(out, open(sys.argv[2], "w"), indent=1)
