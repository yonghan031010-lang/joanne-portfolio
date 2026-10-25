"""Outfit detail call-outs (focus boxes + live magnified insets), as in the reference.

Each outfit plays for 7 beats (≈5s). Beat 1 is clean, then one detail pops in on each of
beats 2-5 and holds to the end. Boxes follow the garment with
optical-flow tracking; insets are cropped live from the full-resolution source.

usage: python3 detail.py <source_dir> <out.mp4>
"""
import sys, subprocess, json, math
import numpy as np, cv2
from vcam import OW, OH, FPS, BEAT, BAR, grade, c01, ease_out

# (clip, start s, speed, colour RGB, [(name, cx, cy, w, h) in frame fractions at start])
OUTFITS = [
    ("IMG_4052", 0.20, 0.82, (233, 196, 106), [
        ("phone", 0.49, 0.42, 0.16, 0.15),
        ("cuff", 0.65, 0.49, 0.14, 0.08),
        ("lace", 0.55, 0.66, 0.22, 0.13),
        ("bag", 0.28, 0.83, 0.20, 0.12)]),
    ("IMG_4180", 0.00, 0.84, (94, 208, 216), [
        ("headphones", 0.54, 0.18, 0.28, 0.14),
        ("belt", 0.39, 0.66, 0.20, 0.09),
        ("bag", 0.81, 0.63, 0.26, 0.14),
        ("boots", 0.50, 0.88, 0.28, 0.14)]),
    ("IMG_4687", 0.20, 1.00, (237, 237, 237), [
        ("scarf", 0.57, 0.07, 0.28, 0.10),
        ("necklace", 0.35, 0.37, 0.16, 0.12),
        ("belt", 0.34, 0.56, 0.24, 0.10),
        ("bag", 0.84, 0.49, 0.26, 0.14)]),
]
SEG = 7 * BEAT          # ≈5.06s per outfit, so every cut still lands on a beat
INSET_W = 0.32
LINE = 3


def probe(path):
    st = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                    "stream=width,height:stream_side_data=rotation", "-of", "json", path],
                                   capture_output=True, text=True).stdout)["streams"][0]
    w, h = st["width"], st["height"]
    rot = next((sd.get("rotation", 0) for sd in st.get("side_data_list", [])), 0)
    return (h, w) if abs(rot) == 90 else (w, h)


def decode(path, ss, dur, size=None):
    w, h = size or probe(path)
    vf = f"fps={FPS}" + (f",scale={w}:{h}" if size else "")
    raw = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{ss:.4f}", "-t", f"{dur:.4f}", "-i", path,
                            "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    while True:
        b = raw.stdout.read(w * h * 3)
        if len(b) < w * h * 3:
            return
        yield np.frombuffer(b, np.uint8).reshape(h, w, 3)


def track(path, ss, dur, boxes):
    """Median Lucas-Kanade flow inside each box → per-frame centre (fractions), smoothed."""
    W, H = 360, 640
    paths = [[(cx, cy)] for _, cx, cy, _, _ in boxes]
    prev = None; pts = [None] * len(boxes)
    for f in decode(path, ss, dur, (W, H)):
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        if prev is not None:
            for k, (_, _, _, bw, bh) in enumerate(boxes):
                cx, cy = paths[k][-1]
                if pts[k] is None or len(pts[k]) < 6:
                    m = np.zeros_like(g)
                    x0, y0 = int((cx - bw / 2) * W), int((cy - bh / 2) * H)
                    m[max(y0, 0):int(y0 + bh * H), max(x0, 0):int(x0 + bw * W)] = 255
                    pts[k] = cv2.goodFeaturesToTrack(prev, 40, 0.01, 4, mask=m)
                if pts[k] is None:
                    paths[k].append((cx, cy)); continue
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, pts[k], None, winSize=(21, 21), maxLevel=3)
                ok = st.ravel() == 1
                if ok.sum() < 3:
                    pts[k] = None; paths[k].append((cx, cy)); continue
                d = np.median((nxt[ok] - pts[k][ok]).reshape(-1, 2), 0)
                paths[k].append((cx + d[0] / W, cy + d[1] / H))
                pts[k] = nxt[ok].reshape(-1, 1, 2)
        prev = g
    out = []
    for p in paths:
        a = np.array(p)
        k = np.ones(7) / 7
        a = np.stack([np.convolve(np.pad(a[:, i], 3, mode="edge"), k, "valid") for i in range(2)], 1)
        out.append(a)
    return out


def place_insets(boxes):
    """Greedy slot choice: minimise overlap with every box and earlier insets."""
    cands = [(x, y) for x in (0.03, 0.65) for y in np.arange(0.04, 0.80, 0.02)]
    taken, res = [], []
    rects = [(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2) for _, cx, cy, w, h in boxes]
    for (_, cx, cy, w, h) in boxes:
        iw = INSET_W; ih = min(iw * (h * OH) / (w * OW) * OW / OH, 0.22)
        def cost(c):
            r = (c[0], c[1], c[0] + iw, c[1] + ih)
            ov = lambda a, b: max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
            side = 0.02 if (c[0] < 0.5) == (cx < 0.5) else 0.0          # prefer the opposite side
            return sum(ov(r, q) for q in rects) * 4 + sum(ov(r, q) for q in taken) * 40 + side + abs(c[1] + ih / 2 - cy) * 0.05
        c = min(cands, key=cost)
        taken.append((c[0], c[1], c[0] + iw, c[1] + ih)); res.append((c[0], c[1], iw, ih))
    return res


def draw_box(img, x0, y0, x1, y1, col, a):
    if a <= 0:
        return
    ov = img.copy()
    cv2.rectangle(ov, (int(x0), int(y0)), (int(x1), int(y1)), col, LINE, cv2.LINE_AA)
    mx, my, t = (x0 + x1) / 2, (y0 + y1) / 2, 16
    for (p, q) in [((mx, y0), (mx, y0 + t)), ((mx, y1), (mx, y1 - t)), ((x0, my), (x0 + t, my)), ((x1, my), (x1 - t, my))]:
        cv2.line(ov, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), col, LINE, cv2.LINE_AA)
    cv2.addWeighted(ov, a, img, 1 - a, 0, img)


def pop(t):
    """0→1 with a small overshoot over ~0.16s."""
    p = c01(t / 0.16)
    return p, 1 + 0.12 * math.sin(p * math.pi) * (1 - p * 0.4) - 0.12 * (1 - p)


def render(src, out):
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17",
                            "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    total = 0
    for o, (clip, ss, speed, col, boxes) in enumerate(OUTFITS):
        path = f"{src}/{clip}.MOV"
        dur = SEG * speed + 0.3
        paths = track(path, ss, dur, boxes)
        frames = list(decode(path, ss, dur))
        H, W = frames[0].shape[:2]
        insets = place_insets(boxes)
        n = round((o + 1) * SEG * FPS) - round(o * SEG * FPS)
        for i in range(n):
            t = i / FPS
            si = min(round(t * speed * FPS), len(frames) - 1)
            f = frames[si]
            img = cv2.resize(f, (OW, OH), interpolation=cv2.INTER_AREA)
            # punch-in on the outfit's first beat
            z = 1 + 0.035 * math.exp(-(t / 0.12) ** 2)
            if z > 1.001:
                M = cv2.getRotationMatrix2D((OW / 2, OH / 2), 0, z)
                img = cv2.warpAffine(img, M, (OW, OH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            for k, (name, _, _, bw, bh) in enumerate(boxes):
                tk = t - (k + 1) * BEAT                     # exact beat
                if tk < 0:
                    continue
                a, s = pop(tk)
                cx, cy = paths[k][min(si, len(paths[k]) - 1)]
                # focus box on the garment (shrinks into place)
                bs = 1 + 0.25 * (1 - ease_out(a))
                x0, y0 = (cx - bw * bs / 2) * OW, (cy - bh * bs / 2) * OH
                draw_box(img, x0, y0, x0 + bw * bs * OW, y0 + bh * bs * OH, col, a)
                # live magnified inset from the full-res frame
                ix, iy, iw, ih = insets[k]
                sx0, sy0 = int(max((cx - bw / 2) * W, 0)), int(max((cy - bh / 2) * H, 0))
                crop = f[sy0:int(min(sy0 + bh * H, H)), sx0:int(min(sx0 + bw * W, W))]
                if crop.size == 0:
                    continue
                pw, ph = int(iw * OW * s), int(ih * OH * s)
                patch = cv2.resize(crop, (pw, ph), interpolation=cv2.INTER_CUBIC)
                cv2.rectangle(patch, (0, 0), (pw - 1, ph - 1), col, LINE, cv2.LINE_AA)
                px = int(ix * OW + (iw * OW - pw) / 2); py = int(iy * OH + (ih * OH - ph) / 2)
                x0c, y0c = max(px, 0), max(py, 0)
                x1c, y1c = min(px + pw, OW), min(py + ph, OH)
                region = img[y0c:y1c, x0c:x1c]
                p = patch[y0c - py:y1c - py, x0c - px:x1c - px]
                # soft shadow then the inset
                sh = img[min(y0c + 8, OH - 1):min(y1c + 8, OH), min(x0c + 8, OW - 1):min(x1c + 8, OW)]
                sh[:] = (sh * (1 - 0.25 * a)).astype(np.uint8)
                region[:] = (region * (1 - a) + p * a).astype(np.uint8)
            enc.stdin.write(grade(img.astype(np.float32)).tobytes())
            total += 1
        print(clip, n, "frames; insets", [tuple(round(v, 2) for v in r) for r in insets], flush=True)
    enc.stdin.close(); enc.wait()
    print("total", total)


if __name__ == "__main__":
    render(sys.argv[1], sys.argv[2])
