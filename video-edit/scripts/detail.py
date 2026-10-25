"""Outfit detail call-outs (focus boxes + live magnified insets), as in the reference.

Each outfit plays for 7 beats (≈5s). Beat 1 is clean, then one detail pops in on each of
beats 2-5 and holds to the end. Boxes follow the garment with
optical-flow tracking; insets are cropped live from the full-resolution source.

usage: python3 detail.py <source_dir> <out.mp4>              (all OUTFITS, back to back)
       python3 detail.py <source_dir> <out.mp4> <callout.json>  (one clip, labels under insets)
"""
import sys, subprocess, json, math
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
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
FONT = ImageFont.truetype(__file__.rsplit("/", 2)[0] + "/fonts/PressStart2P-Regular.ttf", 16)  # 8px grid ×2, crisp
TAG_PAD = 8
TAG_H = 16 + 2 * TAG_PAD               # px
LABEL_H = TAG_H / OH                   # room above an inset for its tag


def load_config(path):
    c = json.load(open(path))
    boxes = [(it["label"], it["cx"], it["cy"], it["w"], it["h"]) for it in c["items"]]
    global FIXED_INSETS
    FIXED_INSETS = [it.get("inset") for it in c["items"]]          # optional [x, y] per item
    global ITEM_OPTS
    ITEM_OPTS = [{k: it[k] for k in ("keys", "freeze", "box_off") if k in it} for it in c["items"]]
    if "inset_w" in c:
        global INSET_W
        INSET_W = c["inset_w"]
    global LINE_COLOR
    LINE_COLOR = tuple(c["line_color"]) if "line_color" in c else None   # lines brighter than tags if set
    return [(c["clip"], c["start"], c["speed"], tuple(c["color"]), boxes)], c["beats"] * BEAT, True


LINE_COLOR = None


FIXED_INSETS = None
ITEM_OPTS = None      # per item: keys [[t, cx, cy]...] (manual path, source-relative s),
                      # freeze t (inset holds the frame at t), box_off t (box fades out after t)


def tag_width(text):
    return int(FONT.getlength(text)) + 2 * TAG_PAD


def draw_tags(img, tags):
    """Filled name tags (white pixel text) sitting on the top-left edge of a box.
    tags: [(text, x_left, y_bottom, colour, alpha)]"""
    if not tags:
        return
    im = Image.fromarray(img).convert("RGBA"); ov = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov); d.fontmode = "1"                   # no anti-aliasing: hard pixel edges
    for text, x, yb, col, a in tags:
        w = tag_width(text)
        x = int(min(max(x, 0), OW - w)); y = int(yb - TAG_H)
        if y < 0:                                                  # no room above: tuck inside the box
            y = int(yb)
        d.rectangle([x, y, x + w, y + TAG_H], fill=(*col, int(255 * a)))
        d.text((x + TAG_PAD, y + TAG_PAD), text, font=FONT, fill=(255, 255, 255, int(255 * a)))
    img[:] = np.asarray(Image.alpha_composite(im, ov).convert("RGB"))


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


def place_insets(boxes, label=False):
    """Greedy slot choice: minimise overlap with every box and earlier insets."""
    cands = [(x, y) for x in (0.03, 0.65) for y in np.arange(0.05, 0.80, 0.01)]
    taken, res = [], []
    rects = [(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2) for _, cx, cy, w, h in boxes]
    for (_, cx, cy, w, h) in boxes:
        iw = INSET_W; ih = min(iw * (h * OH) / (w * OW) * OW / OH, 0.22)
        lh = LABEL_H if label else 0
        def cost(c):
            r = (c[0], c[1] - lh, c[0] + iw, c[1] + ih)
            ov = lambda a, b: max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
            side = 0.02 if (c[0] < 0.5) == (cx < 0.5) else 0.0          # prefer the opposite side
            g = 0.018                                                     # breathing room between insets
            rg = (r[0] - g, r[1] - g, r[2] + g, r[3] + g)
            vert = abs(c[1] + ih / 2 - cy) * (0.6 if label else 0.05)     # stay level with the item
            return sum(ov(r, q) for q in rects) * 4 + sum(ov(rg, q) for q in taken) * 40 + side + vert
        fixed = FIXED_INSETS[len(res)] if FIXED_INSETS else None
        c = tuple(fixed) if fixed else min(cands, key=cost)
        taken.append((c[0], c[1] - lh, c[0] + iw, c[1] + ih)); res.append((c[0], c[1], iw, ih))
    return res


def draw_box(img, x0, y0, x1, y1, col, a):
    if a <= 0:
        return
    ov = img.copy()
    cv2.rectangle(ov, (int(x0), int(y0)), (int(x1), int(y1)), col, LINE, cv2.LINE_AA)
    cv2.addWeighted(ov, a, img, 1 - a, 0, img)


def leader(img, box, ins, col, p):
    """Thin line from the garment box to its inset (nearest corners), drawn out over p∈[0,1]."""
    if p <= 0:
        return
    cb = [(box[0], box[1]), (box[2], box[1]), (box[0], box[3]), (box[2], box[3])]
    ci = [(ins[0], ins[1]), (ins[2], ins[1]), (ins[0], ins[3]), (ins[2], ins[3])]
    a, b = min(((u, v) for u in cb for v in ci), key=lambda uv: (uv[0][0] - uv[1][0]) ** 2 + (uv[0][1] - uv[1][1]) ** 2)
    e = (a[0] + (b[0] - a[0]) * p, a[1] + (b[1] - a[1]) * p)
    cv2.line(img, (int(a[0]), int(a[1])), (int(e[0]), int(e[1])), col, 2, cv2.LINE_AA)


def pop(t):
    """0→1 with a small overshoot over ~0.16s."""
    p = c01(t / 0.16)
    return p, 1 + 0.12 * math.sin(p * math.pi) * (1 - p * 0.4) - 0.12 * (1 - p)


def render(src, out, outfits=None, seg=None, labels=False):
    outfits = outfits or OUTFITS; seg = seg or SEG
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17",
                            "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    total = 0
    for o, (clip, ss, speed, col, boxes) in enumerate(outfits):
        path = f"{src}/{clip}.MOV"
        dur = seg * speed + 0.3
        paths = track(path, ss, dur, boxes)
        frames = list(decode(path, ss, dur))
        H, W = frames[0].shape[:2]
        insets = place_insets(boxes, labels)
        lcol = LINE_COLOR or col
        n = round((o + 1) * seg * FPS) - round(o * seg * FPS)
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
            tags = []
            for k, (name, _, _, bw, bh) in enumerate(boxes):
                tk = t - (k + 1) * BEAT                     # exact beat
                if tk < 0:
                    continue
                a, s = pop(tk)
                opt = ITEM_OPTS[k] if ITEM_OPTS else {}
                pos = lambda j: (tuple(np.interp(j / FPS, [q[0] for q in opt["keys"]], [q[i] for q in opt["keys"]]) for i in (1, 2))
                                 if "keys" in opt else tuple(paths[k][min(j, len(paths[k]) - 1)]))
                cx, cy = pos(si)
                ab = a * (1 - c01((si / FPS - opt["box_off"]) / 0.2)) if "box_off" in opt else a
                # focus box on the garment (shrinks into place)
                bs = 1 + 0.25 * (1 - ease_out(a))
                x0, y0 = (cx - bw * bs / 2) * OW, (cy - bh * bs / 2) * OH
                draw_box(img, x0, y0, x0 + bw * bs * OW, y0 + bh * bs * OH, lcol, ab)
                ix, iy, iw, ih = insets[k]
                if labels and ab > 0:
                    lp = ease_out(c01(tk / 0.22)) * (ab / max(a, 1e-6))
                    leader(img, (x0, y0, x0 + bw * bs * OW, y0 + bh * bs * OH),
                           (ix * OW, iy * OH, (ix + iw) * OW, (iy + ih) * OH), lcol, lp)
                    tags.append((name, x0 - LINE / 2, y0 + LINE / 2, col, ab))
                # live magnified inset from the full-res frame, or a held frame
                if "freeze" in opt:
                    fj = min(round(opt["freeze"] * FPS), len(frames) - 1)
                    fsrc = frames[fj]; cx, cy = pos(fj)
                else:
                    fsrc = f
                sx0, sy0 = int(max((cx - bw / 2) * W, 0)), int(max((cy - bh / 2) * H, 0))
                crop = fsrc[sy0:int(min(sy0 + bh * H, H)), sx0:int(min(sx0 + bw * W, W))]
                if crop.size == 0:
                    continue
                pw, ph = int(iw * OW * s), int(ih * OH * s)
                patch = cv2.resize(crop, (pw, ph), interpolation=cv2.INTER_CUBIC)
                cv2.rectangle(patch, (0, 0), (pw - 1, ph - 1), lcol, LINE, cv2.LINE_AA)
                px = int(ix * OW + (iw * OW - pw) / 2); py = int(iy * OH + (ih * OH - ph) / 2)
                x0c, y0c = max(px, 0), max(py, 0)
                x1c, y1c = min(px + pw, OW), min(py + ph, OH)
                region = img[y0c:y1c, x0c:x1c]
                p = patch[y0c - py:y1c - py, x0c - px:x1c - px]
                if not labels:                                   # soft shadow (v1 look only)
                    sh = img[min(y0c + 8, OH - 1):min(y1c + 8, OH), min(x0c + 8, OW - 1):min(x1c + 8, OW)]
                    sh[:] = (sh * (1 - 0.25 * a)).astype(np.uint8)
                region[:] = (region * (1 - a) + p * a).astype(np.uint8)
            draw_tags(img, tags)
            enc.stdin.write(grade(img.astype(np.float32)).tobytes())
            total += 1
        print(clip, n, "frames; insets", [tuple(round(v, 2) for v in r) for r in insets], flush=True)
    enc.stdin.close(); enc.wait()
    print("total", total)


if __name__ == "__main__":
    if len(sys.argv) > 3:
        outfits, seg, labels = load_config(sys.argv[3])
        render(sys.argv[1], sys.argv[2], outfits, seg, labels)
    else:
        render(sys.argv[1], sys.argv[2])
