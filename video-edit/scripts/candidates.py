"""Numbered candidate boxes on still frames, for picking which items get a call-out."""
import subprocess, sys, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
clip, out = sys.argv[1], sys.argv[2]
spec = json.loads(sys.argv[3])          # [{"t":0.3,"boxes":[[n,cx,cy,w,h],...]}, ...]
font = ImageFont.truetype("/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf", 34)
tiles = []
for s in spec:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(s["t"]), "-i", clip, "-frames:v", "1",
                          "-vf", "scale=720:1280", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    im = Image.fromarray(np.frombuffer(raw, np.uint8).reshape(1280, 720, 3).copy())
    d = ImageDraw.Draw(im)
    for n, cx, cy, w, h in s["boxes"]:
        x0, y0, x1, y1 = (cx - w / 2) * 720, (cy - h / 2) * 1280, (cx + w / 2) * 720, (cy + h / 2) * 1280
        d.rectangle([x0, y0, x1, y1], outline=(255, 214, 0), width=3)
        tb = d.textbbox((0, 0), str(n), font=font)
        d.rectangle([x0, y0 - (tb[3] + 8), x0 + tb[2] + 12, y0], fill=(255, 214, 0))
        d.text((x0 + 6, y0 - tb[3] - 6), str(n), font=font, fill=(0, 0, 0))
    d.text((14, 12), f"t={s['t']}s", font=font, fill=(255, 0, 80))
    tiles.append(im)
W = sum(t.width for t in tiles); canvas = Image.new("RGB", (W + 10 * (len(tiles) - 1), 1280), (0, 0, 0))
x = 0
for t in tiles:
    canvas.paste(t, (x, 0)); x += t.width + 10
canvas.save(out, quality=90)
