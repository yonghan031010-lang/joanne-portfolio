"""Golden-hour run with IMG_4291 in the middle, graded like the reference (cool shadows,
warm highlights, softer saturation, grain, vignette) and no added light:

  beats 0-2  IMG_3994 dusk street, slow push
  beats 2-3  IMG_4291 her face in low sun; the real lens flare peaks on beat 3
  beat 3     cut out of the flare into IMG_4092, exposure settling like a phone after facing the sun
  beats 3-6  IMG_4092 sunlit sandstone; a whip pan carries the real lamp post across frame and
             IMG_4490 is revealed behind it on beat 6
  beats 6-8  IMG_4490 Christ Church in golden light

usage: python3 golden.py <source_dir> <out.mp4>
"""
import sys, subprocess, math
import numpy as np, cv2
from vcam import OW, OH, FPS, BEAT, c01, lerp, ease_in, ease_out
from vcam2 import Clip

POLE_X = 0.09          # lamp post in IMG_4092 (frame fraction), near the left edge
POLE_HALF = 0.035      # half width of the post
FLARE = 0.80           # IMG_4291 time of the brightest sun flare
rng = np.random.default_rng(3)
YY, XX = np.mgrid[0:OH, 0:OW].astype(np.float32)
VIGNETTE = (1 - 0.28 * (((XX / OW - 0.5) * 1.3) ** 2 + ((YY / OH - 0.5) * 0.9) ** 2) * 2.2).clip(0.6, 1)[..., None]


def film(img):
    """Reference look: cool shadows, warm highlights, gentle S-curve, lower saturation, grain, vignette."""
    x = np.clip(img, 0, 255) / 255.0
    lum = x @ np.float32([0.299, 0.587, 0.114])
    x = lum[..., None] + (x - lum[..., None]) * 0.80
    sh = (1 - lum)[..., None] ** 2; hi = lum[..., None] ** 2
    x = x + sh * np.float32([-0.020, 0.010, 0.035]) + hi * np.float32([0.040, 0.012, -0.035])
    x = np.clip(x, 0, 1)
    x = x + 0.10 * (x - 0.5) * (1 - np.abs(2 * x - 1))           # soft S-curve
    x = x * VIGNETTE
    g = rng.normal(0, 0.012, (OH // 2, OW // 2, 1)).astype(np.float32).repeat(2, 0).repeat(2, 1)
    return (np.clip(x + g, 0, 1) * 255).astype(np.uint8)


def view(img, cx, cy, z, sx, sy):
    h, w = img.shape[:2]
    k = z * OW / w
    M = np.float32([[k, 0, sx * OW - k * cx * w], [0, k, sy * OH - k * cy * h]])
    return cv2.warpAffine(img, M, (OW, OH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def main(src, out):
    S1 = Clip(f"{src}/IMG_3994.MOV", 0.0, 1.6)
    S2 = Clip(f"{src}/IMG_4291.MOV", 0.0, 1.0)
    S3 = Clip(f"{src}/IMG_4092.MOV", 2.0, 2.4)
    S4 = Clip(f"{src}/IMG_4490.MOV", 0.3, 1.8)
    B = [k * BEAT for k in range(9)]
    n = round(B[8] * FPS)
    WHIP = 0.32                                                   # length of the whip pan before beat 6

    def shot(t):
        if t < B[2]:                                              # dusk street
            p = t / B[2]
            return view(S1.at(t), 0.5, 0.45, lerp(1.0, 1.08, p), 0.5, 0.45)
        if t < B[3]:                                              # her, flare on beat 3
            s = FLARE - (B[3] - t)
            return view(S2.at(s), 0.5, 0.5, lerp(1.0, 1.05, (t - B[2]) / BEAT), 0.5, 0.5)
        s3 = 4.36 - (B[6] - t)                                    # IMG_4092 runs 1x into the whip
        if t < B[6] - WHIP:
            img = view(S3.at(s3), 0.5, 0.5, 1.12, 0.5, 0.5)
            settle = c01((t - B[3]) / 0.35)                       # exposure comes down after the flare
            return img * lerp(1.65, 1.0, ease_out(settle))
        # whip: post travels from its spot to beyond the right edge; IMG_4490 fills in behind it
        p = ease_in(c01((t - (B[6] - WHIP)) / WHIP))
        sx_pole = lerp(POLE_X, 1.25, p)                           # where the post is on screen
        z = 1.12
        a = view(S3.at(s3), POLE_X, 0.5, z, sx_pole, 0.5)
        # IMG_4490 slides in with the same screen velocity, then settles after the beat
        pan_px = (sx_pole - POLE_X) * OW
        b = view(S4.at(0.3 + max(t - (B[6] - WHIP), 0)), 0.5, 0.45, 1.10, 0.5 + (pan_px - 1.15 * OW) / OW, 0.45)
        edge = sx_pole * OW
        half = POLE_HALF * OW * z
        reveal = np.clip((edge - half - XX) / 6 + 0.5, 0, 1)[..., None]     # left of the post = new scene
        return a * (1 - reveal) + b * reveal

    def after(t):                                                 # IMG_4490 finishing its pan, then holding
        p = ease_out(c01((t - B[6]) / 0.45))
        off = lerp(-0.15 * 1.0, 0.0, p)
        return view(S4.at(0.3 + WHIP + (t - B[6])), 0.5, 0.45, lerp(1.10, 1.16, (t - B[6]) / (B[8] - B[6])), 0.5 + off, 0.45)

    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17",
                            "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    for i in range(n):
        t = i / FPS
        whip = (B[6] - WHIP - 0.05) < t < (B[6] + 0.3)
        K = 10 if whip else 2
        sh = 1.6 if whip else 0.5
        acc = np.zeros((OH, OW, 3), np.float32)
        for j in range(K):
            tj = t + (j / max(K - 1, 1) - 0.5) * sh / FPS
            acc += shot(tj) if tj < B[6] else after(tj)
        enc.stdin.write(film(acc / K).tobytes())
    enc.stdin.close(); enc.wait()
    print("frames", n)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
