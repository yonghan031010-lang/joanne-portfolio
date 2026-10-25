"""E v2: night street → (glow bounce-in crowd) → Dominion → RUNWAY letters as windows → stage.

  beats 0-2  IMG_4196 night street (slit opens from black)
  beat 2     the queue from IMG_4817, cut out, bounces up into frame with a soft white glow
             (overshoot spring); under the crowd the Dominion billboard replaces the street
  beats 3-5  IMG_4817 pushes toward the billboard
  beat 5     IMG_4823: the red RUNWAY letters turn into windows onto the stage (IMG_4834)
  beats 6-8  push into the letters until the stage fills the frame

usage: python3 night2.py <source_dir> <out.mp4>
"""
import sys, subprocess, math
import numpy as np, cv2, mediapipe as mp
from vcam import OW, OH, FPS, BEAT, c01, lerp, ease_in, ease_out
from vcam2 import Clip
from dome import view, segmenter
from golden import film

LETTERS_Y = (0.24, 0.46)     # rows of the RUNWAY letters (IMG_4823)
DIVE = (0.47, 0.35)          # inside the "N" of RUNWAY — where we push through
STAGE = (0.50, 0.55)
YY = np.arange(OH, dtype=np.float32)[:, None, None]


def smooth(p):
    p = c01(p); return p * p * (3 - 2 * p)


def spring(p):
    """0→1 with a bouncy overshoot (damped spring)."""
    p = c01(p)
    return 1 - math.exp(-6 * p) * math.cos(9 * p)


def crowd_mask(frame, seg):
    r = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame)))
    m = (np.squeeze(r.category_mask.numpy_view()) > 0).astype(np.float32)
    return cv2.GaussianBlur(m, (0, 0), 1.5)


def letter_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    m = (((h <= 8) | (h >= 170)) & (s > 110) & (v > 110)).astype(np.uint8)
    H = m.shape[0]
    m[:int(LETTERS_Y[0] * H)] = 0; m[int(LETTERS_Y[1] * H):] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    return cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.2)


def main(src, out):
    S1 = Clip(f"{src}/IMG_4196.MOV", 0.6, 1.8)
    S2 = Clip(f"{src}/IMG_4817.MOV", 0.3, 2.0)
    S3 = Clip(f"{src}/IMG_4823.MOV", 0.0, 2.0)
    S4 = Clip(f"{src}/IMG_4834.MOV", 0.6, 3.0)
    Z4 = (OH / S4.h) * S4.w / OW
    seg = segmenter()
    crowd = [crowd_mask(f, seg) for f in S2.f]
    letters = [letter_mask(f) for f in S3.f]
    BT = [k * BEAT for k in range(9)]
    n = round(BT[8] * FPS)
    sS2 = lambda t: 0.3 + max(t - (BT[2] - 0.2), 0)                # 4817 clock starts as the crowd rises
    i2 = lambda t: min(max(round((sS2(t) - S2.ss) * FPS), 0), len(S2.f) - 1)
    sS3 = lambda t: 0.1 + max(t - BT[5], 0)
    i3 = lambda t: min(max(round((sS3(t) - S3.ss) * FPS), 0), len(S3.f) - 1)

    def street(t):
        return view(S1.at(0.8 + t), 0.5, 0.5, lerp(1.0, 1.1, t / BT[3]), 0.5, 0.5, 0.0)

    def dominion(t, z_extra=1.0):
        p = smooth((t - BT[2]) / (BT[5] - BT[2]))
        z = lerp(1.0, 1.35, p) * z_extra
        return view(S2.at(sS2(t)), 0.5, lerp(0.5, 0.36, p), z, 0.5, 0.5, 0.0)

    def frame(t):
        if t < BT[2] - 0.2:
            return street(t)
        if t < BT[3]:
            # crowd bounces up from below with overshoot, glowing; billboard appears behind it
            p = (t - (BT[2] - 0.2)) / 0.55
            k = spring(p)
            sc = lerp(0.55, 1.0, k)
            dy = lerp(0.55, 0.0, k)                                  # starts low, springs up
            fr = S2.at(sS2(t))
            m = crowd[i2(t)]
            hgt, wid = m.shape
            # place crowd layer: scale about bottom centre, then offset down
            M = cv2.getRotationMatrix2D((wid / 2, hgt), 0, sc)
            M[1, 2] += dy * hgt
            fl = cv2.warpAffine(fr, M, (wid, hgt), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            ml = cv2.warpAffine(m, M, (wid, hgt), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            layer = view(fl, 0.5, 0.5, 1.0, 0.5, 0.5, 0.0)
            ml = view(ml[..., None].repeat(3, 2), 0.5, 0.5, 1.0, 0.5, 0.5, 0.0)[..., :1]
            # soft white glow hugging the cut-out
            halo = cv2.GaussianBlur(ml[..., 0], (0, 0), 12)[..., None]
            glow = np.clip(halo - ml, 0, 1) * 1.4 * (1 - ml) * (1 - 0.7 * smooth((t - BT[2]) / 0.5))   # outside the people only
            back_mix = smooth((t - (BT[2] - 0.02)) / 0.12)            # billboard replaces the street under the crowd
            back = street(t) * (1 - back_mix) + dominion(t) * back_mix
            img = back * (1 - ml) + layer * ml
            return img + glow * np.float32([255, 248, 235]) * 0.6
        if t < BT[5]:
            return dominion(t)
        # RUNWAY: letters are windows onto the stage, then we dive through
        p = ease_in(c01((t - BT[6]) / (BT[7] - BT[6])))
        z = lerp(1.0, 1.08, c01((t - BT[5]) / BEAT)) * (1 + 7.0 * p)
        a = view(S3.at(sS3(t)), DIVE[0], DIVE[1], z, 0.5, lerp(0.5, DIVE[1], 0.0), 0.0)
        lm = view(letters[i3(t)][..., None].repeat(3, 2), DIVE[0], DIVE[1], z, 0.5, 0.5, 0.0)[..., :1]
        open_ = smooth((t - BT[5]) / 0.35)                           # letters turn into windows
        full = smooth((t - (BT[7] - 0.15)) / 0.30)                   # past the letter: stage fills the frame
        st = view(S4.at(1.0 + (t - BT[5])), STAGE[0], STAGE[1], Z4 * lerp(1.25, 1.0, smooth((t - BT[6]) / (2 * BEAT))), 0.5, 0.5, 0.0)
        w = np.clip(lm * open_ + full, 0, 1)
        rim = np.clip(cv2.GaussianBlur(lm[..., 0], (0, 0), 6)[..., None] * 1.8 - lm, 0, 1) * open_ * (1 - full)
        img = a * (1 - w) + st * w
        return img + rim * np.float32([255, 40, 50]) * 0.6           # the neon still glows around each window

    cut2, cut5 = BT[2], BT[5]
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17",
                            "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    for i in range(n):
        t = i / FPS
        busy = abs(t - cut2) < 0.4 or abs(t - BT[7]) < 0.35
        K = 6 if busy else 2
        sh = 0.45 if abs(t - cut2) < 0.4 else 0.9 if busy else 0.5     # short shutter on the bounce so the crowd stays solid
        acc = np.zeros((OH, OW, 3), np.float32)
        for j in range(K):
            tj = t + (j / max(K - 1, 1) - 0.5) * sh / FPS
            acc += frame(tj)
        img = acc / K
        if t < BT[1]:                                              # slit opens from black
            half = ease_out(c01(t / BT[1])) * OH / 2
            img = img * np.clip((half - np.abs(YY - OH / 2)) / 3 + 0.5, 0, 1)
        enc.stdin.write(film(img).tobytes())
    enc.stdin.close(); enc.wait()
    print("frames", n)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
