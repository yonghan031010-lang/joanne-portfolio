"""v4 transitions: speed ramps with real motion blur (sub-frame sampling of both the
source time and the virtual camera), zoom punch + shake on the cut, and a per-frame
wall matte for the gallery wipe.

usage: python3 vcam2.py gallery|boots <source_dir> <out.mp4>
"""
import sys, json, math, subprocess
import numpy as np, cv2
from vcam import (OW, OH, FPS, BEAT, BAR, read_frames, warp, clamp_cam, grade,
                  c01, lerp, ease_in, ease_out, shake)


class Clip:
    """Decoded range of a clip, sampled at fractional source times."""
    def __init__(self, path, ss, dur):
        self.ss = ss
        self.f = read_frames(path, ss, dur)
        self.h, self.w = self.f.shape[1:3]

    def at(self, t):
        x = min(max((t - self.ss) * FPS, 0), len(self.f) - 1.001)
        i = int(x); a = x - i
        return self.f[i].astype(np.float32) * (1 - a) + self.f[i + 1].astype(np.float32) * a


def remap(v, n, t0=0.0):
    """Integrate a speed curve v(t) into source times for every 1/(FPS*8) step."""
    dt = 1 / (FPS * 8)
    ts = np.arange(0, n / FPS + 0.5, dt)
    s = np.concatenate([[0], np.cumsum([v(t) * dt for t in ts[:-1]])]) + t0
    return lambda t: float(np.interp(t, ts, s))


def bump(t, c, w):
    return math.exp(-((t - c) / w) ** 2)


def punch(t, c, amp=0.06, w=0.16, shake_px=12):
    """Global zoom punch + decaying shake centred on the cut."""
    z = 1 + amp * bump(t, c, w)
    k = shake_px * c01(1 - abs(t - c) / 0.35)
    return z, (k * math.sin(t * 97.0), k * 0.8 * math.cos(t * 83.0))


def post(img, z, d):
    if z == 1 and d == (0, 0):
        return img
    return warp(img, z, (OW / 2 - d[0] / z, OH / 2 - d[1] / z), (OW / 2, OH / 2))


def encode(frames, out):
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{OW}x{OH}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                            "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", out],
                           stdin=subprocess.PIPE)
    for f in frames:
        enc.stdin.write(grade(f).tobytes())
    enc.stdin.close(); enc.wait()


def match_stats(src, ref, k):
    """Partial per-channel mean/std transfer so the gallery plate sits in 4305's light."""
    m1, s1 = src.reshape(-1, 3).mean(0), src.reshape(-1, 3).std(0)
    m2, s2 = ref.reshape(-1, 3).mean(0), ref.reshape(-1, 3).std(0)
    gain = lerp(np.ones(3), s2 / s1, k); off = lerp(np.zeros(3), m2 - m1 * s2 / s1, k)
    return gain.astype(np.float32), off.astype(np.float32)


# ---------------------------------------------------------------- gallery

def gallery(src, edge_json):
    n = round(2 * BAR * FPS)
    TM = 3 * BEAT                                   # wall edge crosses mid-frame on beat 4
    vA = lambda t: 0.75 if t < 1.25 else lerp(0.75, 2.2, (lambda p: p * p * (3 - 2 * p))(c01((t - 1.25) / (TM - 1.25))))
    sA = remap(vA, n)
    off = 2.30 - sA(TM)                             # src 2.30 = edge at x≈545
    sA_ = lambda t: sA(t) + off
    A = Clip(f"{src}/IMG_4305.MOV", 0.0, 3.9)

    # wall edge (vertical, sub-pixel) vs 4305 source time; linear lead-in before detection starts
    ed = [(t, x) for t, x in json.load(open(edge_json)) if x is not None and x < 1080]
    et, ex = [p[0] for p in ed], [p[1] for p in ed]
    v0 = (ex[1] - ex[0]) / (et[1] - et[0])
    t_in = et[0] + (1080 - ex[0]) / v0
    v1 = (ex[-1] - ex[-2]) / (et[-1] - et[-2])
    et = [t_in] + et + [et[-1] - ex[-1] / v1]; ex = [1080.0] + ex + [0.0]
    edge = lambda s: 1e4 if s < et[0] else (-1e4 if s > et[-1] else float(np.interp(s, et, ex)))

    # output times where the wipe starts / ends
    ts = np.arange(n) / FPS
    TS = next(t for t in ts if sA_(t) >= et[0])
    TE = next(t for t in ts if sA_(t) >= et[-1])
    vB = lambda t: 2.2 if t < TE else lerp(2.2, 1.0, ease_out(c01((t - TE) / 0.55)))
    sB0 = remap(vB, n)
    sB = lambda t: sB0(t) - sB0(TE) + 1.70             # she reaches the left edge right as the wall clears
    B = Clip(f"{src}/IMG_4719.MOV", 0.3, 5.2)
    gB, oB = match_stats(B.at(2.0), A.at(2.6), 0.35)
    ZB = 1.12
    rng = np.random.default_rng(7)

    def plate(t):
        img = B.at(sB(t)) * gB + oB
        L = 40 * ease_out(c01((t - TS) / (TE - TS + 0.5)))       # camera keeps panning, then settles
        k = ZB * OW / B.w
        P = (L + OW / 2 / k, B.h / 2)
        img = warp(img, ZB, clamp_cam(ZB, P, (OW / 2, OH / 2), B.w, B.h), (OW / 2, OH / 2))
        # the plate inherits the pan's directional blur, fading once the wall has cleared
        w = int(70 * (1 - ease_out(c01((t - TE) / 0.35))) if t > TS else 0)
        return cv2.blur(img, (w, 1)) if w > 2 else img

    xs = np.arange(OW, dtype=np.float32)[None, :, None]
    frames = []
    for i in range(n):
        t = i / FPS
        shutter = 0.5 + 0.7 * bump(t, TM, 0.25)                   # wider shutter at the peak of the ramp
        K = 10 if TS - 0.1 < t < TE + 0.3 else 4
        acc = np.zeros((OH, OW, 3), np.float32)
        for j in range(K):
            tj = t + (j / (K - 1) - 0.5) * shutter / FPS
            sa = sA_(tj)
            e = edge(sa)
            if e >= 1080:
                img = A.at(sa)
            else:
                pb = plate(tj)
                if e <= -50:
                    img = pb
                else:
                    ao = 1 - 0.28 * np.exp(-np.clip(xs - e, 0, None) / 45)   # pillar's contact shadow
                    m = np.clip((xs - e) / 2.5 + 0.5, 0, 1)
                    img = A.at(sa) * (1 - m) + pb * ao * m
            z, d = punch(tj, TM)
            acc += post(img, z, d)
        img = acc / K
        if TS - 0.05 < t:                                          # match phone-sensor grain on the plate
            img += rng.normal(0, 2.2, (OH // 2, OW // 2, 1)).astype(np.float32).repeat(2, 0).repeat(2, 1)
        frames.append(img)
    return frames


# ---------------------------------------------------------------- boots

def boots(src):
    n = round(BAR * FPS)
    ZMAX, ZEND, P0 = 12.0, 1.04, BAR - 2 * BEAT
    anchorA = (0.444 * OW, 0.661 * OH)
    A = Clip(f"{src}/IMG_4278.MOV", 0.0, 3.0)
    # 4306: leaves the black at real speed (boot still in place), surges to 2.4x, settles back to 1x
    vB = lambda t: lerp(1.0, 2.4, ease_in(c01(t / 0.35))) if t < 0.35 else lerp(2.4, 1.0, ease_out(c01((t - 0.35) / 1.0)))
    sB = remap(vB, n)
    B = Clip(f"{src}/IMG_4306.MOV", 0.0, 3.9)
    track = [(0.0, 418, 1340), (0.067, 408, 1358), (0.133, 394, 1374), (0.2, 378, 1394)]
    tt, tx, ty = [k[0] for k in track], [k[1] for k in track], [k[2] for k in track]
    TC = BAR                                                       # the cut sits on the bar line

    def camA(t):
        p = c01((t - P0) / (2 * BEAT))
        return ZMAX ** ease_in(p), anchorA, anchorA

    def camB(t):
        p = c01(t / BEAT)
        z = ZEND * (ZMAX / ZEND) ** ((1 - p) ** 2)
        st = min(sB(t), 0.2)
        P = (float(np.interp(st, tt, tx)), float(np.interp(st, tt, ty)))
        Sc = (lerp(anchorA[0], P[0], ease_out(p)), lerp(anchorA[1], P[1], ease_out(p)))
        return z, P, Sc

    def shot(clip, cam, tsrc, t):
        z, P, Sc = cam(t)
        return warp(clip.at(tsrc(t)), z, clamp_cam(z, P, Sc, clip.w, clip.h), Sc)

    frames = []
    for i in range(2 * n):
        t = i / FPS
        shutter = 0.5 + 1.0 * bump(t, TC, 0.3)
        K = 12 if abs(t - TC) < 0.6 else 4
        acc = np.zeros((OH, OW, 3), np.float32)
        for j in range(K):
            tj = t + (j / (K - 1) - 0.5) * shutter / FPS
            if tj < TC:
                img = shot(A, camA, lambda x: x, tj) * lerp(0.08, 1, ease_out(c01((TC - tj) / 0.35)))
            else:
                tb = tj - TC
                img = shot(B, camB, lambda x: sB(x), tb) * lerp(0.08, 1, ease_in(c01(tb / 0.45)))
            z, d = punch(tj, TC, amp=0.04, w=0.12, shake_px=14)
            acc += post(img, z, d)
        frames.append(acc / K)
    return frames


if __name__ == "__main__":
    which, src, out = sys.argv[1], sys.argv[2], sys.argv[3]
    if which == "gallery":
        frames = gallery(src, sys.argv[4])
    else:
        frames = boots(src)
    encode(frames, out)
    print(which, len(frames), "frames")
