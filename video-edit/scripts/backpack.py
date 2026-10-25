"""Black-backpack fill (v2: the camera first focuses on the train ticket and holds it, then racks
over to the backpack): in IMG_4225 a man's black backpack sweeps right→left past the lens.
The camera chases it and pushes in until it fills the frame (pure black on the beat), then
IMG_4226 opens out of the dark corner of the train window as the train is already moving.

usage: python3 backpack.py <source_dir> <out.mp4>
"""
import sys, subprocess, math
import numpy as np, cv2
from vcam import OW, OH, FPS, BEAT, grade, c01, lerp, ease_in, ease_out, warp, clamp_cam
from vcam2 import Clip

# darkest-patch centre of the backpack (IMG_4225 source time, frame fractions), measured
PACK = [(0.80, 1.00, 0.55), (0.95, 0.83, 0.55), (0.983, 0.674, 0.556), (1.017, 0.533, 0.592),
        (1.05, 0.37, 0.592), (1.083, 0.27, 0.748)]
A_PEAK = 1.02                  # backpack centred and frame-filling
B_DARK = (0.90, 0.90)          # dark interior corner of IMG_4226
ZA, ZB = 3.4, 3.0
# the ticket in her hand (IMG_4225 source time, frame fractions), measured
TICKET = [(0.0, 0.33, 0.80), (0.2, 0.40, 0.80), (0.4, 0.38, 0.87)]
ZT = 1.7                       # ticket close-up
T_HOLD = 0.95                  # ticket held until here (output s), then focus moves to the backpack
YY, XX = np.mgrid[0:OH, 0:OW].astype(np.float32)


def smooth(p):
    p = c01(p); return p * p * (3 - 2 * p)


def main(src, out):
    A = Clip(f"{src}/IMG_4225.MOV", 0.0, 1.4)
    B = Clip(f"{src}/IMG_4226.MOV", 0.0, 3.4)
    TC = 2 * BEAT                               # black fills the frame on beat 2
    n = round(5 * BEAT * FPS)
    # IMG_4225 clock: ~0.3x while the ticket is held, then speeding up into the backpack (hits A_PEAK on TC)
    tt = np.linspace(0, TC, 2000)
    w = np.array([smooth((x - (T_HOLD - 0.2)) / 0.4) for x in tt])
    v_slow = 0.3
    acc_slow = np.cumsum(np.full_like(tt, v_slow) * (1 - w)) * (tt[1] - tt[0])
    acc_fast = np.cumsum(w) * (tt[1] - tt[0])
    v_fast = (A_PEAK - 0.02 - acc_slow[-1]) / acc_fast[-1]
    SA = 0.02 + acc_slow + v_fast * acc_fast
    sA = lambda t: float(np.interp(t, tt, SA)) if t < TC else A_PEAK + (t - TC)
    vB = 1.3
    sB = lambda t: (t - TC) * vB

    def pack(s):
        ts = [p[0] for p in PACK]
        return float(np.interp(s, ts, [p[1] for p in PACK])), float(np.interp(s, ts, [p[2] for p in PACK]))

    def ticket(s):
        ts = [p[0] for p in TICKET]
        return float(np.interp(s, ts, [p[1] for p in TICKET])), float(np.interp(s, ts, [p[2] for p in TICKET]))

    def tk(t):                                   # weight of the ticket framing
        return 1 - smooth((t - T_HOLD) / 0.30)

    def camA(t):
        p = c01((t - (TC - 0.45)) / 0.45)
        z = 1 + (ZA - 1) * ease_in(p)
        px, py = pack(sA(t))
        P = (px * A.w, py * A.h)
        k = ease_in(p)                           # drift the backpack to frame centre as we close in
        Sc = (lerp(px * OW, OW / 2, k), lerp(py * OH, OH / 2, k))
        # ticket close-up first: push in, hold, then hand over to the backpack framing
        q = ease_out(c01(t / 0.30))
        tx, ty = ticket(sA(t))
        zt = 1 + (ZT - 1) * q
        Pt = (tx * A.w, ty * A.h)
        St = (lerp(tx * OW, OW / 2, q), lerp(ty * OH, 0.62 * OH, q))
        w = tk(t)
        return (lerp(z, zt, w), (lerp(P[0], Pt[0], w), lerp(P[1], Pt[1], w)),
                (lerp(Sc[0], St[0], w), lerp(Sc[1], St[1], w)))

    def focus(img, t):
        """Shallow depth of field on the ticket, then a rack: everything softens and the backpack takes focus."""
        dof = c01(t / 0.30) * tk(t)
        rack = math.sin(math.pi * c01((t - T_HOLD) / 0.30))
        if dof < 0.02 and rack < 0.02:
            return img
        soft = cv2.GaussianBlur(img, (0, 0), 9)
        far = np.clip((((XX - OW / 2) / (OW * 0.42)) ** 2 + ((YY - 0.62 * OH) / (OH * 0.22)) ** 2 - 1) / 1.2, 0, 1)[..., None]
        a = np.clip(far * dof + rack * 0.6, 0, 1)
        return img * (1 - a) + soft * a

    def camB(t):
        p = c01((t - TC) / BEAT)
        z = 1 + (ZB - 1) * (1 - ease_out(p))
        P = (B_DARK[0] * B.w, B_DARK[1] * B.h)
        Sc = (lerp(OW / 2, B_DARK[0] * OW, ease_out(p)), lerp(OH / 2, B_DARK[1] * OH, ease_out(p)))
        return z, P, Sc

    def shot(clip, cam, s, t):
        z, P, Sc = cam(t)
        return warp(clip.at(s(t)), z, clamp_cam(z, P, Sc, clip.w, clip.h), Sc)

    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17",
                            "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    for i in range(n):
        t = i / FPS
        near = abs(t - TC) < 0.5
        K = 12 if near else 4
        sh = 0.5 + 1.0 * math.exp(-((t - TC) / 0.25) ** 2)
        acc = np.zeros((OH, OW, 3), np.float32)
        for j in range(K):
            tj = t + (j / (K - 1) - 0.5) * sh / FPS
            if tj < TC:
                img = focus(shot(A, camA, sA, tj), tj) * lerp(0.04, 1, ease_out(c01((TC - tj) / 0.12)))
            else:
                img = shot(B, camB, sB, tj) * lerp(0.04, 1, ease_in(c01((tj - TC) / 0.22)))
            acc += img
        enc.stdin.write(grade(acc / K).tobytes())
    enc.stdin.close(); enc.wait()
    print("frames", n, "IMG_4226 to %.2fs" % sB(n / FPS))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
