"""IMG_4704 → IMG_4781 on St Paul's dome.

Push in on the outfit, rack focus from her (and the room) to the dome through the window,
then a fast clockwise camera roll (real rotational motion blur, no added light) hands over
to IMG_4781 with the dome held at the same place and size, unwinding into the walk.

usage: python3 dome.py <source_dir> <out.mp4>
"""
import sys, subprocess, math
import numpy as np, cv2, mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from vcam import OW, OH, FPS, BEAT, grade, c01, lerp, ease_in, ease_out
from vcam2 import Clip, match_stats

MODEL = __file__.rsplit("/", 2)[0] + "/models/selfie_multiclass.tflite"
HER = (0.55, 0.50)            # her centre in IMG_4704 (frame fractions)
DOME_A = (0.44, 0.23, 0.17)   # dome centre + width in IMG_4704
DOME_B = (0.50, 0.31, 0.30)   # dome centre + width in IMG_4781
SCREEN = (0.50, 0.36)         # where the dome sits on screen through the hand-off
FAR_BAND = (0.17, 0.70)       # rows of window glass (outside view)
ROLL = 28.0                   # degrees at the hand-off


def segmenter():
    return vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=MODEL), output_category_mask=True))


def far_mask(frame, seg):
    """1 where we look through the window at the distance, 0 for people and the room; plus the person mask."""
    r = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame)))
    raw = (np.squeeze(r.category_mask.numpy_view()) > 0).astype(np.uint8)
    person = cv2.dilate(raw, np.ones((9, 9), np.uint8))
    h, w = person.shape
    far = np.zeros((h, w), np.float32)
    far[int(FAR_BAND[0] * h):int(FAR_BAND[1] * h)] = 1
    far[int(0.55 * h):, :int(0.20 * w)] = 0                      # table and bottles, bottom-left
    far *= 1 - person
    return cv2.GaussianBlur(far, (21, 21), 0), raw


def track_dome(frames, i0, cx, cy, size=0.10):
    """Template-track the dome from a seed frame both ways (frame fractions)."""
    W, H = 540, 960
    g = [cv2.cvtColor(cv2.resize(f, (W, H)), cv2.COLOR_RGB2GRAY).astype(np.float32) for f in frames]
    tw, th = int(size * W), int(size * W)
    x, y = cx * W, cy * H
    tmpl = g[i0][int(y - th / 2):int(y + th / 2), int(x - tw / 2):int(x + tw / 2)]
    out = {i0: (cx, cy, 1.0)}
    for step in (1, -1):
        px, py, ps, i = x, y, 1.0, i0 + step
        while 0 <= i < len(g):
            best = (-2, px, py, ps)
            for sc in (ps * 0.97, ps, ps * 1.03):
                t = cv2.resize(tmpl, None, fx=sc, fy=sc)
                R = 30
                x0, y0 = int(max(px - t.shape[1] / 2 - R, 0)), int(max(py - t.shape[0] / 2 - R, 0))
                win = g[i][y0:int(py + t.shape[0] / 2 + R), x0:int(px + t.shape[1] / 2 + R)]
                if win.shape[0] < t.shape[0] or win.shape[1] < t.shape[1]:
                    continue
                r = cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED)
                _, mx, _, loc = cv2.minMaxLoc(r)
                if mx > best[0]:
                    best = (mx, x0 + loc[0] + t.shape[1] / 2, y0 + loc[1] + t.shape[0] / 2, sc)
            _, px, py, ps = best
            out[i] = (px / W, py / H, ps)
            i += step
    a = np.array([out[i] for i in range(len(g))])
    k = np.ones(5) / 5
    return np.stack([np.convolve(np.pad(a[:, j], 2, mode="edge"), k, "valid") for j in range(3)], 1)


def her_track(person_masks, start):
    """Centre of the standing woman: the person blob nearest to where she was last frame."""
    c = start; out = []
    for m in person_masks:
        n, lab, st, cen = cv2.connectedComponentsWithStats(m)
        if n > 1:
            h, w = m.shape
            pts = [(cen[k][0] / w, cen[k][1] / h) for k in range(1, n) if st[k, cv2.CC_STAT_AREA] > 2000]
            if pts:
                c = min(pts, key=lambda p: (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2)
        out.append(c)
    a = np.array(out); k = np.ones(7) / 7
    return np.stack([np.convolve(np.pad(a[:, j], 3, mode="edge"), k, "valid") for j in range(2)], 1)


def dof(img, far, near_px, far_px):
    """Two-layer depth of field: near layer blurred by near_px, far layer by far_px."""
    def blur(x, r):
        return cv2.GaussianBlur(x, (0, 0), r / 2) if r > 0.6 else x
    m = far[..., None]
    return blur(img, near_px) * (1 - m) + blur(img, far_px) * m


def view(img, cx, cy, z, sx, sy, roll):
    """Put source point (cx, cy) [frame fractions] at screen (sx, sy), zoom z, roll degrees."""
    h, w = img.shape[:2]
    k = z * OW / w
    M = cv2.getRotationMatrix2D((cx * w, cy * h), roll, k)
    M[:, 2] += (sx * OW - cx * w, sy * OH - cy * h)
    return cv2.warpAffine(img, M, (OW, OH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


YY, XX = np.mgrid[0:OH, 0:OW].astype(np.float32)


def god_rays(cx, cy, rot, strength):
    th = np.arctan2(YY - cy, XX - cx)
    r = np.hypot(XX - cx, YY - cy) / OH
    rays = (0.5 + 0.5 * np.sin(14 * th + rot)) ** 6 + 0.6 * (0.5 + 0.5 * np.sin(23 * th - 1.7 * rot)) ** 8
    fall = np.exp(-r * 2.2)
    core = np.exp(-(r / 0.08) ** 2)
    return (rays * fall * 0.9 + core * 1.2) * strength


def sweep(pos, strength, angle=-32):
    a = math.radians(angle)
    d = (XX / OW - 0.5) * math.cos(a) + (YY / OH - 0.5) * math.sin(a) * OH / OW - pos
    band = np.exp(-(d / 0.10) ** 2) + 0.35 * np.exp(-((d + 0.16) / 0.03) ** 2)
    return band * strength


def main(src, out):
    A = Clip(f"{src}/IMG_4704.MOV", 0.5, 2.6)
    B = Clip(f"{src}/IMG_4781.MOV", 0.2, 3.0)
    seg = segmenter()
    fm = [far_mask(f, seg) for f in A.f]
    fars = [x[0] for x in fm]
    her = her_track([x[1] for x in fm], HER)
    dome = track_dome(A.f, round((1.5 - A.ss) * FPS), DOME_A[0], DOME_A[1])
    domeB = track_dome(B.f, round((1.0 - B.ss) * FPS), DOME_B[0], DOME_B[1], size=0.16)
    print("IMG_4781 dome x %.2f–%.2f y %.2f–%.2f" % (domeB[:, 0].min(), domeB[:, 0].max(), domeB[:, 1].min(), domeB[:, 1].max()))
    print("dome path x %.2f–%.2f y %.2f–%.2f scale %.2f–%.2f" % (dome[:, 0].min(), dome[:, 0].max(), dome[:, 1].min(),
          dome[:, 1].max(), dome[:, 2].min(), dome[:, 2].max()), flush=True)
    T1, TC, END = BEAT, 2 * BEAT, 4 * BEAT
    n = round(END * FPS)
    sA = lambda t: 0.5 + t * 1.0
    sB = lambda t: 0.2 + (t - TC) * 1.15
    warm = np.float32([1.0, 0.86, 0.62])

    # dome size on screen at the hand-off, so IMG_4781 can match it
    ZA_END = 3.2
    dome_px = DOME_A[2] * ZA_END
    ZB0 = dome_px / DOME_B[2]

    # white balance drifts toward IMG_4781 while the camera spins (no added light)
    gainC, offC = match_stats(A.f[min(round((sA(TC) - A.ss) * FPS), len(A.f) - 1)].astype(np.float32),
                              B.at(sB(TC)), 0.6)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17",
                            "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    for i in range(n):
        t = i / FPS
        K = 16 if abs(t - TC) < 0.3 else 3
        sh = 0.5 + 1.8 * math.exp(-((t - TC) / 0.15) ** 2)      # longer shutter only while spinning
        acc = np.zeros((OH, OW, 3), np.float32)
        for j in range(K):
            tj = t + (j / max(K - 1, 1) - 0.5) * sh / FPS
            def side_a(tj):
                si = min(max(round((sA(tj) - A.ss) * FPS), 0), len(A.f) - 1)
                fr = A.f[si].astype(np.float32)
                p1 = ease_in(c01(tj / T1)); p2 = c01((tj - T1) / (TC - T1))
                hx, hy = her[si]; dx, dy, ds = dome[si]
                if tj < T1:
                    z, c, near_b, far_b = lerp(1.0, 1.6, p1), (hx, hy), 0.0, 6.0
                    s_ = (hx, hy)
                else:
                    e = (lambda p: p * p * (3 - 2 * p))(c01(p2 * 1.35))           # settle on the dome early
                    zd = ZA_END * DOME_A[2] / (DOME_A[2] * ds) if ds > 0 else ZA_END   # keep dome size steady
                    z = lerp(1.6, zd, ease_in(p2))
                    c = (lerp(hx, dx, e), lerp(hy, dy, e))
                    s_ = (lerp(hx, SCREEN[0], e), lerp(hy, SCREEN[1], e))
                    near_b, far_b = lerp(0, 16, e), lerp(6, 0, e)
                img = dof(fr, cv2.resize(fars[si], (fr.shape[1], fr.shape[0])), near_b * fr.shape[1] / OW / z, far_b * fr.shape[1] / OW / z)
                roll = ROLL * ease_in(c01((tj - (TC - 0.30)) / 0.30)) ** 1.5
                img = view(img, c[0], c[1], z, s_[0], s_[1], roll)
                cm = ease_in(c01((tj - (TC - 0.35)) / 0.35))
                if cm > 0:
                    img = img * (1 + (gainC - 1) * cm) + offC * cm
                return img

            def side_b(tj):
                p = c01((tj - TC) / BEAT)
                z = lerp(ZB0, 1.0, ease_out(p))
                roll = -ROLL * (1 - ease_out(c01((tj - TC) / 0.45)))
                bi = min(max(round((sB(tj) - B.ss) * FPS), 0), len(B.f) - 1)
                bx, by = domeB[bi][0], domeB[bi][1]
                s_ = (lerp(SCREEN[0], bx, ease_out(p)), lerp(SCREEN[1], by, ease_out(p)))
                return view(B.at(sB(tj)), bx, by, z, s_[0], s_[1], roll)

            w = (lambda q: q * q * (3 - 2 * q))(c01((tj - (TC - 0.07)) / 0.14))   # ~4-frame dissolve
            if w <= 0:
                img = side_a(min(tj, TC - 1e-3))
            elif w >= 1:
                img = side_b(max(tj, TC))
            else:
                img = side_a(min(tj, TC - 1e-3)) * (1 - w) + side_b(max(tj, TC)) * w
            acc += img
        img = acc / K
        enc.stdin.write(grade(np.clip(img, 0, 255)).tobytes())
    enc.stdin.close(); enc.wait()
    print("frames", n, "dome zoom A %.2f → B %.2f" % (ZA_END, ZB0))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
