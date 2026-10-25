"""Virtual-camera renderer: subpixel zoom/pan with motion blur, exposure ramps and
a short blended cut between two shots. Writes 1080x1920@30 H.264 (no audio).

Camera model per output frame: source point P maps to screen point Sc at scale
k = z * OW / src_w  (z=1 → source fills the 9:16 frame).
"""
import subprocess, sys, json, math
import numpy as np, cv2

OW, OH, FPS = 1080, 1920, 30
BEAT = 60 / 83.0          # measured: onsets fit 83.00 BPM, beat 1 at 8.220s in the mp3
BAR = 4 * BEAT


def read_frames(path, ss, dur):
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height:stream_side_data=rotation", "-of", "json", path],
                       capture_output=True, text=True)
    st = json.loads(p.stdout)["streams"][0]
    w, h = st["width"], st["height"]
    rot = next((sd.get("rotation", 0) for sd in st.get("side_data_list", [])), 0)
    if abs(rot) == 90:
        w, h = h, w
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{ss:.4f}", "-t", f"{dur + 0.2:.4f}", "-i", path,
                          "-vf", f"fps={FPS}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def warp(img, z, P, Sc):
    h, w = img.shape[:2]
    k = z * OW / w
    M = np.float32([[k, 0, Sc[0] - k * P[0]], [0, k, Sc[1] - k * P[1]]])
    return cv2.warpAffine(img, M, (OW, OH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def clamp_cam(z, P, Sc, w, h):
    """Shift the anchor so the source always covers the frame (no borders)."""
    k = z * OW / w
    left = P[0] - Sc[0] / k
    top = P[1] - Sc[1] / k
    left = min(max(left, 0), w - OW / k)
    top = min(max(top, 0), h - OH / k)
    return (left + Sc[0] / k, top + Sc[1] / k)


def render_shot(frames, n_out, cam, gain=lambda t: 1.0, blur=6, shutter=0.5):
    h, w = frames.shape[1:3]
    out = []
    for i in range(n_out):
        t = i / FPS
        src = frames[min(i, len(frames) - 1)].astype(np.float32)
        acc = np.zeros((OH, OW, 3), np.float32)
        for j in range(blur):
            ts = t + (j / max(blur - 1, 1) - 0.5) * shutter / FPS
            z, P, Sc = cam(ts)
            P = clamp_cam(z, P, Sc, w, h)
            acc += warp(src, z, P, Sc)
        out.append(acc / blur * gain(t))
    return out


def ease_in(p):  return p * p
def ease_out(p): return 1 - (1 - p) ** 2
def c01(x):      return min(max(x, 0.0), 1.0)
def lerp(a, b, p): return a + (b - a) * p


def shake(t, amp, seed):
    # smooth pseudo-handheld drift
    return (amp * (math.sin(2.1 * t * 6.28 + seed) * 0.6 + math.sin(3.7 * t * 6.28 + seed * 2) * 0.4),
            amp * 0.7 * (math.sin(1.7 * t * 6.28 + seed * 3) * 0.6 + math.sin(4.3 * t * 6.28 + seed) * 0.4))


def gallery(src):
    """IMG_4305 side walk → IMG_4719 gallery. Push-in on her during the last beat,
    cut on the matching stride, gallery shot opens at the same scale/position and
    tracks her like a follow-cam before easing off."""
    n = round(BAR * FPS)
    a = read_frames(f"{src}/IMG_4305.MOV", 3.47 - BAR, BAR)
    b = read_frames(f"{src}/IMG_4719.MOV", 2.33, BAR)
    ZA = 1.3
    anchorA = (0.55 * OW, 0.32 * OH)          # in 4305 screen/source coords (1080x1920)

    def camA(t):
        p = c01((t - (BAR - BEAT)) / BEAT)
        return 1 + (ZA - 1) * ease_in(p), anchorA, anchorA

    # 4719: her legs centroid at the cut ≈ x145, feet y1434 (source px, 914x1624)
    kB = ZA * OW / 914
    sB = (anchorA[0] - (anchorA[0] - 0.27 * OW) * ZA, anchorA[1] + (0.74 * OH - anchorA[1]) * ZA)
    left0 = 145 - sB[0] / kB
    top0 = 1434 - sB[1] / kB

    def camB(t):
        p = c01(t / 1.36)
        z = lerp(ZA, 1.18, ease_out(c01(t / BAR)))
        left = left0 + 197 * ease_out(p)                # follow-cam: starts at her walking speed
        dx, dy = shake(t, 5 * (1 - c01(t / BEAT)), 1.3)
        cx = left + OW / 2 / (z * OW / 914)
        cy = top0 + OH / 2 / kB
        return z, (cx + dx / kB, cy + dy / kB), (OW / 2, OH / 2)

    A = render_shot(a, n, camA, gain=lambda t: 1 + 0.12 * ease_in(c01((t - (BAR - BEAT)) / BEAT)))
    B = render_shot(b, n, camB, gain=lambda t: lerp(0.84, 1.0, ease_out(c01(t / BEAT))))
    return blend_cut(A, B)


def boots(src):
    """IMG_4278 push into the black boot → pull out of her black trousers in IMG_4719.
    Zoom curves mirror each other around the bar line so the speed feels continuous."""
    n = round(BAR * FPS)
    a = read_frames(f"{src}/IMG_4278.MOV", 0.0, BAR)
    b = read_frames(f"{src}/IMG_4719.MOV", 2.45, BAR)
    ZMAX, P0 = 12.0, BAR - 2 * BEAT
    anchorA = (0.444 * OW, 0.661 * OH)

    def camA(t):
        p = c01((t - P0) / (2 * BEAT))
        return ZMAX ** ease_in(p), anchorA, anchorA

    def camB(t):
        p = c01(t / (2 * BEAT))
        z = 1.15 * (ZMAX / 1.15) ** ((1 - p) ** 2)
        P = (290 + 260 * max(0.0, t - 0.25), 1364)       # dark trouser patch, then follow her loosely
        Sc = (lerp(anchorA[0], 0.42 * OW, ease_out(p)), lerp(anchorA[1], 0.72 * OH, ease_out(p)))
        return z, P, Sc

    dip = lambda t, t0, d: c01(abs(t - t0) / d)
    A = render_shot(a, n, camA, gain=lambda t: lerp(0.08, 1, ease_out(dip(t, BAR, 0.35))))
    B = render_shot(b, n, camB, gain=lambda t: lerp(0.08, 1, ease_in(dip(t, 0, 0.45))))
    return blend_cut(A, B)


def read_remap(path, remap, n):
    """Frames for output frame i taken at source time remap(i/FPS) (nearest frame)."""
    allf = read_frames(path, 0.0, 6.0)
    return np.stack([allf[min(round(remap(i / FPS) * FPS), len(allf) - 1)] for i in range(n)])


def boots2(src):
    """IMG_4278 boots on the sill → IMG_4306 the same boots on her feet (POV).
    Push into the black boot, pull out of the boot she's standing in; 4306 starts
    in slow motion so the pair stays put while the camera pulls back, then ramps to real speed."""
    n = round(BAR * FPS)
    a = read_frames(f"{src}/IMG_4278.MOV", 0.0, BAR)
    v = lambda t: 0.35 + 0.65 * (3 * c01(t / 0.9) ** 2 - 2 * c01(t / 0.9) ** 3)
    ts = [0.0]
    for i in range(n):
        ts.append(ts[-1] + v(i / FPS) / FPS)
    b = read_remap(f"{src}/IMG_4306.MOV", lambda t: ts[min(round(t * FPS), n)], n)
    ZMAX, ZEND, P0 = 12.0, 1.04, BAR - 2 * BEAT
    anchorA = (0.444 * OW, 0.661 * OH)
    track = [(0.0, 418, 1340), (0.067, 408, 1358), (0.133, 394, 1374), (0.2, 378, 1394)]

    def camA(t):
        p = c01((t - P0) / (2 * BEAT))
        return ZMAX ** ease_in(p), anchorA, anchorA

    def camB(t):
        p = c01(t / (2 * BEAT))
        z = ZEND * (ZMAX / ZEND) ** ((1 - p) ** 2)
        st = min(ts[min(round(t * FPS), n)], 0.2)
        P = (np.interp(st, [k[0] for k in track], [k[1] for k in track]),
             np.interp(st, [k[0] for k in track], [k[2] for k in track]))
        # screen anchor drifts to where that point naturally sits, so the pull-out ends un-cropped
        Sc = (lerp(anchorA[0], P[0], ease_out(p)), lerp(anchorA[1], P[1], ease_out(p)))
        return z, P, Sc

    dip = lambda t, t0, d: c01(abs(t - t0) / d)
    A = render_shot(a, n, camA, gain=lambda t: lerp(0.08, 1, ease_out(dip(t, BAR, 0.35))))
    B = render_shot(b, n, camB, gain=lambda t: lerp(0.08, 1, ease_in(dip(t, 0, 0.45))))
    return blend_cut(A, B)


def gallery2(src):
    """IMG_4305 → IMG_4719 hidden behind the stone wall. While the wall pans past,
    the glass behind its trailing edge is replaced by the gallery, so the second time
    she comes out from behind the wall she is already in the gallery — one straight walk."""
    n = round(2 * BAR * FPS)
    offA, offB = 0.14, 0.90                    # wall edge mid-frame lands on beat 4 of bar 1
    a = read_frames(f"{src}/IMG_4305.MOV", offA, 2 * BAR)
    b = read_frames(f"{src}/IMG_4719.MOV", 0.0, 6.0)
    # wall trailing edge (1080px frame), measured per frame from the warm-stone/cool-glass boundary
    et = [1.88, 2.067, 2.1, 2.133, 2.167, 2.2, 2.233, 2.267, 2.3, 2.333, 2.367, 2.4, 2.433, 2.467,
          2.5, 2.533, 2.567, 2.6, 2.633, 2.667, 2.7, 2.733, 2.767]
    ex = [1080, 816, 784, 748, 712, 672, 632, 592, 548, 500, 456, 408, 368, 328,
          288, 248, 212, 176, 140, 108, 72, 32, 0]
    ZB = 1.03
    hB, wB = b.shape[1:3]

    def camB(t):
        dx, dy = shake(t, 4 * (1 - c01((t - 2.7) / BEAT)), 0.7)
        return ZB, (wB / 2 + dx / (ZB * OW / wB), hB / 2 + dy / (ZB * OW / wB)), (OW / 2, OH / 2)

    xs = np.arange(OW, dtype=np.float32)
    out = []
    for i in range(n):
        t = i / FPS
        ta = t + offA
        if ta < et[0]:
            out.append(a[min(i, len(a) - 1)].astype(np.float32)); continue
        e = float(np.interp(ta, et, ex)) if ta <= et[-1] else -1e9
        fb = b[min(round((t - offB) * FPS), len(b) - 1)].astype(np.float32)
        z, P, Sc = camB(t)
        gb = warp(fb, z, clamp_cam(z, P, Sc, wB, hB), Sc) * lerp(0.86, 1.0, ease_out(c01((ta - 2.77) / BEAT)))
        if e < -100:
            out.append(gb); continue
        fa = a[min(i, len(a) - 1)].astype(np.float32)
        m = np.clip((xs - e) / 36 + 0.5, 0, 1)[None, :, None]   # feather ≈ one frame of edge travel
        out.append(fa * (1 - m) + gb * m)
    return out


def blend_cut(A, B):
    """Hard cut softened by one mixed frame each side (reads as motion blur, not a dissolve)."""
    return A[:-1] + [A[-1] * 0.7 + B[0] * 0.3, A[-1] * 0.25 + B[0] * 0.75] + B[1:]


def grade(img):
    # shared look so both locations sit in one palette
    x = np.clip(img, 0, 255) / 255.0
    lum = x @ np.float32([0.299, 0.587, 0.114])
    x = lum[..., None] + (x - lum[..., None]) * 0.88
    x = x * np.float32([1.02, 1.0, 0.97])
    x = np.clip(x, 0, 1) ** 1.04
    return (x * 255).astype(np.uint8)


if __name__ == "__main__":
    which, src, out = sys.argv[1], sys.argv[2], sys.argv[3]
    frames = {"gallery": gallery, "boots": boots, "boots2": boots2, "gallery2": gallery2}[which](src)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{OW}x{OH}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                            "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", out],
                           stdin=subprocess.PIPE)
    for f in frames:
        enc.stdin.write(grade(f).tobytes())
    enc.stdin.close(); enc.wait()
    print(which, len(frames), "frames")
