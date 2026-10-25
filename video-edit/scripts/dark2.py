import subprocess, numpy as np, sys
f=sys.argv[1]; ss=float(sys.argv[2]); n=int(sys.argv[3]); W,H=914,1624
raw=subprocess.run(["ffmpeg","-v","error","-ss",str(ss),"-i",f,"-frames:v",str(n),"-vf","format=gray","-f","rawvideo","-"],capture_output=True).stdout
fr=np.frombuffer(raw,np.uint8).reshape(-1,H,W).astype(np.float32)
ww,wh=48,85   # same 9:16 aspect
for i,a in enumerate(fr[::3]):
    # integral image for window means
    ii=np.pad(a.cumsum(0).cumsum(1),((1,0),(1,0)))
    m=(ii[wh:,ww:]-ii[:-wh,ww:]-ii[wh:,:-ww]+ii[:-wh,:-ww])/(ww*wh)
    y,x=np.unravel_index(m.argmin(),m.shape)
    print(f"t={ss+i*3/30:.2f} mean={m.min():.1f} center=({x+ww/2:.0f},{y+wh/2:.0f})")
