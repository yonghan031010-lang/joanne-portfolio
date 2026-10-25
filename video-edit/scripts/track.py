import subprocess, numpy as np, sys
f=sys.argv[1]; W,H=914,1624; w,h=W//4,H//4
raw=subprocess.run(["ffmpeg","-v","error","-ss","1.5","-t","2.0","-i",f,"-vf",f"scale={w}:{h},format=gray","-f","rawvideo","-"],capture_output=True).stdout
fr=np.frombuffer(raw,np.uint8).reshape(-1,h,w)
for i,a in enumerate(fr):
    reg=a[int(h*0.6):int(h*0.97)]
    ys,xs=np.nonzero(reg<60)
    if len(xs)>30: print(f"{1.5+i/30:.3f} x={xs.mean()*4:.0f} xmin={np.percentile(xs,5)*4:.0f} xmax={np.percentile(xs,95)*4:.0f} ybot={(np.percentile(ys,98)+h*0.6)*4:.0f} n={len(xs)}")
