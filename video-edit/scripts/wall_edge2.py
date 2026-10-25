"""Per-frame stone-pillar edge in IMG_4305 as a line x = a + b*(y-960) (1080x1920 px)."""
import subprocess, numpy as np, sys, json
f=sys.argv[1]; W,H=540,960; ss=1.70
raw=subprocess.run(["ffmpeg","-v","error","-ss",str(ss),"-i",f,"-t","1.2","-vf",f"fps=30,scale={W}:{H}","-f","rawvideo","-pix_fmt","rgb24","-"],capture_output=True).stdout
fr=np.frombuffer(raw,np.uint8).reshape(-1,H,W,3).astype(np.float32)
res=[]
for i,a in enumerate(fr):
    R,G,B=a[...,0],a[...,1],a[...,2]
    stone=(R-B>12)&(np.abs(R-G)<28)&(G>110)
    pts=[]
    for y in range(int(H*0.12),int(H*0.78),6):
        row=np.convolve(stone[y].astype(float),np.ones(7)/7,'same')>0.5
        s=np.nonzero(row)[0]
        if len(s)==0: continue
        # edge = right end of the stone run touching the left side of the frame (the pillar)
        runs=np.split(s,np.nonzero(np.diff(s)>4)[0]+1)
        first=runs[0]
        if first[0]>8: continue
        if first[-1]>=W-3: x=W
        else: x=first[-1]+1
        pts.append((y,x))
    t=round(ss+i/30,4)
    if len(pts)<15: res.append((t,None,None,0)); continue
    ys=np.array([p[0] for p in pts],float)*2; xs=np.array([p[1] for p in pts],float)*2
    full=(xs>=W*2).mean()
    if full>0.6: res.append((t,1080.0,0.0,len(pts))); continue
    keep=xs<W*2
    b,a0=np.polyfit(ys[keep]-960,xs[keep],1)
    r=np.abs(xs[keep]-(a0+b*(ys[keep]-960)))
    ok=r<np.percentile(r,70)+8
    b,a0=np.polyfit(ys[keep][ok]-960,xs[keep][ok],1)
    res.append((t,round(a0,1),round(b,4),int(ok.sum())))
for r in res: print(r)
json.dump(res,open(sys.argv[2],"w"))
