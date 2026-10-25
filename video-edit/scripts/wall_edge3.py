import subprocess, numpy as np, sys, json
f=sys.argv[1]; W,H=1080,1920; ss=1.9
raw=subprocess.run(["ffmpeg","-v","error","-ss",str(ss),"-i",f,"-t","0.95","-vf","fps=30","-f","rawvideo","-pix_fmt","rgb24","-"],capture_output=True).stdout
fr=np.frombuffer(raw,np.uint8).reshape(-1,H,W,3).astype(np.float32)
out=[]
for i,a in enumerate(fr):
    band=a[int(H*0.15):int(H*0.70)]
    s=(band[...,0]-band[...,2]).mean(0)
    s=np.convolve(s,np.ones(9)/9,'same')
    stone=s>15
    # edge: rightmost column of the stone run that touches the left border
    if not stone[:20].any(): out.append((ss+i/30,None)); continue
    nz=np.nonzero(~stone[10:])[0]
    if len(nz)==0: out.append((ss+i/30,1080.0)); continue
    x=nz[0]+10
    # subpixel: interpolate threshold crossing
    x0=x-1; frac=(s[x0]-15)/max(s[x0]-s[x],1e-3)
    out.append((round(ss+i/30,4),round(float(x0+frac),1)))
for o in out: print(o)
json.dump(out,open(sys.argv[2],"w"))
