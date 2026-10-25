import subprocess, numpy as np, sys
f=sys.argv[1]; W,H=540,960; ss=1.80
raw=subprocess.run(["ffmpeg","-v","error","-ss",str(ss),"-i",f,"-t","1.0","-vf",f"fps=30,scale={W}:{H}","-f","rawvideo","-pix_fmt","rgb24","-"],capture_output=True).stdout
fr=np.frombuffer(raw,np.uint8).reshape(-1,H,W,3).astype(np.float32)
for i,a in enumerate(fr):
    band=a[int(H*0.2):int(H*0.7)]
    score=(band[...,0]-band[...,2]).mean(0)          # stone is warm (R>B), glass is cool
    stone=np.convolve(score>15,np.ones(9)/9,'same')>0.5
    # wall occupies the left: edge = first column (from left) where stone stops for good
    nz=np.nonzero(~stone)[0]
    edge = nz[0] if len(nz) else W
    print(f"{ss+i/30:.3f} edge={edge*2} stonefrac={stone.mean():.2f}")
