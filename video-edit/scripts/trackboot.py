import subprocess, numpy as np, cv2, sys
f=sys.argv[1]; W,H=540,960
raw=subprocess.run(["ffmpeg","-v","error","-i",f,"-t","1.5","-vf",f"scale={W}:{H},format=gray","-f","rawvideo","-"],capture_output=True).stdout
fr=np.frombuffer(raw,np.uint8).reshape(-1,H,W)
cx,cy=419/2,1341/2; s=40
tpl=fr[0][int(cy-s):int(cy+s),int(cx-s):int(cx+s)]
for i in range(0,len(fr)):
    r=cv2.matchTemplate(fr[i],tpl,cv2.TM_CCOEFF_NORMED); _,mx,_,loc=cv2.minMaxLoc(r)
    x,y=(loc[0]+s)*2,(loc[1]+s)*2
    if i%2==0: print(f"{i/30:.3f} {x} {y} score={mx:.2f}")
