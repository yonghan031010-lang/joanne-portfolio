import subprocess, numpy as np, cv2, sys
def frames(f,ss,n,w,h):
    raw=subprocess.run(["ffmpeg","-v","error","-ss",str(ss),"-i",f,"-frames:v",str(n),"-vf",f"scale={w}:{h},format=gray","-f","rawvideo","-"],capture_output=True).stdout
    return np.frombuffer(raw,np.uint8).reshape(-1,h,w).astype(np.float32)
fr=frames(sys.argv[1]+"/IMG_4305.MOV",3.0,16,540,960)
print("4305 mean luma",fr.mean())
for i in range(0,15,3):
    # flow of the top band (background reflection/building), in output px per frame (x2 for 1080 width)
    (dx,dy),_=cv2.phaseCorrelate(fr[i][0:450],fr[i+1][0:450]); print("4305 bg dx/frame(1080px)",round(dx*2,1),round(dy*2,1))
g=frames(sys.argv[1]+"/IMG_4719.MOV",2.3,10,914,1624); print("4719 mean luma",g.mean())
b=frames(sys.argv[1]+"/IMG_4278.MOV",1.4,1,1080,1920); print("4278 mean luma",b.mean())
