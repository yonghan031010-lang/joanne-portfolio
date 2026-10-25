import subprocess, numpy as np, cv2, json, sys
f=sys.argv[1]; res={round(r[0],3):r for r in json.load(open(sys.argv[2]))}
old=dict(zip([2.067,2.1,2.2,2.3,2.4,2.5,2.6,2.7],[816,784,672,548,408,288,176,72]))
tiles=[]
for t in [2.1,2.2,2.3,2.4,2.5,2.6]:
    raw=subprocess.run(["ffmpeg","-v","error","-ss",str(t),"-i",f,"-frames:v","1","-f","rawvideo","-pix_fmt","bgr24","-"],capture_output=True).stdout
    a=np.frombuffer(raw,np.uint8).reshape(1920,1080,3).copy()
    r=res.get(round(t,3))
    if r and r[1] is not None:
        for y in (0,1919): pass
        cv2.line(a,(int(r[1]+r[2]*(0-960)),0),(int(r[1]+r[2]*(1919-960)),1919),(0,0,255),6)
    if t in old: cv2.line(a,(old[t],0),(old[t],1919),(0,255,255),4)
    cv2.putText(a,str(t),(20,80),0,3,(0,255,0),6)
    tiles.append(cv2.resize(a,(270,480)))
cv2.imwrite(sys.argv[3],np.hstack(tiles))
