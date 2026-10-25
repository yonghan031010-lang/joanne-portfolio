import subprocess, numpy as np, sys
raw=subprocess.run(["ffmpeg","-v","error","-ss","1.5","-i",sys.argv[1],"-frames:v","1","-vf","scale=108:192,format=gray","-f","rawvideo","-"],capture_output=True).stdout
a=np.frombuffer(raw,np.uint8).reshape(192,108).astype(float)
best=None
for y in range(0,192-24):
  for x in range(0,108-14):
    m=a[y:y+24,x:x+14].mean()
    if best is None or m<best[0]: best=(m,x,y)
m,x,y=best; print("darkest 13% window mean",m,"center frac",(x+7)/108,(y+12)/192)
