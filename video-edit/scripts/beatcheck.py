"""Compare onset peaks in the cut music with the 83.35 BPM grid (t=0 is beat 1)."""
import subprocess, numpy as np, sys
sr=22050; raw=subprocess.run(["ffmpeg","-v","error","-i",sys.argv[1],"-ac","1","-ar",str(sr),"-f","s16le","-"],capture_output=True).stdout
x=np.frombuffer(raw,np.int16).astype(np.float32)/32768
hop=256; fr=np.lib.stride_tricks.sliding_window_view(x,1024)[::hop]
S=np.abs(np.fft.rfft(fr*np.hanning(1024),axis=1)); flux=np.maximum(np.diff(np.log1p(S*10),axis=0),0).sum(1)
ft=(np.arange(len(flux))+1)*hop/sr; beat=60/83.35
errs=[]
for k in range(int(ft[-1]/beat)):
    tb=k*beat; m=(ft>tb-0.08)&(ft<tb+0.08)
    if m.any(): j=np.argmax(flux[m]); errs.append(ft[m][j]-tb)
errs=np.array(errs)*1000
print(f"beats checked={len(errs)} median offset={np.median(errs):.1f}ms  |offset|<20ms: {np.mean(np.abs(errs)<20)*100:.0f}%")
# drift check: strongest onset near each downbeat, plus best-fit BPM over all strong onsets
pk=[(k, e) for k, e in enumerate(errs)]
strong=[]
for k in range(len(errs)):
    tb=k*beat; m=(ft>tb-0.08)&(ft<tb+0.08)
    strong.append(flux[m].max())
strong=np.array(strong); top=strong>np.percentile(strong,60)
ks=np.arange(len(errs))[top]; e=errs[top]
a,b=np.polyfit(ks,e,1)
print(f"strong beats={top.sum()} median={np.median(e):.1f}ms  trend={a:.2f}ms/beat  |e|<25ms: {np.mean(np.abs(e)<25)*100:.0f}%")
for k in range(0,84,12): print(k, round(errs[k],1), round(strong[k],1))
