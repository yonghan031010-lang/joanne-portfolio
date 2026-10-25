"""Tempo/phase fit on the full song, plus per-segment local phase to detect drift."""
import subprocess, numpy as np, sys
sr=22050; raw=subprocess.run(["ffmpeg","-v","error","-i",sys.argv[1],"-ac","1","-ar",str(sr),"-f","s16le","-"],capture_output=True).stdout
x=np.frombuffer(raw,np.int16).astype(np.float32)/32768
hop=128; fr=np.lib.stride_tricks.sliding_window_view(x,1024)[::hop]
S=np.abs(np.fft.rfft(fr*np.hanning(1024),axis=1)); flux=np.maximum(np.diff(np.log1p(S*10),axis=0),0).sum(1)
flux=(flux-flux.mean())/flux.std(); ft=(np.arange(len(flux))+1)*hop/sr+512/sr
def score(bpm,ph,t0,t1):
    beat=60/bpm; k=np.arange(int(np.ceil((t0-ph)/beat)),int((t1-ph)/beat))
    idx=np.clip(np.searchsorted(ft,ph+k*beat),0,len(flux)-1); return flux[idx].mean()
for (t0,t1) in [(8.08,68.54),(0,154)]:
    best=max(((score(b,p,t0,t1),b,p) for b in np.arange(82.5,84.0,0.01) for p in np.arange(0,60/b,0.004)))
    print(f"[{t0}-{t1}] best bpm={best[1]:.2f} first-beat phase={best[2]:.3f}s score={best[0]:.2f}")
    b=best[1]; beat=60/b
    # nearest grid beat to 8.08
    k=round((8.08-best[2])/beat); print(f"   grid beat nearest 8.08s: {best[2]+k*beat:.3f}s; 21 bars at this bpm = {84*beat:.3f}s")
