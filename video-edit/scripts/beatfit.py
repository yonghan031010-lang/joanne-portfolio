"""Fit BPM + phase that best explain the onsets (comb over smoothed spectral flux)."""
import subprocess, numpy as np, sys
sr=22050; raw=subprocess.run(["ffmpeg","-v","error","-i",sys.argv[1],"-ac","1","-ar",str(sr),"-f","s16le","-"],capture_output=True).stdout
x=np.frombuffer(raw,np.int16).astype(np.float32)/32768
hop=128; fr=np.lib.stride_tricks.sliding_window_view(x,1024)[::hop]
S=np.abs(np.fft.rfft(fr*np.hanning(1024),axis=1)); flux=np.maximum(np.diff(np.log1p(S*10),axis=0),0).sum(1)
flux=(flux-flux.mean())/flux.std(); ft=(np.arange(len(flux))+1)*hop/sr+512/sr
best=(-1e9,0,0)
for bpm in np.arange(82.8,83.9,0.01):
    beat=60/bpm; kmax=int((ft[-1]-0.2)/beat)
    for ph in np.arange(-0.06,0.06,0.002):
        idx=np.searchsorted(ft,ph+np.arange(kmax)*beat)
        sc=flux[np.clip(idx,0,len(flux)-1)].mean()
        if sc>best[0]: best=(sc,bpm,ph)
print(f"best bpm={best[1]:.2f} phase={best[2]*1000:.0f}ms score={best[0]:.2f}")
beat=60/83.35; kmax=int((ft[-1]-0.2)/beat)
for ph in (0.0,best[2]):
    idx=np.searchsorted(ft,ph+np.arange(kmax)*beat); print(f"83.35 BPM phase {ph*1000:.0f}ms score={flux[idx].mean():.2f}")
