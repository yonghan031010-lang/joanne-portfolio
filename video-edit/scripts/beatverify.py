import subprocess, numpy as np, sys
sr=22050; raw=subprocess.run(["ffmpeg","-v","error","-i",sys.argv[1],"-ac","1","-ar",str(sr),"-f","s16le","-"],capture_output=True).stdout
x=np.frombuffer(raw,np.int16).astype(np.float32)/32768
hop=128; fr=np.lib.stride_tricks.sliding_window_view(x,1024)[::hop]
S=np.abs(np.fft.rfft(fr*np.hanning(1024),axis=1)); flux=np.maximum(np.diff(np.log1p(S*10),axis=0),0).sum(1)
flux=(flux-np.median(flux))/flux.std(); ft=(np.arange(len(flux))+1)*hop/sr+512/sr
# local onset peaks
pk=[i for i in range(2,len(flux)-2) if flux[i]==flux[i-2:i+3].max() and flux[i]>1.0]
pt=ft[pk]; pv=flux[pk]
def report(name,bpm,ph,t0=8.0,t1=68.6):
    beat=60/bpm; k=np.arange(int(np.ceil((t0-ph)/beat)),int((t1-ph)/beat)); g=ph+k*beat
    d=np.array([np.min(np.abs(pt-x)) for x in g])
    print(f"{name}: beats={len(g)} onset within 25ms={np.mean(d<0.025)*100:.0f}%  within 50ms={np.mean(d<0.05)*100:.0f}%")
report("83.35 from 8.080",83.35,8.08)
report("83.00 from 8.220",83.00,8.220)
# also the half-beat offset option
report("83.00 from 7.859",83.00,8.220-60/83/2)
# strongest onsets between 7 and 9 s
m=(pt>7)&(pt<9.2); print("onsets 7-9.2s:",[(round(a,3),round(b,1)) for a,b in zip(pt[m],pv[m])])
