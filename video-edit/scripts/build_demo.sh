#!/bin/bash
# Two transition demos, 9:16 1080x1920@30, cut on the 83.35 BPM bar grid.
set -e
S=$1; SRC=$S/src; W=$S/work; mkdir -p $W
BAR=2.879; BEAT=0.71986
ENC="-c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -r 30"
OUT="scale=1080:1920:flags=lanczos,setsar=1,fps=30"

# A: gallery — IMG_4305 side walk, hard cut on matching stride (4305@3.47 -> 4719@2.33)
ffmpeg -v error -y -ss 0.591 -t $BAR -i $SRC/IMG_4305.MOV -vf "$OUT" -an $ENC $W/a1.mp4
ffmpeg -v error -y -ss 2.330 -t $BAR -i $SRC/IMG_4719.MOV -vf "$OUT" -an $ENC $W/a2.mp4

# B: boots — IMG_4278 push into the black boot (0.444,0.661) over last 2 beats,
# land on black, then IMG_4719 opens tight on her black trousers and eases out while tracking her.
P0=$(echo "$BAR-2*$BEAT" | bc -l)
ffmpeg -v error -y -ss 0 -t $BAR -i $SRC/IMG_4278.MOV -vf "scale=2160:3840,zoompan=z='if(lt(it,$P0),1,pow(9,pow((it-$P0)/(2*$BEAT),2)))':x='0.444*(iw-iw/zoom)':y='0.661*(ih-ih/zoom)':d=1:s=1080x1920:fps=30,setsar=1" -an $ENC $W/b1.mp4
ffmpeg -v error -y -ss 2.50 -t $BAR -i $SRC/IMG_4719.MOV -vf "scale=1828:3248,zoompan=z='1+2.2*pow(1-min(it/$BEAT,1),2)':x='max(0,min(iw-iw/zoom,(186+300*it)*2-iw/zoom/2))':y='max(0,min(ih-ih/zoom,1240*2-ih/zoom/2))':d=1:s=1080x1920:fps=30,setsar=1" -an $ENC $W/b2.mp4

M=$W/music_cut.wav
for d in a b; do
  off=0; [ $d = b ] && off=$(echo "2*$BAR" | bc -l)
  ffmpeg -v error -y -i $W/${d}1.mp4 -i $W/${d}2.mp4 -ss $off -t $(echo "2*$BAR" | bc -l) -i $M \
    -filter_complex "[0:v][1:v]concat=n=2:v=1[v];[2:a]afade=t=in:d=0.05,afade=t=out:st=$(echo "2*$BAR-0.3" | bc -l):d=0.3[a]" \
    -map "[v]" -map "[a]" $ENC -c:a aac -b:a 192k -shortest $W/demo_$d.mp4
done
