"""Extract training/test frames for plate-detector fine-tuning from the current source video.
Per-file exclusion: this video's first 900 s (protest). Training frames avoid +-10 s around the
12 development + 6 check timestamps; >=1950 s is a locked test block (never used for training)."""
import json, subprocess, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'work/video-tools'))
import imageio_ffmpeg
FF=imageio_ffmpeg.get_ffmpeg_exe()
VIDEO='/Users/md.zarifbinhasnat/Downloads/2025-12-31-111331_2025-12-31-131239_1000004$1$0$0.mp4'
EVAL=[930+90*k for k in range(12)]+[975,1155,1335,1515,1695,1875]
train=[t for t in range(905,1950,5) if all(abs(t-e)>10 for e in EVAL)]
test=list(range(1955,2000,5))
# interleave so partial labelling still spans the whole time range: every 10 s first, then the rest
order=[t for t in train if t%10==5]+[t for t in train if t%10==0]
manifest={'video':VIDEO,'per_file_exclusion_seconds':900,'eval_timestamps_avoided':EVAL,'avoid_window_seconds':10,
          'train_times':order,'locked_test_times':test,
          'limitation':'Same camera/session as development and check frames; stopped vehicles can recur across nearby times.'}
for t in order+test:
    out=HERE/'frames'/f'frame_{t:04d}.jpg'
    if out.exists():continue
    subprocess.run([FF,'-loglevel','error','-ss',str(t),'-i',VIDEO,'-frames:v','1','-q:v','2',str(out)],check=True)
json.dump(manifest,open(HERE/'manifest.json','w'),indent=1)
print('train',len(order),'test',len(test))
