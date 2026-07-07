"""Assemble the narrated presentation video.

Each slide is shown for exactly as long as its narration. The demo beat
(segment 7) plays the real VHS terminal recording with narration over it.
"""
import glob
import os
import subprocess

os.chdir(os.path.dirname(os.path.abspath(__file__)))

NARR = [
 "Frugal Router. A token efficient A I agent for Track 1 of the AMD Developer Hackathon. Built by Bisman Singh and Chirag Sharma.",
 "The track has one rule that decides everything. Pass the accuracy gate, then win on the fewest Fireworks tokens. So accuracy above the bar is wasted money.",
 "The organizers confirmed every scored answer must come from a Fireworks call. So it is not about choosing local or remote. Every answer is remote. The real question is how cheap each call can be. A free local model does the thinking, and Fireworks does the answering.",
 "Here is the pipeline. Each task is classified for free. A local model drafts an answer and measures its own confidence. When it is confident, it sends a compact confirmation call. When it is unsure, it pays for a full remote solve, and only then.",
 "The confidence signals are the ones that hold up on small models. Self consistency voting, and a pessimistic log probability signal. We dropped self verification, because small models say yes to almost everything.",
 "The token savings are measured, not guessed. Capping reasoning effort cut completion tokens by about a third. Remote calls run in parallel, and nothing is cached or hardcoded.",
 "Here it is running the real submission image, on Fireworks. Three tasks, three correct answers, in under four seconds. The math is right, the sentiment answer carries its justification, and the code task returns a runnable function.",
 "It is built for the harness and fails safe. It writes a valid results file after every task, so a timeout never zeroes the run. Model ids come from the environment, and no credentials ship in the image.",
 "Across all eight categories the agent reached full accuracy, while spending under two hundred tokens per task. Seventy five tests, all offline. A lean image, tested against real Fireworks.",
 "Gemma runs on both sides. It is the preferred remote model and the recommended local drafting model. One model family carries the free path and the paid path.",
 "That is Frugal Router. Free intelligence, paid answers, minimum tokens. Thank you.",
]

DEMO_SEGMENT = 7  # 1-indexed slide that shows the live terminal recording
V = "1280:720"

def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def dur(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True).stdout.strip()
    return float(out)

slides = sorted(glob.glob("slide.0*.png"))
assert len(slides) == len(NARR), f"{len(slides)} slides vs {len(NARR)} lines"
os.makedirs("_v", exist_ok=True)

demo_dur = dur("demo.mp4")
segments = []
for i, (slide, text) in enumerate(zip(slides, NARR), 1):
    aiff, wav, seg = f"_v/n{i:02d}.aiff", f"_v/n{i:02d}.wav", f"_v/seg{i:02d}.mp4"
    run(["say", "-v", "Samantha", "-r", "172", "-o", aiff, text])
    run(["ffmpeg", "-y", "-i", aiff, "-af", "apad=pad_dur=0.6",
         "-ar", "44100", "-ac", "2", wav])
    narr_dur = dur(wav)

    if i == DEMO_SEGMENT:
        seg_dur = max(demo_dur, narr_dur)
        # narration padded to the clip length, demo video held to the same length
        padwav = f"_v/n{i:02d}_pad.wav"
        run(["ffmpeg", "-y", "-i", wav, "-af", f"apad=whole_dur={seg_dur:.3f}", padwav])
        run(["ffmpeg", "-y", "-i", "demo.mp4", "-i", padwav,
             "-filter_complex",
             f"[0:v]scale={V}:force_original_aspect_ratio=decrease,"
             f"pad={V}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=25,"
             f"tpad=stop_mode=clone:stop_duration={seg_dur:.3f}[v]",
             "-map", "[v]", "-map", "1:a",
             "-t", f"{seg_dur:.3f}",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
             "-ar", "44100", "-ac", "2", seg])
    else:
        run(["ffmpeg", "-y", "-loop", "1", "-i", slide, "-i", wav,
             "-vf", f"scale={V}:force_original_aspect_ratio=decrease,"
                    f"pad={V}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=25",
             "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-ar", "44100", "-ac", "2", "-shortest", seg])
    segments.append(seg)
    print(f"seg {i:02d}  {dur(seg):5.1f}s  {'DEMO' if i==DEMO_SEGMENT else ''}")

with open("_v/concat.txt", "w") as f:
    for s in segments:
        f.write(f"file '{os.path.basename(s)}'\n")
run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "_v/concat.txt",
     "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
     "frugal-router-video.mp4"])
print("TOTAL", round(dur("frugal-router-video.mp4"), 1), "s ->", "frugal-router-video.mp4")
