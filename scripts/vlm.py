import subprocess, re, shlex, os
ADB=os.environ.get("ADB","adb")
PROMPT="What is the price on the price tag inside the red box? Answer with the number only."
def sh(cmd, timeout=600):
    return subprocess.run([ADB,"shell",cmd],capture_output=True,timeout=timeout).stdout.decode("utf-8","replace")
def run(image, extra="", lm="LFM2-VL-450M-Q8_0.gguf", mm="mmproj-LFM2-VL-450M-Q8_0.gguf", t=6, pre="", n=16, binary="llama-mtmd-cli", dev="none"):
    cmd=(f"cd /data/local/tmp/vlm; {pre} ./{binary} -m {lm} --mmproj {mm} --image {image} "
         f"-p {shlex.quote(PROMPT)} --temp 0 -n {n} -t {t} -dev {dev} {extra} -v 2>&1")
    return sh(cmd)
def ts(s):  # '0.08.061.119' -> ms
    m,sec,ms,us=map(int,s.split(".")); return (m*60+sec)*1000+ms+us/1000
def parse(log):
    d={}
    toks=[int(x) for x in re.findall(r"mtmd_batch_encode_impl: encoding batch with \d+ entries and total (\d+) tokens",log)]
    d["n_img_tokens"]=sum(toks); d["n_encodes"]=len(toks)
    cp=re.findall(r"(\S+) D clip_encode: copying image.*?\(nx=(\d+), ny=(\d+)\)",log)
    out=re.findall(r"(\S+) D clip_encode: output embedding",log)
    d["resized_wh"]="+".join(f"{w}x{h}" for _,w,h in cp)
    d["t_img_encode_ms"]=round(sum(ts(b)-ts(a[0]) for a,b in zip(cp,out)),1)
    d["t_img_decode_ms"]=sum(int(x) for x in re.findall(r"image decoded \(batch \d+/\d+\) in (\d+) ms",log))
    m=re.search(r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens",log)
    if m: d["t_prompt_eval_ms"]=float(m.group(1)); d["n_prompt_tokens"]=int(m.group(2))
    m=re.search(r"\beval time =\s*([\d.]+) ms /\s*(\d+) runs",log)
    if m: d["t_eval_ms"]=float(m.group(1)); d["n_gen_tokens"]=int(m.group(2))
    m=re.search(r"total time =\s*([\d.]+) ms",log)
    if m: d["total_ms"]=float(m.group(1))
    lines=[l for l in log.splitlines() if l.strip() and not re.match(r"^\d+\.\d+\.\d+\.\d+ ",l) and not l.startswith(("<__media__>","main:","llama_","ggml_","build:"))]
    d["answer_raw"]=lines[-1].strip() if lines else ""
    return d
def temps():
    """(battery C, CPU C): battery from dumpsys battery, CPU from hardware_properties (readable without root)."""
    b=int(sh("dumpsys battery | grep temperature").strip().split()[-1])/10
    m=re.search(r"CPU temperatures: \[([\d.]+)",sh("dumpsys hardware_properties"))
    return b,(float(m.group(1)) if m else None)
