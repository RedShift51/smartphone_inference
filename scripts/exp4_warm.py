import sys,time,json,base64,csv,re,subprocess,statistics as st,urllib.request; sys.path.insert(0,"scripts"); from vlm import *
IMG={"D":"data/exp1/tight/09.jpg","A2":"data/exp1/full_a2/10.jpg"}; B64={k:base64.b64encode(open(v,"rb").read()).decode() for k,v in IMG.items()}
CFG={"cpu_t4":("llama-server-cpu","none","","-t 4"),"cpu_big2":("llama-server-cpu","none","taskset c0","-t 2"),"vk_lm_only":("llama-server-vk","Vulkan0","","-t 4 --no-mmproj-offload"),"vk_all":("llama-server-vk","Vulkan0","","-t 4")}
def req(k):
    body=json.dumps({"messages":[{"role":"user","content":[{"type":"text","text":PROMPT},{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+B64[k]}}]}],"max_tokens":8,"temperature":0,"cache_prompt":False}).encode()
    t0=time.time(); r=json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8089/v1/chat/completions",body,{"Content-Type":"application/json"}),timeout=300)); return time.time()-t0,r
rows=[]; base=temps()
for name,(binary,dev,pre,extra) in CFG.items():
    while temps()[0]>=base[0]+3 or temps()[1]>=base[1]+10: time.sleep(15)
    sh("pkill -f llama-server; sleep 1"); t0=time.time()
    subprocess.Popen([ADB,"shell",f"cd /data/local/tmp/vlm; {pre} ./{binary} -m LFM2-VL-450M-Q8_0.gguf --mmproj mmproj-LFM2-VL-450M-Q8_0.gguf --host 127.0.0.1 --port 8089 -dev {dev} {extra} -c 2048 -v > srv_{name}.log 2>&1"])
    while True:
        try: urllib.request.urlopen("http://127.0.0.1:8089/health",timeout=2); break
        except Exception: time.sleep(0.5)
    ready=time.time()-t0; print(name,"ready after",round(ready,1),"s",flush=True)
    seq=[("D","cold1"),("A2","cold1")]+[("D","w"),("A2","w")]*7   # first of each = cold (shaders for that shape), then 2 warm-up + 5 measured
    cnt={"D":0,"A2":0}
    for k,tag in seq:
        wall,r=req(k); cnt[k]+=1; phase="first" if tag=="cold1" else ("warmup" if cnt[k]<=3 else "warm")
        tm=r.get("timings",{}); ans=r["choices"][0]["message"]["content"].strip()
        rows.append(dict(cfg=name,cond=k,phase=phase,n=cnt[k],prompt_ms=round(tm.get("prompt_ms",0)),prompt_n=tm.get("prompt_n"),pred_ms=round(tm.get("predicted_ms",0)),wall_ms=round(wall*1000),answer=ans,ready_s=round(ready,1),batt=temps()[0],cpu=temps()[1]))
        print(rows[-1],flush=True)
    sh("pkill -f llama-server")
w=csv.DictWriter(open("results/exp4_warm.csv","w",newline=""),fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
