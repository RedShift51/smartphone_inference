import sys,csv,os,time,re; sys.path.insert(0,"scripts"); from vlm import *
T=4; OUT="results/exp1_runs.csv"; os.makedirs("results",exist_ok=True)
gt={r["img_id"]:r["gt"] for r in csv.DictReader(open("data/exp1/gt.csv"))}
CONDS=[("A","full",""),("B","full_b",""),("A2","full_a2",""),("C","ctx4",""),("D","tight",""),("E","tight","--image-min-tokens 16 --image-max-tokens 16")]
COLS="img_id,cond,n_encodes,resized_wh,n_img_tokens,t_img_encode_ms,t_img_decode_ms,t_prompt_eval_ms,t_eval_ms,n_gen_tokens,ttft_ms,t_text_prefill_ms,total_ms,answer_raw,answer_norm,gt,correct,batt_temp_before,thermal_max_before,wait_s,threads".split(",")
done=set()
if os.path.exists(OUT): done={(r["img_id"],r["cond"]) for r in csv.DictReader(open(OUT))}
new=not os.path.exists(OUT); f=open(OUT,"a",newline=""); w=csv.DictWriter(f,fieldnames=COLS)
if new: w.writeheader()
def norm(s):
    s=re.sub(r"[^\d.,]","",s).replace(",","."); 
    return s.strip(".")
def level(): return int(re.search(r"level: (\d+)",sh("dumpsys battery")).group(1))
base=temps()[0]; print("baseline",base,"battery",level(),flush=True)
for c,d,ex in CONDS[:1]: pass
# warmups (not recorded)
run("exp1/full/00.jpg",t=T); run("exp1/tight/00.jpg",t=T)
base=temps()[0]; print("baseline after warmup",base,flush=True)
for i in sorted(gt):
    for c,d,ex in CONDS:
        if (i,c) in done: continue
        if level()<8: print("battery low, stop"); sys.exit(0)
        w0=time.time(); tb=temps()
        while tb[0]>=base+3: time.sleep(20); tb=temps()
        wait=round(time.time()-w0)
        log=run(f"exp1/{d}/{i}.jpg",ex,t=T); open(f"logs/main_{i}_{c}.log","w").write(log)
        p=parse(log); a=norm(p.get("answer_raw",""))
        ttft=p.get("t_prompt_eval_ms")
        p.update(img_id=i,cond=c,ttft_ms=ttft,t_text_prefill_ms=round(ttft-p["t_img_encode_ms"]-p["t_img_decode_ms"],1) if ttft else None,
                 answer_norm=a,gt=gt[i],correct=int(bool(a) and a==gt[i]),batt_temp_before=tb[0],thermal_max_before=tb[1],wait_s=wait,threads=T)
        w.writerow({k:p.get(k) for k in COLS}); f.flush()
        print(i,c,p["n_img_tokens"],p["t_img_encode_ms"],ttft,a,gt[i],tb[0],wait,flush=True)
print("DONE")
