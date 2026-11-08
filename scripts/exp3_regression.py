import sys,csv,time,statistics as st; sys.path.insert(0,"scripts"); from vlm import *
rows=list(csv.DictReader(open("results/exp1_runs.csv")))
pick={}
for c,d in(("A2","full_a2"),("D","tight")):
    v=sorted((float(r["ttft_ms"]),r["img_id"]) for r in rows if r["cond"]==c); pick[c]=(d,v[len(v)//2][1])
print("images",pick)
BUILDS={"cur":"llama-mtmd-cli","old":"llama-mtmd-cli-old"}; T=4; REPS=5
COLS=["rep","cond","build","img_id","n_img_tokens","t_img_encode_ms","t_img_decode_ms","t_text_prefill_ms","t_prompt_eval_ms","answer_raw","batt_c","cpu_c","wait_s"]
f=open("results/exp3_regression.csv","w",newline=""); w=csv.DictWriter(f,fieldnames=COLS); w.writeheader()
for c,(d,i) in pick.items():
    for b in BUILDS.values(): run(f"exp1/{d}/{i}.jpg",t=T,binary=b)  # warmup, not recorded
base=temps(); print("base",base,flush=True)
for rep in range(REPS):
    for c,(d,i) in pick.items():
        order=["cur","old"] if rep%2==0 else ["old","cur"]
        for b in order:
            t0=time.time(); tb=temps()
            while tb[0]>=base[0]+3 or (tb[1] and base[1] and tb[1]>=base[1]+10): time.sleep(15); tb=temps()
            log=run(f"exp1/{d}/{i}.jpg",t=T,binary=BUILDS[b]); open(f"logs/reg_{c}_{b}_{rep}.log","w").write(log); p=parse(log)
            ttft=p["t_prompt_eval_ms"]; p.update(rep=rep,cond=c,build=b,img_id=i,t_text_prefill_ms=round(ttft-p["t_img_encode_ms"]-p["t_img_decode_ms"],1),batt_c=tb[0],cpu_c=tb[1],wait_s=round(time.time()-t0))
            w.writerow({k:p.get(k) for k in COLS}); f.flush(); print(rep,c,b,p["n_img_tokens"],p["t_img_encode_ms"],p["t_img_decode_ms"],p["t_text_prefill_ms"],p["answer_raw"],tb,flush=True)
f.close()
R=list(csv.DictReader(open("results/exp3_regression.csv")))
for c in pick:
    for k in("t_img_encode_ms","t_img_decode_ms","t_text_prefill_ms"):
        m={b:[float(r[k]) for r in R if r["cond"]==c and r["build"]==b] for b in BUILDS}
        print(c,k,"cur med",round(st.median(m["cur"])),"[%d,%d]"%(min(m["cur"]),max(m["cur"])),"old med",round(st.median(m["old"])),"[%d,%d]"%(min(m["old"]),max(m["old"])),"cur/old",round(st.median(m["cur"])/st.median(m["old"]),2))
    print(c,"answers",{b:sorted({r['answer_raw'] for r in R if r['cond']==c and r['build']==b}) for b in BUILDS})
