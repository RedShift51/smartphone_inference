import sys,csv,time,statistics as st; sys.path.insert(0,"scripts"); from vlm import *
CF={"base":dict(binary="llama-mtmd-cli-repack"),"repack":dict(binary="llama-mtmd-cli-repack",pre="CLIP_REPACK=1")}
IM={"D":"exp1/tight/09.jpg","A2":"exp1/full_a2/10.jpg"}
COLS=["rep","cond","cfg","n_img_tokens","t_img_encode_ms","t_img_decode_ms","t_text_prefill_ms","t_prompt_eval_ms","answer_raw","batt_c","cpu_c"]
f=open("results/exp5_repack.csv","w",newline=""); w=csv.DictWriter(f,fieldnames=COLS); w.writeheader(); base=temps()
for k in CF: run(IM["D"],t=4,**CF[k])
for rep in range(5):
    for cond,img in IM.items():
        for k in (["base","repack"] if rep%2==0 else ["repack","base"]):
            tb=temps()
            while tb[0]>=base[0]+3 or tb[1]>=base[1]+10: time.sleep(15); tb=temps()
            p=parse(run(img,t=4,**CF[k])); p.update(rep=rep,cond=cond,cfg=k,t_text_prefill_ms=round(p["t_prompt_eval_ms"]-p["t_img_encode_ms"]-p["t_img_decode_ms"],1),batt_c=tb[0],cpu_c=tb[1])
            w.writerow({c:p.get(c) for c in COLS}); f.flush()
f.close()
# correctness: condition D on all 16 images, answers base vs repack
gt={r["img_id"]:r for r in csv.DictReader(open("data/exp1/gt.csv"))}; ids=sorted({r["img_id"] for r in csv.DictReader(open("results/exp1_runs.csv"))})
diff=[]; same=0
for i in ids:
    a=parse(run(f"exp1/tight/{i}.jpg",t=4,**CF["base"]))["answer_raw"]; b=parse(run(f"exp1/tight/{i}.jpg",t=4,**CF["repack"]))["answer_raw"]
    if a==b: same+=1
    else: diff.append((i,a,b,gt[i]["gt"]))
print("answers identical on",same,"of",len(ids),"tight crops; differences:",diff)
R=list(csv.DictReader(open("results/exp5_repack.csv")))
for c in IM:
    for k in CF:
        v=[r for r in R if r["cond"]==c and r["cfg"]==k]; g=lambda x:[float(r[x]) for r in v]
        print(c,k,"enc",round(st.median(g("t_img_encode_ms"))),[round(min(g("t_img_encode_ms"))),round(max(g("t_img_encode_ms")))],"imgLM",round(st.median(g("t_img_decode_ms"))),"text",round(st.median(g("t_text_prefill_ms"))),"TTFT",round(st.median(g("t_prompt_eval_ms"))),"tokens",v[0]["n_img_tokens"],"ans",sorted({r["answer_raw"] for r in v}))
