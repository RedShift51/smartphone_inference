import sys,csv,time,random,statistics as st; sys.path.insert(0,"scripts"); from vlm import *
CF={"base_t4":dict(t=4),"repack_t4":dict(t=4,pre="CLIP_REPACK=1"),"base_big2":dict(t=2,pre="taskset c0"),"repack_big2":dict(t=2,pre="taskset c0 env CLIP_REPACK=1")}
for k in CF: CF[k]["binary"]="llama-mtmd-cli-repack"
IM={"D":"exp1/tight/09.jpg","A2":"exp1/full_a2/10.jpg"}; REPS=10
COLS=["rep","cond","cfg","n_img_tokens","t_img_encode_ms","t_img_decode_ms","t_text_prefill_ms","t_prompt_eval_ms","answer_raw","batt_c","cpu_c"]
f=open("results/exp5_repack_precise.csv","w",newline=""); w=csv.DictWriter(f,fieldnames=COLS); w.writeheader(); base=temps(); names=list(CF)
for k in CF: run(IM["D"],**CF[k])
rng=random.Random(7)
for rep in range(REPS):
    for cond,img in IM.items():
        order=names[:]; rng.shuffle(order)
        for k in order:
            tb=temps()
            while tb[0]>=base[0]+3 or tb[1]>=base[1]+10: time.sleep(15); tb=temps()
            p=parse(run(img,**CF[k])); p.update(rep=rep,cond=cond,cfg=k,t_text_prefill_ms=round(p["t_prompt_eval_ms"]-p["t_img_encode_ms"]-p["t_img_decode_ms"],1),batt_c=tb[0],cpu_c=tb[1])
            w.writerow({c:p.get(c) for c in COLS}); f.flush()
    print("rep",rep,"done",temps(),flush=True)
print("FINISHED",flush=True)
