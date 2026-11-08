import sys,csv,time,statistics as st; sys.path.insert(0,"scripts"); from vlm import *
IMG="exp1/tight/09.jpg"; CFG={"t4_unpinned":dict(t=4),"t2_big_cores":dict(t=2,pre="taskset c0"),"t8_unpinned":dict(t=8)}
COLS=["rep","cfg","n_img_tokens","t_img_encode_ms","t_img_decode_ms","t_text_prefill_ms","t_prompt_eval_ms","answer_raw","batt_c","cpu_c","wait_s"]
f=open("results/exp3_pinning.csv","w",newline=""); w=csv.DictWriter(f,fieldnames=COLS); w.writeheader()
for k in CFG: run(IMG,**CFG[k])
base=temps(); names=list(CFG)
for rep in range(5):
    for k in (names[rep%3:]+names[:rep%3]):
        t0=time.time(); tb=temps()
        while tb[0]>=base[0]+3 or (tb[1] and base[1] and tb[1]>=base[1]+10): time.sleep(15); tb=temps()
        log=run(IMG,**CFG[k]); open(f"logs/pin_{k}_{rep}.log","w").write(log); p=parse(log)
        p.update(rep=rep,cfg=k,t_text_prefill_ms=round(p["t_prompt_eval_ms"]-p["t_img_encode_ms"]-p["t_img_decode_ms"],1),batt_c=tb[0],cpu_c=tb[1],wait_s=round(time.time()-t0))
        w.writerow({c:p.get(c) for c in COLS}); f.flush()
f.close(); R=list(csv.DictReader(open("results/exp3_pinning.csv")))
for k in CFG:
    v=[r for r in R if r["cfg"]==k]; f_=lambda c:[float(r[c]) for r in v]
    print(k,"enc",round(st.median(f_("t_img_encode_ms"))),[round(min(f_("t_img_encode_ms"))),round(max(f_("t_img_encode_ms")))],"imgLM",round(st.median(f_("t_img_decode_ms"))),"text",round(st.median(f_("t_text_prefill_ms"))),"ttft",round(st.median(f_("t_prompt_eval_ms"))),"ans",sorted({r['answer_raw'] for r in v}))
