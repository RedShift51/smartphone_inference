import sys,csv,time,statistics as st; sys.path.insert(0,"scripts"); from vlm import *
CFG={"cpu_t4":dict(t=4),"cpu_big2":dict(t=2,pre="taskset c0"),"vk_all":dict(binary="llama-mtmd-cli-vkfix",dev="Vulkan0",t=4),"vk_lm_only":dict(binary="llama-mtmd-cli-vkfix",dev="Vulkan0",t=4,extra="--no-mmproj-offload")}
IMGS={"D":"exp1/tight/09.jpg","A2":"exp1/full_a2/10.jpg"}
COLS=["rep","cond","cfg","n_img_tokens","t_img_encode_ms","t_img_decode_ms","t_text_prefill_ms","t_prompt_eval_ms","t_eval_ms","answer_raw","batt_c","cpu_c","wait_s"]
f=open("results/exp3_gpu.csv","w",newline=""); w=csv.DictWriter(f,fieldnames=COLS); w.writeheader(); base=temps(); names=list(CFG)
for rep in range(3):
    for cond,img in IMGS.items():
        for k in (names[rep%4:]+names[:rep%4]):
            t0=time.time(); tb=temps()
            while tb[0]>=base[0]+3 or (tb[1] and base[1] and tb[1]>=base[1]+10): time.sleep(15); tb=temps()
            log=run(img,**CFG[k]); open(f"logs/gpu_{cond}_{k}_{rep}.log","w").write(log); p=parse(log)
            if p.get("t_prompt_eval_ms") is None: p.update(t_prompt_eval_ms=0,t_text_prefill_ms=0)
            else: p["t_text_prefill_ms"]=round(p["t_prompt_eval_ms"]-p["t_img_encode_ms"]-p["t_img_decode_ms"],1)
            p.update(rep=rep,cond=cond,cfg=k,batt_c=tb[0],cpu_c=tb[1],wait_s=round(time.time()-t0)); w.writerow({c:p.get(c) for c in COLS}); f.flush()
            print(rep,cond,k,p["n_img_tokens"],p["t_img_encode_ms"],p["t_img_decode_ms"],p["t_text_prefill_ms"],p["answer_raw"][:30],tb,flush=True)
f.close()
