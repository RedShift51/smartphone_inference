import sys,time,json,statistics; sys.path.insert(0,"scripts"); from vlm import *
print("temps",temps())
run("exp1/full/00.jpg"); run("exp1/full/00.jpg")  # warmups
res={2:[],4:[],6:[]}
for rep in range(3):
    for t in (2,4,6):
        while temps()[0]>temps.__defaults__[0] if False else False: pass
        log=run("exp1/full/00.jpg",t=t); open(f"logs/threads_t{t}_{rep}.log","w").write(log)
        d=parse(log); res[t].append(d); print(t,rep,d["n_img_tokens"],d["t_img_encode_ms"],d["t_img_decode_ms"],d["t_prompt_eval_ms"],d["answer_raw"],temps(),flush=True)
for t,v in res.items(): print("t=",t,"median prompt_eval",statistics.median(x["t_prompt_eval_ms"] for x in v),"median enc",statistics.median(x["t_img_encode_ms"] for x in v))
