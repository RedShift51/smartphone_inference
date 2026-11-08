import csv,re,statistics as st,collections,numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R=list(csv.DictReader(open("results/exp1_runs.csv")))
_c=collections.Counter(r["img_id"] for r in R); R=[r for r in R if _c[r["img_id"]]==6]  # only fully measured images (run stopped early)
def rescore(raw):
    s=re.sub(r"[^\d.,]","",raw or "").strip(".,")
    if re.fullmatch(r"\d{1,3}([.,]\d{3})+",s): s=re.sub(r"[.,]","",s)
    else: s=s.replace(",",".")
    try: return float(s)
    except: return None
for r in R:
    v=rescore(r["answer_raw"]); r["ans_num"]=v; r["ok"]=int(v is not None and v==float(r["gt"]))
    for k in("n_img_tokens","t_img_encode_ms","t_img_decode_ms","t_prompt_eval_ms","t_eval_ms","t_text_prefill_ms","ttft_ms"): r[k]=float(r[k])
order=["A","B","A2","C","D","E"]; g=collections.defaultdict(list)
for r in R: g[r["cond"]].append(r)
# detector amortised cost: 805 ms CPU / avg tags per frame (from price-tag dataset manifest)
import os
tags=sum(1 for _ in csv.DictReader(open("data/price_tag_dataset/manifest.csv"))); imgs=len({x["image"] for x in csv.DictReader(open("data/price_tag_dataset/manifest.csv"))})
per=tags/imgs; det_cpu=805/per; det_gpu=300/per
def q(v,p): return float(np.percentile(v,p))
lines=["| cond | n | tokens(med) | encode ms med [p25,p75] | img-prefill ms | text-prefill ms | TTFT ms med [p25,p75] | honest TTFT(+det CPU) | correct |","|---|---|---|---|---|---|---|---|---|"]
for c in order:
    v=g.get(c,[]); 
    if not v: continue
    f=lambda k:[x[k] for x in v]
    hon=st.median(f("ttft_ms"))+(det_cpu if c in("C","D","E") else 0)
    lines.append(f"| {c} | {len(v)} | {st.median(f('n_img_tokens')):.0f} | {st.median(f('t_img_encode_ms')):.0f} [{q(f('t_img_encode_ms'),25):.0f},{q(f('t_img_encode_ms'),75):.0f}] | {st.median(f('t_img_decode_ms')):.0f} | {st.median(f('t_text_prefill_ms')):.0f} | {st.median(f('ttft_ms')):.0f} [{q(f('ttft_ms'),25):.0f},{q(f('ttft_ms'),75):.0f}] | {hon:.0f} | {sum(x['ok'] for x in v)}/{len(v)} |")
open("results/summary_table.md","w").write(f"avg tags/frame in the source dataset: {per:.2f}; detector amortised CPU {det_cpu:.0f} ms, GPU {det_gpu:.0f} ms per tag\n\n"+"\n".join(lines)+"\n")
print("\n".join(lines)); print("tags/frame",round(per,2),"det per tag CPU",round(det_cpu))
col=dict(zip(order,["C0","C1","C2","C3","C4","C5"]))
# 1: tokens vs encoder time
plt.figure(figsize=(6,4)); X=np.array([r["n_img_tokens"] for r in R]); Y=np.array([r["t_img_encode_ms"] for r in R])
for c in order: plt.scatter([r["n_img_tokens"] for r in g[c]],[r["t_img_encode_ms"] for r in g[c]],s=12,label=c,c=col[c])
a,b=np.polyfit(X,Y,1); r2=1-((Y-(a*X+b))**2).sum()/((Y-Y.mean())**2).sum(); xs=np.linspace(X.min(),X.max(),50); plt.plot(xs,a*xs+b,"k--",lw=1)
plt.xlabel("image tokens (all tiles)"); plt.ylabel("encoder ms"); plt.title(f"H1: slope {a:.1f} ms/token, R²={r2:.3f}"); plt.legend(); plt.tight_layout(); plt.savefig("results/plot1_tokens_vs_encode.png",dpi=130); print("H1 slope",round(a,2),"R2",round(r2,3))
# 2: stacked
plt.figure(figsize=(6,4)); cs=[c for c in order if g.get(c)]; bot=np.zeros(len(cs))
for k,l in(("t_img_encode_ms","encoder"),("t_img_decode_ms","image prefill (LM)"),("t_text_prefill_ms","text prefill"),("t_eval_ms","decode")):
    m=np.array([st.median(x[k] for x in g[c]) for c in cs]); plt.bar(cs,m,bottom=bot,label=l); bot+=m
plt.ylabel("ms (median)"); plt.yscale("log" if False else "linear"); plt.legend(); plt.title("H3: time by stage"); plt.tight_layout(); plt.savefig("results/plot2_stages.png",dpi=130)
# 3: accuracy vs tokens
plt.figure(figsize=(6,4))
for c in cs: plt.scatter(st.median(x["n_img_tokens"] for x in g[c]),100*sum(x["ok"] for x in g[c])/len(g[c]),s=60,c=col[c]); plt.annotate(c,(st.median(x["n_img_tokens"] for x in g[c]),100*sum(x["ok"] for x in g[c])/len(g[c])),xytext=(4,4),textcoords="offset points")
plt.xlabel("median image tokens"); plt.ylabel("accuracy %"); plt.title("H2/H4: accuracy vs tokens"); plt.tight_layout(); plt.savefig("results/plot3_accuracy.png",dpi=130)
# disagreements A vs D
by={(r["img_id"],r["cond"]):r for r in R}
print("A vs D disagreements:",[(i,by[(i,'A')]["answer_raw"],by[(i,'D')]["answer_raw"],by[(i,'A')]["gt"]) for i in sorted({r['img_id'] for r in R}) if (i,'A') in by and (i,'D') in by and by[(i,'A')]["ans_num"]!=by[(i,'D')]["ans_num"]][:8])
