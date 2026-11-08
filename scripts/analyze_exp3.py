import csv,re,math,statistics as st,collections,numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R=list(csv.DictReader(open("results/exp1_runs.csv")))
def to_num(raw):
    s=re.sub(r"[^\d.,]","",raw or "")
    s=re.sub(r"[.,]00$","",s).strip(".,")
    if re.fullmatch(r"\d{1,3}([.,]\d{3})+",s): s=re.sub(r"[.,]","",s)
    else: s=s.replace(",",".")
    try: return float(s)
    except ValueError: return None
def dig(raw): return re.sub(r"\D","",raw or "").rstrip("0")
for r in R:
    v=to_num(r["answer_raw"]); g=float(r["gt"])
    r["strict"]=int(v is not None and abs(v-g)<1e-9)
    d=dig(r["answer_raw"]); r["digits"]=int(bool(d) and d==dig(r["gt"]))
    for k in("n_img_tokens","t_img_encode_ms","t_img_decode_ms","t_text_prefill_ms","t_eval_ms","ttft_ms"): r[k]=float(r[k])
LAB={"A":"full shelf (tiled)","A2":"full shelf (1 tile)","B":"shelf ↓ to ~70 tok","C":"tag + context","D":"tight tag crop","E":"tight crop, 20 tok"}
order=["A","A2","B","C","D","E"]; g=collections.defaultdict(list)
for r in R: g[r["cond"]].append(r)
by={(r["img_id"],r["cond"]):r for r in R}; ids=sorted({r["img_id"] for r in R})
# paired sign test B vs D (strict)
onlyD=sum(1 for i in ids if by[(i,"D")]["strict"] and not by[(i,"B")]["strict"]); onlyB=sum(1 for i in ids if by[(i,"B")]["strict"] and not by[(i,"D")]["strict"])
n=onlyD+onlyB; k=min(onlyD,onlyB); p=min(1.0,2*sum(math.comb(n,j) for j in range(k+1))/2**n) if n else 1.0
def q(v,p): return float(np.percentile(v,p))
T=["| condition | N | tokens | encoder ms (med [p25,p75]) | image prefill ms | text prefill ms | TTFT ms (med [p25,p75]) | strict | digits |","|---|---|---|---|---|---|---|---|---|"]
for c in order:
    v=g[c]; f=lambda k:[x[k] for x in v]
    T.append(f"| {c} {LAB[c]} | {len(v)} | {st.median(f('n_img_tokens')):.0f} | {st.median(f('t_img_encode_ms')):.0f} [{q(f('t_img_encode_ms'),25):.0f},{q(f('t_img_encode_ms'),75):.0f}] | {st.median(f('t_img_decode_ms')):.0f} | {st.median(f('t_text_prefill_ms')):.0f} | {st.median(f('ttft_ms')):.0f} [{q(f('ttft_ms'),25):.0f},{q(f('ttft_ms'),75):.0f}] | {sum(x['strict'] for x in v)}/{len(v)} | {sum(x['digits'] for x in v)}/{len(v)} |")
txt="\n".join(T)+f"\n\nPaired B vs D (strict, N={len(ids)}): only D correct {onlyD}, only B correct {onlyB}, both/neither {len(ids)-n}; two-sided exact sign test p = {p:.3f}.\n`digits` = digit string of the answer equals the digit string of GT after stripping trailing zeros; it is more lenient than `strict` and allows false matches (e.g. 127 vs 1270).\n"
open("results/exp3_table.md","w").write(txt); print(txt)
# ---- post plots (A excluded, shown separately) ----
plt.rcParams.update({"font.size":15,"axes.spines.top":False,"axes.spines.right":False})
PH=[("t_img_encode_ms","Vision encoder","#C8553D"),("t_img_decode_ms","Image tokens in LM","#2F6690"),("t_text_prefill_ms","Prompt text","#81B29A"),("t_eval_ms","Answer decode","#BBBBBB")]
cs=["A2","B","C","D","E"]
fig,ax=plt.subplots(figsize=(7,7.5)); bot=np.zeros(len(cs))
for k,l,col in PH:
    m=np.array([st.median(x[k] for x in g[c]) for c in cs])/1000; ax.bar(range(len(cs)),m,bottom=bot,label=l,color=col,width=.65); bot+=m
for i,c in enumerate(cs): ax.text(i,bot[i]+.12,f"{bot[i]:.1f}s",ha="center",fontweight="bold")
ax.set_xticks(range(len(cs))); ax.set_xticklabels(["full shelf\n(1 tile)","shelf ↓\nto ~70 tok","tag +\ncontext","tight tag\ncrop","tight crop,\n20 tok"],fontsize=12)
ax.set_ylabel("time to first token, s (median)"); ax.set_title("Where the time goes\nLFM2-VL-450M, Helio G99, CPU",fontsize=17); ax.legend(frameon=False,loc="upper right"); fig.tight_layout(); fig.savefig("results/post/time_by_phase.png",dpi=160)
# A alone
fig,ax=plt.subplots(figsize=(7,3.6)); v=g["A"]; a=[x["ttft_ms"]/1000 for x in v]
ax.boxplot(a,orientation="horizontal",widths=.5); ax.set_yticks([1]); ax.set_yticklabels(["full shelf\n(tiled)"]); ax.set_xlabel("time to first token, s"); ax.set_title(f"Tiled full frame: median {st.median(a):.1f}s, max {max(a):.0f}s",fontsize=15); fig.tight_layout(); fig.savefig("results/post/full_shelf_tiled_ttft.png",dpi=160)
# tokens vs encoder
fig,ax=plt.subplots(figsize=(7,5)); XX=[];YY=[]
for c,col in zip(cs,["#2F6690","#E0A458","#81B29A","#C8553D","#6D597A"]):
    x=[r["n_img_tokens"] for r in g[c]]; y=[r["t_img_encode_ms"]/1000 for r in g[c]]; XX+=x; YY+=y; ax.scatter(x,y,s=40,c=col,label=LAB[c])
XX=np.array(XX); YY=np.array(YY); s_,b_=np.polyfit(XX,YY,1); r2=1-((YY-(s_*XX+b_))**2).sum()/((YY-YY.mean())**2).sum(); xs=np.linspace(0,XX.max(),20); ax.plot(xs,s_*xs+b_,"k--",lw=1)
ax.set_xlabel("image tokens"); ax.set_ylabel("encoder time, s"); ax.set_title(f"Encoder cost per token: {s_*1000:.0f} ms (R²={r2:.2f})",fontsize=16); ax.legend(frameon=False,fontsize=11); fig.tight_layout(); fig.savefig("results/post/encoder_vs_tokens.png",dpi=160)
print("slope ms/token",round(s_*1000,1),"R2",round(r2,3))
# accuracy
fig,ax=plt.subplots(figsize=(7,5))
for c,col in zip(cs,["#2F6690","#E0A458","#81B29A","#C8553D","#6D597A"]):
    xx=st.median(r["n_img_tokens"] for r in g[c]); ax.scatter(xx,100*sum(r["strict"] for r in g[c])/len(g[c]),s=120,c=col); ax.annotate(LAB[c],(xx,100*sum(r["strict"] for r in g[c])/len(g[c])),xytext=(6,6),textcoords="offset points",fontsize=11)
ax.set_xlabel("image tokens (median)"); ax.set_ylabel("price read exactly, %"); ax.set_ylim(0,100); ax.set_title(f"Accuracy vs token budget (N={len(ids)})",fontsize=16); fig.tight_layout(); fig.savefig("results/post/accuracy_vs_tokens.png",dpi=160)
