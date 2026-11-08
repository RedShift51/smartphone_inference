import sys,csv,statistics as st,collections; sys.path.insert(0,"llama.cpp/gguf-py")
from gguf import GGUFReader
def params(path):
    r=GGUFReader(path); tot=0; emb=0; byname={}
    for t in r.tensors:
        n=int(t.n_elements) if hasattr(t,"n_elements") else int(__import__("numpy").prod(t.shape)); tot+=n
        if "token_embd" in t.name or t.name in("output.weight",): emb+=n
        byname[t.name]=n
    return tot,emb,byname
mt,_,_=params("models/mmproj-LFM2-VL-450M-F16.gguf"); lt,le,lb=params("models/LFM2-VL-450M-F16.gguf")
print(f"mmproj params {mt/1e6:.1f}M; LM total {lt/1e6:.1f}M, embedding/output {le/1e6:.1f}M (names: {[k for k in lb if 'token_embd' in k or k=='output.weight']}), LM w/o embeddings {(lt-le)/1e6:.1f}M")
L,d,dm=12,768,3072; SHUF=4  # pixel shuffle 2x2 -> 4 patches per token
R=list(csv.DictReader(open("results/exp1_runs.csv"))); g=collections.defaultdict(list)
for r in R: g[r["cond"]].append(r)
out=["| cond | tokens | patches N | encoder GFLOP | encoder ms | encoder GOPS | LM prefill GOP | image prefill ms | LM GOPS |","|---|---|---|---|---|---|---|---|---|"]
for c in ["A2","B","C","D","E"]:
    ge=[];gl=[];tk=[];fl=[];fl2=[]
    for r in g[c]:
        t=float(r["n_img_tokens"]); N=t*SHUF
        F=L*(2*N*(4*d*d+2*d*dm)+4*N*N*d); Fl=2*(lt-le)*t
        ge.append(F/(float(r["t_img_encode_ms"])/1000)/1e9); gl.append(Fl/(float(r["t_img_decode_ms"])/1000)/1e9); tk.append(t); fl.append(F/1e9); fl2.append(Fl/1e9)
    out.append(f"| {c} | {st.median(tk):.0f} | {st.median(tk)*SHUF:.0f} | {st.median(fl):.0f} | {st.median(float(r['t_img_encode_ms']) for r in g[c]):.0f} | {st.median(ge):.1f} | {st.median(fl2):.0f} | {st.median(float(r['t_img_decode_ms']) for r in g[c]):.0f} | {st.median(gl):.1f} |")
open("results/exp3_gops.md","w").write("\n".join(out)+"\n"); print("\n".join(out))
