import csv, random, os, collections
from PIL import Image, ImageDraw
SRC="data/price_tag_dataset"; OUT="data/exp1"; N=30; SEED=42
rows=list(csv.DictReader(open(f"{SRC}/manifest.csv")))
by=collections.defaultdict(list)
for r in rows:
    x1,y1,x2,y2=[float(r[k]) for k in("x1","y1","x2","y2")]
    v=r["value"].strip()
    if v.isdigit() and len(v)==4 and (x2-x1)>=60 and (y2-y1)>=25:
        by[r["image"]].append((x1,y1,x2,y2,v))
rng=random.Random(SEED); imgs=sorted(by); rng.shuffle(imgs)
for d in("full","full_b","full_a2","tight","ctx4"): os.makedirs(f"{OUT}/{d}",exist_ok=True)
def frame(im,b,s):  # red box 3px, drawn just outside the tag so digits stay visible
    d=ImageDraw.Draw(im); x1,y1,x2,y2=[c*s for c in b]
    for k in range(3): d.rectangle([x1-1-k,y1-1-k,x2+1+k,y2+1+k],outline=(255,0,0))
def fit(W,H,target):  # w,h multiples of 32, tokens=(w/32)*(h/32) closest to target (<= target if 'max')
    best=None
    for w in range(32,1025,32):
        for h in range(32,1025,32):
            t=(w//32)*(h//32)
            if t>target: continue
            err=abs((w/h)-(W/H))/(W/H)+ (target-t)/target*0.5
            if best is None or err<best[0]: best=(err,w,h)
    return best[1],best[2]
out=[]
for i,name in enumerate(imgs[:N]):
    x1,y1,x2,y2,v=rng.choice(by[name]); im=Image.open(f"{SRC}/{name}").convert("RGB"); W,H=im.size
    iid=f"{i:02d}"; b=(x1,y1,x2,y2)
    s=1024/max(W,H) if max(W,H)>1024 else 1.0
    full=im.resize((round(W*s),round(H*s)),Image.LANCZOS); frame(full,b,s); full.save(f"{OUT}/full/{iid}.jpg",quality=95)
    for d,tg in(("full_b",64),("full_a2",256)):
        fw,fh=fit(W,H,tg); r=im.resize((fw,fh),Image.LANCZOS); frame(r,(x1*fw/W,y1*fh/H,x2*fw/W,y2*fh/H),1); r.save(f"{OUT}/{d}/{iid}.jpg",quality=95)
    bw,bh=x2-x1,y2-y1; cx,cy=(x1+x2)/2,(y1+y2)/2
    tx1,ty1,tx2,ty2=max(0,x1-.1*bw),max(0,y1-.1*bh),min(W,x2+.1*bw),min(H,y2+.1*bh)
    t=im.crop((round(tx1),round(ty1),round(tx2),round(ty2))); frame(t,(x1-round(tx1),y1-round(ty1),x2-round(tx1),y2-round(ty1)),1); t.save(f"{OUT}/tight/{iid}.jpg",quality=95)
    half=2*max(bw,bh); cx1,cy1,cx2,cy2=max(0,cx-half),max(0,cy-half),min(W,cx+half),min(H,cy+half)
    c=im.crop((round(cx1),round(cy1),round(cx2),round(cy2))); frame(c,(x1-round(cx1),y1-round(cy1),x2-round(cx1),y2-round(cy1)),1); c.save(f"{OUT}/ctx4/{iid}.jpg",quality=95)
    out.append(dict(img_id=iid,src=name,gt=v,box=f'{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}',W=W,H=H,full_w=full.width,full_h=full.height,tight_w=t.width,tight_h=t.height,ctx4_w=c.width,ctx4_h=c.height))
w=csv.DictWriter(open(f"{OUT}/gt.csv","w"),fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
print(len(out),"tags;",len(by),"eligible images")
