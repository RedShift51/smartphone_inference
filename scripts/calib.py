import sys; sys.path.insert(0,"scripts"); from vlm import *
conds={"A":("full/00.jpg",""),"B":("full/00.jpg","--image-max-tokens 64"),"C":("ctx4/00.jpg",""),"D":("tight/00.jpg",""),"E":("tight/00.jpg","--image-min-tokens 16 --image-max-tokens 16")}
for c,(img,ex) in conds.items():
    log=run("exp1/"+img,ex); open(f"logs/calib_{c}.log","w").write(log); print(c,parse(log))
