from pathlib import Path
import pandas as pd, numpy as np, torch, time, sys
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image

P=Path(__file__).resolve().parents[1]
M=P/"data/manifest.csv"; MD=P/"models"; CP=MD/"checkpoints"
MD.mkdir(parents=True,exist_ok=True); CP.mkdir(parents=True,exist_ok=True)
C=["Normal","Pneumonia","TB","COVID"]; DEV="cuda" if torch.cuda.is_available() else "cpu"
EPOCHS=8; BATCH=8; WORKERS=0

class DS(Dataset):
    def __init__(self,df,train=False):
        self.df=df.reset_index(drop=True)
        ops=[transforms.Grayscale(3),transforms.Resize((224,224))]
        if train: ops += [transforms.RandomHorizontalFlip(.5),transforms.RandomRotation(5)]
        ops += [transforms.ToTensor(),transforms.Normalize([.485]*3,[.229]*3)]
        self.t=transforms.Compose(ops)
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]
        return self.t(Image.open(r.path).convert("RGB")),C.index(r.disease)

def make_model():
    m=models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    m.classifier[-1]=nn.Linear(m.classifier[-1].in_features,4)
    return m

def save_ckpt(m,opt,epoch,best):
    d={"epoch":epoch,"state_dict":m.state_dict(),"optimizer":opt.state_dict(),
       "best_accuracy":best,"classes":C}
    torch.save(d,CP/f"disease_epoch_{epoch}.pt")
    torch.save(d,CP/"disease_latest.pt")

def main():
    if not M.exists(): raise SystemExit("data/manifest.csv not found.")
    df=pd.read_csv(M); df=df[df.disease.isin(C)].copy()
    if df.disease.nunique()<4: raise SystemExit("All four disease classes were not found.")
    tr,va=train_test_split(df,test_size=.2,random_state=42,stratify=df.disease)
    tl=DataLoader(DS(tr,True),BATCH,shuffle=True,num_workers=WORKERS,pin_memory=DEV=="cuda")
    vl=DataLoader(DS(va),BATCH,shuffle=False,num_workers=WORKERS,pin_memory=DEV=="cuda")
    m=make_model().to(DEV)
    w=compute_class_weight("balanced",classes=np.array(C),y=tr.disease)
    loss=nn.CrossEntropyLoss(weight=torch.tensor(w,dtype=torch.float32,device=DEV))
    opt=torch.optim.AdamW(m.parameters(),2e-4,weight_decay=1e-4)
    latest=CP/"disease_latest.pt"; start=0; best=0.0
    if latest.exists():
        try:
            d=torch.load(latest,map_location=DEV); m.load_state_dict(d["state_dict"]); opt.load_state_dict(d["optimizer"])
            start=int(d["epoch"]); best=float(d.get("best_accuracy",0))
            print(f"Resuming from epoch {start}/{EPOCHS}; best val accuracy={best:.4f}")
        except Exception as e: print("Could not load checkpoint:",e)
    total_start=time.time()
    for ep in range(start,EPOCHS):
        num=ep+1; es=time.time(); m.train(); run=cor=tot=0; nb=len(tl)
        print(f"\n{'='*60}\nEPOCH {num}/{EPOCHS} | OVERALL {(num/EPOCHS)*100:.1f}%\n{'='*60}")
        for bi,(x,y) in enumerate(tl,1):
            x,y=x.to(DEV),y.to(DEV); opt.zero_grad(); z=m(x); l=loss(z,y); l.backward(); opt.step()
            run+=l.item()*len(y); cor+=(z.argmax(1)==y).sum().item(); tot+=len(y)
            if bi%10==0 or bi==nb:
                elapsed=time.time()-es; rate=bi/max(elapsed,.001); eta=(nb-bi)/max(rate,.001)/60
                sys.stdout.write(f"\rBatch {bi}/{nb} ({bi/nb*100:5.1f}%) | loss {run/tot:.4f} | acc {cor/tot:.4f} | ETA {eta:.1f} min")
                sys.stdout.flush()
        print()
        m.eval(); vn=vc=vloss=0
        with torch.no_grad():
            for x,y in vl:
                z=m(x.to(DEV)); y=y.to(DEV); l=loss(z,y); vloss+=l.item()*len(y); vc+=(z.argmax(1)==y).sum().item(); vn+=len(y)
        va_acc=vc/vn; va_loss=vloss/vn; train_acc=cor/tot
        print(f"Train acc={train_acc:.4f} | Val loss={va_loss:.4f} | Val acc={va_acc:.4f} | epoch time={(time.time()-es)/60:.1f} min")
        best=max(best,va_acc); save_ckpt(m,opt,num,best)
        if va_acc>=best:
            torch.save({"state_dict":m.state_dict(),"classes":C,"best_validation_accuracy":best},MD/"disease_model.pt")
        print(f"Checkpoint saved. Overall progress={num/EPOCHS*100:.1f}%")
    print(f"\nDONE. Best validation accuracy={best:.4f}")
    print("Model:",MD/"disease_model.pt")
if __name__=="__main__": main()
