from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from PIL import Image

P=Path(__file__).resolve().parents[1]
C=["Normal","Pneumonia","TB","COVID"]
DEV="cuda" if torch.cuda.is_available() else "cpu"
MANIFEST=P/"data/manifest.csv"
MODEL=P/"models/disease_model.pt"
REPORTS=P/"reports"
REPORTS.mkdir(parents=True,exist_ok=True)

class DS(Dataset):
    def __init__(self,df):
        self.df=df.reset_index(drop=True)
        self.t=transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([.485]*3,[.229]*3)
        ])
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]
        return self.t(Image.open(r.path).convert("RGB")), C.index(r.disease)

def main():
    if not MANIFEST.exists(): raise SystemExit("data/manifest.csv not found.")
    if not MODEL.exists(): raise SystemExit("models/disease_model.pt not found.")
    df=pd.read_csv(MANIFEST)
    df=df[df.disease.isin(C)].copy()
    _,holdout=train_test_split(df,test_size=.20,random_state=42,stratify=df.disease)
    loader=DataLoader(DS(holdout),batch_size=16,shuffle=False,num_workers=0)
    m=models.mobilenet_v3_small(weights=None)
    m.classifier[-1]=nn.Linear(m.classifier[-1].in_features,4)
    ck=torch.load(MODEL,map_location=DEV)
    m.load_state_dict(ck["state_dict"])
    m.to(DEV).eval()
    ys=[]; ps=[]
    with torch.no_grad():
        for x,y in loader:
            p=m(x.to(DEV)).argmax(1).cpu().numpy()
            ys.extend(y.numpy()); ps.extend(p)
    acc=accuracy_score(ys,ps)
    report=classification_report(ys,ps,target_names=C,digits=4)
    cm=confusion_matrix(ys,ps,labels=range(4))
    print("\nDISEASE MODEL HOLDOUT EVALUATION")
    print("="*65)
    print("Device:",DEV)
    print("Holdout images:",len(holdout))
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\nClassification report:\n"+report)
    print("Confusion matrix (rows=true, columns=predicted):")
    print(pd.DataFrame(cm,index=C,columns=C).to_string())
    out=REPORTS/"disease_classification_report.txt"
    out.write_text(
        "DISEASE MODEL HOLDOUT EVALUATION\n\n"+
        f"Holdout images: {len(holdout)}\nAccuracy: {acc:.4f} ({acc*100:.2f}%)\n\n"+
        report+"\nConfusion matrix (rows=true, columns=predicted):\n"+
        pd.DataFrame(cm,index=C,columns=C).to_string()+"\n",
        encoding="utf-8")
    print("\nSaved:",out)
    print("\nNOTE: This is the same 20% holdout split used as validation during training, not an independent untouched test set.")

if __name__=="__main__": main()
