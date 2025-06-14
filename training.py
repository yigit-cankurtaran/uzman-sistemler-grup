# ----------------------------------------------------------------------------
# training.py — weighted RIASEC w/ auto‑label grouping + beefy MLP
# ----------------------------------------------------------------------------
"""
key upgrades for tiny/imbalanced dataset accuracy
-------------------------------------------------
1. **feature scaling** – z‑score the 6‑dim trait vector per column.
2. **roomier net** – 128‑64‑32 hidden dims + dropout(0.2).
3. **min_epochs** – early‑stop can’t trigger before 20 epochs.
4. **class‑weighted CE** – inverse‑freq weights mitigate imbalance.
5. cmd flags unchanged so old calls still run.
"""

import argparse, json, math, pathlib, random
from typing import List, Tuple
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

RIASEC = {
    'realistic':0,'investigative':1,'artistic':2,'social':3,'enterprising':4,'conventional':5
}
TRAIT_COLS=list(RIASEC.keys())
WEIGHTS={'first interest area':4.5,'second interest area':4.0,'third interest area':3.5}

# ───────────────────────── data utils ─────────────────────────

def encode_traits(row:pd.Series)->List[float]:
    vec=np.zeros(6,np.float32)
    for col,w in WEIGHTS.items():
        if col in row and pd.notna(row[col]):
            idx=RIASEC.get(str(row[col]).strip().lower())
            if idx is not None: vec[idx]+=w
    return vec.tolist()


def group_labels(df:pd.DataFrame,label_col:str)->pd.Series:
    cnt=df[label_col].value_counts()
    if cnt.min()>1: return df[label_col]
    if 'O*NET-SOC Code' in df.columns:
        print('⚠️  grouping labels by SOC major group (first 2 digits)')
        return df['O*NET-SOC Code'].astype(str).str[:2]
    print('⚠️  grouping labels by first word of title')
    return df[label_col].str.split().str[0].str.lower()


def scale_matrix(mat:np.ndarray)->np.ndarray:
    mean=mat.mean(0,keepdims=True); std=mat.std(0,keepdims=True)+1e-6
    return (mat-mean)/std


def load_dataset(csv: pathlib.Path,label_col:str)->Tuple[np.ndarray,np.ndarray,List[str]]:
    df=pd.read_csv(csv)
    if label_col not in df.columns:
        label_col='O*NET-SOC Title' if 'O*NET-SOC Title' in df.columns else df.columns[-1]
        print(f"⚠️  using '{label_col}' as label column")
    traits=df.apply(encode_traits,axis=1,result_type='expand'); traits.columns=TRAIT_COLS
    x_raw=traits.values.astype(np.float32)
    x=scale_matrix(x_raw)
    y=group_labels(df,label_col).astype(str).values
    return x,y,TRAIT_COLS

# ───────────────────────── torch classes ─────────────────────────
class DS(torch.utils.data.Dataset):
    def __init__(self,x,y): self.x=torch.tensor(x); self.y=torch.tensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return self.x[i],self.y[i]

class MLP(nn.Module):
    def __init__(self,d_in,n_cls):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(d_in,128),nn.ReLU(),nn.Dropout(.2),
            nn.Linear(128,64),nn.ReLU(),nn.Dropout(.2),
            nn.Linear(64,32),nn.ReLU(),
            nn.Linear(32,n_cls))
    def forward(self,x): return self.net(x)

class Stopper:
    def __init__(self,patience,min_epochs=20):
        self.p=patience; self.m=min_epochs; self.best=0.; self.c=0
    def step(self,ep,acc):
        if ep<self.m: return False
        if acc>self.best+1e-4: self.best, self.c=acc,0; return False
        self.c+=1; return self.c>=self.p

# ───────────────────────── training fn ─────────────────────────

def train(csv,out,label_col,epochs=200,lr=1e-3,patience=10):
    x,y_str,feat_cols=load_dataset(csv,label_col)
    le=LabelEncoder(); y=le.fit_transform(y_str).astype(np.int64)

    # class weights
    unique,counts=np.unique(y,return_counts=True)
    w=torch.tensor(1.0/counts,dtype=torch.float32)

    n=len(x); test_sz=max(1,math.floor(.1*n)); val_sz=max(1,math.floor(.15*n))
    x_tmp,x_test,y_tmp,y_test=train_test_split(x,y,test_size=test_sz,random_state=SEED)
    x_tr,x_val,y_tr,y_val=train_test_split(x_tmp,y_tmp,test_size=val_sz,random_state=SEED)

    tr=torch.utils.data.DataLoader(DS(x_tr,y_tr),batch_size=32,shuffle=True)
    val=torch.utils.data.DataLoader(DS(x_val,y_val),batch_size=32)
    tst=torch.utils.data.DataLoader(DS(x_test,y_test),batch_size=32)

    model=MLP(len(feat_cols),len(le.classes_))
    opt=optim.Adam(model.parameters(),lr=lr)
    crit=nn.CrossEntropyLoss(weight=w)
    stop=Stopper(patience)

    for ep in range(1,epochs+1):
        model.train(); loss_sum=0.
        for xb,yb in tr:
            opt.zero_grad(); l=crit(model(xb), yb); l.backward(); opt.step(); loss_sum+=l.item()*xb.size(0)
        model.eval(); correct=0
        with torch.no_grad():
            for xb,yb in val: correct+=(model(xb).argmax(1)==yb).sum().item()
        acc=correct/len(y_val)
        if ep==1 or ep%10==0: print(f"ep {ep:03d} loss {loss_sum/len(y_tr):.3f} val_acc {acc:.3f}")
        if stop.step(ep,acc):
            print(f"early stop @ {ep} best_val {stop.best:.3f}"); break

    model.eval(); correct=0
    with torch.no_grad():
        for xb,yb in tst: correct+=(model(xb).argmax(1)==yb).sum().item()
    print(f"test_acc {correct/len(y_test):.3f}")

    torch.save({
        'model_state_dict':model.state_dict(),
        'label_encoder':json.dumps(le.classes_.tolist()),
        'feature_cols':json.dumps(feat_cols),
    },out)
    print('model saved →',out.resolve())

# ───────────────────────── cli ─────────────────────────
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data_path',type=pathlib.Path,required=True)
    p.add_argument('--output_path',type=pathlib.Path,default='model.pth')
    p.add_argument('--label_col',type=str,default='O*NET-SOC Title')
    p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--patience',type=int,default=10)
    a=p.parse_args(); train(a.data_path,a.output_path,a.label_col,a.epochs,a.lr,a.patience)

# ----------------------------------------------------------------------------
# inference.py unchanged
# ----------------------------------------------------------------------------

