"""GPU step v2: longer, hard-case-weighted fine-tuning of the cross-encoder.
Paste into ONE Kaggle notebook cell (GPU on; Internet on, or the model attached as an input).

Inputs (Kaggle inputs, found automatically):
  - the first dataset (train/test text tables, val_pairs, test_pairs)       from export_pairs.py
  - train_pairs_v2.parquet (a second small dataset, or added to the first)  from export_pairs_hard.py
Outputs (/kaggle/working): val_scores_e1.parquet, test_scores_e1.parquet  (after epoch 1)
                           val_scores_e2.parquet, test_scores_e2.parquet  (after epoch 2)
If the session runs short, the epoch-1 files are already usable.
"""
import glob
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_local = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/config.json", recursive=True)
          if "minilm" in p.lower() or "marco" in p.lower() or "cross" in p.lower()]
if _local:
    MODEL = _local[0]
MAXLEN, BS_TRAIN, BS_INFER, LR, EPOCHS = 128, 256, 1024, 3e-5, 2
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
DATA = os.path.dirname((glob.glob("/kaggle/input/**/test_pairs.parquet", recursive=True) or glob.glob("**/test_pairs.parquet", recursive=True))[0])
V2 = (glob.glob("/kaggle/input/**/train_pairs_v2.parquet", recursive=True) or glob.glob("**/train_pairs_v2.parquet", recursive=True))[0]
dev = torch.device("cuda")
torch.manual_seed(42); np.random.seed(42)
print("model:", MODEL, "\ndata:", DATA, "\ntrain pairs:", V2, "\ngpu:", torch.cuda.get_device_name(0))


def texts(split, who):
    t = pd.read_parquet(f"{DATA}/{split}_{who}_text.parquet")
    arr = np.empty(int(t["row"].max()) + 1, dtype=object)
    arr[t["row"].to_numpy()] = t["text"].to_numpy()
    return arr


class Pairs(Dataset):
    def __init__(self, a, b, y=None):
        self.a, self.b, self.y = a, b, y

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        return i


tok = AutoTokenizer.from_pretrained(MODEL)


def collate_for(ds):
    def collate(idx):
        enc = tok(list(ds.a[idx]), list(ds.b[idx]), truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt")
        if ds.y is not None:
            enc["labels"] = torch.tensor(ds.y[idx], dtype=torch.float32)
        return enc
    return collate


@torch.no_grad()
def score(model, ds, t0):
    """length-sorted batches (much less padding), results returned in the original order"""
    model.eval()
    order = np.argsort(np.fromiter((len(a) + len(b) for a, b in zip(ds.a, ds.b)), dtype=np.int32, count=len(ds)))
    sds = Pairs(ds.a[order], ds.b[order])
    out = np.empty(len(ds), dtype=np.float32)
    dl = DataLoader(sds, batch_size=BS_INFER, shuffle=False, collate_fn=collate_for(sds), num_workers=2)
    k = 0
    for i, b in enumerate(dl):
        b = {kk: v.to(dev) for kk, v in b.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            p = torch.sigmoid(model(**b).logits.squeeze(-1).float()).cpu().numpy()
        out[order[k:k + len(p)]] = p
        k += len(p)
        if i % 2000 == 0:
            print(f"  scored {k:,}/{len(ds):,}  {time.time() - t0:.0f}s", flush=True)
    model.train()
    return out


t0 = time.time()
s1_tr, r_tr = texts("train", "s1"), texts("train", "r")
s1_te, r_te = texts("test", "s1"), texts("test", "r")
tr = pd.read_parquet(V2)
hold = tr.sample(20000, random_state=1); tr = tr.drop(hold.index)
mk = lambda df, s1t, rt, lab: Pairs(s1t[df["s1row"].to_numpy()], rt[df["rrow"].to_numpy()],
                                    df["label"].to_numpy() if lab else None)
dtr, dho = mk(tr, s1_tr, r_tr, True), mk(hold, s1_tr, r_tr, True)
va = pd.read_parquet(f"{DATA}/val_pairs.parquet"); dva = mk(va, s1_tr, r_tr, False)
te = pd.read_parquet(f"{DATA}/test_pairs.parquet"); dte = mk(te, s1_te, r_te, False)
print(f"train {len(dtr):,}  val {len(dva):,}  test {len(dte):,}")

model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
dl = DataLoader(dtr, batch_size=BS_TRAIN, shuffle=True, collate_fn=collate_for(dtr), num_workers=2)
steps = EPOCHS * len(dl)
sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
scaler = torch.cuda.amp.GradScaler()
lossf = torch.nn.BCEWithLogitsLoss()
model.train()
for ep in range(EPOCHS):
    for it, b in enumerate(dl):
        y = b.pop("labels").to(dev)
        b = {k: v.to(dev) for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            loss = lossf(model(**b).logits.squeeze(-1), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sch.step()
        if it % 1000 == 0:
            done = ep * len(dl) + it + 1
            eta = (time.time() - t0) / done * (steps - done)
            print(f"epoch {ep + 1} step {it}/{len(dl)} loss {loss.item():.4f}  {time.time() - t0:.0f}s  train ETA {eta / 60:.0f} min", flush=True)
    ph = score(model, dho, t0)
    print(f"== epoch {ep + 1}: holdout accuracy {((ph >= 0.5) == (dho.y == 1)).mean():.4f}  {time.time() - t0:.0f}s", flush=True)
    e = ep + 1
    va["pT"] = score(model, dva, t0).astype(np.float16)
    va[["s1row", "rrow", "pT"]].to_parquet(f"{OUT}/val_scores_e{e}.parquet", index=False)
    te["pT"] = score(model, dte, t0).astype(np.float16)
    te[["s1row", "rrow", "pT"]].to_parquet(f"{OUT}/test_scores_e{e}.parquet", index=False)
    model.save_pretrained(f"{OUT}/crossencoder_e{e}"); tok.save_pretrained(f"{OUT}/crossencoder_e{e}")
    print(f"== epoch {e} scores saved  {time.time() - t0:.0f}s", flush=True)
print("ALL DONE", time.time() - t0)
