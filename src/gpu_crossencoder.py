"""GPU step: fine-tune a small pretrained cross-encoder on labelled candidate pairs, then score
the validation and test candidate pairs. Paste into ONE Kaggle notebook cell (GPU + Internet ON)
or run as a script on any CUDA machine.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (Apache-2.0, ~22M parameters).
Input:  the files produced by export_pairs.py (a Kaggle dataset, found automatically).
Output: val_scores.parquet, test_scores.parquet (s1row, rrow, pT) in the working directory.
"""
import glob
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Without notebook internet: attach the model as an input (Kaggle Model or dataset) and it is used from there.
_local = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/config.json", recursive=True)
          if "minilm" in p.lower() or "marco" in p.lower() or "cross" in p.lower()]
if _local:
    MODEL = _local[0]
print("model:", MODEL)
MAXLEN, BS_TRAIN, BS_INFER, LR, EPOCHS = 128, 256, 1024, 3e-5, 1
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
hits = glob.glob("/kaggle/input/**/train_pairs.parquet", recursive=True) or glob.glob("**/train_pairs.parquet", recursive=True)
DATA = os.path.dirname(hits[0])
dev = torch.device("cuda")
torch.manual_seed(42); np.random.seed(42)
print("data:", DATA, " gpu:", torch.cuda.get_device_name(0))


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


# ---------------------------------------------------------------- train
t0 = time.time()
s1t, rt = texts("train", "s1"), texts("train", "r")
tr = pd.read_parquet(f"{DATA}/train_pairs.parquet")
hold = tr.sample(20000, random_state=1); tr = tr.drop(hold.index)
mk = lambda df, lab: Pairs(s1t[df["s1row"].to_numpy()], rt[df["rrow"].to_numpy()], df["label"].to_numpy() if lab else None)
dtr, dho = mk(tr, True), mk(hold, True)
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
        if it % 500 == 0:
            print(f"epoch {ep} step {it}/{len(dl)} loss {loss.item():.4f}  {time.time() - t0:.0f}s", flush=True)
model.eval()
model.save_pretrained(f"{OUT}/crossencoder_model"); tok.save_pretrained(f"{OUT}/crossencoder_model")


@torch.no_grad()
def score(ds):
    out = np.empty(len(ds), dtype=np.float32)
    dl = DataLoader(ds, batch_size=BS_INFER, shuffle=False, collate_fn=collate_for(ds), num_workers=2)
    k = 0
    for i, b in enumerate(dl):
        b.pop("labels", None)
        b = {kk: v.to(dev) for kk, v in b.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            p = torch.sigmoid(model(**b).logits.squeeze(-1).float()).cpu().numpy()
        out[k:k + len(p)] = p; k += len(p)
        if i % 2000 == 0:
            print(f"  scored {k:,}/{len(ds):,}  {time.time() - t0:.0f}s", flush=True)
    return out


ph = score(dho)
acc = ((ph >= 0.5) == (dho.y == 1)).mean()
print(f"holdout accuracy {acc:.4f}  (positives {dho.y.mean():.2f})  {time.time() - t0:.0f}s")

# ---------------------------------------------------------------- score validation + test candidates
va = pd.read_parquet(f"{DATA}/val_pairs.parquet")
va["pT"] = score(mk(va, False)).astype(np.float16)
va[["s1row", "rrow", "pT"]].to_parquet(f"{OUT}/val_scores.parquet", index=False)
print("val done", time.time() - t0)
s1t, rt = texts("test", "s1"), texts("test", "r")
te = pd.read_parquet(f"{DATA}/test_pairs.parquet")
te["pT"] = score(Pairs(s1t[te["s1row"].to_numpy()], rt[te["rrow"].to_numpy()])).astype(np.float16)
te[["s1row", "rrow", "pT"]].to_parquet(f"{OUT}/test_scores.parquet", index=False)
print("ALL DONE", time.time() - t0)
