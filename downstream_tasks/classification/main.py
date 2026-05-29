"""
ABMIL downstream classification entrypoint.
See README.md for usage examples.
"""
import os
import sys
import time
import glob
import traceback

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

from utils.options import parse_args


# ---------------- small utils ----------------
def _bool(v):
    try:
        return bool(v)
    except Exception:
        return False


def _list_ckpt_candidates(path, patterns=("model_best*.pth*", "*.pth*", "*.ckpt*", "*.pt*", "*.tar*")):
    """Return checkpoint candidates (newest first). Accepts file or directory."""
    path = os.path.expanduser(path)
    if os.path.isfile(path):
        return [path]
    cands = []
    if os.path.isdir(path):
        for pat in patterns:
            cands.extend(glob.glob(os.path.join(path, pat)))
    cands = sorted(cands, key=os.path.getmtime, reverse=True)
    return cands


def _resolve_resume_to_checkpoint(resume_path: str):
    """Pick newest checkpoint under a dir; or return the file itself."""
    cands = _list_ckpt_candidates(resume_path)
    print(f"[ckpt-check] resume={resume_path}", flush=True)
    if not cands:
        print("[ckpt-check] No checkpoint files found (looked for model_best*.pth*, *.pth*, *.ckpt*, *.pt*, *.tar*)", flush=True)
        return None
    for i, f in enumerate(cands[:10]):
        print(f"[ckpt-check] cand[{i}] {time.ctime(os.path.getmtime(f))}  {f}", flush=True)
    print(f"[ckpt-check] --> using: {cands[0]}", flush=True)
    return cands[0]


def _count_labels(ds, indices):
    """Returns {class_id: count}. Robust to different label types."""
    import collections
    import numpy as np
    import torch as _torch

    def _to_class_index(y):
        if isinstance(y, dict) and "label" in y:
            y = y["label"]
        if _torch.is_tensor(y):
            if y.ndim == 0:
                return int(y.item())
            y = y.reshape(-1)
            if y.numel() == 1:
                return int(y.item())
            return int(_torch.argmax(y).item())
        if isinstance(y, (np.generic, np.ndarray, list, tuple)):
            arr = np.array(y).reshape(-1)
            if arr.size == 1:
                return int(arr[0])
            return int(np.argmax(arr))
        return int(y)

    c = collections.Counter()
    for i in indices:
        y = ds.get_label(i) if hasattr(ds, "get_label") else ds[i][2]
        try:
            c[_to_class_index(y)] += 1
        except Exception as e:
            print(f"[warn] could not parse label at idx={i}: type={type(y)} err={e}", flush=True)
    return dict(c)


# ---------------- dataset / loaders ----------------
def build_dataset(args):
    from datasets.Dataset_Classification import Dataset_Classification

    root = getattr(args, "root", "") or ""
    feature = args.feature
    excel = args.excel_file

    print(f"[dataset] root     : {root}", flush=True)
    print(f"[dataset] feature  : {feature}", flush=True)
    print(f"[dataset] excel    : {excel}", flush=True)

    if not excel or not os.path.isfile(excel):
        raise FileNotFoundError(f"Excel split not found: {excel}")
    if root and not os.path.isdir(root):
        print(f"[warn] root does not exist (ok if --feature is an absolute dir): {root}", flush=True)

    kwargs = dict(root=root, excel_file=excel, feature=feature)
    if hasattr(args, "split_keep"):
        kwargs["split_keep"] = getattr(args, "split_keep", None)
    if hasattr(args, "label_column"):
        kwargs["label_column"] = getattr(args, "label_column", "label")
    try:
        ds = Dataset_Classification(**kwargs)
    except TypeError:
        print("[warn] Dataset_Classification(*) does not accept split_keep/label_column; fallback ctor used.", flush=True)
        ds = Dataset_Classification(root=root, excel_file=excel, feature=feature)

    if not hasattr(ds, "num_folds"):
        ds.num_folds = 1
    if not hasattr(ds, "get_fold") and hasattr(ds, "get_split"):
        def _get_fold(_=0): return ds.get_split()
        ds.get_fold = _get_fold

    print(f"[dataset] num_classes : {getattr(ds, 'num_classes', 'NA')}", flush=True)
    print(f"[dataset] n_features  : {getattr(ds, 'n_features', 'NA')}", flush=True)
    print(f"[dataset] len(samples): {len(ds)}", flush=True)
    print(f"[dataset] num_folds   : {ds.num_folds}", flush=True)

    return ds


def build_loaders(ds, batch_size=1):
    splits = ds.get_fold(0)  # (train, val, test)
    if isinstance(splits, (list, tuple)):
        if len(splits) == 3:
            train_idx, val_idx, test_idx = splits
        elif len(splits) == 2:
            train_idx, test_idx = splits
            val_idx = []
        else:
            raise ValueError(f"Unexpected split tuple length: {len(splits)}")
    else:
        raise ValueError("Dataset.get_fold(0) must return a tuple/list of index lists")

    pin = _bool(torch.cuda.is_available())

    def _loader(indices):
        if not indices:
            return DataLoader(ds, batch_size=batch_size, num_workers=0, sampler=SubsetRandomSampler([]))
        return DataLoader(ds, batch_size=batch_size, num_workers=4, pin_memory=pin,
                          sampler=SubsetRandomSampler(indices))

    train_loader = _loader(train_idx)
    val_loader   = _loader(val_idx)
    test_loader  = _loader(test_idx)

    def _len_of(loader):
        try:
            return len(loader.sampler.indices)
        except Exception:
            return 0

    print(f"[dataloader] sizes -> train:{_len_of(train_loader)}  val:{_len_of(val_loader)}  test:{_len_of(test_loader)}", flush=True)
    return (train_idx, val_idx, test_idx), (train_loader, val_loader, test_loader)


# ---------------- model / engine ----------------
def build_model_and_engine(args, results_dir, fold):
    if args.model != "ABMIL":
        raise NotImplementedError(f"model [{args.model}] is not implemented here (only ABMIL).")

    from models.ABMIL.network import DAttention
    from models.ABMIL.engine import Engine

    n_classes  = args.num_classes
    n_features = args.n_features
    if n_classes is None or n_features is None:
        raise ValueError(f"Invalid model dims: n_classes={n_classes}, n_features={n_features}")

    model = DAttention(n_classes=n_classes, dropout=0.25, act="relu", n_features=n_features)
    engine = Engine(args, results_dir, fold)
    print("[model] ABMIL ready", flush=True)
    return model, engine


# ---------------- main ----------------
def main(args):
    print("[entry] starting main()", flush=True)
    print(f"[env]  torch={torch.__version__}  cuda_build={torch.version.cuda}  cuda_avail={torch.cuda.is_available()}", flush=True)

    # -------- Results dir policy --------
    if args.evaluate:
        external_root = os.environ.get("EXTERNAL_RESULTS_ROOT", "").strip()
        if external_root:
            if not getattr(args, "study", None):
                raise ValueError("--evaluate requires --study to place outputs under external results root")
            results_dir = os.path.join(
                external_root, "results", f"results_{args.seed}",
                str(args.study), f"[{args.model}]", f"[{args.feature}]"
            )
        else:
            resume_path = getattr(args, "resume", "")
            if not resume_path:
                raise ValueError("--evaluate requires --resume")
            results_dir = resume_path if os.path.isdir(resume_path) else os.path.dirname(resume_path)
    else:
        results_dir = "./results/results_{seed}/{study}/[{model}]/[{feature}]-[{time}]".format(
            seed=args.seed,
            study=args.study,
            model=args.model,
            feature=args.feature,
            time=time.strftime("%Y-%m-%d]-[%H-%M-%S"),
        )
    os.makedirs(results_dir, exist_ok=True)
    print(f"[log dir] {results_dir}", flush=True)

    # -------- Dataset / Loaders --------
    ds = build_dataset(args)
    args.num_classes = getattr(ds, "num_classes", None)
    args.n_features  = getattr(ds, "n_features", None)
    args.num_folds   = getattr(ds, "num_folds", 1)

    (train_idx, val_idx, test_idx), (train_loader, val_loader, test_loader) = build_loaders(ds, batch_size=args.batch_size)

    eval_split = getattr(args, "eval_split", None)
    if eval_split is None:
        eval_split = "cohort" if getattr(args, "split_keep", None) else "test"
    eval_split = str(eval_split).lower()

    pin = _bool(torch.cuda.is_available())
    def _mk_loader(idxs):
        if not idxs:
            return DataLoader(ds, batch_size=args.batch_size, num_workers=0, sampler=SubsetRandomSampler([]))
        return DataLoader(ds, batch_size=args.batch_size, num_workers=4, pin_memory=pin, sampler=SubsetRandomSampler(idxs))

    if eval_split == "cohort":
        eval_indices = list(train_idx) + list(val_idx) + list(test_idx)
        eval_loader  = _mk_loader(eval_indices)
    elif eval_split == "train":
        eval_indices, eval_loader = train_idx, train_loader
    elif eval_split == "val":
        eval_indices, eval_loader = val_idx, val_loader
    else:
        eval_indices, eval_loader = test_idx, test_loader

    if len(eval_indices) == 0:
        fallback = list(train_idx) + list(val_idx) + list(test_idx)
        print(f"[warn] eval_split '{eval_split}' is empty; fallback to 'cohort' with {len(fallback)} samples.", flush=True)
        eval_indices = fallback
        eval_loader  = _mk_loader(eval_indices)
        eval_split   = "cohort"

    dist = _count_labels(ds, eval_indices)
    print(f"[sanity] eval_split={eval_split} size={len(eval_indices)}  label dist={dist}", flush=True)

    loaders_all = [train_loader, val_loader, test_loader]  

    # -------- Model / Engine / Optim --------
    model, engine = build_model_and_engine(args, results_dir, fold=0)

    from utils.loss import define_loss
    from utils.optimizer import define_optimizer
    from utils.scheduler import define_scheduler

    criterion = define_loss(args)
    optimizer = define_optimizer(args, model)
    scheduler = define_scheduler(args, optimizer)
    print(f"[train] loss={args.loss} opt={args.optimizer} lr={args.lr} wd={args.weight_decay} sched={args.scheduler}", flush=True)

    # Optional: CV meter if present
    meter = None
    try:
        from utils.util import CV_Meter
        meter = CV_Meter(args.num_folds)
    except Exception:
        pass

    # -------- Train / Evaluate --------
    if not args.evaluate:
        out = engine.learning(model, loaders_all, criterion, optimizer, scheduler)
        if isinstance(out, tuple) and meter:
            if len(out) == 2:
                val_scores, best_epoch = out
                meter.updata(best_epoch, val_scores)
            elif len(out) == 3:
                val_scores, test_scores, best_epoch = out
                meter.updata(best_epoch, val_scores, test_scores)
        if meter:
            try:
                meter.save(os.path.join(results_dir, "result.csv"))
                print(f"[result] saved: {os.path.join(results_dir, 'result.csv')}", flush=True)
            except Exception as e:
                print(f"[warn] meter.save failed: {e}", flush=True)

    else:
        # ---------- Evaluate-only ----------
        resume_path = getattr(args, "resume", "")
        if not resume_path:
            raise ValueError("--evaluate requires --resume (file or directory)")
        cand = _resolve_resume_to_checkpoint(resume_path)
        if not cand or not os.path.isfile(cand):
            raise FileNotFoundError(f"Could not resolve checkpoint from: {resume_path}")

        print(f"=> loading checkpoint '{cand}'", flush=True)
        checkpoint = torch.load(cand, map_location="cpu")
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

        model = model.to(engine.device)

        try:
            engine.validate(eval_loader, model, criterion, status="test")
        except Exception as e:
            print(f"[eval] engine.validate failed (continue to preds): {e}", flush=True)

        import pandas as pd, numpy as np

        all_ids, all_labels, all_probs = [], [], []
        model.eval()
        with torch.no_grad():
            for data_ID, data_WSI, data_Label in eval_loader:
                data_WSI = data_WSI.to(engine.device)
                logits = model(data_WSI)
                probs  = torch.softmax(logits, dim=-1).cpu().numpy()
                for b in range(probs.shape[0]):
                    case_id = data_ID[b] if isinstance(data_ID, (list, tuple)) else data_ID
                    if isinstance(case_id, (list, tuple)):
                        case_id = case_id[b]
                    all_ids.append(str(case_id))
                    y = data_Label[b] if hasattr(data_Label, "__len__") else data_Label
                    if torch.is_tensor(y):
                        y = int(y.item())
                    else:
                        y = int(y)
                    all_labels.append(y)
                    all_probs.append(probs[b])

        all_probs = np.asarray(all_probs)
        if all_probs.ndim == 1:
            all_probs = all_probs[:, None]
        C = all_probs.shape[1]
        prob_cols = {f"prob_class_{i}": all_probs[:, i] for i in range(C)}
        df = pd.DataFrame({
            "ID": all_ids,
            "label": [int(x) for x in all_labels],
            "pred": all_probs.argmax(axis=1),
            **prob_cols
        })
        out_csv = os.path.join(results_dir, "preds.csv")
        df.to_csv(out_csv, index=False)
        print(f"[evaluate] wrote predictions to {out_csv}", flush=True)

    return 0


if __name__ == "__main__":
    try:
        args = parse_args()
        try:
            from utils.util import set_seed
            set_seed(args.seed)
        except Exception:
            pass
        rc = main(args)
        print("finished!", flush=True)
        sys.exit(rc)
    except SystemExit:
        raise
    except Exception:
        print("[FATAL] Uncaught exception:\n" + "".join(traceback.format_exc()), file=sys.stderr, flush=True)
        sys.exit(1)
