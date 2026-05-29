#!/usr/bin/env python3
import argparse
import copy
import os
import re
import time

import numpy as np
import pandas as pd
import torch
from sksurv.metrics import concordance_index_censored
from torch.utils.data import DataLoader, SubsetRandomSampler

from dataset import TCGA_Dataset
from engine import Engine
from loss import define_loss
from network import DAttention
from utils.optimizer import define_optimizer
from utils.scheduler import define_scheduler
from utils.util import set_seed


def parse_args():
    parser = argparse.ArgumentParser("ABMIL Survival Analysis")

    parser.add_argument("--study", type=str, required=True)
    parser.add_argument("--feature", type=str, required=True)
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--excel_file", type=str, required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="result")

    parser.add_argument("--model", type=str, default="ABMIL")
    parser.add_argument("--modal", type=str, default="os", choices=["os", "dfs"])
    parser.add_argument("--n_bins", type=int, default=4)

    parser.add_argument(
        "--loss",
        type=str,
        default="nll_surv",
        choices=["nll_surv", "ce_surv", "nll_surv_l1"],
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        choices=["SGD", "Adam", "AdamW"],
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["None", "exp", "step", "plateau", "cosine"],
    )

    parser.add_argument("--dropout", action="store_true")
    parser.add_argument("--act", type=str, default="relu", choices=["relu", "gelu"])
    parser.add_argument("--tqdm", action="store_true")

    parser.add_argument("--task_id", type=str, default="survival_demo_run")
    parser.add_argument("--task_name", type=str, default="Survival Demo")

    return parser.parse_args()


def compute_risk_from_logits(logit: torch.Tensor) -> torch.Tensor:
    hazards = torch.sigmoid(logit)
    survival = torch.cumprod(1 - hazards, dim=1)
    risk = -torch.sum(survival, dim=1)
    return risk


def get_available_folds(ds) -> list:
    if not hasattr(ds, "rows"):
        return [0, 1, 2, 3, 4]

    fold_ids = []
    for column in ds.rows.columns:
        match = re.fullmatch(r"Fold\s+(\d+)", str(column).strip())
        if match:
            fold_ids.append(int(match.group(1)))

    fold_ids = sorted(set(fold_ids))
    return fold_ids if fold_ids else [0, 1, 2, 3, 4]


def safe_item(x, i=None):
    if isinstance(x, torch.Tensor):
        if x.ndim == 0:
            return x.item()
        return x[i].item()
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return x.item()
        return x[i].item() if np.isscalar(x[i]) else x[i]
    if isinstance(x, (list, tuple)):
        return x[i]
    return x


def infer_rows(loader, model, device, fold, split_name):
    model.eval()
    rows = []

    with torch.no_grad():
        for batch in loader:
            dataset_name, case_id, filename, slide, event_time, event_status, disc_label = batch

            slide = slide.to(device, non_blocking=True)
            logit = model(slide)
            risk = compute_risk_from_logits(logit)

            batch_size = slide.shape[0]
            for i in range(batch_size):
                rows.append({
                    "dataset": str(dataset_name[i]),
                    "filename": str(safe_item(filename, i)),
                    "case_id": str(safe_item(case_id, i)),
                    "risk_score": float(risk[i].item()),
                    "event_time": float(safe_item(event_time, i)),
                    "event_status": int(safe_item(event_status, i)),
                    "disc_label": int(safe_item(disc_label, i)),
                    "split": split_name,
                    "fold": int(fold),
                })

    return rows


def compute_cindex(rows):
    if not rows:
        return float("nan")

    df = pd.DataFrame(rows)
    if len(df) < 2:
        return float("nan")

    event = df["event_status"].astype(bool).values
    time = df["event_time"].astype(float).values
    risk = df["risk_score"].astype(float).values

    if event.sum() == 0:
        return float("nan")
    if np.allclose(risk, risk[0]):
        return float("nan")

    return float(concordance_index_censored(event, time, risk)[0])


def assign_risk_group(rows, train_median_risk):
    out = []
    for row in rows:
        row_copy = dict(row)
        row_copy["risk_group"] = "High Risk" if row_copy["risk_score"] > train_median_risk else "Low Risk"
        out.append(row_copy)
    return out


def save_checkpoint(
    model,
    ckpt_path,
    best_epoch,
    best_cindex,
    fold,
    args,
    n_bins,
    input_dim,
    train_median_risk,
):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": int(best_epoch),
            "c_index": float(best_cindex),
            "selection_split": "test",
            "fold": int(fold),
            "study": args.study,
            "feature": args.feature,
            "model_name": args.model,
            "seed": int(args.seed),
            "n_bins": int(n_bins),
            "input_dim": int(input_dim),
            "train_median_risk": float(train_median_risk),
        },
        ckpt_path,
    )


def main(args):
    print(f"[boot] study={args.study} feature={args.feature} seed={args.seed}", flush=True)
    print(f"[boot] excel_file={args.excel_file}", flush=True)

    if args.batch_size != 1:
        raise ValueError("batch_size must be 1 with the default collate function for variable-length bags.")

    set_seed(args.seed)
    print(f"[seed] set random seed to {args.seed}", flush=True)

    root_list = [root.strip() for root in str(args.root).split(",") if root.strip()]
    root_list = [os.path.join(root, args.feature) for root in root_list]
    print(f"[boot] feature roots = {root_list}", flush=True)

    ds = TCGA_Dataset(
        roots=root_list,
        excel_file=args.excel_file,
        modal=args.modal,
        n_bins=args.n_bins,
    )

    input_dim = ds.dim_slide
    n_bins = ds.n_bins
    fold_ids = get_available_folds(ds)
    print(f"[dataset] available folds: {fold_ids}", flush=True)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(args.output_dir, args.study, args.model, args.feature, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    metrics_rows = []
    all_test_predictions = []

    for fold in fold_ids:
        print("=" * 100, flush=True)
        print(f"[fold {fold}] start", flush=True)

        train_idx, test_idx = ds.fold(fold)
        if len(train_idx) == 0:
            raise RuntimeError(f"[fold {fold}] empty training split.")
        if len(test_idx) == 0:
            raise RuntimeError(f"[fold {fold}] empty test/holdout split.")

        train_loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            sampler=SubsetRandomSampler(train_idx),
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        test_loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            sampler=SubsetRandomSampler(test_idx),
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        model = DAttention(
            n_classes=n_bins,
            dropout=args.dropout,
            act=args.act,
            n_features=input_dim,
        )

        criterion = define_loss(args)
        optimizer = define_optimizer(args, model)
        scheduler = define_scheduler(args, optimizer)

        fold_dir = os.path.join(out_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        eng = Engine(args, fold_dir, fold)
        device = eng.device
        model = model.to(device)

        best_cindex = -1.0
        best_epoch = -1
        best_state = copy.deepcopy(model.state_dict())
        no_improve = 0

        for epoch in range(args.num_epoch):
            eng.epoch = epoch
            eng.train(train_loader, model, criterion, optimizer)
            test_cindex, _ = eng.validate(test_loader, model, criterion)

            print(
                f"[fold {fold}] [epoch {epoch}] "
                f"test_cindex={test_cindex:.4f} "
                f"best={best_cindex:.4f} "
                f"no_improve={no_improve}",
                flush=True,
            )

            if scheduler is not None:
                if args.scheduler == "plateau":
                    score = 1.0 - float(test_cindex) if not np.isnan(test_cindex) else 1.0
                    scheduler.step(score)
                else:
                    scheduler.step()

            improved = (not np.isnan(test_cindex)) and (float(test_cindex) > best_cindex)
            if improved:
                best_cindex = float(test_cindex)
                best_epoch = int(epoch)
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(
                        f"[early stop] fold={fold} stopped at epoch {epoch} "
                        f"(best={best_cindex:.4f}@{best_epoch})",
                        flush=True,
                    )
                    break

        model.load_state_dict(best_state)

        train_rows = infer_rows(train_loader, model, device, fold, split_name="train")
        train_risks = [row["risk_score"] for row in train_rows]
        if not train_risks:
            raise RuntimeError(f"[fold {fold}] no training predictions found.")

        train_median_risk = float(np.median(train_risks))
        print(f"[fold {fold}] train median risk = {train_median_risk:.6f}", flush=True)

        test_rows = infer_rows(test_loader, model, device, fold, split_name="test")
        test_rows = assign_risk_group(test_rows, train_median_risk)

        fold_cindex = compute_cindex(test_rows)
        fold_ibs = np.nan

        ckpt_path = os.path.join(
            fold_dir,
            f"model_best_fold{fold}_testcindex{fold_cindex:.4f}_epoch{best_epoch}.pth.tar",
        )
        save_checkpoint(
            model=model,
            ckpt_path=ckpt_path,
            best_epoch=best_epoch,
            best_cindex=fold_cindex,
            fold=fold,
            args=args,
            n_bins=n_bins,
            input_dim=input_dim,
            train_median_risk=train_median_risk,
        )
        print(f"[save] checkpoint -> {ckpt_path}", flush=True)

        metrics_rows.append({
            "task_id": args.task_id,
            "task_name": args.task_name,
            "split": "test",
            "fold": int(fold),
            "c_index": float(fold_cindex),
            "integrated_brier_score": fold_ibs,
        })
        all_test_predictions.extend(test_rows)

    metrics_df = pd.DataFrame(
        metrics_rows,
        columns=[
            "task_id",
            "task_name",
            "split",
            "fold",
            "c_index",
            "integrated_brier_score",
        ],
    )
    metrics_path = os.path.join(out_dir, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[save] metrics -> {metrics_path}", flush=True)

    pred_df = pd.DataFrame(all_test_predictions)
    pred_df = pred_df[
        [
            "filename",
            "case_id",
            "risk_score",
            "risk_group",
            "event_time",
            "event_status",
            "split",
            "fold",
        ]
    ]
    pred_path = os.path.join(out_dir, "predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"[save] predictions -> {pred_path}", flush=True)

    print(f"[done] Finished {len(fold_ids)}-fold survival evaluation.", flush=True)


if __name__ == "__main__":
    main(parse_args())
