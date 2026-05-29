import copy
import os, numpy as np
from tqdm import tqdm
import torch
from sksurv.metrics import concordance_index_censored

class Engine(object):
    def __init__(self, args, results_dir, fold):
        self.args = args
        # keep your original prefixing behavior (main already gives "results_42/..."):
        self.results_dir = results_dir
        self.fold = fold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {"c-index": 0.0, "checkpoint": None, "epoch": 0}

    # ---- small helper so nll_surv_l1 (list) also works ----
    def _compute_loss(self, criterion, logit, hazards, S, Y, c, model):
        # single callable path
        if callable(criterion):
            return criterion(hazards=hazards, S=S, Y=Y, c=c), {}

        # list path: [base_loss_callable, l1_loss_obj]
        if isinstance(criterion, (list, tuple)) and len(criterion) == 2:
            base_loss_fn, l1_loss_obj = criterion
            base_loss = base_loss_fn(hazards=hazards, S=S, Y=Y, c=c)

            # default: L1 on hazards toward zero (no target)
            target = torch.zeros_like(hazards)
            l1_term = l1_loss_obj(hazards, target)

            # weight & target selection (optional, safe defaults if args not set)
            w = getattr(self.args, "l1_weight", 1e-6)
            which = getattr(self.args, "l1_target", "hazards")
            if which == "logit":
                l1_term = l1_loss_obj(logit, torch.zeros_like(logit))
            elif which == "params":
                # mean |param| over all params
                l1_term = sum(p.abs().mean() for p in model.parameters())

            total = base_loss + w * l1_term
            return total, {"base_loss": float(base_loss.item()), "l1": float(l1_term.item()), "l1_weight": float(w)}

        raise TypeError("Unsupported criterion type for survival loss.")

    def learning(self, model, train_loader, val_loader, criterion, optimizer, scheduler):
        model = model.to(self.device)
        print(f"[engine] results_dir={self.results_dir}", flush=True)
        os.makedirs(self.results_dir, exist_ok=True)
        best_state = None

        for epoch in range(self.args.num_epoch):
            self.epoch = epoch
            train_stats = self.train(train_loader, model, criterion, optimizer)
            val_score, val_stats = self.validate(val_loader, model, criterion)

            if val_score > self.results["c-index"]:
                self.results["c-index"] = float(val_score)
                self.results["checkpoint"] = None
                self.results["epoch"] = int(epoch)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            # log LR too (if present)
            try:
                lrs = [pg.get("lr", None) for pg in optimizer.param_groups]
                if hasattr(self.args, "run") and self.args.run:
                    self.args.run.log({f"lr/{i}": float(lr) for i, lr in enumerate(lrs) if lr is not None}, step=epoch)
            except Exception:
                pass

            print(f"[best] c-index={self.results['c-index']:.4f} @ epoch {self.results['epoch']}", flush=True)
            if scheduler is not None:
                scheduler.step()

        self.save_checkpoint()
        if best_state is not None:
            model.load_state_dict(best_state)
        return {"c-index": float(val_score)}

    def train(self, loader, model, criterion, optimizer):
        model.train()
        total_loss = 0.0
        risks, events, times = [], [], []
        it = tqdm(loader, desc=f"train {self.epoch}") if self.args.tqdm else loader

        for _, _, _, slide, t, e, y in it:
            optimizer.zero_grad(set_to_none=True)
            slide = slide.to(self.device)     # [N,D] bag (batch_size=1)
            y = y.to(self.device).long()      # discrete bin
            e = e.to(self.device).float()     # 1=event, 0=censored

            logit = model(slide)              # [1, n_bins]
            hazards = torch.sigmoid(logit)    # [1, n_bins]
            S = torch.cumprod(1 - hazards, dim=1)

            c = 1.0 - e                       # 1=censored, 0=event
            loss, aux = self._compute_loss(criterion, logit, hazards, S, y, c, model)

            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            risks.append(risk)
            events.append(e.detach().cpu().numpy())
            times.append(t.detach().cpu().numpy())

            total_loss += float(loss.item())
            loss.backward()
            optimizer.step()

        risks  = np.concatenate(risks,  axis=0)
        events = np.concatenate(events, axis=0).astype(bool)
        times  = np.concatenate(times,  axis=0)

        cidx = concordance_index_censored(events, times, risks, tied_tol=1e-08)[0]
        avg_loss = total_loss / max(1, len(loader))
        print(f"[train] loss={avg_loss:.4f} c-index={cidx:.4f}", flush=True)

        # --- logging ---
        if hasattr(self.args, "run") and self.args.run:
            payload = {"train/loss": float(avg_loss), "train/c_index": float(cidx), "epoch": int(self.epoch)}
            # log aux terms if present (e.g., base_loss, l1, l1_weight)
            if isinstance(aux, dict):
                payload.update({f"train/{k}": v for k, v in aux.items()})
            self.args.run.log(payload, step=int(self.epoch))

        return {"loss": float(avg_loss), "c_index": float(cidx)}

    def validate(self, loader, model, criterion):
        model.eval()
        total_loss = 0.0
        risks, events, times = [], [], []
        it = tqdm(loader, desc=f"val {self.epoch}") if self.args.tqdm else loader

        with torch.no_grad():
            for _, _, _, slide, t, e, y in it:
                slide = slide.to(self.device)
                y = y.to(self.device).long()
                e = e.to(self.device).float()

                logit = model(slide)
                hazards = torch.sigmoid(logit)
                S = torch.cumprod(1 - hazards, dim=1)

                c = 1.0 - e
                loss, aux = self._compute_loss(criterion, logit, hazards, S, y, c, model)

                risk = -torch.sum(S, dim=1).detach().cpu().numpy()
                risks.append(risk)
                events.append(e.detach().cpu().numpy())
                times.append(t.detach().cpu().numpy())
                total_loss += float(loss.item())

        risks  = np.concatenate(risks,  axis=0)
        events = np.concatenate(events, axis=0).astype(bool)
        times  = np.concatenate(times,  axis=0)

        cidx = concordance_index_censored(events, times, risks, tied_tol=1e-08)[0]
        avg_loss = total_loss / max(1, len(loader))
        print(f"[val]   loss={avg_loss:.4f} c-index={cidx:.4f}", flush=True)

        # --- logging ---
        if hasattr(self.args, "run") and self.args.run:
            payload = {"val/loss": float(avg_loss), "val/c_index": float(cidx), "epoch": int(self.epoch)}
            if isinstance(aux, dict):
                payload.update({f"val/{k}": v for k, v in aux.items()})
            self.args.run.log(payload, step=int(self.epoch))

        return float(cidx), {"loss": float(avg_loss), "c_index": float(cidx)}

    def save_checkpoint(self):
        fn = os.path.join(self.results_dir, f"[fold-{self.fold}]-[model_best_{round(self.results['c-index'],4)}_{self.results['epoch']}].pth.tar")
        print(f"[save] {fn}", flush=True)
        torch.save(self.results, fn)
