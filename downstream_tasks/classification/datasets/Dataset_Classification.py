import os, glob, time, pickle
import torch
import numpy as np
import pandas as pd
import torch.utils.data as data
from typing import List, Tuple

# ---------------- helpers ----------------
def _resolve_feature_base(root: str, feature: str) -> str:
    """Join root and feature; if feature is absolute, root is ignored by join."""
    return feature if os.path.isabs(feature) else os.path.join(root or "", feature)

def _tensor_from_any(obj) -> torch.Tensor:
    """Convert common .pt payloads (tensor/ndarray/list/dict) into a torch.FloatTensor."""
    if isinstance(obj, torch.Tensor):
        return obj.float()
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj).float()
    if isinstance(obj, (list, tuple)):
        return torch.as_tensor(obj).float()
    if isinstance(obj, dict):
        for k in ("features", "feats", "feat", "embedding", "embeddings", "data", "x"):
            if k in obj:
                v = obj[k]
                return v if isinstance(v, torch.Tensor) else torch.as_tensor(v).float()
    return torch.as_tensor(obj).float()

def _to_slide_basenames(slide_field: str) -> List[str]:
    """
    Accepts one cell from the Excel 'slide' column.
    It may be a single path or multiple paths separated by ';' or '/'.
    Return a list of base names (no extension).
    """
    names: List[str] = []
    slide_field = str(slide_field).replace("/", ";")
    for part in slide_field.split(";"):
        s = part.strip()
        if not s:
            continue
        base = os.path.splitext(os.path.basename(s))[0]
        names.append(base)
    return names

def _infer_n_features_from_dir(feature_base: str, slide_basenames: List[str]) -> int:
    """Infer feature dim by loading one .pt: prefer slide match, else any .pt."""
    for name in slide_basenames:
        p = os.path.join(feature_base, f"{name}.pt")
        if os.path.exists(p):
            t = _tensor_from_any(torch.load(p, map_location="cpu"))
            return int(t.shape[-1])

    candidates = glob.glob(os.path.join(feature_base, "*.pt"))
    if not candidates:
        candidates = glob.glob(os.path.join(feature_base, "**", "*.pt"), recursive=True)
    if not candidates:
        raise FileNotFoundError(f"No .pt files found under feature directory: {feature_base}")

    t = _tensor_from_any(torch.load(candidates[0], map_location="cpu"))
    return int(t.shape[-1])

def _safe_torch_load(pt_path: str, max_retries: int = 2, sleep_base: float = 0.1) -> torch.Tensor:
    """Robust torch.load with retries."""
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            if os.path.getsize(pt_path) <= 0:
                raise EOFError(f"empty file: {pt_path}")
            with open(pt_path, "rb") as f:
                obj = torch.load(f, map_location="cpu")
            return _tensor_from_any(obj)
        except (EOFError, OSError, RuntimeError, pickle.UnpicklingError) as e:
            last_err = e
            time.sleep(sleep_base * (attempt + 1))
    raise last_err

def _load_slide_feature(base_dir: str, slide_basename: str) -> torch.Tensor:
    """Load a single slide's feature tensor by <base_dir>/<basename>.pt."""
    p = os.path.join(base_dir, f"{slide_basename}.pt")
    if os.path.exists(p):
        return _safe_torch_load(p)
    matches = glob.glob(os.path.join(base_dir, "**", f"{slide_basename}.pt"), recursive=True)
    if matches:
        return _safe_torch_load(matches[0])
    raise FileNotFoundError(f"Feature .pt not found for slide '{slide_basename}' under {base_dir}")

# ---------------- dataset ----------------
class Dataset_Classification(data.Dataset):
    """
    Classification dataset with internal/external split logic.

    Excel must contain at least:
      - 'slide' : WSI filename(s) or paths.
      - 'label' : class label (string or int).
      - 'split' : 'train' / 'val' / 'test' OR cohort/source names.

    Optional:
      - 'ID' : row identifier.
    """
    def __init__(self, root: str, excel_file: str, feature: str,
                 split_keep=None, label_column: str = "label",
                 evaluate: bool = False):

        self.feature = feature
        self.root = root.split(",") if (isinstance(root, str) and "," in root) else [root]

        # read excel
        if str(excel_file).lower().endswith(".csv"):
            self.data = pd.read_csv(excel_file)
        else:
            self.data = pd.read_excel(excel_file)

        # case-insensitive column map
        lower_map = {c.lower(): c for c in self.data.columns}
        lc_key = (label_column or "label").lower()
        required = {"slide", "split", lc_key}
        missing = {r for r in required if r not in lower_map}
        if missing:
            raise KeyError(f"Excel must contain columns: {required} (missing: {missing})")

        # optional --split_keep filtering
        if split_keep:
            if isinstance(split_keep, str):
                split_keep = [s.strip() for s in split_keep.split(",") if s.strip()]
            cand_keys = ["dataset", "study", "cohort", "source", "split"]
            cols = [(k, lower_map[k]) for k in cand_keys if k in lower_map]
            if cols:
                mask = pd.Series(False, index=self.data.index)
                for _, col in cols:
                    mask |= self.data[col].astype(str).isin(split_keep)
                self.data = self.data[mask].copy()
                self._used_filter_col = cols[0][1]
                self._used_filter_tags = list(split_keep)

        self.data.reset_index(drop=True, inplace=True)

        # handle labels
        label_col = lower_map[lc_key]
        try:
            self.data[label_col] = pd.to_numeric(self.data[label_col], errors="raise").astype(int)
            self.num_classes = int(self.data[label_col].max()) + 1 if len(self.data) else 0
        except Exception:
            cat = pd.Categorical(self.data[label_col])
            self.data[label_col] = cat.codes.astype(int)
            self.num_classes = len(cat.categories)

        id_col    = lower_map.get("case", None)
        slide_col = lower_map.get("slide", "slide")
        split_col = lower_map.get("split", "split")

        # build cases
        self.cases: List[Tuple[str, List[str], int]] = []
        for idx in range(len(self.data)):
            row = self.data.iloc[idx]
            ID = str(row[id_col]) if id_col else str(idx)
            slide_bases = _to_slide_basenames(row[slide_col])
            label_code = int(row[label_col])
            self.cases.append((ID, slide_bases, label_code))

        print(f"[dataset] dataset from {excel_file}")
        print(f"[dataset] number of cases={len(self.cases)}")
        print(f"[dataset] number of classes={self.num_classes}")
        print(f"[dataset] feature dir: {self.feature}")

        # count slides
        total_slides = sum(len(s_list) for _, s_list, _ in self.cases)
        if len(self.cases) > 0:
            print(f"[check] total slides={total_slides}, avg per case={total_slides/len(self.cases):.2f}")

        # --- split handling ---
        roles = {"train", "val", "valid", "validation", "test"}
        split_vals = self.data[split_col].astype(str).str.lower()

        if set(split_vals.unique()).issubset(roles):
            if evaluate:
                # evaluation mode: only test rows
                self.train, self.val, self.test = [], [], self.data.index[split_vals == "test"].tolist()
                print(f"[dataset] eval mode: using only test split ({len(self.test)} rows)")
            else:
                # training mode: honor train/val/test
                self.train = self.data.index[split_vals == "train"].tolist()
                self.val   = self.data.index[split_vals.isin(["val", "valid", "validation"])].tolist()
                self.test  = self.data.index[split_vals == "test"].tolist()
                print(f"[dataset] training split: {len(self.train)}, val: {len(self.val)}, test: {len(self.test)}")
        else:
            # external eval fallback
            self.train, self.val, self.test = [], [], list(range(len(self.data)))
            msg_col = getattr(self, "_used_filter_col", split_col)
            msg_tags = getattr(self, "_used_filter_tags", None)
            if msg_tags:
                print(f"[dataset] external eval: using {msg_col} in {msg_tags}; assigning ALL {len(self.test)} rows to test.")
            else:
                print(f"[dataset] external eval: assigning ALL {len(self.test)} rows to test.")

        # --- n_features inference ---
        self.n_features = None
        all_basenames = []
        for _, s_list, _ in self.cases:
            all_basenames.extend(s_list)
        seen, uniq_basenames = set(), []
        for s in all_basenames:
            if s not in seen:
                seen.add(s); uniq_basenames.append(s)

        for r in self.root:
            base_dir = _resolve_feature_base(r, self.feature)
            try:
                self.n_features = _infer_n_features_from_dir(base_dir, uniq_basenames)
                break
            except FileNotFoundError:
                continue

        if self.n_features is None:
            for r in self.root:
                base_dir = _resolve_feature_base(r, self.feature)
                try:
                    self.n_features = _infer_n_features_from_dir(base_dir, [])
                    break
                except FileNotFoundError:
                    continue

        if self.n_features is None:
            raise RuntimeError(f"Failed to infer feature dimensionality. roots={self.root}, feature={self.feature}")

        print(f"[dataset] number of features={self.n_features}")

    # API
    def get_split(self):
        return self.train, self.val, self.test

    @property
    def num_folds(self):
        return 1

    def get_fold(self, fold=0):
        assert fold == 0, "only fixed split supported"
        return self.get_split()

    def __getitem__(self, index: int):
        ID, slide_basenames, class_code = self.cases[index]
        tensors, missing = [], []

        for r in self.root:
            base_dir = _resolve_feature_base(r, self.feature)
            for s in slide_basenames:
                try:
                    tensors.append(_load_slide_feature(base_dir, s))
                except Exception as e:
                    missing.append((base_dir, s, str(e)))
                    continue

        if not tensors:
            D = int(self.n_features or 1)
            print(f"[warn] no readable features for ID={ID}, slides={slide_basenames}. Using dummy [1,{D}]. Missing {missing[:2]} ...")
            Slide = torch.zeros((1, D), dtype=torch.float32)
        else:
            D_target = int(self.n_features)
            normed = []
            for t in tensors:
                t = t.float()
                if t.dim() == 1:
                    t = t.unsqueeze(0)
                elif t.dim() > 2:
                    t = t.view(-1, t.shape[-1])
                d = t.shape[-1]
                if d < D_target:
                    pad = torch.zeros((t.shape[0], D_target - d), dtype=t.dtype)
                    t = torch.cat([t, pad], dim=-1)
                elif d > D_target:
                    t = t[:, :D_target]
                normed.append(t)
            Slide = torch.cat(normed, dim=0)

        label = torch.tensor(class_code, dtype=torch.int64)
        return ID, Slide, label

    def __len__(self):
        return len(self.cases)
