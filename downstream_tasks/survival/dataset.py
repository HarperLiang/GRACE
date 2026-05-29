
import os
import re
import numpy as np
import pandas as pd

import torch
import torch.utils.data as data


class TCGA_Dataset(data.Dataset):
    """
    Single-modality survival dataset for OS / DFS / general survival tasks.

    Supported input table formats

    New / full format:
      - dataset
      - case   (or case_id)
      - slide  (or filename)
      - event_time   (or time (month))
      - event_status (or label)
      - Fold 0 ... Fold 4

    Simpler format also supported:
      - filename
      - event_time
      - event_status
      - case_id
      - split   (e.g. train / val / test / fold_0 / fold_1 / ...)
    """

    def __init__(self, roots, excel_file, modal="os", n_bins=4):
        if isinstance(roots, str):
            roots = [r.strip() for r in roots.split(",") if r.strip()]
        self.roots = roots
        self.modal = str(modal).lower().strip()
        assert self.modal in ("os", "dfs"), "modal must be 'os' or 'dfs'"

        print(f"[dataset] loading dataset from {excel_file}")
        assert os.path.exists(excel_file), f"file [{excel_file}] not found"

        if excel_file.lower().endswith(".csv"):
            rows = pd.read_csv(excel_file)
        else:
            rows = pd.read_excel(excel_file)

        rows = self._normalize_columns(rows, excel_file)

        self._check_required_columns(rows)

        # 关键：先把 row-level 表聚合成 case-level
        rows = self._aggregate_to_case_level(rows)

        # 再在 case-level 上离散化 survival label
        self.rows = self.__disc_label__(rows, n_bins=n_bins)
        self.n_bins = int(self.rows["disc_label"].max()) + 1

        first_ds = self.rows.iloc[0]["dataset"]
        first_slides = self.rows.iloc[0]["slide"]
        self.dim_slide = self.__read_slide__(first_ds, first_slides).size(-1)

        label_dist = self.rows["event_status"].value_counts().sort_index()
        print(f"[dataset] dim_slide: {self.dim_slide}")
        print(f"[dataset] event distribution (0=censored,1=event):\n{label_dist}")
        print(f"[dataset] number of cases = {len(self.rows)}")

    def _split_slide_field(self, x):
    
        if pd.isna(x):
            return []

        if isinstance(x, (list, tuple)):
            return [str(v).strip() for v in x if str(v).strip()]

        s = str(x).strip()
        if not s:
            return []

        parts = re.split(r"[;,]", s)
        return [p.strip() for p in parts if p.strip()]


    def _aggregate_to_case_level(self, rows: pd.DataFrame) -> pd.DataFrame:
        """
        Convert row-level table to case-level table.

        Rules:
        - One output row per case
        - All slides/filenames under the same case are merged
        - event_time must be identical within a case
        - event_status must be identical within a case
        - Fold 0~4 assignments must be identical within a case
        """
        rows = rows.copy()

        # standardize some text columns
        rows["case"] = rows["case"].astype(str).str.strip()
        rows["dataset"] = rows["dataset"].astype(str).str.strip()
        rows["slide"] = rows["slide"].astype(str).str.strip()

        fold_cols = [f"Fold {k}" for k in range(5)]

        case_rows = []
        bad_cases = []

        for case_id, g in rows.groupby("case", sort=False):
            g = g.reset_index(drop=True)

            # -------- dataset consistency --------
            ds_values = g["dataset"].dropna().astype(str).str.strip().unique().tolist()
            if len(ds_values) != 1:
                bad_cases.append(
                    f"case={case_id}: inconsistent dataset values = {ds_values}"
                )
                continue
            dataset_name = ds_values[0]

            # -------- event_time consistency --------
            time_values = pd.to_numeric(g["event_time"], errors="coerce").dropna().unique()
            if len(time_values) != 1:
                bad_cases.append(
                    f"case={case_id}: inconsistent event_time values = {time_values.tolist()}"
                )
                continue
            event_time = float(time_values[0])

            # -------- event_status consistency --------
            status_values = pd.to_numeric(g["event_status"], errors="coerce").dropna().unique()
            if len(status_values) != 1:
                bad_cases.append(
                    f"case={case_id}: inconsistent event_status values = {status_values.tolist()}"
                )
                continue
            event_status = int(status_values[0])

            # -------- fold consistency --------
            fold_dict = {}
            fold_inconsistent = False
            for col in fold_cols:
                vals = (
                    g[col]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .unique()
                    .tolist()
                )
                if len(vals) == 0:
                    bad_cases.append(f"case={case_id}: missing split values in {col}")
                    fold_inconsistent = True
                    break
                if len(vals) != 1:
                    bad_cases.append(
                        f"case={case_id}: inconsistent {col} values = {vals}"
                    )
                    fold_inconsistent = True
                    break
                fold_dict[col] = vals[0]

            if fold_inconsistent:
                continue

            # -------- merge all slides --------
            slide_list = []
            for x in g["slide"].tolist():
                slide_list.extend(self._split_slide_field(x))

            # deduplicate while preserving order
            seen = set()
            slide_list_unique = []
            for s in slide_list:
                if s not in seen:
                    seen.add(s)
                    slide_list_unique.append(s)

            if len(slide_list_unique) == 0:
                bad_cases.append(f"case={case_id}: no valid slide filenames found")
                continue

            case_row = {
                "dataset": dataset_name,
                "case": case_id,
                "slide": ";".join(slide_list_unique),   # keep as one field
                "filename": ";".join(slide_list_unique),
                "event_time": event_time,
                "event_status": event_status,
                "n_slides": len(slide_list_unique),
            }
            case_row.update(fold_dict)
            case_rows.append(case_row)

        if len(bad_cases) > 0:
            msg = "\n".join(bad_cases[:20])
            more = "" if len(bad_cases) <= 20 else f"\n... and {len(bad_cases)-20} more bad cases."
            raise ValueError(
                "[dataset] case-level aggregation failed. "
                "The following cases have inconsistent labels/splits:\n"
                f"{msg}{more}"
            )

        case_df = pd.DataFrame(case_rows)

        print(f"[dataset] raw rows = {len(rows)}")
        print(f"[dataset] aggregated case-level rows = {len(case_df)}")
        print(f"[dataset] mean slides per case = {case_df['n_slides'].mean():.2f}")
        print(f"[dataset] median slides per case = {case_df['n_slides'].median():.2f}")

        return case_df

    def _infer_dataset_name(self, excel_file: str) -> str:
        """
        Infer dataset name when no 'dataset' column exists.
        Priority:
          1) if only one root -> use parent folder name of that root
          2) else use excel file parent folder
        """
        if len(self.roots) == 1:
            root_parent = os.path.basename(os.path.dirname(self.roots[0]))
            if root_parent:
                return root_parent

        excel_parent = os.path.basename(os.path.dirname(excel_file))
        if excel_parent:
            return excel_parent

        return "UNKNOWN_DATASET"

    def _build_fold_columns_from_split(self, rows: pd.DataFrame) -> pd.DataFrame:
        """
        Support CSVs that only contain one 'split' column.

        Cases handled:
        1) split contains train/val/test
           -> replicate to Fold 0..4
              train stays train
              val/test become holdout

        2) split contains fold_0/fold_1/.../fold_4
           -> create 5-fold CV columns:
              Fold k == holdout if split == fold_k else train
        """
        rows = rows.copy()

        if "split" not in rows.columns:
            return rows

        split_series = rows["split"].astype(str).str.strip()
        split_lower = split_series.str.lower()

        # Case 1: train/val/test style
        basic_tokens = {"train", "val", "valid", "validation", "test"}
        if split_lower.isin(basic_tokens).all():
            for k in range(5):
                col = f"Fold {k}"
                rows[col] = split_lower.replace({
                    "train": "train",
                    "val": "test",
                    "valid": "test",
                    "validation": "test",
                    "test": "test",
                })
            return rows

        # Case 2: fold_0/fold_1/... style
        fold_pat = re.compile(r"^fold[_\s-]*(\d+)$", flags=re.IGNORECASE)
        parsed = split_series.apply(lambda x: fold_pat.match(x))
        if parsed.notna().all():
            fold_ids = split_series.apply(lambda x: int(fold_pat.match(x).group(1)))
            max_fold = int(fold_ids.max())

            for k in range(max(5, max_fold + 1)):
                col = f"Fold {k}"
                rows[col] = np.where(fold_ids == k, "test", "train")
            return rows

        return rows

    def _normalize_columns(self, rows: pd.DataFrame, excel_file: str) -> pd.DataFrame:
        rows = rows.copy()

        # strip whitespace from column names
        rows.columns = [str(c).strip() for c in rows.columns]

        # case_id -> case
        if "case" not in rows.columns and "case_id" in rows.columns:
            rows["case"] = rows["case_id"]

        # filename -> slide fallback
        if "slide" not in rows.columns and "filename" in rows.columns:
            rows["slide"] = rows["filename"]

        # old -> new survival columns
        if "event_time" not in rows.columns and "time (month)" in rows.columns:
            rows["event_time"] = rows["time (month)"]

        if "event_status" not in rows.columns and "label" in rows.columns:
            rows["event_status"] = rows["label"]

        # dataset fallback
        if "dataset" not in rows.columns:
            rows["dataset"] = self._infer_dataset_name(excel_file)

        # build Fold 0..Fold 4 from split if needed
        has_any_fold = any(f"Fold {k}" in rows.columns for k in range(5))
        if not has_any_fold and "split" in rows.columns:
            rows = self._build_fold_columns_from_split(rows)

        # sanitize types
        rows["event_time"] = pd.to_numeric(rows["event_time"], errors="coerce")
        rows["event_status"] = pd.to_numeric(rows["event_status"], errors="coerce")

        if rows["event_time"].isna().any():
            bad_n = int(rows["event_time"].isna().sum())
            raise ValueError(f"'event_time' contains {bad_n} invalid/missing values.")

        if rows["event_status"].isna().any():
            bad_n = int(rows["event_status"].isna().sum())
            raise ValueError(f"'event_status' contains {bad_n} invalid/missing values.")

        rows["event_status"] = rows["event_status"].astype(int)

        return rows

    def _check_required_columns(self, rows: pd.DataFrame):
        required_cols = ["dataset", "case", "slide", "event_time", "event_status"]
        for col in required_cols:
            if col not in rows.columns:
                raise KeyError(f"Table missing required column: {col}")

        has_fold_cols = all(f"Fold {k}" in rows.columns for k in range(5))
        if not has_fold_cols:
            raise KeyError(
                "Table must contain either 'Fold 0'...'Fold 4' columns "
                "or a valid 'split' column that can be converted to folds."
            )

    def fold(self, fold=0):
        assert 0 <= fold <= 4, "fold should be in 0 ~ 4"
        split = self.rows[f"Fold {fold}"].astype(str).str.lower().tolist()

        train_split = [i for i, x in enumerate(split) if x == "train"]
        val_split = [i for i, x in enumerate(split) if x in ("val", "test")]

        print(f"[dataset] (Fold {fold}) train={len(train_split)}, holdout={len(val_split)}")
        return train_split, val_split

    def __disc_label__(self, rows, n_bins=4):
        eps = 1e-6

        uncensored_df = rows[rows["event_status"] == 1]
        if len(uncensored_df) == 0:
            raise ValueError("No event samples found; cannot build discrete time bins.")

        _, q_bins = pd.qcut(
            uncensored_df["event_time"],
            q=n_bins,
            retbins=True,
            labels=False,
            duplicates="drop"
        )

        q_bins = np.unique(q_bins)
        if len(q_bins) < 2:
            raise ValueError("Not enough unique event_time values to construct bins.")

        q_bins[-1] = rows["event_time"].max() + eps
        q_bins[0] = rows["event_time"].min() - eps

        disc_labels, _ = pd.cut(
            rows["event_time"],
            bins=q_bins,
            labels=False,
            right=False,
            include_lowest=True,
            retbins=True
        )

        disc_labels = np.asarray(disc_labels).astype(int)
        disc_labels[disc_labels < 0] = -1

        rows = rows.copy()
        rows["disc_label"] = disc_labels
        return rows

    def _map_dataset_to_root(self, dataset_name: str):
        ds = str(dataset_name).strip()

        # single-root case: no mapping needed
        if len(self.roots) == 1:
            return self.roots[0]

        # direct containment
        for r in self.roots:
            if ds in r:
                return r

        # normalized matching
        ds_norm = ds.replace("_", "-").replace(" ", "-").upper()
        for r in self.roots:
            r_norm = r.replace("_", "-").replace(" ", "-").upper()
            if ds_norm in r_norm:
                return r

        if ds in ("UCEC_SUR (TCGA)", "UCEC-SUR", "UCEC_SUR_TCGA", "TCGA-UCEC", "UCEC-SUR-TCGA"):
            for r in self.roots:
                if "UCEC" in r.upper():
                    return r

        raise RuntimeError(f"Cannot map dataset '{ds}' to any of roots {self.roots}")

    def _normalize_slide_name(self, slide_name: str):
        slide_name = str(slide_name).strip()

        raw_name = slide_name

        if slide_name.endswith(".svs"):
            feature_name = slide_name[:-4] + ".pt"
        elif slide_name.endswith(".pt"):
            feature_name = slide_name
            raw_name = slide_name[:-3] + ".svs"
        else:
            feature_name = slide_name + ".pt"
            raw_name = slide_name + ".svs"

        return raw_name, feature_name

    def __read_slide__(self, dataset_name, slide_field):
    
        root = self._map_dataset_to_root(dataset_name)
        slide_list = self._split_slide_field(slide_field)

        feats = []
        missing_files = []

        for s in slide_list:
            raw_name, feature_name = self._normalize_slide_name(s)
            slide_path = os.path.join(root, feature_name)

            if not os.path.exists(slide_path):
                missing_files.append(slide_path)
                continue

            x = torch.load(slide_path, map_location="cpu")

            if isinstance(x, dict):
                if "feature" in x:
                    x = x["feature"]
                elif "features" in x:
                    x = x["features"]
                else:
                    raise ValueError(f"Unsupported dict tensor format in file: {slide_path}")

            if not torch.is_tensor(x):
                x = torch.tensor(x)

            if x.ndim == 1:
                x = x.unsqueeze(0)

            feats.append(x.float())

        if len(feats) == 0:
            raise FileNotFoundError(
                f"No slide features could be loaded for case. "
                f"dataset={dataset_name}, slides={slide_list}, "
                f"missing={missing_files[:5]}"
            )

        return torch.cat(feats, dim=0)

    def _get_export_filename(self, slide_value: str):
        export_names = []
        for slide_name in self._split_slide_field(slide_value):
            raw_name, _ = self._normalize_slide_name(slide_name)
            export_names.append(raw_name)
        return "/".join(export_names)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]

        dataset = str(row["dataset"])
        case_id = str(row["case"])
        slide_field = str(row["slide"])

        filename = self._get_export_filename(slide_field)
        slide_tensor = self.__read_slide__(dataset, slide_field)

        event_time = float(row["event_time"])
        event_status = int(row["event_status"])
        disc_label = int(row["disc_label"])

        event_time_t = torch.tensor(event_time, dtype=torch.float32)
        event_status_t = torch.tensor(event_status, dtype=torch.float32)
        disc_label_t = torch.tensor(disc_label, dtype=torch.long)

        return (
            dataset,         # 0
            case_id,         # 1
            filename,        # 2
            slide_tensor,    # 3
            event_time_t,    # 4
            event_status_t,  # 5
            disc_label_t     # 6
        )

    def __len__(self):
        return len(self.rows)

if __name__ == "__main__":
    raise SystemExit("Use main.py or run_survival.sh to launch training and evaluation.")
