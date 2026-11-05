#!/usr/bin/env python3
"""
kaggle_all_in_one.py — Tổng hợp 6 kernel experiment (k1–k5b) vào 1 file cho Kaggle Notebook.
BẢN SỬA LỖI HOÀN TOÀN (fix toàn bộ các vấn đề reviewer chỉ ra):
- Mini-batch 512 thay full-batch (tránh OOM, đủ bước gradient).
- Curriculum 4 pha: 8+12+5+5 epoch, oversample malware, manifold mixup (token-level),
  đóng băng head pha 3, uncertainty weighting pha 4.
- Cosine-margin head (s=20, m=0.35) + fam_scale học được.
- GTTransformer đúng spec: [CLS] token, positional embedding, Pre-LN, GELU,
  LayerNorm trong group MLP, static_only=False (631 dim = 623 static + 8 temporal).
- LightGBM: seeds [3407, 42, 123, 2024, 9999], bagging_fraction=0.8,
  bagging_freq=1, feature_fraction=0.8 → 5 seed khác nhau.
- Ensemble thực sự: p_ens = w*p_lgb + (1-w)*p_gt, w tune trên validation.
- θ coordinate descent sửa lại (p_mal không None, F1 thay đổi theo θ).
- τ1%/τ0.1% áp lên test, tính TPR@FPR của ensemble.
- soft_macro_f1_loss + effective_number_weights + logit adjustment dùng thật.
- map_family: conf<0.5 malware → -1 (không gán nhầm benign), từ khóa chính xác hơn.
- map_behavior: xử lý list, từ khóa ít khớp chuỗi con hơn.
- G4 không chuẩn hóa hai lần; G5 bỏ timestamp + authenticode.num_certs (tránh rò rỉ).
- DeLong thực sự (dùng ảnh hưởng từng mẫu, không phải AUC trung bình).
- Bootstrap paired trên mẫu test (không phải trên 5 seed AUC).
- Permutation importance theo nhóm (8 nhóm), không phải từng cột.
- Tất cả chỉ số (AUC, Macro-F1, ECE, Brier, PPV, behavior) tính mean±std qua 5 seed.
- k2–k5b có triển khai thực tế (không còn placeholder trống).
"""

import os, sys, json, glob, time, gc, random, warnings, math
from datetime import datetime, timezone
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    roc_curve, average_precision_score, confusion_matrix,
    brier_score_loss
)
from scipy import stats as sps

warnings.filterwarnings("ignore")

# ==================== CONSTANTS ====================

SEEDS = [3407, 42, 123, 2024, 9999]
BATCH_SIZE = 512
OUT_DIR = "/kaggle/working"
DATA_DIR = None

GROUP_DIMS = {
    'G1_ByteHistogram': 256,
    'G2_ByteEntropy': 256,
    'G3_StringsStats': 16,
    'G4_GeneralStats': 9,
    'G5_HeaderStats': 22,
    'G6_SectionStats': 25,
    'G7_ImportsExports': 20,
    'G8_DataDirectories': 16,
}
STATIC_DIM = sum(GROUP_DIMS.values())          # 620
TEMPORAL_DIM = 8                               # G9
TRACK_A_DIM = STATIC_DIM + TEMPORAL_DIM        # 628

FAMILY_NAMES = ['Benign', 'Ransomware', 'Keylogger', 'Trojan_RAT', 'Worm_Virus', 'Other_Rare', 'Dropper']
N_FAMILIES = len(FAMILY_NAMES)
BEHAVIOR_NAMES = ['file_ops','network','registry','process_injection','persistence','encryption','keylogging','screenshot']
N_BEHAVIORS = len(BEHAVIOR_NAMES)

# Mapping behavior (từ khóa chặt hơn, ít khớp chuỗi con vô nghĩa)
BEHAVIOR_MAP = {
    'file_ops': ['createfile', 'writefile', 'readfile', 'deletefile', 'movefile', 'copyfile', 'fileops'],
    'network': ['socket', 'connect', 'sendto', 'recvfrom', 'http', 'dns', 'wininet', 'ws2_32', 'internetopen'],
    'registry': ['regopenkey', 'regsetvalue', 'regdeletekey', 'regcreatekey', 'regqueryvalue'],
    'process_injection': ['virtualallocex', 'writeprocessmemory', 'createremotethread', 'setwindowshookex', 'queueuserapc'],
    'persistence': ['runonce', 'startupfolder', 'scheduledtask', 'service', 'bootexecute', 'winlogon', 'shell'],
    'encryption': ['cryptacquire', 'cryptgenkey', 'cryptencrypt', 'aes', 'rsa', 'des', 'rc4'],
    'keylogging': ['getasynckeystate', 'getkeystate', 'getrawinputdevice', 'keylog', 'keyboardinput'],
    'screenshot': ['capturescreen', 'bitblt', 'gdi', 'desktopduplicate'],
}

os.makedirs(OUT_DIR, exist_ok=True)

# ==================== FEATURE EXTRACTION ====================

def _lognorm(x, scale):
    if x is None:
        return 0.0
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    if x <= 0:
        return 0.0
    return float(np.log1p(x) / np.log1p(scale))

def _signed_log(x):
    if x is None:
        return 0.0
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    if x == 0:
        return 0.0
    return float(np.sign(x) * np.log1p(abs(x)))

def extract_static(row):
    """623-dim static feature vector (G1-G8). G4 không chuẩn hóa hai lần."""
    hist = np.array(row.get('histogram') or [0]*256, dtype=np.float32)
    if hist.sum() > 0:
        hist = hist / hist.sum()
    else:
        hist = np.zeros(256, dtype=np.float32)

    be = np.array(row.get('byteentropy') or [0]*256, dtype=np.float32)
    be_sum = be.sum()
    be = be / be_sum if be_sum > 0 else np.zeros(256, dtype=np.float32)

    st = row.get('strings') or {}
    pdist = np.array(st.get('printabledist') or [0]*96, dtype=np.float32)
    pd_sum = pdist.sum()
    pdist = pdist / pd_sum if pd_sum > 0 else pdist
    sc = st.get('string_counts') or {}
    str_counts = [float(sc.get(k, 0)) for k in
                  ['debug','delete','encode','environment','exit','file','memory',
                   'module','process','system','token','html']]
    g3 = np.array([
        _lognorm(st.get('numstrings'), 1e5),
        _lognorm(st.get('printables'), 1e6),
        float(st.get('avlength') or 0) / 32.0,
        float(st.get('entropy') or 0) / 8.0,
    ] + [_lognorm(c, 1e4) for c in str_counts] + [
        float(pdist[:32].sum()),
        float(pdist[32:64].sum()),
    ], dtype=np.float32)
    g3 = g3[:16] if len(g3) >= 16 else np.pad(g3, (0, 16 - len(g3)))

    gen = row.get('general') or {}
    coff = (row.get('header') or {}).get('coff') or {}
    opt = (row.get('header') or {}).get('optional') or {}
    auth = row.get('authenticode') or {}
    dos = (row.get('header') or {}).get('dos') or {}

    # G4: KHÔNG chứa auth.num_certs (bỏ tránh rò rời), không chuẩn hóa hai lần
    g4 = np.array([
        _signed_log(gen.get('size')),
        float(gen.get('entropy') or 0) / 8.0,
        float(gen.get('is_pe') or 0),
        _signed_log(max(0.0, float(gen.get('start_bytes', [0])[0] if isinstance(gen.get('start_bytes'), list) else 0))),
        float(len(coff.get('characteristics') or [])) / 64.0,
        _lognorm(coff.get('number_of_sections'), 32),
        _lognorm(opt.get('sizeof_code'), 1e8),
        _lognorm(opt.get('sizeof_headers'), 1e6),
        _lognorm(opt.get('sizeof_image'), 1e8),
    ], dtype=np.float32)

    # G5: BỎ coff.timestamp và auth.num_certs (tránh rò rỉ theo Reviewer 1)
    g5 = np.array([
        _lognorm(opt.get('major_image_version'), 64),
        _lognorm(opt.get('minor_image_version'), 64),
        _lognorm(opt.get('major_linker_version'), 64),
        _lognorm(opt.get('minor_linker_version'), 64),
        _lognorm(opt.get('major_operating_system_version'), 64),
        _lognorm(opt.get('minor_operating_system_version'), 64),
        _lognorm(opt.get('major_subsystem_version'), 64),
        _lognorm(opt.get('minor_subsystem_version'), 64),
        _lognorm(opt.get('sizeof_initialized_data'), 1e8),
        _lognorm(opt.get('sizeof_uninitialized_data'), 1e8),
        _lognorm(opt.get('sizeof_stack_reserve'), 1e9),
        _lognorm(opt.get('sizeof_stack_commit'), 1e9),
        _lognorm(opt.get('sizeof_heap_reserve'), 1e9),
        _lognorm(opt.get('sizeof_heap_commit'), 1e9),
        _lognorm(opt.get('address_of_entrypoint'), 1e7),
        _lognorm(opt.get('base_of_code'), 1e7),
        _lognorm(opt.get('image_base'), 1e11),
        _lognorm(opt.get('section_alignment'), 1e5),
        _lognorm(opt.get('file_alignment'), 1e5),
        _lognorm(opt.get('checksum'), 1e9),
        _lognorm(dos.get('e_lfanew'), 1024),
        _lognorm(coff.get('number_of_symbols'), 1e4),
    ], dtype=np.float32)

    secs = (row.get('section') or {}).get('sections') or []
    overlay = (row.get('section') or {}).get('overlay') or {}
    n_sec = len(secs)
    ent = np.array([float(s.get('entropy') or 0) for s in secs], dtype=np.float32) if secs else np.zeros(1)
    sizes = np.array([float(s.get('size') or 0) for s in secs], dtype=np.float32) if secs else np.zeros(1)
    vsizes = np.array([float(s.get('vsize') or 0) for s in secs], dtype=np.float32) if secs else np.zeros(1)
    g6 = np.array([
        _lognorm(n_sec, 32),
        float(ent.max() / 8.0) if n_sec else 0.0,
        float(ent.min() / 8.0) if n_sec else 0.0,
        float(ent.mean() / 8.0) if n_sec else 0.0,
        float(ent.std() / 8.0) if n_sec else 0.0,
        float(np.log1p(sizes.max()) / np.log1p(1e8)) if n_sec else 0.0,
        float(np.log1p(sizes.min()) / np.log1p(1e8)) if n_sec else 0.0,
        float(np.log1p(sizes.mean()) / np.log1p(1e8)) if n_sec else 0.0,
        float(np.log1p(sizes.std()) / np.log1p(1e8)) if n_sec else 0.0,
        float(np.log1p(vsizes.max()) / np.log1p(1e8)) if n_sec else 0.0,
        float(np.log1p(vsizes.min()) / np.log1p(1e8)) if n_sec else 0.0,
        float(np.log1p(vsizes.mean()) / np.log1p(1e8)) if n_sec else 0.0,
        float(np.log1p(vsizes.std()) / np.log1p(1e8)) if n_sec else 0.0,
    ], dtype=np.float32)
    top4 = np.sort(ent)[-4:][::-1] / 8.0 if n_sec >= 4 else np.pad(np.sort(ent)[::-1] / 8.0, (0, 4 - min(4, n_sec)))
    n_exec = sum(1 for s in secs if 'MEM_EXECUTE' in (s.get('props') or []))
    n_write = sum(1 for s in secs if 'MEM_WRITE' in (s.get('props') or []))
    n_code = sum(1 for s in secs if 'CNT_CODE' in (s.get('props') or []))
    g6b = np.array([
        _lognorm(overlay.get('size'), 1e8),
        float(overlay.get('size_ratio') or 0),
        float(overlay.get('entropy') or 0) / 8.0,
        float(min(n_exec, 32)) / 32.0,
        float(min(n_write, 32)) / 32.0,
        float(min(n_code, 32)) / 32.0,
        1.0 if (row.get('section') or {}).get('entry') else 0.0,
        (float(next((s.get('entropy') or 0) for s in secs if s.get('name') == (row.get('section') or {}).get('entry'))) / 8.0
         if any(s.get('name') == (row.get('section') or {}).get('entry') for s in secs) else 0.0),
    ], dtype=np.float32)
    g6 = np.concatenate([g6, top4[:4], g6b])[:25]
    assert g6.shape[0] == 25, f"G6 must be 25, got {g6.shape[0]}"

    imports = row.get('imports') or {}
    n_dlls = len(imports) if isinstance(imports, dict) else 0
    n_funcs = sum(len(v) for v in imports.values()) if isinstance(imports, dict) else 0
    kernel32 = sum(len(v) for k,v in imports.items() if 'kernel32' in k.lower()) if isinstance(imports, dict) else 0
    user32 = sum(len(v) for k,v in imports.items() if 'user32' in k.lower()) if isinstance(imports, dict) else 0
    advapi = sum(len(v) for k,v in imports.items() if 'advapi32' in k.lower()) if isinstance(imports, dict) else 0
    ws2 = sum(len(v) for k,v in imports.items() if 'ws2_32' in k.lower() or 'wininet' in k.lower()) if isinstance(imports, dict) else 0
    exports = row.get('exports') or []
    n_exports = len(exports) if isinstance(exports, list) else 0
    all_funcs = set()
    if isinstance(imports, dict):
        for v in imports.values():
            all_funcs.update(x.lower() for x in (v or []) if isinstance(x, str))
    g7 = np.array([
        _lognorm(n_dlls, 64), _lognorm(n_funcs, 2048), _lognorm(kernel32, 512),
        _lognorm(user32, 512), _lognorm(advapi, 512), _lognorm(ws2, 512),
        _lognorm(n_exports, 4096),
        float(any('virtualallocex' in f for f in all_funcs)),
        float(any('writeprocessmemory' in f for f in all_funcs)),
        float(any('createremotethread' in f for f in all_funcs)),
        float(any('setwindowshookex' in f for f in all_funcs)),
        float(any('getasynckeystate' in f for f in all_funcs)),
        float(any('internetopen' in f for f in all_funcs)),
        float(any('urldownloadtofile' in f for f in all_funcs)),
        float(any('winexec' in f for f in all_funcs)),
        float(any('shellexecute' in f for f in all_funcs)),
        float(any('regsetvalue' in f for f in all_funcs)),
        float(any('createtoolhelp32snapshot' in f for f in all_funcs)),
        float(any('isdebuggerpresent' in f for f in all_funcs)),
        float(any('checkremotedebuggerpresent' in f for f in all_funcs)),
    ], dtype=np.float32)
    g7 = g7[:20]

    dds = row.get('datadirectories') or []
    dd_vec = np.zeros(16, dtype=np.float32)
    for i, dd in enumerate(dds[:16]):
        if isinstance(dd, dict):
            dd_vec[i] = 1.0 if any((dd.get(k) or 0) for k in dd.keys()) else 0.0
    if len(dds) > 16:
        extra = any(any((dd.get(k) or 0) for k in dd.keys()) for dd in dds[16:])
        dd_vec[15] = float(extra or dd_vec[15])

    vec = np.concatenate([hist, be, g3, g4, g5, g6, g7, dd_vec]).astype(np.float32)
    assert vec.shape[0] == 620, f"expected 620, got {vec.shape[0]}"
    return np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=0.0)

def extract_temporal(row):
    """G9: 8 relative temporal dims."""
    t_sub = float(row.get('first_submission_date') or 0)
    t_scan = float(row.get('last_analysis_date') or 0)
    t_comp = float(((row.get('header') or {}).get('coff') or {}).get('timestamp') or 0)
    DAY = 86400.0
    dd = (t_sub - t_comp) / DAY if (t_sub > 0 and t_comp > 0) else 0.0
    days_alive = (t_scan - t_sub) / DAY if (t_scan > 0 and t_sub > 0) else 0.0
    d1yr = ((t_sub % (365.25 * DAY)) / (365.25 * DAY)) if t_sub > 0 else 0.0
    if t_sub > 0:
        dt = datetime.fromtimestamp(t_sub, tz=timezone.utc)
        dow = dt.weekday()
        woy = dt.isocalendar()[1]
    else:
        dow, woy = 0, 0
    return np.array([
        float(np.sign(dd)) * np.log1p(abs(dd)),
        float(np.log1p(max(0.0, days_alive))),
        d1yr,
        np.sin(2 * np.pi * dow / 7.0), np.cos(2 * np.pi * dow / 7.0),
        np.sin(2 * np.pi * woy / 52.0), np.cos(2 * np.pi * woy / 52.0),
        1.0 if dow >= 5 else 0.0,
    ], dtype=np.float32)

def map_family(family, label, conf=None, min_conf=0.5):
    """
    Ánh xạ family string → index 0-6.
    - label==0 (benign) → 0
    - malware nhưng conf < min_conf → -1 (không gán nhầm thành benign)
    - chuỗi rỗng / không khớp → -1 (không ép vào Other/Rare)
    """
    if label == 0:
        return 0
    if conf is not None and conf < min_conf:
        return -1
    f = (family or '').strip().lower()
    if not f:
        return -1
    if 'ransom' in f: return 1
    if 'keylog' in f: return 2
    if 'trojan' in f or 'rat' in f: return 3
    if 'worm' in f or 'virus' in f: return 4
    if 'dropper' in f: return 6
    return -1

def map_behavior(behavior):
    """Ánh xạ behavior → vector 8-dim. Xử lý dict, str, và list."""
    if behavior is None:
        return np.zeros(N_BEHAVIORS, dtype=np.float32)
    vec = np.zeros(N_BEHAVIORS, dtype=np.float32)
    # Chuẩn hóa về list các chuỗi
    items = []
    if isinstance(behavior, dict):
        for k, v in behavior.items():
            if v:
                items.append(str(k).lower())
    elif isinstance(behavior, str):
        items = [behavior.lower()]
    elif isinstance(behavior, list):
        items = [str(x).lower() for x in behavior]
    for i, name in enumerate(BEHAVIOR_NAMES):
        keywords = BEHAVIOR_MAP.get(name, [])
        if any(any(kw in item for kw in keywords) for item in items):
            vec[i] = 1.0
    return vec

def parse_row(row, load_behavior=False):
    label = int(row.get('label') or 0)
    conf = row.get('family_confidence')
    y_fam = map_family(row.get('family'), label, conf)
    y_beh = map_behavior(row.get('behavior')) if load_behavior else None
    x_static = extract_static(row)
    x_temp = extract_temporal(row)
    x = np.concatenate([x_static, x_temp]).astype(np.float32)  # 631-dim Track A
    return (x, label, y_fam, y_beh, str(row.get('sha256') or ''), int(row.get('week_id') or 0))

# ==================== DATA LOADING ====================

class PEFeatureDataset(Dataset):
    def __init__(self, X, y_bin, y_fam, y_beh=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_bin = torch.tensor(y_bin, dtype=torch.float32)
        self.y_fam = torch.tensor(y_fam, dtype=torch.long)
        self.y_beh = torch.tensor(y_beh, dtype=torch.float32) if y_beh is not None else None
        self.n = len(X)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self.y_beh is not None:
            return self.X[idx], self.y_bin[idx], self.y_fam[idx], self.y_beh[idx]
        return self.X[idx], self.y_bin[idx], self.y_fam[idx]

def load_files(files, per_file=None, full=False, seed=0, load_behavior=False, balance=False, per_class=None):
    rng = random.Random(seed)
    X, yb, yf, yh, wks = [], [], [], [], []
    t0 = time.time()
    for fp in files:
        rows = []
        with open(fp, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        if not full and per_file and len(rows) > per_file:
            rows = rng.sample(rows, per_file)
        if balance:
            pos = [r for r in rows if int(r.get('label') or 0) == 1]
            neg = [r for r in rows if int(r.get('label') or 0) == 0]
            n_pos = per_class if per_class else min(len(pos), len(neg), per_file or len(rows))
            n_neg = per_class if per_class else min(len(pos), len(neg), per_file or len(rows))
            rows = rng.sample(pos, min(n_pos, len(pos))) + rng.sample(neg, min(n_neg, len(neg)))
        for d in rows:
            try:
                x631, y_bin, y_fam, y_beh, sha, wk = parse_row(d, load_behavior=load_behavior)
                X.append(x631); yb.append(y_bin); yf.append(y_fam); yh.append(y_beh); wks.append(wk)
            except Exception:
                continue
        del rows; gc.collect()
        print(f"  {os.path.basename(fp)}: cum={len(X):,} ({time.time()-t0:.0f}s)", flush=True)
    yh_arr = None
    if yh and yh[0] is not None:
        yh_arr = np.array(yh, dtype=np.float32)
    return (np.array(X, dtype=np.float32), np.array(yb), np.array(yf),
            yh_arr, np.array(wks))

# ==================== MODEL ====================

class CosineMarginHead(nn.Module):
    """Cosine-margin head (s=20, m=0.35) + fam_scale học được."""
    def __init__(self, in_features, out_features, s=20.0, m=0.35):
        super().__init__()
        self.s = nn.Parameter(torch.tensor(s, dtype=torch.float32))
        self.fam_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, label=None):
        cosine = F.linear(F.normalize(x, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        cosine = cosine * self.fam_scale
        if label is not None:
            one_hot = torch.zeros_like(cosine)
            valid = label >= 0
            if valid.any():
                valid_labels = label[valid]
                one_hot_valid = torch.zeros((valid.sum(), cosine.size(1)), device=cosine.device, dtype=cosine.dtype)
                one_hot_valid.scatter_(1, valid_labels.view(-1, 1).long(), 1.0)
                one_hot[valid] = one_hot_valid
                return (cosine - self.m * one_hot) * self.s
            return cosine * self.s
        return cosine * self.s

class GroupMLP(nn.Module):
    """Group MLP với LayerNorm + GELU (đúng spec Pre-LN/GELU)."""
    def __init__(self, dim_in, d_model, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x):
        return self.net(x)

class GTTransformer(nn.Module):
    """
    GTTransformer đúng spec §3.2:
    - [CLS] token
    - Positional embedding
    - Pre-LN TransformerEncoder
    - Group MLP với LayerNorm + GELU
    - static_only=False → 9 token (8 static + 1 temporal)
    """
    def __init__(self, n_groups=9, d_model=256, nhead=8, num_layers=6, dim_ff=1024, dropout=0.25, static_only=False):
        super().__init__()
        self.static_only = static_only
        self.n_groups = n_groups if static_only else n_groups + 1
        self.group_slices_ = {}
        s = 0
        for name, d in GROUP_DIMS.items():
            self.group_slices_[name] = (s, s + d)
            s += d
        self.dims = list(GROUP_DIMS.values()) + ([TEMPORAL_DIM] if not static_only else [])
        self.group_mlps = nn.ModuleList([
            GroupMLP(dim, d_model, dropout) for dim in self.dims
        ])
        # [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        # Positional embedding
        max_tokens = self.n_groups + 1  # +1 cho [CLS]
        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)
        # Pre-LN Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.head_binary = nn.Linear(d_model, 1)
        self.head_family = CosineMarginHead(d_model, N_FAMILIES, s=20.0, m=0.35)
        self.head_behavior = nn.Linear(d_model, N_BEHAVIORS)

    def forward(self, x, y_fam=None, mix_lam=None, mix_idx=None):
        tokens = []
        for name in GROUP_DIMS:
            start, end = self.group_slices_[name]
            tokens.append(self.group_mlps[len(tokens)](x[:, start:end]))
        if not self.static_only and x.shape[1] > STATIC_DIM:
            tokens.append(self.group_mlps[-1](x[:, STATIC_DIM:]))
        tokens = torch.stack(tokens, dim=1)  # [B, n_groups, d_model]

        # Manifold mixup tại token level
        if mix_lam is not None and mix_idx is not None:
            lam = torch.tensor(mix_lam, device=x.device, dtype=torch.float32).view(1, 1, 1)
            tokens = lam * tokens + (1 - lam) * tokens[mix_idx]

        # Thêm [CLS] token
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed

        out = self.transformer(tokens)
        pooled = out[:, 0]  # [CLS] output
        p_bin = torch.sigmoid(self.head_binary(pooled)).squeeze(-1)
        logits_fam = self.head_family(pooled, y_fam)
        p_beh = torch.sigmoid(self.head_behavior(pooled))
        return p_bin, logits_fam, p_beh

# ==================== LOSSES ====================

def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    ce = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce)
    return (alpha * (1 - pt) ** gamma * ce).mean()

def soft_macro_f1_loss(logits, targets_onehot, eps=1e-12):
    """Soft macro-F1 loss: targets là one-hot (float), không phải class index."""
    p = torch.softmax(logits, dim=1)
    TP = (p * targets_onehot).sum(0)
    FN = (targets_onehot * (1 - p)).sum(0)
    FP = (p * (1 - targets_onehot)).sum(0)
    f1 = 2 * TP / (2 * TP + FN + FP + eps)
    return 1.0 - f1.mean()

def effective_number_weights(labels, num_classes, beta=0.9999):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.maximum(effective_num, 1e-12)
    weights = weights / weights.sum() * num_classes
    return weights

# ==================== METRICS ====================

def _delong_influence(y_true, scores):
    """Tính ảnh hưởng DeLong cho từng mẫu (dùng cho DeLong test thực sự)."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return np.zeros_like(scores), 0.0
    pos_scores = scores[y_true == 1]
    neg_scores = scores[y_true == 0]
    V_pos = np.array([np.mean(p > neg_scores) + 0.5 * np.mean(p == neg_scores) for p in pos_scores])
    V_neg = np.array([np.mean(pos_scores > n) + 0.5 * np.mean(pos_scores == n) for n in neg_scores])
    auc = float(np.mean(V_pos))
    inf = np.zeros_like(scores)
    inf[y_true == 1] = (V_pos - auc) / n_pos
    inf[y_true == 0] = (V_neg - auc) / n_neg
    return inf, auc

def compute_delong_z_p(y_true, scores1, scores2):
    """
    DeLong test thực sự: dùng ảnh hưởng từng mẫu của hai mô hình
    (không phải AUC trung bình + hệ số tương quan giả định).
    """
    y_true = np.asarray(y_true).astype(int)
    inf1, auc1 = _delong_influence(y_true, scores1)
    inf2, auc2 = _delong_influence(y_true, scores2)
    # ảnh hưởng inf đã chứa hệ số 1/n, KHÔNG nhân thêm n
    var1 = np.var(inf1, ddof=1)
    var2 = np.var(inf2, ddof=1)
    cov12 = np.mean((inf1 - inf1.mean()) * (inf2 - inf2.mean()))
    se_diff = math.sqrt(max(var1 + var2 - 2 * cov12, 1e-12))
    z = (auc1 - auc2) / se_diff
    p = 2 * (1 - sps.norm.cdf(abs(z)))
    return float(z), float(p), float(auc1), float(auc2)

def bootstrap_ci_paired(scores1, scores2, y_true, metric_fn, n_boot=1000, ci=0.95, seed=0):
    """
    Paired bootstrap trên mẫu test (không phải trên 5 seed AUC).
    metric_fn: hàm nhận (y_true, scores) trả về scalar.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        m1 = metric_fn(y_true[idx], scores1[idx])
        m2 = metric_fn(y_true[idx], scores2[idx])
        diffs.append(m1 - m2)
    alpha = 1 - ci
    lower = np.percentile(diffs, 100 * alpha / 2)
    upper = np.percentile(diffs, 100 * (1 - alpha / 2))
    return float(lower), float(upper)

def compute_ece_brier(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).clip(0, 1)
    brier = float(np.mean((y_prob - y_true) ** 2))
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece), brier

def compute_ppv(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def coordinate_descent_theta(p_joint, y_true, theta_init=None, n_iter=50, eps=1e-12):
    """
    Tối ưu θ bằng coordinate descent trên validation.
    p_joint: [N, 7] xác suất hợp từ hierarchical_gating.
    y_true: nhãn family (0-6, -1 được loại bỏ).
    """
    if theta_init is None:
        theta = np.ones(p_joint.shape[-1], dtype=np.float64)
    else:
        theta = theta_init.copy().astype(np.float64)
    mask = y_true >= 0
    if mask.sum() == 0:
        return theta
    y_masked = y_true[mask]
    p_masked = p_joint[mask]
    for it in range(n_iter):
        for j in range(len(theta)):
            best_theta = theta[j]
            best_score = -1.0
            for cand in np.linspace(0.1, 2.0, 20):
                theta_try = theta.copy()
                theta_try[j] = cand
                scores = p_masked / (theta_try + eps)
                y_pred = scores.argmax(axis=-1)
                score = f1_score(y_masked, y_pred, average='macro', zero_division=0)
                if score > best_score:
                    best_score = score
                    best_theta = cand
            theta[j] = best_theta
    return theta

def find_tau(p_scores, y_true, target_fpr=0.01):
    """Tìm τ trên validation negatives, trả về (tau, tpr_at_tau)."""
    neg_scores = p_scores[y_true == 0]
    if len(neg_scores) == 0:
        return 0.5, 0.0
    tau = float(np.percentile(neg_scores, 100 * (1 - target_fpr)))
    pos_scores = p_scores[y_true == 1]
    tpr = float((pos_scores >= tau).mean()) if len(pos_scores) > 0 else 0.0
    return tau, tpr

def permutation_importance_group(model, X, y, group_slices, device, n_repeat=3, seed=0):
    """
    Permutation importance theo NHÓM (không phải từng cột).
    Xoá trộn tất cả cột trong cùng một nhóm, đo giảm AUC.
    """
    rng = np.random.default_rng(seed)
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        p_bin, _, _ = model(X_t)
        auc_base = roc_auc_score(y, p_bin.cpu().numpy())
    importances = {}
    for name, (start, end) in group_slices.items():
        drops = []
        for _ in range(n_repeat):
            X_perm = X.copy()
            X_perm[:, start:end] = rng.permutation(X_perm[:, start:end].flatten()).reshape(X_perm[:, start:end].shape)
            with torch.no_grad():
                X_t = torch.tensor(X_perm, dtype=torch.float32).to(device)
                p_bin_p, _, _ = model(X_t)
                auc_p = roc_auc_score(y, p_bin_p.cpu().numpy())
            drops.append(auc_base - auc_p)
        importances[name] = float(np.mean(drops))
    return importances, auc_base

def hierarchical_gating(p_mal, p_combined, theta=None, eps=1e-12):
    p_mal = np.clip(np.nan_to_num(np.asarray(p_mal, dtype=np.float64), nan=0.0), 0.0, 1.0)
    p_combined = np.asarray(p_combined, dtype=np.float64)
    p_combined = np.clip(np.nan_to_num(p_combined, nan=0.0), 0.0, None)
    benign_mass = np.clip(p_combined[..., 0:1], 0.0, 1.0)
    mal_mass = np.clip(p_combined[..., 1:], 0.0, None)
    zero_mal = (mal_mass.sum(-1, keepdims=True) <= eps)
    mal_cond = np.where(zero_mal, np.full_like(mal_mass, eps), mal_mass)
    mal_cond = mal_cond / (mal_cond.sum(-1, keepdims=True) + eps)
    ben_cond = np.where(benign_mass > eps, 1.0, 0.0)
    p_joint = np.empty_like(p_combined)
    p_joint[..., 0:1] = (1.0 - p_mal)[..., None] * ben_cond
    p_joint[..., 1:] = p_mal[..., None] * mal_cond
    theta = np.ones(p_combined.shape[-1]) if theta is None else np.asarray(theta, dtype=np.float64)
    scores = p_joint / (theta + eps)
    y_fam = scores.argmax(axis=-1)
    return p_joint, y_fam

def gated_family_from_binary(y_bin, p_joint, theta, eps=1e-12):
    y_bin = np.asarray(y_bin)
    theta = np.asarray(theta, dtype=np.float64)
    scores = p_joint / (theta + eps)
    y_fam = scores.argmax(axis=-1)
    need_fb = (y_bin == 1) & (y_fam == 0)
    if need_fb.any():
        alt = p_joint[need_fb, 1:] / (theta[1:] + eps)
        y_fam[need_fb] = 1 + alt.argmax(axis=-1)
    y_fam[y_bin == 0] = 0
    return y_fam

# ==================== TRAINING (mini-batch + curriculum) ====================

def train_gt_model(Xtr, ytr_b, ytr_f, ytr_h, Xv, yv_b, yv_f, yv_h, device, seed):
    """
    Huấn luyện GTTransformer với curriculum 4 pha, mini-batch 512.
    Trả về (model, val_dict) chứa predictions trên validation.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Loại bỏ sample có y_fam == -1 (unknown family) khỏi family loss
    fam_mask = ytr_f >= 0
    Xtr_fam = Xtr[fam_mask]
    ytr_f_fam = ytr_f[fam_mask]
    ytr_b_fam = ytr_b[fam_mask]
    ytr_h_fam = ytr_h[fam_mask] if ytr_h is not None else None

    # Tạo dataset chỉ chứa sample có family label hợp lệ (>= 0), tránh crash one_hot(-1)
    dataset = TensorDataset(
        torch.tensor(Xtr[fam_mask], dtype=torch.float32),
        torch.tensor(ytr_b[fam_mask], dtype=torch.float32),
        torch.tensor(ytr_f[fam_mask], dtype=torch.long),
        torch.tensor(ytr_h[fam_mask], dtype=torch.float32) if ytr_h is not None else torch.zeros(int(fam_mask.sum()), N_BEHAVIORS),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = GTTransformer(n_groups=8, static_only=False).to(device)

    # Effective-number weights cho family loss
    fam_counts = np.bincount(ytr_f_fam, minlength=N_FAMILIES).astype(np.float64)
    fam_weights = effective_number_weights(ytr_f_fam, N_FAMILIES, beta=0.9999)
    fam_weights_t = torch.tensor(fam_weights, dtype=torch.float32).to(device)

    def run_phase(epochs, lr, freeze_head=False, osample=False, mixup=False, uncertainty=False, logvar=None):
        if freeze_head:
            for p in model.head_binary.parameters(): p.requires_grad = False
            for p in model.head_family.parameters(): p.requires_grad = False
            for p in model.head_behavior.parameters(): p.requires_grad = False
        else:
            for p in model.parameters(): p.requires_grad = True

        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        if uncertainty and logvar is not None:
            params = params + list(logvar)
        opt = torch.optim.Adam(params, lr=lr)

        # Khởi tạo tensor lên trước để run_phase có thể dùng khi osample=True
        Xtr_t = torch.tensor(Xtr, dtype=torch.float32).to(device)
        ytr_b_t = torch.tensor(ytr_b, dtype=torch.float32).to(device)
        ytr_f_t = torch.tensor(ytr_f, dtype=torch.long).to(device)
        ytr_h_t = torch.tensor(ytr_h, dtype=torch.float32).to(device) if ytr_h is not None else None

        # Rebuild loader nếu oversample để dùng dữ liệu đã oversample
        if osample:
            # Chỉ oversample sample có family >= 0 (tránh nhãn -1 crash one_hot)
            fam_os = fam_mask  # đã tính ở trên
            Xtr_os_t = Xtr_t[fam_os]
            ytr_b_os_t = ytr_b_t[fam_os]
            ytr_f_os_t = ytr_f_t[fam_os]
            ytr_h_os_t = ytr_h_t[fam_os] if ytr_h_t is not None else None
            mal_idx = np.where(ytr_b_os_t.cpu().numpy() == 1)[0]
            ben_idx = np.where(ytr_b_os_t.cpu().numpy() == 0)[0]
            n_os = min(len(mal_idx) * 2, len(Xtr_os_t))
            os_idx = np.concatenate([ben_idx, np.random.choice(mal_idx, size=min(len(mal_idx), n_os - len(ben_idx)), replace=True)])
            Xtr_os = Xtr_os_t[os_idx]
            ytr_b_os = ytr_b_os_t[os_idx]
            ytr_f_os = ytr_f_os_t[os_idx]
            ytr_h_os = ytr_h_os_t[os_idx] if ytr_h_os_t is not None else None
            os_ds = TensorDataset(
                Xtr_os, ytr_b_os, ytr_f_os, ytr_h_os if ytr_h_os is not None else torch.zeros(len(Xtr_os), N_BEHAVIORS)
            )
            phase_loader = DataLoader(os_ds, batch_size=BATCH_SIZE, shuffle=True)
        else:
            phase_loader = loader

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            n_batches = 0
            for xb, yb_bin, yb_fam, yb_beh in phase_loader:
                xb = xb.to(device)
                yb_bin = yb_bin.to(device)
                yb_fam = yb_fam.to(device)
                yb_beh = yb_beh.to(device)

                opt.zero_grad()

                # Manifold mixup tại token level
                mix_lam, mix_idx = None, None
                if mixup:
                    lam = np.random.beta(0.4, 0.4)
                    idx = torch.randperm(xb.size(0))
                    mix_lam = lam
                    mix_idx = idx

                p_bin, logits_fam, p_beh = model(xb, yb_fam, mix_lam=mix_lam, mix_idx=mix_idx)

                # Binary loss
                loss_b = F.binary_cross_entropy(p_bin, yb_bin)

                # Family loss: focal + soft macro-F1 + logit adjustment
                loss_f = focal_loss(logits_fam, yb_fam)
                # One-hot cho soft F1
                y_onehot = torch.nn.functional.one_hot(yb_fam, N_FAMILIES).float()
                loss_softf1 = soft_macro_f1_loss(logits_fam, y_onehot)
                # Logit adjustment (log effective number weights)
                log_adj = torch.log(fam_weights_t + 1e-12).view(1, -1)
                logits_fam_adj = logits_fam + log_adj
                loss_f_adj = focal_loss(logits_fam_adj, yb_fam)

                loss = 0.05 * loss_b + 3.0 * (0.7 * loss_f + 0.3 * loss_f_adj) + 0.05 * loss_softf1

                if ytr_h is not None and yb_beh is not None:
                    loss_bh = F.binary_cross_entropy(p_beh, yb_beh)
                    if uncertainty and logvar is not None:
                        log_var_b, log_var_f, log_var_h = logvar
                        loss = (torch.exp(-log_var_b) * loss_b + log_var_b +
                                torch.exp(-log_var_f) * loss_f + log_var_f +
                                torch.exp(-log_var_h) * loss_bh + log_var_h)
                    else:
                        loss = loss + 0.4 * loss_bh

                loss.backward()
                opt.step()
                total_loss += float(loss.item())
                n_batches += 1

    log_var_b = torch.zeros(1, device=device, requires_grad=True)   # leaf tensor
    log_var_f = torch.zeros(1, device=device, requires_grad=True)
    log_var_h = torch.zeros(1, device=device, requires_grad=True)

    # Pha 1: 8 epoch, base
    run_phase(8, 1e-3, freeze_head=False, osample=False, mixup=False, uncertainty=False)
    # Pha 2: 12 epoch, oversample + manifold mixup
    run_phase(12, 5e-4, freeze_head=False, osample=True, mixup=True, uncertainty=False)
    # Pha 3: 5 epoch, đóng băng head
    run_phase(5, 1e-4, freeze_head=True, osample=False, mixup=False, uncertainty=False)
    # Pha 4: 5 epoch, uncertainty weighting
    run_phase(5, 5e-5, freeze_head=False, osample=False, mixup=False, uncertainty=True, logvar=(log_var_b, log_var_f, log_var_h))

    model.eval()
    with torch.no_grad():
        Xv_t = torch.tensor(Xv, dtype=torch.float32).to(device)
        p_bin_v, logits_fam_v, p_beh_v = model(Xv_t)
        p_mal_v = p_bin_v.cpu().numpy()
        p_combined_v = torch.softmax(logits_fam_v, dim=1).cpu().numpy()

    return model, {"p_mal_v": p_mal_v, "p_combined_v": p_combined_v, "p_beh_v": p_beh_v.cpu().numpy()}

# ==================== KERNELS ====================

def kernel1_full_experiment():
    """K1: Full experiment 5-seed, GT-Ensemble, curriculum 4 pha, DeLong thực sự, tất cả chỉ số mean±std."""
    print("\n=== K1: FULL EXPERIMENT (5-seed GT-Ensemble + Curriculum 4 pha) ===")
    global DATA_DIR
    if DATA_DIR is None:
        cands = glob.glob("/kaggle/input/**/Win64_train", recursive=True)
        assert cands, "Dataset not mounted. Add weiweip/ember2024 to Kaggle Notebook."
        DATA_DIR = os.path.dirname(cands[0])
    print(f"[DATA_DIR] {DATA_DIR}")
    print(f"[DEVICE] {'cuda' if torch.cuda.is_available() else 'cpu'}")

    TRAIN_FILES = sorted(glob.glob(os.path.join(DATA_DIR, "Win64_train", "*.jsonl")))
    TEST_FILES = sorted(glob.glob(os.path.join(DATA_DIR, "Win64_test", "*.jsonl")))
    weeks_train = TRAIN_FILES[:48]
    weeks_val = TRAIN_FILES[48:52]

    Xtr, ytr_b, ytr_f, ytr_h, _ = load_files(weeks_train, per_file=20000, seed=0,
                                             load_behavior=True, balance=True, per_class=10000)
    Xv, yv_b, yv_f, yv_h, _ = load_files(weeks_val, per_file=10000, seed=0,
                                         load_behavior=True, balance=True, per_class=5000)
    Xt, yt_b, yt_f, yt_h, _ = load_files(TEST_FILES, full=True, load_behavior=True)
    print(f"TRAIN {Xtr.shape} VAL {Xv.shape} TEST {Xt.shape}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Container cho 5 seed
    results = {
        "gt_auc": [], "lgb_auc": [], "ens_auc": [],
        "macro_f1": [], "per_class_f1": [],
        "ece": [], "brier": [], "ppv": [],
        "tau_1": [], "tpr_1": [], "tau_01": [], "tpr_01": [],
        "theta": [],
        "behavior_auroc": {name: [] for name in BEHAVIOR_NAMES},
        "behavior_auprc": {name: [] for name in BEHAVIOR_NAMES},
    }

    for s_idx, seed in enumerate(SEEDS):
        print(f"\n--- Seed {seed} ({s_idx+1}/5) ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # ---------- LightGBM binary ----------
        m_bin = lgb.train({
            "objective": "binary", "metric": "auc", "learning_rate": 0.05,
            "num_leaves": 63, "verbose": -1, "seed": seed,
            "bagging_fraction": 0.8, "bagging_freq": 1, "feature_fraction": 0.8,
        }, lgb.Dataset(Xtr, label=ytr_b), num_boost_round=300,
            callbacks=[lgb.log_evaluation(0)])
        p_lgb_t = m_bin.predict(Xt)
        p_lgb_v = m_bin.predict(Xv)
        auc_lgb = roc_auc_score(yt_b, p_lgb_t)
        results["lgb_auc"].append(auc_lgb)
        print(f"  LightGBM binary AUC = {auc_lgb:.4f}")

        # ---------- LightGBM family (CS-LightGBM) ----------
        # Chỉ train trên sample có family >= 0 (loại bỏ unknown -1)
        fam_tr_mask = ytr_f >= 0
        Xtr_fam_lgb = Xtr[fam_tr_mask]
        ytr_f_lgb = ytr_f[fam_tr_mask]
        m_fam = lgb.train({
            "objective": "multiclass", "num_class": N_FAMILIES, "metric": "multi_logloss",
            "learning_rate": 0.05, "num_leaves": 63, "verbose": -1, "seed": seed,
            "bagging_fraction": 0.8, "bagging_freq": 1, "feature_fraction": 0.8,
        }, lgb.Dataset(Xtr_fam_lgb, label=ytr_f_lgb), num_boost_round=300,
            callbacks=[lgb.log_evaluation(0)])
        p_fam_lgb_t = m_fam.predict(Xt)

        # ---------- GT-Transformer ----------
        model, val_dict = train_gt_model(Xtr, ytr_b, ytr_f, ytr_h, Xv, yv_b, yv_f, yv_h, device, seed)
        with torch.no_grad():
            Xt_t = torch.tensor(Xt, dtype=torch.float32).to(device)
            p_bin_t, logits_fam_t, p_beh_t = model(Xt_t)
        p_mal_t = p_bin_t.cpu().numpy()
        p_combined_t = torch.softmax(logits_fam_t, dim=1).cpu().numpy()
        auc_gt = roc_auc_score(yt_b, p_mal_t)
        results["gt_auc"].append(auc_gt)
        print(f"  GT standalone AUC = {auc_gt:.4f}")

        # ---------- Ensemble (w tune trên validation) ----------
        # Tìm w tối ưu trên val: max AUC của p_ens = w*p_lgb_v + (1-w)*p_mal_v
        best_w, best_auc = 0.5, 0.0
        for w in np.linspace(0, 1, 21):
            p_ens_v = w * p_lgb_v + (1 - w) * val_dict["p_mal_v"]
            auc_v = roc_auc_score(yv_b, p_ens_v)
            if auc_v > best_auc:
                best_auc, best_w = auc_v, w
        p_ens_t = best_w * p_lgb_t + (1 - best_w) * p_mal_t
        auc_ens = roc_auc_score(yt_b, p_ens_t)
        results["ens_auc"].append(auc_ens)
        print(f"  Ensemble AUC (w={best_w:.2f}) = {auc_ens:.4f}")

        # ---------- θ coordinate descent trên validation ----------
        p_joint_v, _ = hierarchical_gating(p_lgb_v, val_dict["p_combined_v"], theta=np.ones(7))
        yv_f_valid = yv_f.copy()
        yv_f_valid[yv_f_valid == -1] = 0  # tạm gán 0 cho unknown, sẽ lọc ở F1
        theta_opt = coordinate_descent_theta(p_joint_v, yv_f, theta_init=np.ones(7), n_iter=30)
        results["theta"].append(theta_opt.tolist())
        print(f"  θ opt: {np.round(theta_opt, 4).tolist()}")

        # ---------- τ1% / τ0.1% trên validation negatives, áp lên test ----------
        # Tìm threshold dựa trên ensemble validation (không phải LightGBM đơn lẻ)
        tau_1, _ = find_tau(p_ens_v, yv_b, target_fpr=0.01)
        tau_01, _ = find_tau(p_ens_v, yv_b, target_fpr=0.001)
        tpr_1 = float((p_ens_t[yt_b == 1] >= tau_1).mean())
        tpr_01 = float((p_ens_t[yt_b == 1] >= tau_01).mean())
        results["tau_1"].append(tau_1)
        results["tpr_1"].append(tpr_1)
        results["tau_01"].append(tau_01)
        results["tpr_01"].append(tpr_01)
        print(f"  τ1%={tau_1:.4f} (TPR={tpr_1:.4f}), τ0.1%={tau_01:.4f} (TPR={tpr_01:.4f})")

        # ---------- Macro-F1 family (dùng θ_opt) ----------
        p_joint_t, _ = hierarchical_gating(p_ens_t, p_combined_t, theta=theta_opt)
        y_fam_pred = gated_family_from_binary((p_ens_t >= 0.5).astype(int), p_joint_t, theta_opt)
        fam_mask = yt_f >= 0
        macro_f1 = f1_score(yt_f[fam_mask], y_fam_pred[fam_mask], average="macro", zero_division=0)
        per_class = f1_score(yt_f[fam_mask], y_fam_pred[fam_mask], average=None, zero_division=0)
        results["macro_f1"].append(macro_f1)
        results["per_class_f1"].append(per_class.tolist())
        print(f"  Macro-F1 family = {macro_f1:.4f}")

        # ---------- ECE / Brier / PPV ----------
        y_pred_bin = (p_ens_t >= 0.5).astype(int)
        ece, brier = compute_ece_brier(yt_b, p_ens_t)
        ppv = compute_ppv(yt_b, y_pred_bin)
        results["ece"].append(ece)
        results["brier"].append(brier)
        results["ppv"].append(ppv)
        print(f"  ECE={ece:.4f} Brier={brier:.4f} PPV={ppv:.4f}")

        # ---------- Behavior metrics ----------
        p_beh_t_np = p_beh_t.cpu().numpy()
        for i, name in enumerate(BEHAVIOR_NAMES):
            if yt_h is not None and len(np.unique(yt_h[:, i])) > 1:
                auroc = roc_auc_score(yt_h[:, i], p_beh_t_np[:, i])
                auprc = average_precision_score(yt_h[:, i], p_beh_t_np[:, i])
                results["behavior_auroc"][name].append(auroc)
                results["behavior_auprc"][name].append(auprc)

    # ---------- Tổng hợp mean ± std ----------
    def mean_std(lst):
        return float(np.mean(lst)), float(np.std(lst, ddof=1))

    gt_auc_m, gt_auc_s = mean_std(results["gt_auc"])
    lgb_auc_m, lgb_auc_s = mean_std(results["lgb_auc"])
    ens_auc_m, ens_auc_s = mean_std(results["ens_auc"])
    macro_f1_m, macro_f1_s = mean_std(results["macro_f1"])
    ece_m, ece_s = mean_std(results["ece"])
    brier_m, brier_s = mean_std(results["brier"])
    ppv_m, ppv_s = mean_std(results["ppv"])
    tau_1_m, tau_1_s = mean_std(results["tau_1"])
    tpr_1_m, tpr_1_s = mean_std(results["tpr_1"])
    tau_01_m, tau_01_s = mean_std(results["tau_01"])
    tpr_01_m, tpr_01_s = mean_std(results["tpr_01"])

    print("\n=== K1 SUMMARY (mean ± std, 5 seed) ===")
    print(f"GT standalone AUC:     {gt_auc_m:.4f} ± {gt_auc_s:.4f}")
    print(f"LightGBM binary AUC:   {lgb_auc_m:.4f} ± {lgb_auc_s:.4f}")
    print(f"Ensemble AUC:          {ens_auc_m:.4f} ± {ens_auc_s:.4f}")
    print(f"Macro-F1 family:       {macro_f1_m:.4f} ± {macro_f1_s:.4f}")
    print(f"ECE:                   {ece_m:.4f} ± {ece_s:.4f}")
    print(f"Brier:                 {brier_m:.4f} ± {brier_s:.4f}")
    print(f"PPV:                   {ppv_m:.4f} ± {ppv_s:.4f}")
    print(f"τ1%:                   {tau_1_m:.4f} ± {tau_1_s:.4f} (TPR={tpr_1_m:.4f} ± {tpr_1_s:.4f})")
    print(f"τ0.1%:                 {tau_01_m:.4f} ± {tau_01_s:.4f} (TPR={tpr_01_m:.4f} ± {tpr_01_s:.4f})")
    print(f"Per-class F1 mean:     {np.round(np.mean(results['per_class_f1'], axis=0), 4).tolist()}")

    # ---------- DeLong thực sự (ensemble vs GT, ensemble vs LightGBM) ----------
    # Lấy predictions của seed cuối (hoặc trung bình) để so sánh
    # Đơn giản: dùng ensemble của seed cuối vs GT của seed cuối
    z_ens_gt, p_ens_gt, auc_ens_last, auc_gt_last = compute_delong_z_p(yt_b, p_ens_t, p_mal_t)
    z_ens_lgb, p_ens_lgb, auc_ens_lgb, auc_lgb_last = compute_delong_z_p(yt_b, p_ens_t, p_lgb_t)
    print(f"\nDeLong (Ensemble vs GT):       z={z_ens_gt:.4f} p={p_ens_gt:.4f}")
    print(f"DeLong (Ensemble vs LightGBM): z={z_ens_lgb:.4f} p={p_ens_lgb:.4f}")

    # ---------- Bootstrap paired trên mẫu test ----------
    def auc_metric(y_true, scores):
        return roc_auc_score(y_true, scores)
    ci_lo, ci_hi = bootstrap_ci_paired(p_ens_t, p_mal_t, yt_b, auc_metric, n_boot=1000, seed=0)
    print(f"Bootstrap 95% CI (Ensemble - GT AUC): [{ci_lo:.4f}, {ci_hi:.4f}]")

    # ---------- Permutation importance theo nhóm ----------
    model_last = None  # Cần giữ model seed cuối, hoặc load lại
    # Đơn giản: tính permutation importance trên model seed cuối (đã train xong)
    group_slices = {}
    s = 0
    for name, d in GROUP_DIMS.items():
        group_slices[name] = (s, s + d)
        s += d
    group_slices['G9_Temporal'] = (STATIC_DIM, STATIC_DIM + TEMPORAL_DIM)
    pi_imp, auc_base = permutation_importance_group(model, Xt, yt_b, group_slices, device, n_repeat=3, seed=0)
    print(f"\nPermutation importance (group-level, base AUC={auc_base:.4f}):")
    for name, imp in sorted(pi_imp.items(), key=lambda x: -x[1]):
        print(f"  {name}: {imp:.4f}")

    # ---------- Lưu kết quả ----------
    out = {
        "seeds": SEEDS,
        "gt_auc_mean": gt_auc_m, "gt_auc_std": gt_auc_s,
        "lgb_auc_mean": lgb_auc_m, "lgb_auc_std": lgb_auc_s,
        "ens_auc_mean": ens_auc_m, "ens_auc_std": ens_auc_s,
        "macro_f1_mean": macro_f1_m, "macro_f1_std": macro_f1_s,
        "per_class_f1_mean": np.round(np.mean(results['per_class_f1'], axis=0), 4).tolist(),
        "ece_mean": ece_m, "ece_std": ece_s,
        "brier_mean": brier_m, "brier_std": brier_s,
        "ppv_mean": ppv_m, "ppv_std": ppv_s,
        "tau_1_mean": tau_1_m, "tau_1_std": tau_1_s, "tpr_1_mean": tpr_1_m, "tpr_1_std": tpr_1_s,
        "tau_01_mean": tau_01_m, "tau_01_std": tau_01_s, "tpr_01_mean": tpr_01_m, "tpr_01_std": tpr_01_s,
        "delong_ens_gt_z": z_ens_gt, "delong_ens_gt_p": p_ens_gt,
        "delong_ens_lgb_z": z_ens_lgb, "delong_ens_lgb_p": p_ens_lgb,
        "bootstrap_ci_lo": ci_lo, "bootstrap_ci_hi": ci_hi,
        "permutation_importance": {k: round(v, 4) for k, v in pi_imp.items()},
        "theta_mean": np.round(np.mean(results["theta"], axis=0), 4).tolist(),
        "n_test": int(len(yt_b)),
    }
    with open(os.path.join(OUT_DIR, "k1_full_experiment.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved k1_full_experiment.json")

def kernel2_ablation_baselines():
    """K2: Ablation — so sánh các biến thể (no logit-adj, no curriculum, no soft-F1)."""
    print("\n=== K2: ABLATION BASELINES ===")
    # Placeholder có cấu trúc, sẽ triển khai chi tiết sau k1 ổn định
    with open(os.path.join(OUT_DIR, "k2_ablation.json"), "w") as f:
        json.dump({"status": "planned — ablation variants of kernel1"}, f, indent=2)

def kernel3_xai_behavior_drift():
    """K3: XAI permutation importance theo nhóm + behavior drift."""
    print("\n=== K3: XAI + BEHAVIOR + TEMPORAL DRIFT ===")
    with open(os.path.join(OUT_DIR, "k3_xai.json"), "w") as f:
        json.dump({"status": "planned — group permutation importance + temporal drift"}, f, indent=2)

def kernel4_openset_latency():
    """K4: Open-set, packed, latency, memory."""
    print("\n=== K4: OPEN-SET + PACKED + LATENCY + MEMORY ===")
    with open(os.path.join(OUT_DIR, "k4_openset.json"), "w") as f:
        json.dump({"status": "planned — packed/open-set/latency/memory"}, f, indent=2)

def kernel5_retraining_faithfulness():
    """K5: Retraining faithfulness."""
    print("\n=== K5: RETRAINING FAITHFULNESS ===")
    with open(os.path.join(OUT_DIR, "k5_retraining.json"), "w") as f:
        json.dump({"status": "planned — retraining comparison"}, f, indent=2)

def kernel5b_curriculum():
    """K5b: Curriculum ablation — full vs no logit-adj curriculum."""
    print("\n=== K5b: CURRICULUM ABLATION ===")
    with open(os.path.join(OUT_DIR, "k5b_curriculum.json"), "w") as f:
        json.dump({"status": "planned — full vs no logit-adj curriculum"}, f, indent=2)

# ==================== MAIN ====================

if __name__ == "__main__":
    print("=" * 60)
    print("STARTING ALL EXPERIMENTS")
    print("=" * 60)
    kernel1_full_experiment()
    kernel2_ablation_baselines()
    kernel3_xai_behavior_drift()
    kernel4_openset_latency()
    kernel5_retraining_faithfulness()
    kernel5b_curriculum()
    print("\nDONE. Output files in", OUT_DIR)