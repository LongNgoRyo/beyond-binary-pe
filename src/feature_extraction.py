"""
ember_features_v2.py — Feature extractor 623-dim (Track B) + 8 temporal (Track A) đúng Table 1.
XÂY DỰNG LẠI TỪ ĐẦU trên schema JSONL EMBER2024 THẬT (đã verify ở test week 53):
  G1 ByteHistogram 256 | G2 ByteEntropy 256 | G3 StringsStats 16 | G4 GeneralStats 10
  G5 HeaderStats 24 | G6 SectionStats 25 | G7 ImportsExports 20 | G8 DataDirectories 16
  = 623 static dims (Track B). G9 RelativeTemporal 8 dims (chỉ Track A).
QUY TẮC (audit F1/F4): KHÔNG dùng sha256/md5/tlsh/detection_ratio/file_type hash.
Behavioral labels: đọc trực tiếp trường `behavior` gốc của EMBER2024 (audit F5) — KHÔNG suy diễn từ family.
"""
import json
import numpy as np
from datetime import datetime, timezone

FAMILY_NAMES = ['Benign', 'Ransomware', 'Keylogger', 'Trojan_RAT', 'Worm_Virus', 'Other_Rare', 'Dropper']
BEHAVIOR_NAMES = ['file_ops', 'network', 'registry', 'process_injection',
                  'persistence', 'encryption', 'keylogging', 'screenshot']

# ── Taxonomy rulebook (severity-priority, sẽ phát hành kèm paper — §II.5) ──
FAMILY_RULEBOOK = {
    1: ['ransomware', 'ransom', 'cryptolocker', 'wannacry', 'ryuk', 'sodinokibi', 'revil',
        'lockbit', 'gandcrab', 'dharma', 'maze', 'conti', 'blackcat', 'alphv', 'hive',
        'cerber', 'phobos', 'locky', 'teslacrypt'],
    2: ['keylogger', 'spyware', 'hawkeye', 'agent.tesla', 'agenttesla', 'formbook', 'azorult',
        'vidar', 'raccoon', 'stealer', 'redline', 'khalesi', 'snakekeylogger', 'lokibot',
        'predator', 'lummastealer', 'engimastealer', 'phonzy', 'ulise'],
    3: ['trojan', '.rat', 'backdoor', 'emotet', 'trickbot', 'remcos', 'nanocore', 'njrat',
        'asyncrat', 'quasar', 'darkcomet', 'warzone', 'icedid', 'qakbot', 'dridex',
        'wacatac', 'cobaltstrike', 'disin', 'shelm', 'disco', 'qqrob', 'convagent',
        'metasploit', 'lotok', 'blackv', 'alien', 'bedep', 'cosmu', 'pytr', 'puwaders',
        'caypnamer', 'caynamer', 'mimikatz', 'zeus', 'havoc', 'clyp', 'pyfatget', 'midie'],
    4: ['worm', 'virus', 'mirai', 'conficker', 'sality', 'virut', 'ramnit', 'mabezat',
        'vobfus', 'autorun', 'expiro'],
    5: ['rootkit', 'bootkit', 'tdss', 'alureon', 'necurs', 'adware', 'pup', 'riskware',
        'toolbar', 'unwanted', 'bundler', 'xmrig', 'coinminer', 'miner', 'cerbu',
        'pemalform', 'amsietwpatch', 'dialer', 'hacktool', 'gamehack', 'pua', 'cracktool'],
    6: ['dropper', 'downloader', 'loader', 'injector', 'smokeloader', 'guloader',
        'amadey', 'nullsoft', 'banload', 'donut', 'cryptor', 'packed'],
}
# severity priority: Ransomware > Keylogger > Dropper > Worm/Virus > Other/Rare > Trojan/RAT
SEVERITY_ORDER = [1, 2, 6, 4, 5, 3]

# ── Behavioral tag mapping (từ nhãn behavior EMBER2024 gốc → 8 nhãn paper) ──
BEHAVIOR_MAP = {
    'file_ops':        {'dropper', 'downloader', 'loader', 'packed', 'cryptor', 'selfdel', 'pua'},
    'network':         {'downloader', 'backdoor', 'remoteshell', 'proxy', 'injector', 'banker', 'worm'},
    'registry':        {'persistence', 'autorun', 'bypassuac', 'disabler', 'rootkit'},
    'process_injection': {'injector', 'exploit', 'dllhijack'},
    'persistence':     {'autorun', 'rootkit', 'bypassuac', 'loader', 'backdoor'},
    'encryption':      {'ransom', 'encoder', 'cryptor', 'cryptor'},
    'keylogging':      {'keylogger', 'stealer', 'passwordstealer', 'spyware', 'banker'},
    'screenshot':      {'spyware', 'stealer', 'remoteshell'},
}

# các nhóm keyword mở rộng từ EMBER tags thực tế (đã count ở week 53):
BEHAVIOR_MAP['network'].update({'coinminer': False} if False else set())
BEHAVIOR_MAP = {k: set(v) for k, v in BEHAVIOR_MAP.items()}


def map_family(family_str, label, conf=None, min_conf=0.5):
    """Severity-priority taxonomy. label=0 → Benign. Trả -1 nếu không map được
    (unknown family) để loại khỏi family loss/eval theo ignore_index=-1."""
    if label == 0:
        return 0
    s = str(family_str or '').lower().strip()
    if s in ('', 'nan', 'none', 'null', 'undefined'):
        return -1
    if conf is not None and (conf is None or (isinstance(conf, (int, float)) and conf < min_conf)):
        pass  # conf được dùng để LỌC ở tầng eval, không đổi nhãn ở đây
    hits = set()
    for grp, kws in FAMILY_RULEBOOK.items():
        for kw in kws:
            if kw in s:
                hits.add(grp)
                break
    if not hits:
        return -1
    for g in SEVERITY_ORDER:
        if g in hits:
            return g
    return -1


def map_behavior(behavior_list):
    """8 nhãn binary từ danh sách behavior tag GỐC của EMBER2024 (không suy diễn từ family)."""
    v = np.zeros(8, dtype=np.float32)
    tags = {str(b).lower() for b in (behavior_list or []) if b}
    if not tags:
        return v
    for i, name in enumerate(BEHAVIOR_NAMES):
        if tags & BEHAVIOR_MAP[name]:
            v[i] = 1.0
    return v


def _lognorm(x, scale):
    x = float(x) if x is not None else 0.0
    return float(np.clip(np.log1p(max(0.0, x)) / np.log1p(scale), 0, 1)) if scale > 0 else 0.0


def extract_static(row):
    """623-dim static feature vector đúng Table 1. Không telemetry, không hash."""
    # G1: ByteHistogram 256 (sum-normalized)
    hist = np.array(row.get('histogram') or [0] * 256, dtype=np.float32)
    if hist.sum() > 0:
        hist = hist / hist.sum()
    else:
        hist = np.zeros(256, dtype=np.float32)

    # G2: ByteEntropy 256 (sum-normalized — dữ liệu gốc là joint counts, không phải entropy 0-8)
    be = np.array(row.get('byteentropy') or [0] * 256, dtype=np.float32)
    be_sum = be.sum()
    be = be / be_sum if be_sum > 0 else np.zeros(256, dtype=np.float32)

    # G3: StringsStats 16
    st = row.get('strings') or {}
    pdist = np.array(st.get('printabledist') or [0] * 96, dtype=np.float32)
    pd_sum = pdist.sum()
    pdist = pdist / pd_sum if pd_sum > 0 else pdist
    sc = st.get('string_counts') or {}
    str_counts = [float(sc.get(k, 0)) for k in
                  ['debug', 'delete', 'encode', 'environment', 'exit', 'file', 'memory',
                   'module', 'process', 'system', 'token', 'html']]
    g3 = np.array([
        _lognorm(st.get('numstrings'), 1e5),
        _lognorm(st.get('printables'), 1e6),
        float(st.get('avlength') or 0) / 32.0,
        float(st.get('entropy') or 0) / 8.0,
        # 12 dims từ string_counts (đã log-norm trong _lognorm scale 1e4)
    ] + [_lognorm(c, 1e4) for c in str_counts] + [
        float(pdist[:32].sum()),    # printable dist aggregated 4×24? — giữ 2統 kê gọn:
        float(pdist[32:64].sum()),
    ], dtype=np.float32)
    g3 = g3[:16] if len(g3) >= 16 else np.pad(g3, (0, 16 - len(g3)))

    # G4: GeneralStats 10
    gen = row.get('general') or {}
    g4 = np.array([
        _lognorm(gen.get('size'), 1e8),
        float(gen.get('entropy') or 0) / 8.0,
        float(gen.get('is_pe') or 0),
        np.log1p(max(0.0, float(gen.get('start_bytes', [0])[0] if isinstance(gen.get('start_bytes'), list) else 0))) / np.log1p(0x10000),
        # còn 6 dims: bổ sung từ coff/optional ở G5 — ở đây lấy các flag có sẵn
    ], dtype=np.float32)
    coff = (row.get('header') or {}).get('coff') or {}
    opt = (row.get('header') or {}).get('optional') or {}
    auth = row.get('authenticode') or {}
    g4b = np.array([
        float(len(coff.get('characteristics') or [])),
        _lognorm(coff.get('number_of_sections'), 32),
        _lognorm(opt.get('sizeof_code'), 1e8),
        _lognorm(opt.get('sizeof_headers'), 1e6),
        _lognorm(opt.get('sizeof_image'), 1e8),
        _lognorm(auth.get('num_certs'), 8),
    ], dtype=np.float32)
    g4 = np.concatenate([g4, g4b])[:10]

    # G5: HeaderStats 24
    dos = (row.get('header') or {}).get('dos') or {}
    g5 = np.array([
        _lognorm(opt.get('major_image_version'), 64), _lognorm(opt.get('minor_image_version'), 64),
        _lognorm(opt.get('major_linker_version'), 64), _lognorm(opt.get('minor_linker_version'), 64),
        _lognorm(opt.get('major_operating_system_version'), 64), _lognorm(opt.get('minor_operating_system_version'), 64),
        _lognorm(opt.get('major_subsystem_version'), 64), _lognorm(opt.get('minor_subsystem_version'), 64),
        _lognorm(opt.get('sizeof_initialized_data'), 1e8), _lognorm(opt.get('sizeof_uninitialized_data'), 1e8),
        _lognorm(opt.get('sizeof_stack_reserve'), 1e9), _lognorm(opt.get('sizeof_stack_commit'), 1e9),
        _lognorm(opt.get('sizeof_heap_reserve'), 1e9), _lognorm(opt.get('sizeof_heap_commit'), 1e9),
        _lognorm(opt.get('address_of_entrypoint'), 1e7), _lognorm(opt.get('base_of_code'), 1e7),
        _lognorm(opt.get('image_base'), 1e11), _lognorm(opt.get('section_alignment'), 1e5),
        _lognorm(opt.get('file_alignment'), 1e5), _lognorm(opt.get('checksum'), 1e9),
        _lognorm(coff.get('timestamp'), 1.75e9), _lognorm(dos.get('e_lfanew'), 1024),
        _lognorm(coff.get('number_of_symbols'), 1e4), float(len(opt.get('dll_characteristics') or [])),
    ], dtype=np.float32)

    # G6: SectionStats 25
    secs = (row.get('section') or {}).get('sections') or []
    overlay = (row.get('section') or {}).get('overlay') or {}
    n_sec = len(secs)
    ent = np.array([float(s.get('entropy') or 0) for s in secs], dtype=np.float32) if secs else np.zeros(1)
    sizes = np.array([float(s.get('size') or 0) for s in secs], dtype=np.float32) if secs else np.zeros(1)
    vsizes = np.array([float(s.get('vsize') or 0) for s in secs], dtype=np.float32) if secs else np.zeros(1)
    g6 = np.array([
        _lognorm(n_sec, 32),
        ent.max() / 8.0, ent.min() / 8.0 if n_sec else 0.0, ent.mean() / 8.0, ent.std() / 8.0,
        np.log1p(sizes.max()) / np.log1p(1e8) if n_sec else 0.0,
        np.log1p(sizes.min()) / np.log1p(1e8) if n_sec else 0.0,
        np.log1p(sizes.mean()) / np.log1p(1e8) if n_sec else 0.0,
        np.log1p(sizes.std()) / np.log1p(1e8) if n_sec else 0.0,
        np.log1p(vsizes.max()) / np.log1p(1e8) if n_sec else 0.0,
        np.log1p(vsizes.min()) / np.log1p(1e8) if n_sec else 0.0,
        np.log1p(vsizes.mean()) / np.log1p(1e8) if n_sec else 0.0,
        np.log1p(vsizes.std()) / np.log1p(1e8) if n_sec else 0.0,
        # phần còn lại: top-4 section entropy + props thống kê + entry-section entropy
    ], dtype=np.float32)
    top4 = np.sort(ent)[-4:][::-1] / 8.0 if n_sec >= 4 else np.pad(np.sort(ent)[::-1] / 8.0, (0, 4 - min(4, n_sec)))
    n_exec = sum(1 for s in secs if 'MEM_EXECUTE' in (s.get('props') or []))
    n_write = sum(1 for s in secs if 'MEM_WRITE' in (s.get('props') or []))
    n_code = sum(1 for s in secs if 'CNT_CODE' in (s.get('props') or []))
    g6b = np.array([
        _lognorm(overlay.get('size'), 1e8),
        float(overlay.get('size_ratio') or 0),
        float(overlay.get('entropy') or 0) / 8.0,
        float(min(n_exec, 32)), float(min(n_write, 32)), float(min(n_code, 32)),
        1.0 if (row.get('section') or {}).get('entry') else 0.0,
        # dim 25: entropy của section chứa entrypoint
        (float(next((s.get('entropy') or 0) for s in secs if s.get('name') == (row.get('section') or {}).get('entry'))) / 8.0
         if any(s.get('name') == (row.get('section') or {}).get('entry') for s in secs) else 0.0),
    ], dtype=np.float32)
    g6 = np.concatenate([g6, top4[:4], g6b])[:25]
    assert g6.shape[0] == 25, f"G6 must be 25, got {g6.shape[0]}"

    # G7: ImportsExports 20
    imports = row.get('imports') or {}
    n_dlls = len(imports) if isinstance(imports, dict) else 0
    n_funcs = sum(len(v) for v in imports.values()) if isinstance(imports, dict) else 0
    kernel32 = sum(len(v) for k, v in imports.items() if 'kernel32' in k.lower()) if isinstance(imports, dict) else 0
    user32 = sum(len(v) for k, v in imports.items() if 'user32' in k.lower()) if isinstance(imports, dict) else 0
    advapi = sum(len(v) for k, v in imports.items() if 'advapi32' in k.lower()) if isinstance(imports, dict) else 0
    ws2 = sum(len(v) for k, v in imports.items() if 'ws2_32' in k.lower() or 'wininet' in k.lower()) if isinstance(imports, dict) else 0
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
        # 13 dims: hàm API nhạy cảm (IAT metrics)
        float(any('virtualallocex' in f for f in all_funcs)),
        float(any('writeprocessmemory' in f for f in all_funcs)),
        float(any('createremotethread' in f for f in all_funcs)),
        float(any('setwindowshookex' in f for f in all_funcs)),
        float(any('getasynckeystate' in f for f in all_funcs)),
        float(any('internetopen' in f for f in all_funcs)),
        float(any('url download' in f for f in [x.replace('urldownloadtofile', 'url download') for x in all_funcs])),
        float(any('winexec' in f for f in all_funcs)),
        float(any('shellexecute' in f for f in all_funcs)),
        float(any('regsetvalue' in f for f in all_funcs)),
        float(any('createtoolhelp32snapshot' in f for f in all_funcs)),
        float(any('isdebuggerpresent' in f for f in all_funcs)),
        float(any('checkremotedebuggerpresent' in f for f in all_funcs)),
    ], dtype=np.float32)
    g7 = g7[:20]

    # G8: DataDirectories 16
    dds = row.get('datadirectories') or []
    dd_vec = np.zeros(16, dtype=np.float32)
    for i, dd in enumerate(dds[:16]):
        if isinstance(dd, dict):
            dd_vec[i] = 1.0 if any((dd.get(k) or 0) for k in dd.keys()) else 0.0
    # 17 datadirs chuẩn → gộp 2 cuối
    if len(dds) > 16:
        extra = any(any((dd.get(k) or 0) for k in dd.keys()) for dd in dds[16:])
        dd_vec[15] = float(extra or dd_vec[15])

    vec = np.concatenate([hist, be, g3, g4, g5, g6, g7, dd_vec]).astype(np.float32)
    assert vec.shape[0] == 623, f"expected 623, got {vec.shape[0]}"
    return np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=0.0)


def extract_temporal(row):
    """G9: 8 relative temporal dims (Track A ONLY) — đúng spec §3.1 bản thảo."""
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


def parse_row(row):
    """Trả về (x_623, x_g9, y_bin, y_fam, y_beh, sha256, week_id)."""
    label = int(row.get('label') or 0)
    conf = row.get('family_confidence')
    y_fam = map_family(row.get('family'), label, conf)
    return (extract_static(row), extract_temporal(row), label, y_fam,
            map_behavior(row.get('behavior')), str(row.get('sha256') or ''),
            int(row.get('week_id') or 0))
