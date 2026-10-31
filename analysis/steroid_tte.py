# -*- coding: utf-8 -*-
"""Bounded, outcome-blind steroid TTE feasibility check; Python 3.8 / CPU.

One canonical script/report/table. Native rows and read provenance stay private.
The prior withdrawal studies and their caches are read-only dependencies.
"""
import os
for _key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[_key] = '1'
import argparse
import ast
import ctypes
import datetime
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
from hc_paths import configured_path
import re
import sys
import time
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parent
LOCAL = configured_path('legacy_workspace', '../restricted/legacy-workspace')
OLD = configured_path('legacy_source_dir', '../restricted/legacy-workspace/clinical_tte_20261005')
RAW = configured_path('mimic_dir', '../restricted/mimic-iv-3.1')
SOURCE = OLD / 'icu_vp_policy' / 'private' / 'mimic31_extension.pkl.gz'
PRIVATE = configured_path('private_dir', '../restricted/hc-private') / 'steroid_tte.pkl.gz'
REPORT = ROOT / 'steroid_tte.md'
TABLE = ROOT / 'steroid_tte_aggregates.csv'
DEFS = configured_path('sepsis_defs', '../restricted/sepsis3_defs.py')
CONFIG = {
    'data_version': 'MIMIC-IV 3.1 only',
    'index': 'first observed continuous positive ICU NE episode reaching four hours, still active at t0',
    'ne_observed_hours': 4, 'action_window_hours': 6,
    'gap_bridge_seconds': 0, 'one_index_per_hospital_admission': True,
    'baseline_drug_lookback': 'whole observed current hospitalization before t0',
    'antibacterial_proxy_lookback_hours': 24,
    'steroid_action': 'documented systemic hydrocortisone; IV and formulation-only evidence distinguished',
    'primary_clock': 'reported event time; labs additionally require storetime before t0',
    'dose_threshold_for_eligibility': None,
    'outcome_estimation': False, 'models_fitted': 0,
    'delta': 'availability check only; no mandatory eligibility and no classifier retraining',
}
PROTOCOL = '''# 激素启动分层TTE：有限可行性验证

本次只检验一个候选：在ICU已连续记录NE支持四小时、仍有NE需求，且治疗前已有抗菌治疗证据的成人中，能否在共同节点比较启动氢化可的松与暂缓启动，并按该节点前可观察的状态分层。不是恢复ED一小时δ方案，也不是恢复VP追加或首次停药政策。

## 先于新队列给药路径统计保存的规则

- t0为首次连续、正数NE记录达到四小时的时刻，仍需有覆盖t0的正数记录；记录空隙不填补。只取每次医院住院首个合格节点，不能以未来用药、存活或停药成功选人。该节点是四小时**ICU可观察支持**，不是保证真实休克或NE首次发作。
- 四小时是本候选的研究者复评设定，不冒充2026指南硬性阈值。2021的≥0.25 μg/kg/min与持续时间有其特定语义，本次不把任何剂量四小时等同于该条件，也不进行未知盐基换算。
- 先保留所有NE技术候选。院内t0前明确系统性糖皮质激素给药、既往记录不确定、eMAR捕获不足及未检出既往给药分别记录。没有记录不能证明未用药；院外慢性使用仍未知。
- 抗菌药只采用t0前24小时的实际给药证据。抗菌治疗是感染评估的代理，不等于确诊感染、充分复苏或Sepsis-3；外科预防用药与其他休克病因仍须区分。同版培养表目前未定位。
- 固定描述随后六小时的HC启动、其他激素、尚无给药记录、明确NE停止、再启动、死亡/出院/转出与观察不足。这里是路径支持检查，尚不是两组患者数量或最终克隆兼容数。不限制为六小时存活或最终接受治疗者。
- 六小时是文献中常见早期启动窗口所启发的研究者检查窗口，尚未获临床规则认可。不能因行动数更好而改窗口。最终策略是否允许NE改善退出、何时救援、其他激素怎样处理，须由可观察状态与临床理由共同确定。
- 基线信息严格早于t0。检验主口径为charttime与storetime均早于t0；storetime为数据库存储时间，不自动等同医生首次可见。生命体征分开报告测量时钟与入库时钟。没有测量不填正常。
- δ只检查AST、碳酸氢盐与乳酸能否及时取得；不强制完整病例入组，不将未来轨迹作为基线，不据死亡效果选分层。NE剂量历史可描述治疗前状态，不能自动称为已验证治疗反应表型。
- 本阶段不估计死亡效果、不拟合PS/权重、不训练模型。旧停药研究与所有冻结结果保留。若连行动或共同节点都不成立，停止；若仅感染资格等有限缺口，给具体补查，不滚动扩大工程。

## 读取顺序与上限

先复用3.1全成人升压药投影与小表，避免重新扫描inputevents。eMAR 6.25GB、明细8.68GB各读一次，只留候选住院的目标药物及捕获摘要。只有药物路径有支持，才读18.40GB检验和必要的41.94GB chart字段；这些是文本CSV的完整顺序字节读取，列投影不会减少磁盘读取量。患者行仅存本机private缓存，不公开临床原文。
'''
COLS = {
    'emar': ['subject_id', 'hadm_id', 'emar_id', 'emar_seq', 'charttime', 'medication', 'event_txt', 'storetime'],
    'emar_detail': ['subject_id', 'emar_id', 'emar_seq', 'parent_field_ordinal', 'administration_type',
                    'complete_dose_not_given', 'dose_given', 'dose_given_unit', 'product_amount_given',
                    'product_unit', 'product_description', 'infusion_rate', 'infusion_rate_unit', 'route'],
    'labs': ['labevent_id', 'subject_id', 'hadm_id', 'specimen_id', 'itemid', 'charttime', 'storetime', 'valuenum', 'valueuom'],
}
PATHS = {'emar': RAW/'icu'/'emar.csv', 'emar_detail': RAW/'icu'/'emar_detail.csv',
         'labs': RAW/'hosp'/'labevents.csv'/'labevents.csv'}
LAB_MAP = {
    'AST': [50878,53088], 'bicarbonate':[50882], 'lactate':[50813,52442,53154],
    'creatinine':[50912,52546], 'bilirubin':[50885,53089], 'hemoglobin':[51222,51640],
    'platelet':[51265,53189], 'wbc':[51301,51755,51756], 'bun':[51006,52647],
    'pH_bloodgas':[50820], 'pCO2_bloodgas':[50818], 'pO2_bloodgas':[50821],
    'albumin':[50862,53085], 'ALT':[50861], 'INR':[51237,51675], 'CRP':[50889],
    'lymphocytes_pct':[51244,53188], 'neutrophils_pct':[51256,53164],
    'glucose':[50931,52569], 'sodium':[50983,52623], 'potassium':[50971,52610],
}
LAB_HOURS = dict((k,24) for k in LAB_MAP)
LAB_HOURS.update(lactate=6,pH_bloodgas=6,pCO2_bloodgas=6,pO2_bloodgas=6)
CHART_MAP = {
    'MAP_arterial':[220052], 'MAP_noninvasive':[220181],
    'HR':[220045], 'RR':[220210], 'SpO2':[220277],
    'temperature_F':[223761], 'temperature_C':[223762],
    'GCS_eye':[220739], 'GCS_verbal':[223900], 'GCS_motor':[223901],
    'RASS':[228096], 'oxygen_device':[226732], 'ventilator_mode':[223849,229314],
    'FiO2':[223835], 'PEEP':[220339], 'code_status':[223758,228687],
    'intubation_status':[229784],
    'ECMO_flow':[224660,229270], 'Impella_flow':[228154,229671],
    'IABP_evidence':[224322,225984], 'CRRT_mode':[227290],
}
CHART_HOURS = dict((k,4) for k in CHART_MAP)
CHART_HOURS.update(HR=1,RR=1,SpO2=1,MAP_arterial=1,MAP_noninvasive=1,
                   temperature_F=4,temperature_C=4,code_status=720,intubation_status=720)
COLS['chart']=['subject_id','hadm_id','stay_id','itemid','charttime','storetime','value','valuenum','valueuom']
PATHS['chart']=RAW/'icu'/'chartevents.csv'

# An additive qualification check. The original 4h/6h protocol and signature
# remain unchanged; this supplement is saved before reading culture records.
CULTURE_COLS = ['microevent_id','subject_id','hadm_id','micro_specimen_id',
                'chartdate','charttime','spec_itemid','spec_type_desc','test_seq',
                'storedate','storetime','test_itemid','test_name']
CULTURE_PATH = RAW/'hosp'/'microbiologyevents.csv'
CULTURE_CONFIG = {
    'version':'MIMIC-IV 3.1 only', 'candidate_scope':'existing technical cohort; no new t0 or drug mapping',
    'antibacterial_lookback_h':24, 'culture_to_antibacterial_max_h':72,
    'antibacterial_to_culture_max_h':24, 'collection_projection_lookback_h':96,
    'baseline_cutoff':'both actual antibacterial and specimen collection strictly before t0',
    'pair_limits':'inclusive at 72h/24h; t0 itself excluded',
    'specimen_unit':'micro_specimen_id, not susceptibility or organism rows',
    'date_only':'whole-day uncertainty interval; never impute midnight as exact collection',
    'result_requirement':'none; no organism, susceptibility or comments are read',
    'primary_microbiology':'diagnostic bacterial cultures; surveillance and nonculture tests separate',
    'missing_hadm':'unique full-admission subject/time link; otherwise unresolved',
    'no_effect_estimation':True,
}
CULTURE_PROTOCOL = '''本补充只核验既有抗菌药代理候选的基线培养配对，不修改4小时NE节点、6小时行动窗或既有实际抗菌药映射。培养先于抗菌药时允许72小时，抗菌药先于培养时允许24小时；两者必须严格早于t0。既有抗菌药最多回看24小时，因此培养投影最多回看96小时。用诊断性细菌培养而非筛查、病毒检测或培养阳性定义感染评估代理，不能称确诊感染性休克。采集时间依charttime，日期记录保留全天区间；storetime是最后结果更新，不作为采集或首次临床可见时刻。缺记录、未配对及时间不确定分别保留。所有患者继续留在原始技术候选及代理分母内，不按未来治疗或死亡删除。'''


def culture_signature():
    return hashlib.sha256((CULTURE_PROTOCOL+json.dumps(CULTURE_CONFIG,sort_keys=True)).encode('utf-8')).hexdigest()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def say(x):
    print(json.dumps(x, ensure_ascii=False, default=str), flush=True)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for v in iter(lambda: f.read(1024*1024), b''):
            h.update(v)
    return h.hexdigest()


def signature():
    return hashlib.sha256((PROTOCOL+json.dumps(CONFIG, sort_keys=True)).encode('utf-8')).hexdigest()


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sec(x):
    dt = pd.to_datetime(x, errors='coerce')
    return pd.Series(dt.to_numpy(dtype='datetime64[s]').astype('int64'), index=x.index, dtype=float).where(dt.notna())


def save(b):
    PRIVATE.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(b, PRIVATE, compression={'method': 'gzip', 'compresslevel': 1})


def load():
    b = pd.read_pickle(PRIVATE)
    assert b['protocol_signature'] == signature(), 'Frozen feasibility rules changed'
    assert b['effects_estimated'] == 0 and b['models_fitted'] == 0
    for p,h in b['dependencies'].items():
        assert sha(p)==h, 'Read-only dependency changed: '+p
    return b


def count(n):
    n = int(n)
    return '1–10（抑制）' if 0 < n <= 10 else str(10*int(math.floor(n/10.+.5)))


def pct(n, d):
    return '' if not d or 0 < n <= 10 or 0 < d-n <= 10 else round(100.*n/d, 1)


def row(b, section, metric, n=None, d=None, value='', note='', stratum='all'):
    b['aggregates'].append(dict(section=section, metric=metric, stratum=stratum,
                                exact_n=n, exact_d=d, value=value, note=note))


def peak_mb():
    class M(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong)] + [
            (n, ctypes.c_size_t) for n in ['PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
            'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage']]
    m = M(); m.cb = ctypes.sizeof(m)
    k = ctypes.windll.kernel32; k.GetCurrentProcess.restype = ctypes.c_void_p
    p = ctypes.windll.psapi; p.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    return round(m.PeakWorkingSetSize/1048576., 1) if p.GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(m), m.cb) else None


class HashReader(io.RawIOBase):
    def __init__(self, path):
        self.f = open(path, 'rb'); self.h = hashlib.sha256(); self.n = 0
    def readable(self):
        return True
    def readinto(self, buf):
        n = self.f.readinto(buf)
        if n:
            self.h.update(memoryview(buf)[:n]); self.n += n
        return n
    def close(self):
        self.f.close(); super().close()


def first_four_hours(g, lo, hi):
    intervals = sorted((max(float(r.start), lo), min(float(r.end), hi)) for r in g.itertuples()
                       if r.end > lo and r.start < hi and np.isfinite(r.rate) and r.rate > 0)
    merged = []
    for s, e in intervals:
        if e <= s:
            continue
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(e, merged[-1][1])
        else:
            merged.append([s, e])
    for s, e in merged:
        t = s + CONFIG['ne_observed_hours']*3600
        if t < e:  # still supported; ending exactly at four hours is not active
            return s, t
    return None


def tests():
    def g(rows):
        return pd.DataFrame(rows, columns=['start', 'end', 'rate'])
    assert first_four_hours(g([[0, 18000, .1]]), 0, 20000) == (0, 14400)
    assert first_four_hours(g([[0, 14400, .1]]), 0, 20000) is None
    assert first_four_hours(g([[0, 7200, .1], [7200, 18000, .2]]), 0, 20000) == (0, 14400)
    assert first_four_hours(g([[0, 7200, .1], [7201, 18000, .2]]), 0, 20000) is None
    assert first_four_hours(g([[-3600, 18000, .1]]), 0, 20000) == (0, 14400)
    assert first_four_hours(g([[0, 7200, .1], [8000, 28000, .2]]), 0, 30000) == (8000, 22400)
    return ['ongoing_at_t0', 'no_future_survival_filter', 'no_gap_bridge', 'contiguous_dose_change', 'ICU_clip', 'first_eligible_episode']


def prepare():
    if PRIVATE.exists():
        say({'stage': 'prepare', 'reuse': True}); return
    assert sys.version_info[:2] == (3, 8), sys.executable
    begin = time.perf_counter()
    gate = import_file('reused_clinical_gate', OLD/'clinical_tte_gate.py')
    src = pd.read_pickle(SOURCE)
    meta = src['adult_stays'].copy()
    adm = src['admissions'][['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime']].copy()
    adm['admit'] = sec(adm.admittime); adm['discharge'] = sec(adm.dischtime); adm['death_time'] = sec(adm.deathtime)
    meta = meta.merge(adm.drop(columns='subject_id'), on='hadm_id', validate='many_to_one')
    raw = src['events']; ne = gate.normalize(raw[raw.itemid.eq(221906)], src['icustays'])
    refs = meta.set_index('stay_id'); candidates = []; reasons = dict(no_four_hours=0, invalid_hospital_time=0)
    for stay, g in ne.groupby('stay_id', sort=False):
        if stay not in refs.index:
            continue
        m = refs.loc[stay]
        pair = first_four_hours(g, max(m.icu_in, m.admit), min(m.icu_out, m.discharge))
        if pair is None:
            reasons['no_four_hours'] += 1; continue
        start, t0 = pair
        if not (m.admit <= t0 < m.discharge and m.icu_in <= t0 < m.icu_out) or (np.isfinite(m.death_time) and m.death_time <= t0):
            reasons['invalid_hospital_time'] += 1; continue
        state = gate.State(g)
        dose, flag, active = state.at(t0)
        path = gate.dose_path(state, t0, hours=4)
        stops = sorted(set(g.loc[g.statusdescription.astype(str).eq('Stopped') & g.end.gt(t0) &
                                 g.end.le(t0+21600), 'end']))
        stops = [v for v in stops if state.at(v)[1] == 'Stopped']
        stop = stops[0] if stops else np.nan
        restart = float(g.loc[g.start.gt(stop) & g.start.le(t0+21600), 'start'].min()) if np.isfinite(stop) else np.nan
        candidates.append(dict(subject_id=int(m.subject_id), hadm_id=int(m.hadm_id), stay_id=int(stay),
            t0=t0, ne_start=start, age=m.age, gender=m.gender, anchor_year_group=m.anchor_year_group,
            first_careunit=m.first_careunit, icu_in=m.icu_in, icu_out=m.icu_out, admit=m.admit,
            discharge=m.discharge, death_time=m.death_time, ne_dose=dose, ne_dose_state=flag,
            ne_recorded_by_t0=bool(g.loc[active, 'store'].le(t0).any()) if active.any() else False,
            response_pattern=path['pattern'], relative_ne_change=path['relative_change'],
            ne_known_min=path['known_min'], first_recorded_ne_stop_6h=stop, recorded_ne_restart_6h=restart))
    q = pd.DataFrame(candidates).sort_values(['t0', 'stay_id']).drop_duplicates('hadm_id', keep='first').reset_index(drop=True)
    assert q.hadm_id.is_unique and q.t0.gt(q.ne_start).all()
    services = RAW/'hosp'/'services.csv'/'services.csv'
    if services.exists():
        ss = pd.read_csv(services, usecols=['hadm_id', 'transfertime', 'curr_service'])
        ss = ss[ss.hadm_id.isin(q.hadm_id)].copy(); ss['at'] = sec(ss.transfertime)
        ss['t0'] = ss.hadm_id.map(q.set_index('hadm_id').t0)
        sv = ss[ss['at'].lt(ss.t0)].sort_values('at').drop_duplicates('hadm_id', keep='last').set_index('hadm_id').curr_service
        q['service_before_t0'] = q.hadm_id.map(sv).fillna('UNKNOWN')
    else:
        q['service_before_t0'] = 'UNKNOWN'
    defs = {}
    for node in ast.parse(DEFS.read_text(encoding='utf-8-sig')).body:
        if isinstance(node, ast.Assign):
            for name in node.targets:
                if isinstance(name, ast.Name) and name.id in ['ANTIBIOTICS', 'ANTIBIOTIC_EXCLUDE_NAME']:
                    defs[name.id] = ast.literal_eval(node.value)
    plans = {k: dict(path=str(p), bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns, columns=COLS[k]) for k,p in PATHS.items()}
    b = dict(protocol_text=PROTOCOL, config=CONFIG, protocol_signature=signature(), frozen_utc=utc(),
             candidates=q, plans=plans, definitions=defs, aggregates=[], runs=[], effects_estimated=0,
             models_fitted=0, source_cache_sha256=sha(SOURCE), source_manifest=src['manifest'],
             dependencies={str(p): sha(p) for p in [OLD/'clinical_tte_gate.py', OLD/'withdrawal_association'/'supplement_analysis.py', DEFS]},
             tests=tests(), source_version='user-provided MIMIC-IV 3.1, native-key verified; no cross-version linkage')
    row(b, 'flow', '有NE记录的成人ICU住院', ne.stay_id.nunique(), note='ICU住院，不是医院住院或合格sepsis队列')
    row(b, 'flow', '有NE记录的医院住院', ne.hadm_id.nunique())
    row(b, 'flow', '连续NE满四小时且起点时仍有记录支持：每医院住院首个', len(q), ne.hadm_id.nunique())
    row(b, 'flow', '未达到严格连续四小时的ICU住院', reasons['no_four_hours'], note='未桥接记录空隙；不等同真实NE使用不足四小时')
    for x,g in q.groupby('anchor_year_group'):
        row(b, 'calendar', '四小时技术候选', len(g), len(q), stratum=str(x))
    for x,g in q.groupby('response_pattern'):
        row(b, 'NE_history', '治疗前四小时剂量记录模式', len(g), len(q), stratum=str(x), note='不是已验证的治疗反应分型')
    b['runs'].append(dict(stage='prepare', utc=utc(), elapsed_s=round(time.perf_counter()-begin,2), peak_MB=peak_mb(), raw_large_scans=0))
    save(b); report()
    say({'stage': 'prepare', 'hospital_candidates_rounded': count(len(q)), 'baseline_rules_saved': True,
         'raw_large_scans': 0, 'seconds': b['runs'][-1]['elapsed_s'], 'peak_MB': peak_mb()})


def scan(name):
    b = load()
    if name in b:
        say({'stage': name, 'reuse': True}); return
    assert name in ['emar', 'emar_detail']
    path = PATHS[name]; plan = b['plans'][name]
    assert path.stat().st_size == plan['bytes'] and path.stat().st_mtime_ns == plan['mtime_ns']
    q = b['candidates']; refs = q.set_index('hadm_id'); adms = set(q.hadm_id); subs = set(q.subject_id)
    begin = time.perf_counter(); last = begin; chunks = []; total = 0; conflicts = 0; missing_unlinked = 0
    if name == 'emar':
        cap = pd.DataFrame(0, index=refs.index, columns=['pre24_records', 'pre4_records', 'post6_records', 'pre24_mask', 'post6_mask'])
        abx = re.compile('|'.join(map(re.escape, b['definitions']['ANTIBIOTICS'])), re.I)
        excluded = re.compile(b['definitions']['ANTIBIOTIC_EXCLUDE_NAME'], re.I)
        supp = import_file('reused_steroid_parser', OLD/'withdrawal_association'/'supplement_analysis.py')
        steroid = re.compile(supp.STEROID_PATTERN, re.I)
    else:
        assert 'emar' in b
        ids = set(b['emar'].emar_id)
    f = HashReader(path); stream = io.BufferedReader(f, buffer_size=4*1024*1024)
    try:
        for raw in pd.read_csv(stream, usecols=COLS[name], dtype=str, chunksize=150000):
            total += len(raw)
            if name == 'emar_detail':
                z = raw[raw.emar_id.isin(ids)].copy()
            else:
                h = pd.to_numeric(raw.hadm_id, errors='coerce'); s = pd.to_numeric(raw.subject_id, errors='coerce')
                z = raw[h.isin(adms) | (h.isna() & s.isin(subs))].copy()
                if len(z):
                    z['hadm_id'] = h.loc[z.index]; z['subject_id'] = s.loc[z.index]
                    z['chart'] = sec(z.charttime); z['store'] = sec(z.storetime)
                    z['join_method'] = 'native_hadm'
                    for ix,r in z[z.hadm_id.isna()].iterrows():
                        poss = q[q.subject_id.eq(r.subject_id) & q.admit.le(r.chart) & q.t0.add(21600).ge(r.chart)]
                        if len(poss) == 1:
                            z.at[ix, 'hadm_id'] = poss.hadm_id.iloc[0]; z.at[ix, 'join_method'] = 'unique_subject_admission_time'
                        else:
                            missing_unlinked += 1
                    z = z[z.hadm_id.isin(adms)].copy()
                    good = z.subject_id.eq(z.hadm_id.map(refs.subject_id)); conflicts += int((~good).sum()); z = z[good]
                    z = z[z.chart.ge(z.hadm_id.map(refs.admit)) & z.chart.le(z.hadm_id.map(refs.t0)+21600)].copy()
                    z['hadm_id'] = z.hadm_id.astype('int64'); z['subject_id'] = z.subject_id.astype('int64')
                    delta = z.chart-z.hadm_id.map(refs.t0)
                    for label,mask in [('pre24_records', delta.ge(-86400)&delta.lt(0)), ('pre4_records', delta.ge(-14400)&delta.lt(0)),
                                       ('post6_records', delta.ge(0)&delta.le(21600))]:
                        co = z.loc[mask].groupby('hadm_id').size()
                        cap.loc[co.index, label] += co
                    for label,mask,origin in [('pre24_mask',delta.ge(-86400)&delta.lt(0),24), ('post6_mask',delta.ge(0)&delta.lt(21600),0)]:
                        zz = z.loc[mask, ['hadm_id']].copy(); zz['bin'] = (np.floor(delta.loc[mask]/3600)+origin).astype(int)
                        zz = zz.drop_duplicates()
                        for h,binid in zz.itertuples(index=False, name=None):
                            cap.at[h, label] = int(cap.at[h,label]) | (1 << int(binid))
                    names = z.medication.fillna('')
                    z['is_steroid_name'] = names.str.contains(steroid)
                    z['is_abx_name'] = names.str.contains(abx) & ~names.str.contains(excluded) & delta.loc[z.index].ge(-86400) & delta.loc[z.index].lt(0)
                    z = z[z.is_steroid_name | z.is_abx_name]
            if len(z):
                chunks.append(z)
            now = time.perf_counter()
            if now-last >= 30:
                say({'stage': name, 'percent_bytes': round(100.*f.n/plan['bytes'],1), 'seconds':round(now-begin), 'peak_MB':peak_mb()}); last=now
        assert f.n == plan['bytes']
        digest = f.h.hexdigest()
    finally:
        stream.close()
    assert path.stat().st_size == plan['bytes'] and path.stat().st_mtime_ns == plan['mtime_ns']
    assert conflicts == 0, 'Native patient/admission conflict'
    d = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=COLS[name])
    before=len(d); d=d.drop_duplicates().reset_index(drop=True); b[name]=d
    if name == 'emar':
        b['capture'] = cap.reset_index()
    b['plans'][name].update(sha256=digest, rows_traversed=total, rows_retained=len(d), exact_duplicates_removed=before-len(d), parsed_to_EOF=True)
    b['runs'].append(dict(stage=name, utc=utc(), seconds=round(time.perf_counter()-begin,2), bytes_read=plan['bytes'],
                          peak_MB=peak_mb(), source_key_conflicts=conflicts, missing_hadm_unlinked=missing_unlinked))
    save(b)
    say({'stage': name, 'completed':True, 'retained_rows_rounded':count(len(d)), 'seconds':b['runs'][-1]['seconds'], 'peak_MB':peak_mb()})


def resolve_abx(e, g, supp, b):
    status=str(e['event_txt']).strip().lower()
    result=dict(status='UNCERTAIN', route='UNKNOWN', dose_mg=np.nan, reason='no_actual_detail')
    if status in supp.NOT_GIVEN:
        return dict(result, status='NOT_ADMIN', reason='explicit_not_given')
    if g is None or not len(g):
        return result
    g=g.fillna('').drop_duplicates()
    if g.duplicated('parent_field_ordinal', keep=False).any():
        return dict(result, reason='conflicting_detail_ordinal')
    routes=set(g.route.str.strip().str.upper())-{'','NAN'}
    result['route']='|'.join(sorted(routes)) if routes else 'UNKNOWN'
    if routes and routes <= supp.LOCAL_ROUTES:
        return dict(result, status='LOCAL_ONLY', reason='local_route')
    if routes and not routes <= supp.SYSTEMIC:
        return dict(result, reason='route_mixed_or_unknown')
    if status not in supp.GIVEN and status != 'confirmed administered':
        return dict(result, reason='execution_not_confirmed')
    child=g[g.parent_field_ordinal.ne('')].copy()
    if not len(child):
        return result
    positive=child.dose_given.map(supp.number).gt(0)
    rate=status in ['started','delayed started','restarted'] and child.infusion_rate.map(supp.number).gt(0).any()
    if not positive.any() and not rate:
        return dict(result, reason='no_positive_actual_amount')
    prod=child.product_description.str.cat(sep=' | ')
    if re.search(b['definitions']['ANTIBIOTIC_EXCLUDE_NAME'],prod,re.I):
        return dict(result, status='LOCAL_ONLY', reason='local_product')
    formulation=bool(re.search(r'\b(?:vial|syringe|bag|tablet|capsule|tab|cap)\b|\b(?:mg|gm|g)\s+(?:in|/)',prod,re.I))
    if not routes and not formulation:
        return dict(result, reason='route_and_formulation_unknown')
    return dict(result, status='ADMIN_DOCUMENTED' if routes else 'ADMIN_SYSTEMIC_FORMULATION_ROUTE_UNKNOWN', reason='actual_status_positive_detail')


def named_injectable_salt(name):
    return bool(re.search(r'hydrocortisone\s+(?:na|sod(?:ium)?)\.?\s+succ(?:inate)?|methylprednisolone\s+(?:na|sod(?:ium)?)\.?\s+succ(?:inate)?',str(name),re.I))


def resolve_gc(e,g,supp,allow_named_salt=False):
    result=supp.resolve_event(e,g)
    if not allow_named_salt or not named_injectable_salt(e['medication']):
        return result
    if result['status']!='UNCERTAIN' or result['reason']!='route_missing_no_systemic_formulation':
        return result
    if str(e['event_txt']).strip().lower() not in supp.GIVEN or g is None:
        return result
    gg=g.fillna('').drop_duplicates()
    if gg.duplicated('parent_field_ordinal',keep=False).any():
        return result
    routes=set(gg.route.str.strip().str.upper())-{'','NAN'}
    if routes:
        return result  # this revision resolves missing route, never overrides a recorded route
    child=gg[gg.parent_field_ordinal.ne('')].copy()
    compatible_product=child.product_description.isin(['','___'])|child.product_description.str.contains(supp.STEROID_PATTERN,case=False,regex=True,na=False)
    amount=child.dose_given.map(supp.number)
    unit=child.dose_given_unit.str.strip().str.lower()
    mg=child[compatible_product&amount.gt(0)&unit.eq('mg')].copy()
    if not len(mg):
        return result
    result.update(status='ADMIN_SYSTEMIC_FORMULATION_ROUTE_UNKNOWN',
        reason='actual_status_named_injectable_salt_positive_mg_route_unknown',route='UNKNOWN',dose_unit='mg',
        dose_mg=float(mg.dose_given.map(supp.number).sum()),
        partial=bool(gg.complete_dose_not_given.str.lower().eq('yes').any() or 'partial' in str(e['event_txt']).lower()))
    return result


def medication_tests():
    supp=import_file('reused_parser_tests',OLD/'withdrawal_association'/'supplement_analysis.py')
    cols=COLS['emar_detail']
    parent={k:'' for k in cols}; child=dict(parent,parent_field_ordinal='1.1',dose_given='100',dose_given_unit='mg')
    ev=dict(medication='Hydrocortisone Na Succ.',event_txt='Administered')
    g=pd.DataFrame([parent,child])
    out=resolve_gc(ev,g,supp,True)
    assert out['status']=='ADMIN_SYSTEMIC_FORMULATION_ROUTE_UNKNOWN' and out['route']=='UNKNOWN' and out['dose_mg']==100
    assert resolve_gc(dict(ev,medication='Hydrocortisone'),g,supp,True)['status']=='UNCERTAIN'
    assert resolve_gc(dict(ev,event_txt='Not Given'),g,supp,True)['status']=='NOT_ADMIN'
    assert resolve_gc(dict(ev,medication='Hydrocortisone study drug'),g,supp,True)['status']=='UNCERTAIN'
    assert resolve_gc(ev,pd.DataFrame([parent,dict(child,dose_given='0')]),supp,True)['status']=='UNCERTAIN'
    assert resolve_gc(ev,pd.DataFrame([parent,dict(child,dose_given_unit='mL')]),supp,True)['status']=='UNCERTAIN'
    assert resolve_gc(ev,pd.DataFrame([parent,dict(child,product_description='Sodium Chloride')]),supp,True)['status']=='UNCERTAIN'
    assert resolve_gc(ev,pd.DataFrame([parent,child,dict(child,dose_given='50')]),supp,True)['status']=='UNCERTAIN'
    assert resolve_gc(ev,pd.DataFrame([dict(parent,route='TOPICAL'),child]),supp,True)['status']=='LOCAL_ONLY'
    assert not named_injectable_salt('Hydrocortisone 5mg tablet')
    return ['specific_salt_positive_actual_mg','route_remains_unknown','generic_not_upgraded','not_given_not_exposure',
            'study_drug_not_upgraded','zero_not_exposure','mL_not_mg','diluent_not_drug','ordinal_conflict_unknown','local_route_not_overridden']


def revise_med_mapping():
    b=load()
    if 'named_salt_revision' in b:
        say({'stage':'revise_med_mapping','reuse':True}); return
    assert 'medication_features' in b and 'clinical_features' not in b
    b['named_salt_revision']=dict(utc=utc(),before_any_outcome_model=True,
        reason='Local eMAR explicitly names injectable sodium succinate and confirms administration with positive mg child detail; route and product description are often blank. Missing product text must not erase independent named-drug evidence. Specific IV/IM route stays UNKNOWN.',
        unchanged=['population','four_hour_index','six_hour_window','outcomes','treatment_strategy'],
        prior_aggregates=b['aggregates'].copy(),
        prior_medication_features=b.pop('medication_features'),prior_resolved_medications=b.pop('resolved_medications'))
    save(b)
    say({'stage':'revise_med_mapping','reason_saved_before_reclassification':True,'new_raw_scans':0,'outcome_models':0})


def med_features():
    b=load(); assert 'emar_detail' in b
    if 'medication_features' in b:
        say({'stage':'med_features','reuse':True}); return
    begin=time.perf_counter()
    if 'anchor_year' not in b['candidates']:
        meta=pd.read_pickle(SOURCE)['patients'][['subject_id','anchor_year']]
        cq=b['candidates'].merge(meta,on='subject_id',validate='many_to_one')
        shift=pd.to_datetime(cq.t0,unit='s').dt.year-cq.anchor_year
        bounds=cq.anchor_year_group.str.extract(r'(\d{4})\s*-\s*(\d{4})').astype(int)
        cq['calendar_year_lo']=bounds[0]+shift; cq['calendar_year_hi']=bounds[1]+shift
        cq['calendar_era']=np.where(cq.calendar_year_hi.lt(2016),'entire_interval_before_2016',
            np.where(cq.calendar_year_lo.ge(2017),'entire_interval_2017_or_later','interval_overlaps_2016_boundary'))
        b['candidates']=cq
        for rr in b['aggregates']:
            if rr['section']=='calendar':
                rr['note']='患者anchor年组，不能直接当成该次住院年份；参见经t0年份差校正的calendar_coverage'
    supp=import_file('reused_steroid_parser',OLD/'withdrawal_association'/'supplement_analysis.py')
    e=b['emar']; d=b['emar_detail']
    headers=e.drop_duplicates('emar_id').set_index('emar_id')
    assert d.subject_id.eq(d.emar_id.map(headers.subject_id).astype(str)).all(), 'Detail native subject mismatch'
    assert pd.to_numeric(d.emar_seq).eq(pd.to_numeric(d.emar_id.map(headers.emar_seq))).all()
    conflicts=set(e.loc[e.duplicated('emar_id',keep=False),'emar_id'])
    groups={k:g for k,g in d.groupby('emar_id',sort=False)}
    rows=[]
    feature_last=time.perf_counter()
    for ev_index,ev in enumerate(e.drop_duplicates('emar_id').to_dict('records')):
        detail=groups.get(ev['emar_id'])
        if ev['is_steroid_name']:
            adapted=dict(ev)
            if str(ev['event_txt']).strip().lower()=='confirmed administered':
                adapted['event_txt']='Administered'
            resolved=resolve_gc(adapted,detail,supp,allow_named_salt='named_salt_revision' in b)
        else:
            resolved=resolve_abx(ev,detail,supp,b)
        if ev['emar_id'] in conflicts:
            resolved.update(status='UNCERTAIN',reason='conflicting_header')
        prod='' if detail is None else detail.product_description.fillna('').str.cat(sep=' | ')
        hydro=bool(re.search(r'hydrocortis|solu.?cort|cortef',str(ev['medication']),re.I))
        injectable=bool(re.search(r'hydrocortisone\s+(?:na|sodium)\s+succ',prod,re.I) and re.search(r'vial|syringe|\bmg\s+in',prod,re.I))
        record={k:ev[k] for k in ['subject_id','hadm_id','emar_id','chart','store','is_steroid_name','is_abx_name']}
        record.update(resolved); record.update(hydrocortisone=hydro, injectable_hc_formulation=injectable,
            header_named_injectable_salt=named_injectable_salt(ev['medication']),
            glucocorticoid=bool(ev['is_steroid_name'] and supp.drug_class(ev['medication'])=='glucocorticoid'))
        rows.append(record)
        if time.perf_counter()-feature_last>=30:
            say({'stage':'resolve_medication_events','events_processed_rounded':count(ev_index+1),'percent':round(100.*(ev_index+1)/len(e),1),'seconds':round(time.perf_counter()-begin),'peak_MB':peak_mb()})
            feature_last=time.perf_counter()
    r=pd.DataFrame(rows)
    r['actual_systemic']=r.status.isin(['ADMIN_DOCUMENTED','ADMIN_SYSTEMIC_FORMULATION_ROUTE_UNKNOWN'])
    iv={'IV','IVP','IV PUSH','IV DRIP','IVPB','INTRAVENOUS'}
    r['IV_confirmed']=r.route.map(lambda x: bool(set(str(x).split('|'))<=iv)) & r.actual_systemic
    injection_evidence=r.injectable_hc_formulation|(r.header_named_injectable_salt if 'named_salt_revision' in b else False)
    r['hc_action']=r.actual_systemic & r.hydrocortisone & (r.IV_confirmed | (r.route.eq('UNKNOWN') & injection_evidence))
    cap=b['capture'].set_index('hadm_id'); rg={h:g for h,g in r.groupby('hadm_id',sort=False)}
    features=[]
    for c in b['candidates'].itertuples():
        g=rg.get(c.hadm_id,r.iloc[:0]); pre=g[g.chart.lt(c.t0)]; post=g[g.chart.ge(c.t0)&g.chart.le(c.t0+21600)]
        gc=pre[pre.glucocorticoid]
        prior_actual=gc[gc.actual_systemic]; prior_uncertain=gc[gc.status.eq('UNCERTAIN')]
        if len(prior_actual): state='PRIOR_SYSTEMIC_GC_ADMIN'
        elif len(prior_uncertain): state='PRIOR_GC_RECORD_UNCERTAIN'
        elif cap.loc[c.hadm_id,'pre24_records']==0: state='BASELINE_CAPTURE_INSUFFICIENT'
        else: state='NO_PRIOR_GC_IDENTIFIED_CAPTURE_PRESENT'
        ab=pre[pre.is_abx_name & pre.actual_systemic & pre.chart.ge(c.t0-86400)]
        hc=post[post.hc_action].sort_values('chart')
        other=post[post.glucocorticoid & post.actual_systemic & ~post.hc_action].sort_values('chart')
        uncertain=post[post.glucocorticoid & post.status.eq('UNCERTAIN')]
        until=min(c.t0+21600,c.discharge,c.death_time if np.isfinite(c.death_time) else np.inf)
        postmortem_hc=bool(len(hc) and hc.chart.iloc[0]>until)
        hc=hc[hc.chart.le(until)]; other=other[other.chart.le(until)]
        first_hc=float(hc.chart.iloc[0]) if len(hc) else np.nan
        first_other=float(other.chart.iloc[0]) if len(other) else np.nan
        if np.isfinite(first_hc) and (not np.isfinite(first_other) or first_hc<first_other): action='HC_DOCUMENTED_WITHIN_6H'
        elif np.isfinite(first_other): action='OTHER_GC_DOCUMENTED_WITHIN_6H'
        elif len(uncertain): action='GC_RECORD_UNCERTAIN_WITHIN_6H'
        elif cap.loc[c.hadm_id,'post6_records']==0: action='FOLLOWUP_EMAR_CAPTURE_INSUFFICIENT'
        else: action='NO_GC_ADMIN_IDENTIFIED_CAPTURE_PRESENT'
        features.append(dict(hadm_id=c.hadm_id, baseline_gc_state=state,
            prior_systemic_gc=bool(len(prior_actual)), prior_gc_uncertain=bool(len(prior_uncertain)),
            baseline_abx_actual=bool(len(ab)), baseline_abx_IV=bool(ab.IV_confirmed.any()),
            baseline_abx_strict=bool((ab.store.lt(c.t0)&ab.store.notna()).any()),
            candidate_proxy=bool(state=='NO_PRIOR_GC_IDENTIFIED_CAPTURE_PRESENT' and len(ab)),
            action_state=action, first_hc=first_hc, first_other_gc=first_other,
            hc_first_IV=bool(hc.IV_confirmed.iloc[0]) if len(hc) else False,
            hc_time_tie=bool(len(hc) and first_hc==c.t0),
            hc_documentation_lag_min=float((hc.store.iloc[0]-first_hc)/60) if len(hc) else np.nan,
            hc_after_hospital_observation=postmortem_hc,
            hospital_observation_6h=until-c.t0,
            early_death=bool(np.isfinite(c.death_time) and c.t0<c.death_time<=c.t0+21600),
            early_discharge=bool(c.t0<c.discharge<=c.t0+21600), early_icu_exit=bool(c.t0<c.icu_out<=c.t0+21600)))
    q=b['candidates'].merge(pd.DataFrame(features),on='hadm_id',validate='one_to_one').merge(b['capture'],on='hadm_id',validate='one_to_one')
    b['resolved_medications']=r; b['medication_features']=q
    b['aggregates']=[x for x in b['aggregates'] if x['section'] not in ['capture','drug_eligibility','actions','calendar_actions','calendar_coverage','response_actions','early_events']]
    for x,g in q.groupby('baseline_gc_state'):
        row(b,'drug_eligibility','院内既往激素与捕获状态',len(g),len(q),stratum=x)
    for name,mask in [('t0前24h有任意eMAR',q.pre24_records.gt(0)),('t0前4h有任意eMAR',q.pre4_records.gt(0)),
                      ('t0前实际系统性抗菌药',q.baseline_abx_actual),('t0前明确IV抗菌药',q.baseline_abx_IV),
                      ('t0前抗菌药且已入库',q.baseline_abx_strict)]:
        row(b,'capture',name,int(mask.sum()),len(q),note='任意eMAR记录仅是捕获机会，不证明完整性；抗菌药不等于Sepsis-3')
    p=q[q.candidate_proxy]
    row(b,'drug_eligibility','无检出院内既往GC且有t0前实际抗菌药的代理候选',len(p),len(q),note='尚非临床核验感染性休克人群')
    for x,g in p.groupby('action_state'):
        row(b,'actions','随后6h药物路径',len(g),len(p),stratum=x,note='六小时内死亡/出院未按未来状态排除；没有记录不等于没有治疗')
    for period,g in p.groupby('calendar_era'):
        for action,z in g.groupby('action_state'):
            row(b,'calendar_actions','随后6h药物路径',len(z),len(g),stratum=str(period)+' / '+action)
    for period,g in q.groupby('calendar_era'):
        row(b,'calendar_coverage','t0前24h任意eMAR',int(g.pre24_records.gt(0).sum()),len(g),stratum=period,
            note='anchor_year_group界限加(t0移位年-anchor_year)，得到就诊年区间；不把患者anchor组直接当就诊年份')
    for pattern,g in p.groupby('response_pattern'):
        for action,z in g.groupby('action_state'):
            row(b,'response_actions','随后6h药物路径',len(z),len(g),stratum=str(pattern)+' / '+action)
    hc=p[p.action_state.eq('HC_DOCUMENTED_WITHIN_6H')]
    row(b,'actions','HC启动有明确IV途径',int(hc.hc_first_IV.sum()),len(hc),note='余者为注射制剂、具体途径未知，不自动当成IV')
    row(b,'actions','HC与t0精确同刻',int(hc.hc_time_tie.sum()),len(hc),note='无基于同刻记录顺序的因果推断')
    for name,mask in [('六小时内死亡',p.early_death),('六小时内医院出院',p.early_discharge),('六小时内ICU转出',p.early_icu_exit),
                      ('六小时内明确NE停止记录',p.first_recorded_ne_stop_6h.notna()),('停止后NE重新启动记录',p.recorded_ne_restart_6h.notna())]:
        row(b,'early_events',name,int(mask.sum()),len(p),note='只检查风险集结构；非分组死亡效果或临床恢复标签')
    b['runs'].append(dict(stage='med_features',utc=utc(),seconds=round(time.perf_counter()-begin,2),peak_MB=peak_mb()))
    save(b); report()
    say({'stage':'med_features','proxy_candidates_rounded':count(len(p)), 'HC_path_rounded':count(len(hc)),
         'path_states':{k:count(len(g)) for k,g in p.groupby('action_state')},'models':0,'effects':0,'seconds':b['runs'][-1]['seconds']})


def scan_cultures():
    """One projected CSV pass; never read organisms, susceptibilities or text."""
    b=load()
    if 'cultures_raw' in b:
        assert b['culture_supplement']['signature']==culture_signature()
        st=CULTURE_PATH.stat(); old=b['plans']['cultures']
        assert st.st_size==old['bytes'] and st.st_mtime_ns==old['mtime_ns'], 'Culture source changed; do not silently reuse or overwrite projection'
        say({'stage':'cultures','reuse':True}); return
    assert CULTURE_PATH.is_file()
    header=pd.read_csv(CULTURE_PATH,nrows=0).columns
    assert set(CULTURE_COLS)<=set(header)
    q=b['candidates']; refs=q.set_index('hadm_id'); subs=set(q.subject_id); adms=set(q.hadm_id)
    if 'culture_supplement' not in b:
        b['culture_supplement']=dict(utc=utc(),signature=culture_signature(),config=CULTURE_CONFIG,
            protocol=CULTURE_PROTOCOL,official_source='https://mimic.mit.edu/docs/iv/modules/hosp/microbiologyevents.html')
        b['culture_baseline_snapshot']=dict(utc=utc(),decision=b.get('decision'),decision_text=b.get('decision_text'),
            aggregates=list(b['aggregates']),report_text=REPORT.read_text(encoding='utf-8'),
            table_text=TABLE.read_text(encoding='utf-8-sig'))
        st=CULTURE_PATH.stat()
        b['plans']['cultures']=dict(path=str(CULTURE_PATH),bytes=st.st_size,mtime_ns=st.st_mtime_ns,columns=CULTURE_COLS)
        save(b)  # the additional rules precede new sample/action counts
    assert b['culture_supplement']['signature']==culture_signature()
    plan=b['plans']['cultures']; st=CULTURE_PATH.stat()
    assert st.st_size==plan['bytes'] and st.st_mtime_ns==plan['mtime_ns']
    begin=time.perf_counter(); last=begin; pieces=[]; total=0; conflicts=0
    missing_links={}; unmatched=set(); interval_ambiguous=set(); alladm=None
    qsubs={s:g for s,g in q.groupby('subject_id',sort=False)}

    def admission_intervals():
        # The small same-version administrative table prevents a candidate-only
        # join from falsely declaring uniqueness in patients with other stays.
        path=RAW/'hosp'/'admissions.csv'/'admissions.csv'; st=path.stat(); started=time.perf_counter()
        cols=['subject_id','hadm_id','admittime','dischtime']; parts=[]; f=HashReader(path)
        stream=io.BufferedReader(f,buffer_size=4*1024*1024)
        try:
            for z in pd.read_csv(stream,usecols=cols,chunksize=150000):
                z=z[z.subject_id.isin(subs)].copy()
                if len(z): parts.append(z)
            assert f.n==st.st_size
            digest=f.h.hexdigest()
        finally: stream.close()
        assert path.stat().st_mtime_ns==st.st_mtime_ns
        zz=pd.concat(parts,ignore_index=True); zz['admit']=sec(zz.admittime); zz['discharge']=sec(zz.dischtime)
        b['plans']['culture_admission_link']=dict(path=str(path),bytes=st.st_size,mtime_ns=st.st_mtime_ns,
            columns=cols,sha256=digest,parsed_to_EOF=True)
        b['runs'].append(dict(stage='culture_admission_link',utc=utc(),seconds=round(time.perf_counter()-started,2),
            bytes_read=st.st_size,peak_MB=peak_mb()))
        return {s:g for s,g in zz.groupby('subject_id',sort=False)}

    f=HashReader(CULTURE_PATH); stream=io.BufferedReader(f,buffer_size=4*1024*1024)
    try:
        for raw in pd.read_csv(stream,usecols=CULTURE_COLS,dtype=str,chunksize=150000):
            total+=len(raw)
            h=pd.to_numeric(raw.hadm_id,errors='coerce'); s=pd.to_numeric(raw.subject_id,errors='coerce')
            z=raw[h.isin(adms)|(h.isna()&s.isin(subs))].copy()
            if not len(z): continue
            z['hadm_id']=h.loc[z.index]; z['subject_id']=s.loc[z.index]
            z['collection']=sec(z.charttime); z['collection_day']=sec(z.chartdate)
            z['collection_lo']=z.collection.fillna(z.collection_day)
            z['collection_hi']=z.collection.where(z.collection.notna(),z.collection_day+86399)
            z['result_store']=sec(z.storetime); z['result_store_day']=sec(z.storedate)
            z['join_method']='native_hadm'
            missing=z[z.hadm_id.isna()]
            if len(missing) and alladm is None: alladm=admission_intervals()
            for ix,r in missing.iterrows():
                key=(int(r.subject_id),str(r.micro_specimen_id),r.collection_lo,r.collection_hi)
                if key not in missing_links:
                    cq=qsubs[r.subject_id]
                    near=cq.t0.gt(r.collection_lo)&cq.t0.sub(96*3600).le(r.collection_hi)
                    hh=np.nan; method='unlinked'
                    if near.any() and np.isfinite(r.collection_lo):
                        aa=alladm.get(r.subject_id)
                        full=aa[aa.admit.le(r.collection_lo)&aa.discharge.gt(r.collection_hi)]
                        possible=aa[aa.admit.le(r.collection_hi)&aa.discharge.gt(r.collection_lo)]
                        if len(full)==1 and len(possible)==1:
                            hh=full.hadm_id.iloc[0]; method='unique_subject_full_admission_interval'
                        elif len(possible): interval_ambiguous.add((int(r.subject_id),str(r.micro_specimen_id)))
                        else: unmatched.add((int(r.subject_id),str(r.micro_specimen_id)))
                    missing_links[key]=(hh,method)
                hh,method=missing_links[key]; z.at[ix,'hadm_id']=hh; z.at[ix,'join_method']=method
            z=z[z.hadm_id.isin(adms)].copy()
            if len(z):
                good=z.subject_id.eq(z.hadm_id.map(refs.subject_id)); conflicts+=int((~good).sum()); z=z[good].copy()
                lower=z.hadm_id.map(refs.t0)-96*3600; upper=z.hadm_id.map(refs.t0)
                z=z[(z.collection_hi.ge(lower)&z.collection_lo.lt(upper))|z.collection_lo.isna()].copy()
                z['hadm_id']=z.hadm_id.astype('int64'); z['subject_id']=z.subject_id.astype('int64')
                if len(z): pieces.append(z)
            now=time.perf_counter()
            if now-last>=30:
                say({'stage':'cultures','percent_bytes':round(100*f.n/plan['bytes'],1),'seconds':round(now-begin),'peak_MB':peak_mb()}); last=now
        assert f.n==plan['bytes']; digest=f.h.hexdigest()
    finally: stream.close()
    assert CULTURE_PATH.stat().st_mtime_ns==plan['mtime_ns']
    assert conflicts==0, 'Native patient/admission conflict'
    d=pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame(columns=CULTURE_COLS)
    before=len(d); d=d.drop_duplicates().reset_index(drop=True)
    assert not d.microevent_id.duplicated().any(), 'Conflicting source microevent rows'
    b['cultures_raw']=d
    b['culture_dictionary']=d[['spec_itemid','spec_type_desc','test_itemid','test_name']].drop_duplicates().reset_index(drop=True)
    plan.update(sha256=digest,rows_traversed=total,rows_retained=len(d),exact_duplicates_removed=before-len(d),
        parsed_to_EOF=True,source_key_conflicts=conflicts,missing_hadm_unlinked_specimens=len(unmatched),
        missing_hadm_ambiguous_specimens=len(interval_ambiguous))
    b['runs'].append(dict(stage='cultures',utc=utc(),seconds=round(time.perf_counter()-begin,2),
        bytes_read=plan['bytes'],peak_MB=peak_mb()))
    b['final_entry_sha256']=sha(Path(__file__)); save(b)
    say({'stage':'cultures','completed':True,'retained_rows_rounded':count(len(d)),
        'distinct_specimens_rounded':count(d.micro_specimen_id.nunique()),'seconds':b['runs'][-1]['seconds'],'peak_MB':peak_mb(),
        'models':0,'effects':0})


def culture_kind(specimen, test):
    """Classify the test order label, never its eventual result."""
    s=str(specimen or '').strip().upper(); t=str(test or '').strip().upper(); both=s+' '+t
    if re.search(r'MRSA|VANCOMYCIN RESISTANT ENTEROCOCC|STAPH AUREUS (SCREEN|PREOP)|SURVEILL|GROUP B.*STREP|ANORECTAL',both):
        return 'SURVEILLANCE_OR_COLONIZATION_SCREEN'
    if re.search(r'POST.?MORT|ACID.?FAST|\bAFB\b|FUNG|YEAST|VIRAL|VIRUS|ENTEROVIR|VARICELLA|CYTOMEGAL|NEOPLAST|CHROMOSOM|CYTOGENET|TISSUE CULTURE[- ]BONE MARROW',both):
        return 'NONBACTERIAL_OR_NONINFECTIOUS_TEST'
    if 'CULTURE' in t or t in ['AEROBIC BOTTLE','ANAEROBIC BOTTLE','R/O BETA STREP GROUP A']:
        return 'DIAGNOSTIC_BACTERIAL_CULTURE'
    if re.search(r'DIFFICILE|LEGIONELLA URINARY|NEISSERIA GONORRHOEAE|CHLAMYDIA TRACHOMATIS',t):
        return 'BACTERIAL_TEST_NOT_CULTURE'
    return 'OTHER_OR_AMBIGUOUS_NONCULTURE_TEST'


def culture_pair_interval(lo, hi, t0, antibiotics):
    """0=no compatible time, 1=possible, 2=certain across the full interval.

    For a fixed antibiotic A, culture C can be in [A-72h,A+24h]. Both
    events must precede t0. A date-only interval is not collapsed to midnight.
    """
    if not np.isfinite(lo) or not np.isfinite(hi) or hi<lo: return 0
    windows=sorted((float(a)-72*3600,min(float(a)+24*3600,float(t0)-1))
                   for a in antibiotics if np.isfinite(a) and a<t0)
    merged=[]
    for s,e in windows:
        if s>e: continue
        if merged and s<=merged[-1][1]+1: merged[-1][1]=max(e,merged[-1][1])
        else: merged.append([s,e])
    if any(lo>=s and hi<=e for s,e in merged): return 2
    if any(max(lo,s)<=min(hi,e) for s,e in merged): return 1
    return 0


def culture_tests():
    a=300000.; t=800000.
    assert culture_pair_interval(a-72*3600,a-72*3600,t,[a])==2
    assert culture_pair_interval(a-72*3600-1,a-72*3600-1,t,[a])==0
    assert culture_pair_interval(a+24*3600,a+24*3600,t,[a])==2
    assert culture_pair_interval(a+24*3600+1,a+24*3600+1,t,[a])==0
    assert culture_pair_interval(a,a,t,[a])==2
    assert culture_pair_interval(t,t,t,[t-1000])==0
    assert culture_pair_interval(t-1,t-1,t,[t-1000])==2
    assert culture_pair_interval(t-100,t+100,t,[t-1000])==1
    assert culture_pair_interval(t-90000,t-3601,t,[t-4000])==2
    assert culture_pair_interval(t-1000,t-1000,t,[t+10])==0
    assert culture_pair_interval(np.nan,np.nan,t,[a])==0
    assert culture_pair_interval(t-100,t-100,t,[])==0
    assert culture_kind('BLOOD CULTURE','AEROBIC BOTTLE')=='DIAGNOSTIC_BACTERIAL_CULTURE'
    assert culture_kind('SWAB','MRSA SCREEN')=='SURVEILLANCE_OR_COLONIZATION_SCREEN'
    assert culture_kind('SPUTUM','FUNGAL CULTURE')=='NONBACTERIAL_OR_NONINFECTIOUS_TEST'
    assert culture_kind('BONE MARROW','Tissue Culture-Bone Marrow')=='NONBACTERIAL_OR_NONINFECTIOUS_TEST'
    assert culture_kind('STOOL','C. difficile PCR')=='BACTERIAL_TEST_NOT_CULTURE'
    return ['72h_inclusive','72h_outside','24h_inclusive','24h_outside','simultaneous_pre_t0',
            'collection_at_t0_excluded','last_second_before_t0','date_straddles_t0_unknown',
            'whole_date_certain','future_antibiotic_excluded','missing_time_unknown','no_antibiotic',
            'routine_blood_culture','screen_not_diagnostic','fungal_not_bacterial',
            'cytogenetic_culture_not_infection','bacterial_PCR_separate']


def culture_features():
    b=load(); assert 'cultures_raw' in b
    assert b['culture_supplement']['signature']==culture_signature()
    b['culture_tests']=culture_tests()
    if 'culture_qualification' in b:
        say({'stage':'culture_features','reuse':True}); return
    begin=time.perf_counter(); raw=b['cultures_raw'].copy()
    catalog=b['culture_dictionary'].copy()
    catalog['kind']=[culture_kind(s,t) for s,t in zip(catalog.spec_type_desc.fillna(''),catalog.test_name.fillna(''))]
    # Mapping is saved before joining these tests to HC paths, covariates, or
    # precision. This implements the stated diagnostic-culture concept.
    b['culture_test_mapping']=dict(utc=utc(),source='test/specimen labels only, no result or treatment-path review',
        catalog=catalog,source_sha256=b['plans']['cultures']['sha256'],code_sha256=sha(Path(__file__)))
    save(b)
    raw['kind']=[culture_kind(s,t) for s,t in zip(raw.spec_type_desc.fillna(''),raw.test_name.fillna(''))]
    specimens=[]; last=time.perf_counter()
    for i,(sid,g) in enumerate(raw.groupby('micro_specimen_id',sort=False)):
        hh=g.hadm_id.unique(); ss=g.subject_id.unique()
        key_conflict=len(hh)!=1 or len(ss)!=1
        exact=sorted(g.collection.dropna().unique()); days=set(g.collection_day.dropna().unique())
        if len(exact)==1 and (not days or days=={math.floor(exact[0]/86400)*86400}):
            lo=hi=float(exact[0]); state='EXACT_COLLECTION_TIME'
        elif len(exact)==0 and len(days)==1:
            lo=float(next(iter(days))); hi=lo+86399; state='DATE_ONLY_INTERVAL'
        elif not exact and not days:
            lo=hi=np.nan; state='UNKNOWN_COLLECTION_TIME'
        else:
            lo=hi=np.nan; state='CONFLICTING_COLLECTION_TIME'
        diagnostic=bool(g.kind.eq('DIAGNOSTIC_BACTERIAL_CULTURE').any())
        ds=g.loc[g.kind.eq('DIAGNOSTIC_BACTERIAL_CULTURE'),'spec_type_desc'].fillna('').str.upper()
        specimens.append(dict(micro_specimen_id=sid,hadm_id=hh[0],subject_id=ss[0],collection_lo=lo,collection_hi=hi,
            time_state=state,key_conflict=key_conflict,diagnostic_bacterial_culture=diagnostic,
            blood_culture=bool(ds.str.fullmatch(r'BLOOD CULTURE(?: - NEONATE)?').any()),
            surveillance=bool(g.kind.eq('SURVEILLANCE_OR_COLONIZATION_SCREEN').any()),
            bacterial_nonculture=bool(g.kind.eq('BACTERIAL_TEST_NOT_CULTURE').any()),
            test_kinds='|'.join(sorted(g.kind.unique())),source_rows=len(g),
            native_hadm=bool(g.join_method.eq('native_hadm').any()),
            result_first_stored=g.result_store.min(),result_last_stored=g.result_store.max()))
        if time.perf_counter()-last>=30:
            say({'stage':'culture_specimen_collapse','percent':round(100*(i+1)/raw.micro_specimen_id.nunique(),1),'seconds':round(time.perf_counter()-begin),'peak_MB':peak_mb()}); last=time.perf_counter()
    d=pd.DataFrame(specimens)
    b['culture_specimens']=d
    meds=b['resolved_medications']; aa=meds[meds.is_abx_name&meds.actual_systemic].copy()
    ag={h:g for h,g in aa.groupby('hadm_id',sort=False)}
    dg={h:g for h,g in d.groupby('hadm_id',sort=False)}; empty=d.iloc[:0]; emptya=aa.iloc[:0]
    features=[]
    for c in b['medication_features'].itertuples():
        ab=ag.get(c.hadm_id,emptya); ab=ab[ab.chart.ge(c.t0-86400)&ab.chart.lt(c.t0)]
        at=ab.chart.to_numpy(); strict_at=ab.loc[ab.store.lt(c.t0)&ab.store.notna(),'chart'].to_numpy()
        g=dg.get(c.hadm_id,empty).copy(); g['pair_status']=0; g['strict_abx_pair_status']=0
        for ix,s in g[~g.key_conflict].iterrows():
            g.at[ix,'pair_status']=culture_pair_interval(s.collection_lo,s.collection_hi,c.t0,at)
            g.at[ix,'strict_abx_pair_status']=culture_pair_interval(s.collection_lo,s.collection_hi,c.t0,strict_at)
        diagnostic=g[g.diagnostic_bacterial_culture]
        certain=diagnostic[diagnostic.pair_status.eq(2)]; possible=diagnostic[diagnostic.pair_status.eq(1)]
        if not len(ab): state='NO_ACCEPTED_PREBASE_ANTIBACTERIAL'
        elif len(certain): state='CERTAIN_BASELINE_DIAGNOSTIC_PAIR'
        elif len(possible): state='POSSIBLE_PAIR_DATE_INTERVAL'
        elif bool((diagnostic.key_conflict|diagnostic.time_state.isin(['UNKNOWN_COLLECTION_TIME','CONFLICTING_COLLECTION_TIME'])).any()): state='CULTURE_KEY_OR_TIME_UNRESOLVED'
        elif len(diagnostic): state='DIAGNOSTIC_CULTURE_NOT_TIME_PAIRED'
        else: state='NO_DIAGNOSTIC_CULTURE_IDENTIFIED_IN_PROJECTION'
        features.append(dict(hadm_id=c.hadm_id,culture_pair_state=state,
            paired_culture=bool(len(certain)),paired_exact_culture=bool(certain.time_state.eq('EXACT_COLLECTION_TIME').any()),
            paired_date_only_culture=bool(certain.time_state.eq('DATE_ONLY_INTERVAL').any()),
            paired_blood_culture=bool(certain.blood_culture.any()),
            paired_strict_abx=bool(diagnostic.strict_abx_pair_status.eq(2).any()),
            any_micro_test_pair=bool(g.pair_status.eq(2).any()),
            possible_diagnostic_pair=bool(len(possible)),
            culture_date_straddles_t0=bool((diagnostic.time_state.eq('DATE_ONLY_INTERVAL')&diagnostic.collection_lo.lt(c.t0)&diagnostic.collection_hi.ge(c.t0)).any()),
            culture_key_or_time_conflict=bool((g.key_conflict|g.time_state.isin(['UNKNOWN_COLLECTION_TIME','CONFLICTING_COLLECTION_TIME'])).any()),
            matching_specimens=len(certain),baseline_specimens=len(g),diagnostic_specimens=len(diagnostic),
            paired_culture_result_update_after_t0=bool(certain.result_last_stored.ge(c.t0).any()),
            paired_culture_all_result_store_unknown=bool(len(certain) and certain.result_last_stored.isna().all()),
            paired_culture_native_hadm=bool(certain.native_hadm.any()),
            paired_collection_pre_hospital_admit=bool((certain.native_hadm&certain.collection_hi.lt(c.admit)).any()),
            latest_paired_collection=float(certain.collection_hi.max()) if len(certain) else np.nan))
    f=pd.DataFrame(features)
    q=b['medication_features'].merge(f,on='hadm_id',validate='one_to_one')
    b['culture_qualification']=q
    p=b['clinical_features'].merge(f,on='hadm_id',validate='one_to_one')
    assert set(p.hadm_id)==set(b['clinical_features'].hadm_id)
    assert np.array_equal(p.t0.to_numpy(),b['clinical_features'].t0.to_numpy())
    assert not p.paired_culture.gt(p.baseline_abx_actual).any()
    b['culture_clinical_features']=p
    b['runs'].append(dict(stage='culture_features',utc=utc(),seconds=round(time.perf_counter()-begin,2),peak_MB=peak_mb()))
    b['final_entry_sha256']=sha(Path(__file__)); save(b)
    say({'stage':'culture_features','completed':True,'existing_proxy_rounded':count(len(p)),
        'certain_pair_rounded':count(int(p.paired_culture.sum())),
        'paired_HC_path_rounded':count(int((p.paired_culture&p.action_state.eq('HC_DOCUMENTED_WITHIN_6H')).sum())),
        'seconds':b['runs'][-1]['seconds'],'models':0,'effects':0})


def assess_cultures():
    """Recount fixed paths and pretreatment information, not causal estimates."""
    b=load(); assert 'culture_clinical_features' in b
    p=b['culture_clinical_features']; q=p[p.paired_culture].copy()
    hc=q[q.action_state.eq('HC_DOCUMENTED_WITHIN_6H')]
    wait=q[q.action_state.eq('NO_GC_ADMIN_IDENTIFIED_CAPTURE_PRESENT')]
    d=b['culture_specimens']; raw=b['cultures_raw']
    b['aggregates']=[r for r in b['aggregates'] if not r['section'].startswith('culture_')]
    note='同版3.1、固定t0及原抗菌药映射；培养采集与实际抗菌药均早于t0；不是确诊感染性休克或最终TTE资格'
    row(b,'culture_source','培养文件实际字节',value=b['plans']['cultures']['bytes'],note=str(CULTURE_PATH))
    row(b,'culture_source','保留投影行：不是培养次数',len(raw),note='全部4h技术候选的96h投影；未读取菌名、药敏、comments')
    row(b,'culture_source','按micro_specimen_id折叠的标本',len(d),note='药敏、多个菌株、多个检验不重复计数')
    for state,g in d.groupby('time_state'):
        row(b,'culture_source','标本采集时间口径',len(g),len(d),stratum=state)
    row(b,'culture_source','标本native键冲突',int(d.key_conflict.sum()),len(d))
    row(b,'culture_source','按完整住院区间补链的标本',int((~d.native_hadm).sum()),len(d),note='唯一subject/住院时间匹配；日期区间必须完整落在同一住院')
    for label,key in [('缺hadm且未能链接的标本','missing_hadm_unlinked_specimens'),('缺hadm且日期/住院链接歧义的标本','missing_hadm_ambiguous_specimens')]:
        row(b,'culture_source',label,b['plans']['cultures'].get(key,0),note='候选患者的近节点记录；不能作为未采样')
    classified=raw[['micro_specimen_id','spec_type_desc','test_name']].copy()
    classified['kind']=[culture_kind(s,t) for s,t in zip(classified.spec_type_desc.fillna(''),classified.test_name.fillna(''))]
    for kind,g in classified.groupby('kind'):
        row(b,'culture_source','检测类别标本',g.micro_specimen_id.nunique(),len(d),stratum=kind,note='同一标本可有多种检测，不要求相加为100%')
    # The classified dictionary is joined without exposing source patient rows.
    cats=b['culture_test_mapping']['catalog']
    for kind,g in cats.groupby('kind'):
        row(b,'culture_source','不同检验/标本类型映射',value=len(g),stratum=kind,note='计数为字典组合，不是患者数')
    for state,g in p.groupby('culture_pair_state'):
        row(b,'culture_qualification','原抗菌药代理候选的培养配对状态',len(g),len(p),stratum=state,note=note)
    for label,mask in [('原实际抗菌药代理候选',pd.Series(True,index=p.index)),
                       ('确定的基线诊断性培养配对',p.paired_culture),
                       ('至少一项精确采集时刻的确定配对',p.paired_exact_culture),
                       ('仅整日区间可确定且没有精确采集时刻的配对',p.paired_culture&~p.paired_exact_culture),
                       ('血培养配对敏感性',p.paired_blood_culture),
                       ('抗菌药在t0前已入库的配对敏感性',p.paired_culture&p.paired_strict_abx),
                       ('直接用全部微生物检测配对的宽泛上界',p.any_micro_test_pair),
                       ('存在日期横跨t0的诊断性培养',p.culture_date_straddles_t0),
                       ('存在键或时间冲突的标本',p.culture_key_or_time_conflict)]:
        row(b,'culture_qualification',label,int(mask.sum()),len(p),note=note)
    row(b,'culture_qualification','确定配对候选的独立患者',q.subject_id.nunique(),len(q),note='分母为医院住院，不是患者分母')
    row(b,'culture_qualification','确定配对仅依赖subject/住院时间补链',int((~q.paired_culture_native_hadm).sum()),len(q),
        note='其他候选至少一个配对标本有native hadm；链接修复不等于临床资格已经确认')
    row(b,'culture_qualification','存在重复候选住院的患者对应住院',int(q.subject_id.duplicated(keep=False).sum()),len(q),note='并非独立患者数；未来区间需按患者聚类')
    for label,mask in [('所配标本最后结果更新时间至少一项晚于t0',q.paired_culture_result_update_after_t0),
                       ('所配标本所有精确结果更新时间缺失',q.paired_culture_all_result_store_unknown),
                       ('存在native同次住院但采集早于医院admittime的配对',q.paired_collection_pre_hospital_admit)]:
        row(b,'culture_qualification',label,int(mask.sum()),len(q),note='结果时间不进入采集资格；native住院可含入院前ED采样，不据结果阳性选人')
    for name,z in [('抗菌药代理、培养配对前',p),('确定基线培养配对',q),('未确定配对',p[~p.paired_culture]),
                   ('血培养配对子集',p[p.paired_blood_culture]),('抗菌药严格入库配对子集',p[p.paired_culture&p.paired_strict_abx])]:
        for action,g in z.groupby('action_state'):
            row(b,'culture_paths','固定6h药物记录路径',len(g),len(z),stratum=name+' / '+action,
                note='包含未治疗、观察不足及早期终止；这是描述路径，不是CCW兼容人数、已验证未治疗或ESS')
    for variable in ['AST','bicarbonate','lactate','creatinine','platelet','bilirubin','hemoglobin','wbc']:
        col=variable+'__strict'
        row(b,'culture_labs','t0前采样且已入库：'+variable,int(q[col].notna().sum()),len(q),note='已有缓存；未重读18.4GB labevents；单位与原规则不变')
    for name,z in [('全部确定配对',q),('六小时HC路径',hc),('未检出GC且有给药捕获路径',wait)]:
        row(b,'culture_labs','δ三项输入严格时钟且单位可解释',int(z.delta_three_positive_plausible__strict.sum()),len(z),stratum=name,
            note='可用率不是分型标签或验证性能；不要求完整病例入组')
    for pattern,g in q.groupby('response_pattern'):
        for action,z in g.groupby('action_state'):
            if action in ['HC_DOCUMENTED_WITHIN_6H','NO_GC_ADMIN_IDENTIFIED_CAPTURE_PRESENT']:
                row(b,'culture_support','NE治疗前记录模式',len(z),len(g),stratum=pattern+' / '+action,
                    note='未拟合分配模型；只报告支持的初步分布')
    dose_cell=pd.cut(q.ne_dose,[0,.1,.25,.5,np.inf],right=False).astype(str).replace('nan','dose_unknown')
    for cell,g in q.groupby(dose_cell):
        row(b,'culture_support','既定NE显示剂量单元：全部路径',len(g),len(q),stratum=cell,
            note='沿用培养前描述界点，不作为新入组阈值或盐基换算')
        for action,z in g.groupby('action_state'):
            row(b,'culture_support','既定NE显示剂量单元：记录路径',len(z),len(g),stratum=cell+' / '+action,
                note='不是临床治疗可互换性、动态策略人数或ESS')
    for variable in ['ne_dose','relative_ne_change','age','lactate__strict','creatinine__strict','platelet__strict','bicarbonate__strict']:
        aa=pd.to_numeric(hc[variable],errors='coerce').dropna(); bb=pd.to_numeric(wait[variable],errors='coerce').dropna()
        if len(aa)>10 and len(bb)>10:
            sd=math.sqrt((aa.var(ddof=1)+bb.var(ddof=1))/2.)
            if sd>0: row(b,'culture_support','未调整SMD '+variable,value=round(float((aa.mean()-bb.mean())/sd),3),
                note='HC减未检出GC且有其他给药捕获；非加权平衡或无未测混杂证明')
    for label,mask in [('6h内死亡',q.early_death),('6h内医院出院',q.early_discharge),('6h内ICU转出',q.early_icu_exit),
                       ('6h内明确NE停止记录',q.first_recorded_ne_stop_6h.notna()),('NE停止后重新启动记录',q.recorded_ne_restart_6h.notna())]:
        row(b,'culture_early_events',label,int(mask.sum()),len(q),note='全配对分母，不分组比较死亡；未据此排除人群')
    for risk in [.1,.2,.3,.4]:
        for retention in [1.,.75,.5]:
            if len(hc)>10 and len(wait)>10:
                half=1.96*math.sqrt(risk*(1-risk)*(1./(len(hc)*retention)+1./(len(wait)*retention)))*100
                row(b,'culture_precision','总体RD 95%区间规划半宽百分点',value=round(half,2),
                    stratum='risk={}; information_retained={}'.format(risk,retention),
                    note='假定两路径相同风险；独立住院正态近似；未用实际死亡或权重；保留系数不是实际ESS')
    if len(hc)>10 and len(wait)>10:
        half=1.96*math.sqrt(.3*.7*(1/len(hc)+1/len(wait)))*100
        row(b,'culture_precision','理想均分两亚组：RD交互差半宽百分点',value=round(half*2,2),
            note='假定风险30%、所有分型完整、未损失信息；不是实际交互估计')
    for rr in b['aggregates']:
        if rr['section']=='unresolved' and rr['metric']=='t0前培养采集与配对':
            rr.update(value='BASELINE_PAIR_PROXY_DATA_VALIDATED',note='同版microbiologyevents已取得并验证；诊断性培养配对是感染评估代理，不证明Sepsis-3或充分复苏')
    b['precision_uses_observed_outcome']=False
    b['culture_assessment_utc']=utc(); b['final_entry_sha256']=sha(Path(__file__))
    save(b); report()
    say({'stage':'culture_assess','paired_candidates_rounded':count(len(q)), 'HC_path_rounded':count(len(hc)),
        'other_captured_no_GC_path_rounded':count(len(wait)),'models':0,'effects':0})


def culture_clinical_support():
    """Coverage and severity diagnostics, including the observed grace period.

    No strategy assignment model or outcome contrast is constructed here.
    """
    b=load(); assert 'chart' in b and 'culture_clinical_features' in b
    q=b['culture_clinical_features']; q=q[q.paired_culture].copy()
    b['aggregates']=[r for r in b['aggregates'] if r['section'] not in ['culture_chart','culture_dynamic','culture_context','culture_dose']]
    for clock in ['reported','strict']:
        for variable in ['MAP','HR','RR','SpO2','temperature_C','temperature_F','GCS_eye','GCS_verbal','GCS_motor',
                         'RASS','oxygen_device','ventilator_mode','FiO2','PEEP','code_status','intubation_status','CRRT_mode']:
            col=variable+'__'+clock
            row(b,'culture_chart',variable+' / '+clock,int(q[col].notna().sum()),len(q),
                note='基线严格charttime<t0；reported为测量代理时钟；strict另要求storetime<t0，均不证明医生首次可见')
            agecol=col+'_age_h'
            if agecol in q:
                age=q.loc[q[col].notna(),agecol].dropna()
                if len(age)>10: row(b,'culture_chart',variable+'最近测量距t0小时 / '+clock,
                    value='median {:.2f}; P90 {:.2f}'.format(age.median(),age.quantile(.9)))
        for status in ['DNR','DNI','CMO','full_code_record']:
            row(b,'culture_context',status+' / '+clock,int(q[status+'__'+clock].sum()),len(q),
                note='各状态分开，不把DNR或DNI当CMO，不把未知当full code；仅t0前记录')
        code=q['code_status__'+clock].notna()|q['intubation_status__'+clock].notna()
        row(b,'culture_context','一般/插管Code Status均无记录 / '+clock,int((~code).sum()),len(q))
        anyflag=q[['DNR__'+clock,'DNI__'+clock,'CMO__'+clock,'full_code_record__'+clock]].any(axis=1)
        row(b,'culture_context','有Code Status记录但未匹配明确状态 / '+clock,int((code&~anyflag).sum()),len(q),note='不自动当full code；需要检查枚举字典')
        for path,g in q.groupby('action_state'):
            if path not in ['HC_DOCUMENTED_WITHIN_6H','NO_GC_ADMIN_IDENTIFIED_CAPTURE_PRESENT']: continue
            for metric,mask in [('MAP有测量',g['MAP__'+clock].notna()),('MAP<65',g['MAP__'+clock].lt(65)),
                                ('明确CMO记录',g['CMO__'+clock]),('DNR记录',g['DNR__'+clock]),('DNI记录',g['DNI__'+clock])]:
                row(b,'culture_context',metric+' / '+clock,int(mask.sum()),len(g),stratum=path,note='描述记录差异，不按治疗后结局筛人')
        cmofree=q[~q['CMO__'+clock]]
        row(b,'culture_context','没有检出明确CMO记录的候选 / '+clock,len(cmofree),len(q),
            note='包括治疗限制未知，不能称full code或已临床确认；当前只作可观察资格情景，不改变主分母')
        row(b,'culture_context','没有检出明确CMO记录的HC路径 / '+clock,
            int(cmofree.action_state.eq('HC_DOCUMENTED_WITHIN_6H').sum()),len(cmofree),note='不是按死亡排除')
    hc=q[q.action_state.eq('HC_DOCUMENTED_WITHIN_6H')]
    wait=q[q.action_state.eq('NO_GC_ADMIN_IDENTIFIED_CAPTURE_PRESENT')]
    for clock in ['reported','strict']:
        for variable in ['MAP','HR','RR','SpO2']:
            col=variable+'__'+clock
            aa=pd.to_numeric(hc[col],errors='coerce').dropna(); bb=pd.to_numeric(wait[col],errors='coerce').dropna()
            if len(aa)>10 and len(bb)>10:
                sd=math.sqrt((aa.var(ddof=1)+bb.var(ddof=1))/2.)
                if sd>0: row(b,'culture_chart','未调整SMD '+col,value=round(float((aa.mean()-bb.mean())/sd),3),
                    note='HC记录路径减未检出GC且有其他给药捕获；不是加权诊断')
    r=b['resolved_medications']; events=r[r.hc_action].merge(hc[['hadm_id','first_hc']],on='hadm_id')
    events=events[events.chart.eq(events.first_hc)].drop_duplicates('hadm_id')
    assert len(events)==len(hc)
    for dose,g in events.groupby('dose_mg',dropna=False):
        row(b,'culture_dose','首个HC记录剂量mg',len(g),len(hc),stratum=str(dose),
            note='实际给药解析剂量，不用未来24h累积剂量筛资格，不推定200mg/day或IV途径')
    row(b,'culture_dose','明确IV的HC路径',int(events.IV_confirmed.sum()),len(hc),note='注射剂型与具体途径分开；未补造IV')
    d=b['chart']; d=d[d.hadm_id.isin(set(q.hadm_id))].copy(); refs=q.set_index('hadm_id')
    for h in range(6):
        point=q.t0+h*3600
        # An observation opportunity, not an informative-censoring rule or a
        # trial eligibility restriction: everyone remains in the baseline flow.
        alive=q.death_time.isna()|q.death_time.gt(point)
        inhospital=q.discharge.gt(point)
        inicu=q.icu_out.gt(point)
        gc_before=(q.first_hc.lt(point)|q.first_other_gc.lt(point))
        opportunity=alive&inhospital&inicu&~gc_before
        active=q[opportunity]
        label='hour '+str(h)
        row(b,'culture_dynamic','可观察ICU且未检出此前明确GC的时点风险集',len(active),len(q),stratum=label,
            note='仅覆盖诊断；未知给药保留，离ICU不等于删失医院死亡随访；不是CCW兼容人数')
        row(b,'culture_dynamic','该风险集中随后1h HC记录',int((opportunity&q.first_hc.ge(point)&q.first_hc.lt(point+3600)).sum()),len(active),stratum=label,
            note='最后6h边界另在总体行动路径按既定<=6h处理；稀疏小单元抑制')
        row(b,'culture_dynamic','时点前已离开ICU观察范围',int((alive&inhospital&~inicu).sum()),len(q),stratum=label)
        for clock in ['reported','strict']:
            known={}
            for variable in ['MAP_arterial','MAP_noninvasive','HR','RR','SpO2']:
                z=d[d.variable.eq(variable)&d.hadm_id.isin(set(active.hadm_id))].copy()
                decision=z.hadm_id.map(refs.t0)+h*3600
                z=z[z.chart.lt(decision)&z.chart.ge(decision-3600)&z.valuenum.notna()]
                if clock=='strict': z=z[z.store.notna()&z.store.lt(z.hadm_id.map(refs.t0)+h*3600)]
                known[variable]=set(z.hadm_id)
            known['MAP']=known['MAP_arterial']|known['MAP_noninvasive']
            for variable in ['MAP','HR','RR','SpO2']:
                row(b,'culture_dynamic','此前1h '+variable+'记录 / '+clock,len(known[variable]),len(active),stratum=label,
                    note='各时点只用此前记录；后面返回或发生的记录不回填；未把无测量填正常')
            joint=set(active.hadm_id)
            for variable in ['MAP','HR','RR','SpO2']: joint &= known[variable]
            row(b,'culture_dynamic','四项核心床旁记录联合覆盖 / '+clock,len(joint),len(active),stratum=label,
                note='不以联合完整病例定义主分析人群，不将该覆盖当无未测混杂证明')
    row(b,'culture_dynamic','宽限期新返回检验',value='SOURCE_PRESENT_NOT_EXTRACTED',
        note='labevents缓存只覆盖t0前；不能把未提取当没有复查或将未来结果填基线')
    dictionary=RAW/'icu'/'d_items.csv'/'d_items.csv'
    items=pd.read_csv(dictionary)
    steroid_items=items[items.label.fillna('').str.contains('hydrocort|methylpred|dexameth|predniso|solu.?cort',case=False,regex=True)]
    b['culture_supplement']['ICU_steroid_dictionary_check']=dict(path=str(dictionary),sha256=sha(dictionary),
        matched_items=steroid_items.to_dict('records'),meaning='No named systemic GC item found; this is not proof that no GC was administered.')
    for rr in b['aggregates']:
        if rr['section']=='unresolved' and rr['metric']=='新队列MAP、呼吸状态与code status':
            rr.update(value='SOURCE_PROJECTED_BASELINE_AND_6H',note='同版限定字段投影已完成；基线与宽限期分开，reported/strict时钟并列；未拟合权重或确认临床可互换性')
    b['culture_support_utc']=utc(); b['final_entry_sha256']=sha(Path(__file__))
    save(b); report()
    say({'stage':'culture_support','completed':True,'paired_candidates_rounded':count(len(q)),
        'HC_path_rounded':count(len(hc)),'models':0,'effects':0})


def decide_cultures():
    b=load(); assert 'culture_clinical_features' in b
    p=b['culture_clinical_features']; q=p[p.paired_culture]
    hc=q[q.action_state.eq('HC_DOCUMENTED_WITHIN_6H')]; wait=q[q.action_state.eq('NO_GC_ADMIN_IDENTIFIED_CAPTURE_PRESENT')]
    def half(r,keep=1.):
        return 1.96*math.sqrt(r*(1-r)*(1/(len(hc)*keep)+1/(len(wait)*keep)))*100
    txt='''**当前决定：培养配对核验已完成；总体HC启动候选保持CONDITIONAL GO，不恢复δ为主要效应修饰因素，尚不进入死亡效果估计。**

本轮使用新增MIMIC-IV3.1培养表，保持同一4小时NE复评节点、6小时行动窗和实际抗菌药/激素映射。原规则、解析前后结果及培养补充前状态保留在同一private缓存；旧NE/VP研究未更动。

|固定问题下的前后对照|培养补充前|确定t0前诊断性培养配对后|
|---|---:|---:|
|医院住院候选约|%s|%s|
|独立患者约|%s|%s|
|6h HC启动记录路径约|%s|%s|
|未检出GC、随访期有eMAR记录的路径约|%s|%s|
|δ三项治疗前有值、单位可解释且已入库|%s%%|%s%%|
|HC路径中三项输入严格可用约|%s|%s|

**已解决的缺口：**约%s%%的原代理候选已有t0前诊断性培养与实际抗菌治疗配对。采集早于给药最多72小时，或给药早于采集最多24小时；两项必须均在t0前。按标本去重，未读取菌名、药敏或原始临床文本，不要求培养阳性。没有确定配对者仍保留在原始分母，未被解释为没有感染或没有采样。血培养限制、日期不确定及抗菌药入库时钟分别报告，不据行动数量选择定义。

**时间语义：**%s%%的配对候选至少有一个所配标本的最后结果更新时间晚于t0。最后更新不是标本采集，也不能代表最早结果可见；本研究只据采集时钟建立感染评估代理，不用后来的培养结果反向创造资格。日期记录作为全天区间核验，不默认零点采集。该检查没有补出完整Sepsis-3、充分液体复苏或休克病因。

**信息量：**在两路径结局风险都假定20%%/30%%、独立住院且不损失信息的规划下，总体RD区间半宽约±%.2f/±%.2f个百分点；只保留一半信息时约±%.2f/±%.2f。理想分型完全、均分且无损失时，两个RD之差的半宽已约±%.2f个百分点。上述不是实际死亡效果或ESS；±5/±10只是用户指定的并列规划情景。培养表改善了资格解释，没有扩大HC行动规模，不支持将分型异质性作为当前主分析承诺。

**未解决的关键限制：**HC路径的注射盐型与正数实际给药剂量可解释，但具体IV/IM途径仍缺失；若坚持IV-only，定义尚未成立。当前抗菌药＋诊断性培养＋NE是感染评估伴循环支持的记录代理，不是所有非感染性原因已排除。NE剂量与乳酸等治疗前差异仍明显，培养配对不自动解决治疗选择混杂。接下来的正式策略仍需明确改善退出、救援、其他GC及动态记录不足的处理，不能用简单六小时分组冒充随机策略或CCW兼容人数。

**捕获口径：**capture统计任意eMAR行，可能包含due、held或未执行记录，只证明系统记录机会。它不能证明给药捕获完整，也不能将“未检出GC”升级为确定未治疗；此前表格中的“有给药捕获”均按这一记录口径解释。
''' % (count(len(p)),count(len(q)),count(p.subject_id.nunique()),count(q.subject_id.nunique()),
       count(int(p.action_state.eq('HC_DOCUMENTED_WITHIN_6H').sum())),count(len(hc)),
       count(int(p.action_state.eq('NO_GC_ADMIN_IDENTIFIED_CAPTURE_PRESENT').sum())),count(len(wait)),
       str(pct(int(p.delta_three_positive_plausible__strict.sum()),len(p))),str(pct(int(q.delta_three_positive_plausible__strict.sum()),len(q))),
       count(int((p.action_state.eq('HC_DOCUMENTED_WITHIN_6H')&p.delta_three_positive_plausible__strict).sum())),
       count(int(hc.delta_three_positive_plausible__strict.sum())),str(pct(len(q),len(p))),
       str(pct(int(q.paired_culture_result_update_after_t0.sum()),len(q))),half(.2),half(.3),half(.2,.5),half(.3,.5),half(.3)*2)
    if 'culture_support_utc' in b:
        txt+='\n**床旁资料核验也已完成：**限定字段的MAP、呼吸状态及分别记录的DNR/DNI/CMO已从现有3.1 chartevents取得，同时保留6h内纵向投影。基线只用t0前记录，测量与入库口径并列，未知不填正常或full code；实际覆盖和时点风险集见culture_chart/culture_context/culture_dynamic。动态记录覆盖不是策略共同支持或混杂已充分调整的证明。\n'
        txt+='\n|床旁核验：确定培养配对分母|测量时钟|另要求t0前入库|\n|---|---:|---:|\n'
        for variable in ['MAP','HR','RR','SpO2']:
            txt+='|{}|{}%|{}%|\n'.format(variable,pct(int(q[variable+'__reported'].notna().sum()),len(q)),pct(int(q[variable+'__strict'].notna().sum()),len(q)))
        unknown=q['code_status__reported'].isna()&q['intubation_status__reported'].isna()
        txt+='\n约{}%没有t0前Code Status记录；本次未检出CMO不等于已排除限制生命支持。宽限期内四项床旁记录的测量时钟联合覆盖约90%，严格入库口径约52%–55%，为各时点尚在可观察ICU、未检出此前明确GC的记录风险集，不能当作完整临床随访或因果共同支持。\n'.format(pct(int(unknown.sum()),len(q)))
    txt+='\n本轮完成限定的培养和床旁资料核查后停止新增原始读取，不新增模型或死亡比较。现有chartevents、药物和检验资产不等于正式TTE已经成立：总体候选若继续，应先在这些既有投影上写清唯一可执行的HC启动/暂缓策略，评估其兼容性与固定调整集。院外激素、确切给药途径和临床病因仍为独立限制，不以继续下载文件替代设计。\n'
    b['decision_text']=txt; b['decision']='CONDITIONAL_GO_SUPPORT_NOT_ESTABLISHED'; b['decision_utc']=utc()
    b['raw_scans_stopped_after_labs']=False
    b['raw_scans_stopped_after_culture']=('chart' not in b)
    b['final_entry_sha256']=sha(Path(__file__)); save(b); report()
    say({'stage':'culture_decide','decision':b['decision'],'models':0,'effects':0,'delta_primary':False})


def scan_clinical(name):
    b=load(); assert 'medication_features' in b
    if name in b:
        say({'stage':name,'reuse':True}); return
    assert name in ['labs','chart']
    q=b['medication_features']; q=q[q.candidate_proxy].copy(); refs=q.set_index('hadm_id')
    if not len(q):
        raise ValueError('No drug-proxy candidates; do not scan clinical source')
    mapping=LAB_MAP if name=='labs' else CHART_MAP
    ids={i:k for k,values in mapping.items() for i in values}
    path=PATHS[name]
    plan=dict(path=str(path),bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns,columns=COLS[name],
              filter=('candidate proxy admissions; selected itemids; labs strictly pre-t0; <=24h labs' if name=='labs' else
                      'same candidate ICU stay; selected items; t0-4h through t0+6h; code status from admission; future rows never baseline'),
              candidate_count=len(q),mapping=mapping,mapping_saved_utc=utc())
    b['plans'][name]=plan; save(b)
    begin=time.perf_counter(); last=begin; chunks=[]; total=0; conflicts=0; missing_unlinked=0
    adms=set(q.hadm_id); subs=set(q.subject_id); candidate_stays=set(q.stay_id)
    f=HashReader(path); stream=io.BufferedReader(f,buffer_size=4*1024*1024)
    try:
        for raw in pd.read_csv(stream,usecols=COLS[name],dtype=str,chunksize=150000):
            total+=len(raw)
            item=pd.to_numeric(raw.itemid,errors='coerce'); z=raw[item.isin(ids)].copy()
            if len(z):
                z['itemid']=item.loc[z.index].astype('int64')
                h=pd.to_numeric(z.hadm_id,errors='coerce'); s=pd.to_numeric(z.subject_id,errors='coerce')
                z=z[h.isin(adms)|(h.isna()&s.isin(subs))].copy()
                z['hadm_id']=h.loc[z.index]; z['subject_id']=s.loc[z.index]
                z['chart']=sec(z.charttime); z['store']=sec(z.storetime)
                z['join_method']='native_hadm'
                if name=='labs':
                    for ix,r in z[z.hadm_id.isna()].iterrows():
                        poss=q[q.subject_id.eq(r.subject_id)&q.admit.le(r.chart)&q.t0.gt(r.chart)&q.t0.sub(86400).le(r.chart)]
                        if len(poss)==1:
                            z.at[ix,'hadm_id']=poss.hadm_id.iloc[0]; z.at[ix,'join_method']='unique_subject_admission_time'
                        else:
                            missing_unlinked+=1
                z=z[z.hadm_id.isin(adms)].copy()
                good=z.subject_id.eq(z.hadm_id.map(refs.subject_id)); conflicts+=int((~good).sum()); z=z[good]
                z['variable']=z.itemid.map(ids)
                cutoff=z.hadm_id.map(refs.t0)
                lo=np.maximum(z.hadm_id.map(refs.admit),cutoff-(86400 if name=='labs' else 14400))
                if name=='chart':
                    code=z.variable.isin(['code_status','intubation_status']); lo=np.where(code,z.hadm_id.map(refs.admit),lo)
                    # A different ICU stay's records must not silently become this ICU's bedside measurements.
                    stay=pd.to_numeric(z.stay_id,errors='coerce')
                    z['source_stay_matches']=stay.eq(z.hadm_id.map(refs.stay_id))
                    valid_stay=z.source_stay_matches|code
                else:
                    valid_stay=pd.Series(True,index=z.index)
                upper_ok=z.chart.lt(cutoff) if name=='labs' else z.chart.le(cutoff+21600)
                z=z[z.chart.ge(lo)&upper_ok&valid_stay].copy()
                z['hadm_id']=z.hadm_id.astype('int64'); z['subject_id']=z.subject_id.astype('int64')
                z['valuenum']=pd.to_numeric(z.valuenum,errors='coerce')
                if name=='chart': z['phase']=np.where(z.chart.lt(z.hadm_id.map(refs.t0)),'before_t0','t0_through_6h')
                if len(z): chunks.append(z)
            now=time.perf_counter()
            if now-last>=30:
                say({'stage':name,'percent_bytes':round(100*f.n/plan['bytes'],1),'seconds':round(now-begin),'peak_MB':peak_mb()}); last=now
        assert f.n==plan['bytes']; digest=f.h.hexdigest()
    finally:
        stream.close()
    assert path.stat().st_size==plan['bytes'] and path.stat().st_mtime_ns==plan['mtime_ns']
    assert conflicts==0, 'Same-version source key conflict'
    data=pd.concat(chunks,ignore_index=True) if chunks else pd.DataFrame(columns=COLS[name]+['chart','store','variable'])
    before=len(data); data=data.drop_duplicates().reset_index(drop=True)
    b[name]=data
    b['plans'][name].update(sha256=digest,rows_traversed=total,rows_retained=len(data),exact_duplicates_removed=before-len(data),parsed_to_EOF=True)
    b['runs'].append(dict(stage=name,utc=utc(),seconds=round(time.perf_counter()-begin,2),bytes_read=plan['bytes'],
                          peak_MB=peak_mb(),source_key_conflicts=conflicts,missing_hadm_unlinked=missing_unlinked))
    save(b)
    say({'stage':name,'completed':True,'retained_rows_rounded':count(len(data)),'seconds':b['runs'][-1]['seconds'],'peak_MB':peak_mb()})


def clinical_features():
    b=load(); q=b['medication_features']; q=q[q.candidate_proxy].copy(); refs=q.set_index('hadm_id')
    b['medication_tests']=medication_tests()
    b['aggregates']=[x for x in b['aggregates'] if x['section'] not in ['lab_coverage','delta_coverage','chart_coverage','clinical_context','state_support']]
    numeric=[]
    for source,mapping,hours in [('labs',LAB_MAP,LAB_HOURS),('chart',CHART_MAP,CHART_HOURS)]:
        if source not in b:
            continue
        d=b[source].copy()
        d['age_h']=(d.hadm_id.map(refs.t0)-d.chart)/3600
        d['lag_min']=(d.store-d.chart)/60
        d['strict']=d.store.notna()&d.store.lt(d.hadm_id.map(refs.t0))
        for var_index,variable in enumerate(mapping):
            if var_index % 4 == 0:
                q=q.copy()  # consolidate blocks; do not emit hundreds of fragmentation warnings
            pre=d.chart.lt(d.hadm_id.map(refs.t0))
            v=d[d.variable.eq(variable)&pre&d.age_h.le(hours[variable])].copy()
            if variable in ['code_status','intubation_status']:
                v=d[d.variable.eq(variable)&pre].copy()
            for clock in ['reported','strict']:
                z=v if clock=='reported' else v[v.strict]
                is_text=source=='chart' and variable in ['code_status','intubation_status','oxygen_device','ventilator_mode','CRRT_mode','IABP_evidence']
                z=z[z.value.fillna('').ne('')] if is_text else z[z.valuenum.notna()]
                last=z.sort_values(['chart','store']).drop_duplicates('hadm_id',keep='last')
                q[variable+'__'+clock]=q.hadm_id.map(last.set_index('hadm_id')['value' if is_text else 'valuenum'])
                q[variable+'__'+clock+'_age_h']=q.hadm_id.map(last.set_index('hadm_id').age_h)
                q[variable+'__'+clock+'_chart']=q.hadm_id.map(last.set_index('hadm_id').chart)
                q[variable+'__'+clock+'_unit']=q.hadm_id.map(last.set_index('hadm_id').valueuom)
                row(b,'lab_coverage' if source=='labs' else 'chart_coverage',variable+' / '+clock,
                    len(last),len(q),note='时长<=%sh；报告时钟不等同临床首次可见；严格口径要求已入库'%hours[variable])
            if len(v):
                for col,label in [('age_h','样本/测量距t0小时'),('lag_min','记录入库滞后分钟')]:
                    vv=v[col].dropna()
                    if len(vv)>10:
                        row(b,'lab_coverage' if source=='labs' else 'chart_coverage',variable+' / '+label,
                            value='median {:.2f}; P90 {:.2f}'.format(vv.median(),vv.quantile(.9)),note='所有选定窗口内记录，不是患者效应')
        numeric.append(d)
    if 'labs' in b:
        for clock in ['reported','strict']:
            cols=[v+'__'+clock for v in ['AST','bicarbonate','lactate']]
            complete=q[cols].notna().all(axis=1)
            ast_unit=q['AST__'+clock+'_unit'].fillna('').str.lower().str.replace(' ','',regex=False).isin(['iu/l','u/l'])
            bicarb_unit=q['bicarbonate__'+clock+'_unit'].fillna('').str.lower().str.replace(' ','',regex=False).isin(['meq/l','mmol/l'])
            lactate_unit=q['lactate__'+clock+'_unit'].fillna('').str.lower().str.replace(' ','',regex=False).eq('mmol/l')
            units=ast_unit&bicarb_unit&lactate_unit
            plausible=complete&units&q[cols[0]].gt(0)&q[cols[0]].le(100000)&q[cols[1]].gt(0)&q[cols[1]].le(70)&q[cols[2]].gt(0)&q[cols[2]].le(50)
            q['delta_three_available__'+clock]=complete
            q['delta_three_positive_plausible__'+clock]=plausible
            row(b,'delta_coverage','三项决策前输入均有记录 / '+clock,int(complete.sum()),len(q),note='AST/碳酸氢盐24h、乳酸6h；非原论文ICU24h极值分类器验证')
            row(b,'delta_coverage','三项单位可解释、正值且宽范围内 / '+clock,int(plausible.sum()),len(q),note='AST IU/L或U/L；碳酸氢盐mEq/L或mmol/L；乳酸mmol/L；未知单位不自动换算')
            for action,g in q.groupby('action_state'):
                row(b,'delta_coverage','三项决策前输入 / '+clock,int(g['delta_three_available__'+clock].sum()),len(g),stratum=action)
            tc=[v+'__'+clock+'_chart' for v in ['AST','bicarbonate','lactate']]
            spread=(q[tc].max(axis=1)-q[tc].min(axis=1))/3600
            spread=spread[complete]
            if len(spread)>10:
                row(b,'delta_coverage','三项采样时差小时 / '+clock,value='median {:.2f}; P90 {:.2f}'.format(spread.median(),spread.quantile(.9)))
            row(b,'delta_coverage','乳酸>=2且已有抗菌药 / '+clock,int(q['lactate__'+clock].ge(2).sum()),len(q),note='只有此前测量；仍不是充分复苏或完整Sepsis-3证明')
    if 'chart' in b:
        for clock in ['reported','strict']:
            code=q['code_status__'+clock].fillna('').astype(str)
            q['CMO__'+clock]=code.str.contains(r'comfort|\bcmo\b',case=False,regex=True)
            q['DNR__'+clock]=code.str.contains(r'\bdnr\b|do not resusc',case=False,regex=True)
            intub=q['intubation_status__'+clock].fillna('').astype(str)
            q['DNI__'+clock]=(code+' | '+intub).str.contains(r'\bdni\b|do not intubat',case=False,regex=True)
            q['full_code_record__'+clock]=code.str.contains(r'full code|full resusc',case=False,regex=True)
            for label in ['CMO','DNR','DNI','full_code_record']:
                row(b,'clinical_context',label+' / '+clock,int(q[label+'__'+clock].sum()),len(q),note='DNR、DNI和CMO分开；不同标志可并存，不把未知当full code')
            row(b,'clinical_context','没有可用code记录 / '+clock,int(code.eq('').sum()),len(q))
            q['MAP__'+clock]=q['MAP_arterial__'+clock].fillna(q['MAP_noninvasive__'+clock])
            q['GCS_complete__'+clock]=q[[v+'__'+clock for v in ['GCS_eye','GCS_verbal','GCS_motor']]].notna().all(axis=1)
            row(b,'chart_coverage','任一MAP / '+clock,int(q['MAP__'+clock].notna().sum()),len(q))
            row(b,'chart_coverage','GCS三个分量 / '+clock,int(q['GCS_complete__'+clock].sum()),len(q))
            # These are transparent cross-tabs, not response phenotypes or exchangeability tests.
            for action,g in q.groupby('action_state'):
                for label,mask in [('MAP<65',g['MAP__'+clock].lt(65)),('MAP已测',g['MAP__'+clock].notna()),
                                   ('乳酸已测',g['lactate__'+clock].notna() if 'labs' in b else pd.Series(False,index=g.index))]:
                    row(b,'state_support',label+' / '+clock,int(mask.sum()),len(g),stratum=action)
        for label,mask in [('医疗服务MED/OMED/NMED',q.service_before_t0.isin(['MED','OMED','NMED'])),
                           ('医疗ICU或混合MICU/SICU',q.first_careunit.astype(str).str.contains('Medical',case=False,na=False))]:
            row(b,'clinical_context',label,int(mask.sum()),len(q),note='当前服务/位置代理，不是休克病因或最终诊断')
    if 'culture_qualification' in b:
        if 'clinical_features_before_chart' not in b['culture_baseline_snapshot']:
            b['culture_baseline_snapshot']['clinical_features_before_chart']=b['clinical_features']
        extra=[c for c in b['culture_qualification'].columns if c not in b['medication_features'].columns]
        f=b['culture_qualification'][['hadm_id']+extra]
        b['culture_clinical_features']=q.merge(f,on='hadm_id',validate='one_to_one')
    b['clinical_features']=q
    save(b); report()
    say({'stage':'clinical_features','proxy_candidates_rounded':count(len(q)),'models':0,'effects':0})


def assess():
    """Descriptive support and optimistic precision bounds, never outcome fitting."""
    b=load(); allq=b['medication_features']; q=b.get('clinical_features',allq[allq.candidate_proxy]).copy()
    b['aggregates']=[x for x in b['aggregates'] if x['section'] not in ['precision','support_detail','timing','technical_upper_bound','unresolved']]
    hc=q[q.action_state.eq('HC_DOCUMENTED_WITHIN_6H')]
    waiting=q[q.action_state.eq('NO_GC_ADMIN_IDENTIFIED_CAPTURE_PRESENT')]
    row(b,'support_detail','代理候选独立患者',q.subject_id.nunique(),len(q),note='分母为医院住院；不是独立住院假设')
    row(b,'support_detail','同一患者有多次候选住院',int(q.subject_id.duplicated(keep=False).sum()),len(q),note='按患者聚类；本轮未bootstrap')
    for action,g in q.groupby('action_state'):
        for variable in ['ne_dose','relative_ne_change','age']:
            z=pd.to_numeric(g[variable],errors='coerce').dropna()
            row(b,'support_detail',variable+'已知',len(z),len(g),stratum=action)
            if len(z)>10:
                row(b,'support_detail',variable+'分布',value='median {:.3f}; IQR {:.3f}–{:.3f}'.format(z.median(),z.quantile(.25),z.quantile(.75)),stratum=action,
                    note='NE为本地显示剂量；未做未知盐基换算；不是充分复苏或治疗可互换证明')
    for variable in ['ne_dose','relative_ne_change','age']+([k+'__strict' for k in ['lactate','creatinine','platelet','bicarbonate']] if 'clinical_features' in b else []):
        aa=pd.to_numeric(hc[variable],errors='coerce').dropna(); bb=pd.to_numeric(waiting[variable],errors='coerce').dropna()
        if len(aa)>10 and len(bb)>10:
            sd=math.sqrt((aa.var(ddof=1)+bb.var(ddof=1))/2.)
            if sd>0:
                row(b,'support_detail','未调整SMD '+variable,value=round(float((aa.mean()-bb.mean())/sd),3),
                    note='HC记录路径减未检出GC且有其他给药捕获路径；缺失与未知路径保留；非PS或加权诊断')
    for action,g in q.groupby('action_state'):
        cells=pd.cut(g.ne_dose,[0,.1,.25,.5,np.inf],right=False).astype(str).fillna('dose_unknown')
        for cell,n in cells.value_counts().items():
            row(b,'support_detail','t0 NE显示剂量单元',int(n),len(g),stratum=action+' / '+cell,note='既定SOFA/指南常见界点的描述；不按结果改变资格，不推算ESS')
    for risk in [.1,.2,.3,.4]:
        for retention in [1.,.75,.5]:
            if len(hc)>10 and len(waiting)>10:
                half=1.96*math.sqrt(risk*(1-risk)*(1./(len(hc)*retention)+1./(len(waiting)*retention)))*100
                label='risk={}; information_retained={}'.format(risk,retention)
                row(b,'precision','总体RD 95%区间半宽百分点',value=round(half,2),stratum=label,
                    note='两组相同假定风险；独立住院正态近似的乐观规划；未使用实际死亡，未拟合权重，保留系数不是实际ESS')
                row(b,'precision','理想均分两亚组：每亚组RD半宽百分点',value=round(half*math.sqrt(2),2),stratum=label,
                    note='分型完全可得且两组均分的理想情景，不是实际分型结果')
                row(b,'precision','理想均分两亚组：RD交互差半宽百分点',value=round(half*2,2),stratum=label,
                    note='分型完全可得、两亚组独立且均分的情景；不把±5/±10规划目标冒充临床标准')
    if 'clinical_features' in b and 'labs' in b:
        joint=q.delta_three_positive_plausible__strict
        for action,g in q.groupby('action_state'):
            row(b,'technical_upper_bound','三项单位可解释严格时钟：记录路径人数',int(g.delta_three_positive_plausible__strict.sum()),len(g),stratum=action,
                note='只是信息规模，不以完整病例定义最终资格或拟合效果')
        nh=int((hc.delta_three_positive_plausible__strict).sum()); nw=int((waiting.delta_three_positive_plausible__strict).sum())
        if nh>10 and nw>10:
            half=1.96*math.sqrt(.3*.7*(1/nh+1/nw))*100
            row(b,'precision','完整三项子集：假定风险30%总体RD半宽百分点',value=round(half,2),note='无信息损失的乐观情景，非效应估计，不推荐complete-case主分析')
    naive=allq[allq.baseline_gc_state.eq('NO_PRIOR_GC_IDENTIFIED_CAPTURE_PRESENT')]
    for label,g in [('已有实际抗菌药代理',naive[naive.baseline_abx_actual]),('未满足该抗菌药映射',naive[~naive.baseline_abx_actual])]:
        row(b,'technical_upper_bound','六小时HC记录路径',int(g.action_state.eq('HC_DOCUMENTED_WITHIN_6H').sum()),len(g),stratum=label,
            note='后者不自动新增为感染患者；培养、途径和非感染性休克仍需核验，不能把无匹配写成未治疗')
    if len(hc)>10:
        for label,z in [('HC启动距t0小时',(hc.first_hc-hc.t0)/3600),('HC入库减报告给药分钟',hc.hc_documentation_lag_min)]:
            z=z.dropna()
            row(b,'timing',label,value='median {:.2f}; P90 {:.2f}'.format(z.median(),z.quantile(.9)),note='数据库时间差，不证明医生作出决定的时间')
        for label,mask in [('HC入库滞后>30分钟',hc.hc_documentation_lag_min.gt(30)),('HC入库早于报告给药',hc.hc_documentation_lag_min.lt(0))]:
            row(b,'timing',label,int(mask.sum()),len(hc))
    row(b,'timing','四小时NE本身有t0前入库记录',int(q.ne_recorded_by_t0.sum()),len(q),note='其余不等于临床不可见，只能说明存储时间不能替代床旁给药可见时钟')
    for label,value,note in [
        ('t0前培养采集与配对','UNKNOWN_SOURCE_NOT_LOCATED','当前已指定3.1目录未定位hosp/microbiologyevents；不是断言数据不存在；不借用1.0键'),
        ('新队列MAP、呼吸状态与code status','SOURCE_PRESENT_NOT_EXTRACTED','3.1 chartevents存在，本轮没有读取41.94GB；不能借用停药队列较晚时点缓存填补'),
        ('新队列液体输入','SOURCE_PRESENT_NOT_EXTRACTED','原始inputevents存在；已有全成人NE投影不含完整液体，不能把无缓存当0'),
        ('新队列尿量','SOURCE_PRESENT_NOT_EXTRACTED','3.1 outputevents存在；新队列尚未定向提取，不作无尿或正常处理'),
        ('院外慢性激素','UNOBSERVED','本轮能检查当前医院住院记录；不能证明院外未用药'),
        ('第一剂明确IV途径','NOT_CONFIRMED','当前HC路径中明确IV为0；其余有执行与注射盐型证据，具体途径缺失，不填IV'),
        ('原RI/RW模型治疗前迁移','NOT_RUN','没有加载公共pickle或训练模型；24h模型输入改为治疗前并不自动保持原模型定义/性能')]:
        row(b,'unresolved',label,value=value,note=note)
    b['medication_tests']=medication_tests()
    b['precision_uses_observed_outcome']=False
    b['final_entry_sha256']=sha(Path(__file__))
    services=RAW/'hosp'/'services.csv'/'services.csv'
    if services.exists():
        b['small_source_services']=dict(path=str(services),bytes=services.stat().st_size,sha256=sha(services))
    save(b); report()
    say({'stage':'assess','hospital_proxy_candidates_rounded':count(len(q)), 'HC_path_rounded':count(len(hc)),
         'other_captured_no_GC_path_rounded':count(len(waiting)),'observed_outcome_used_for_precision':False,'models':0,'effects':0})


def decide():
    b=load()
    if 'culture_clinical_features' in b:
        return decide_cultures()
    assert 'clinical_features' in b and 'labs' in b and 'chart' not in b
    q=b['clinical_features']; hc=q[q.action_state.eq('HC_DOCUMENTED_WITHIN_6H')]
    waiting=q[q.action_state.eq('NO_GC_ADMIN_IDENTIFIED_CAPTURE_PRESENT')]
    joint=q.delta_three_positive_plausible__strict
    risk=.3
    half=1.96*math.sqrt(risk*(1-risk)*(1/len(hc)+1/len(waiting)))*100
    nh=int(hc.delta_three_positive_plausible__strict.sum()); nw=int(waiting.delta_three_positive_plausible__strict.sum())
    jhalf=1.96*math.sqrt(risk*(1-risk)*(1/nh+1/nw))*100 if nh>10 and nw>10 else np.nan
    txt='''**本轮决定：CONDITIONAL GO，仅保留“总体HC启动TTE”的有限候选；不恢复δ为主要效应修饰因素，不直接进入死亡效果估计。**

已经实测的改善：现有3.1全成人NE资产可以恢复统一节点，实际给药解析也可以修复；这不是master宽表无法使用，也不代表MIMIC不能做TTE。但这次具体候选尚未完成正式立项。

|本轮关键结果|实际结果（计数约）|
|---|---:|
|连续NE四小时、节点时仍有记录支持|%s次医院住院|
|无检出院内既往GC、有t0前实际抗菌治疗代理|%s次医院住院、%s位患者|
|随后六小时HC启动记录路径|%s|
|未检出GC且有其他给药捕获的路径|%s|
|其他GC／给药记录不确定／随访eMAR捕获不足|%s／%s／%s|
|三项δ输入治疗前有数值且已入库|%s%%|
|HC路径中三项δ输入已入库且单位可解释|%s，%s%%|
|HC路径明确IV途径|0；注射盐型及实际剂量可识别，具体途径缺失|

以上是记录路径，不能把“未检出”写成确定未治疗，不能把抗菌药代理写成确诊感染性休克，也不是CCW策略样本量或ESS。当前人群、数据版本和节点均与旧ED一小时方案不同，不能直接把输入可用率相减作为改善幅度；输入可用也不等于δ标签已验证。

**为何暂不恢复分型主线：**在假定两路径风险均为30%%、未损失任何信息的乐观规划下，总体RD区间半宽约±%.2f个百分点；保留一半信息时约±%.2f。即使分型完全可得且两亚组理想均分，每亚组约±%.2f，两个RD之差约±%.2f。实际三项输入齐全子集的总体RD半宽已约±%.2f，尚未再按分型拆分。±5/±10只是用户指定并列规划情景，不是事后设定的临床或人数门槛。缺失插补能处理部分测量问题，不能增加独立接受HC的患者信息。

**总体比较的主要风险：**未调整NE剂量SMD为1.139、乳酸0.638、碳酸氢盐−0.688，表明有明显治疗选择差异；这不是已证明不能调整，也不是共同支持已经通过。MAP、呼吸状态和治疗限制源表存在，但尚未为新节点读取。当前培养资格未核验，抗菌药可能包括预防性使用及其他休克病因；院外激素史也未观测。具体IV途径缺失，若研究坚持IV-only，则当前不能声称该定义已核实。

**相对已有论文的增量候选：**把临床问题明确为初始NE支持后的同一复评节点、只使用激素前状态、区分实际启动与记录捕获，研究早期循环支持需求是否修饰启动选择。这比重用“激素×分型”的方法名称具体，但尚未证明有足够科学增量；Nature2025及Toppin2026已覆盖分型/指标×激素研究，不能把换3.1、换δ或换结局当主要创新。当前不承诺期刊层级。

**唯一允许继续的有限步骤：**先用同版培养采集表，核验固定t0前感染评估配对是否已完成，给出同一窗口下的资格与行动数；未满足原抗菌药映射的约50条HC技术路径只是待核验上界，不能直接加进样本。只有资格和信息量仍值得投入，才对现有chartevents作一次限定字段提取，检查MAP、呼吸状态及独立的DNR/DNI/CMO，并完成固定调整集的支持度判断。保持4h/6h和主要结局，不为增加行动改窗口，不搜索阈值、模型或效果。

**当前在此停止新增计算。**未定位培养表时，不等待、不下载全库，也不继续读取41.94GB chartevents。培养文件的最小需求为MIMIC-IV3.1 `hosp/microbiologyevents.csv(.gz)`；若文件另存于嵌套同名目录也可。需要subject/hadm、标本ID、采集charttime/chartdate、标本/检验类型及storetime；资格只用采集时刻，不要求未来培养阳性。当前获取体积未核实，标UNKNOWN，取得路径后先读表头和实际文件体积。若资格明显不足或后续固定诊断仍由少数行动主导，就结束这个版本；不自动重开旧协议或第三个题目。
''' % (count(len(b['candidates'])),count(len(q)),count(q.subject_id.nunique()),count(len(hc)),count(len(waiting)),
       count(q.action_state.eq('OTHER_GC_DOCUMENTED_WITHIN_6H').sum()),count(q.action_state.eq('GC_RECORD_UNCERTAIN_WITHIN_6H').sum()),
       count(q.action_state.eq('FOLLOWUP_EMAR_CAPTURE_INSUFFICIENT').sum()),str(pct(int(joint.sum()),len(q))),
       count(nh),str(pct(nh,len(hc))),half,half*math.sqrt(2),half*math.sqrt(2),half*2,jhalf)
    b['decision_text']=txt; b['decision_utc']=utc(); b['decision']='CONDITIONAL_GO_LIMITED_QUALIFICATION_ONLY'
    b['raw_scans_stopped_after_labs']=True; b['final_entry_sha256']=sha(Path(__file__))
    save(b); report()
    say({'stage':'decide','decision':b['decision'],'formal_effect_analysis':False,'delta_as_primary_modifier':False,'chartevents_scan':False})


def report():
    b=load(); public=[]
    for r in b['aggregates']:
        public.append(dict(section=r['section'],metric=r['metric'],stratum=r['stratum'],
                           n='' if r['exact_n'] is None else count(r['exact_n']),
                           denominator='' if r['exact_d'] is None else count(r['exact_d']),
                           percent='' if r['exact_n'] is None or r['exact_d'] is None else pct(r['exact_n'],r['exact_d']),
                           value=r['value'],note=r['note'],
                           evidence_state='PLANNING_ASSUMPTION_NOT_EFFECT' if r['section'] in ['precision','culture_precision'] else (r['value'] if r['section']=='unresolved' or r['value']=='SOURCE_PRESENT_NOT_EXTRACTED' else 'DATA_VALIDATED_AGGREGATE')))
    table=pd.DataFrame(public)
    assert not set(table.columns)&{'subject_id','hadm_id','stay_id','emar_id'}
    table.to_csv(TABLE,index=False,encoding='utf-8-sig')
    title,body=PROTOCOL.split('\n',1)
    if 'culture_supplement' in b:
        body=body.replace('同版培养表目前未定位。','同版培养表已取得；额外的基线配对核验见本轮结果，未改变4h/6h或实际药物映射。')
    text=title+'\n\n'+(b['decision_text']+'\n\n' if 'decision_text' in b else '')+body+'\n## 当前实际结果\n\n'
    text+='规则保存UTC：`'+b['frozen_utc']+'`；签名：`'+b['protocol_signature']+'`。计数1–10抑制，其余取整到十位，百分比按实际分母；不反算小单元。\n\n'
    if 'culture_supplement' in b:
        text+='培养补充保存UTC：`'+b['culture_supplement']['utc']+'`；补充签名：`'+b['culture_supplement']['signature']+'`。原规则及培养补充前报告、聚合表和决定均在同一private缓存留存。\n\n'
    sections=list(table.section.drop_duplicates())
    if 'culture_supplement' in b:
        sections=[x for x in sections if x.startswith('culture_')]+[x for x in sections if not x.startswith('culture_')]
    original_heading_added=False
    for section in sections:
        if 'culture_supplement' in b and not section.startswith('culture_') and not original_heading_added:
            text+='## 原始技术队列与抗菌药代理分母\n\n以下flow为全部4h技术候选，actions、lab_coverage、chart_coverage等为未加培养限制的原抗菌药代理分母；上文culture表为培养配对分母，两者不相互替代。unresolved列出当前字段状态。\n\n'
            original_heading_added=True
        g=table[table.section.eq(section)]
        text+='### '+section+'\n\n|检查|分层|人数约|分母约|百分比|数值|\n|---|---|---:|---:|---:|---|\n'
        for r in g.to_dict('records'):
            text+='|'+ '|'.join(str(r[k]) for k in ['metric','stratum','n','denominator','percent','value'])+'|\n'
        text+='\n'
    if 'culture_supplement' in b:
        text+='## 培养核验的范围与来源\n\n'+CULTURE_PROTOCOL+'\n\n'
        text+='[MIMIC官方培养表说明](https://mimic.mit.edu/docs/iv/modules/hosp/microbiologyevents.html)区分采集时刻、日期精度及最后结果更新；同一标本可产生多行。本次按标本折叠，无法解释的键/时间冲突保留未知；缺hadm记录只在同患者完整住院区间唯一匹配时补链，未链接不表示未采样。\n\n'
        text+='72h/24h范围同时见既有XLab定义与[官方suspected infection代码](https://github.com/MIT-LCP/mimic-code/blob/main/mimic-iv/concepts/sepsis/suspicion_of_infection.sql)。本研究使用实际抗菌药、要求采集与给药均早于t0、分开诊断培养与筛查，并用日期区间而非零点处理缺时间；因此是基线可完成的感染评估代理，不宣称原样复现官方回顾性Sepsis-3。对同时记录且均早于t0的采样/给药，作为配对保留；这也是明确的本地操作化选择。\n\n'
        text+='培养采集仍为结构化时间代理，未逐例确认成功采样；本次没有用后来取消或培养结果筛选基线。对未链接或无目标培养的记录，结论限于未确定配对，不能断言没有感染评估。\n\n'
        if 'culture_tests' in b:
            text+='新增培养配对与检验类型边界检查'+str(len(b['culture_tests']))+'项通过；源表只读过一次完整投影，未读取菌名、药敏或comments。\n\n'
        if 'chart' in b:
            text+='[MIMIC床旁记录说明](https://mimic.mit.edu/docs/iv/modules/icu/chartevents.html)将charttime解释为观察代理时刻、storetime解释为人工录入或核验时刻；二者都不能直接证明医生首次获知。本次保留两种口径，不依覆盖率选时钟；一般Code Status与Intubation Code Status分别取值，DNR/DNI/CMO不合并。6h后续记录与基线分开，不能借后续CMO或治疗结果重定资格。\n\n'
        if 'culture_quality_checks' in b:
            ck=b['culture_quality_checks']
            text+='NE触发和点剂量表示t0当时输注状态；约'+count(ck['ne_record_start_at_t0'])+'次配对住院在t0恰有NE记录切换，未把边界状态称为严格提前测量。HC与t0同刻的记录为'+count(ck['HC_exactly_at_t0'])+'。检验和床旁基线时间戳严格早于t0的独立检查通过，精确培养时间的独立配对实现与主入口一致；这些检查不证明医生决策时间已被观测。\n\n'
    text+='## 参照论文与代码：来源核对\n\n'
    text+='[Rajendran等，Nature Communications 2025](https://doi.org/10.1038/s41467-025-59643-z)已有分型×激素TTE，不能把相同框架当新创新。其激素窗口为ICU前10小时至后24小时，预测分型输入取入ICU后24小时；作者承认治疗可能先于分组并影响预测轨迹。判定CONFIRMED。72小时轨迹是分型建立目标，不能误写为所有患者直接用未来72小时资料作治疗前预测。\n\n'
    text+='[公开处方notebook，固定commit](https://github.com/surajraj99/Corticosteroids-in-Patients-with-Critical-Illness/blob/e96952407e38b52807ecd521372eeb005c993f1f/data/MIMIC/scipts/MIMIC%20Steroid%20Use%20-%20Prescription.ipynb)读取prescriptions及处方日剂量，非eMAR；只就该脚本而言CONFIRMED。把它概括为已确认实际给药被REJECTED；未检查的下游处理不能据此一并否定。\n\n'
    text+='[SSC2026官方推荐](https://www.sccm.org/clinical-resources/guidelines/guidelines/surviving-sepsis-campaign-international-guidelines-for-management-of-sepsis-and-septic-shock-2026)为成人感染性休克建议IV激素，conditional、low certainty；公开推荐页未规定本候选四小时/六小时，不能称其为2026硬性方案。2026全文时点细节当前BLOCKED。[2021原始指南](https://doi.org/10.1007/s00134-021-06506-y)的剂量与持续时长有明确条件，不能简化为任何剂量NE四小时；错误等同2026阈值被REJECTED。独立来源核对3组，拒绝2种过度配对。\n\n'
    text+='[Toppin等，ICM Experimental 2026](https://doi.org/10.1186/s40635-026-00972-8)已研究MIMIC3.1激素与血细胞指标分层；其定义含ICU住院≥24h、乳酸资格与主要激素窗口延至ICU出院，另有24h暴露敏感性。与本候选的前瞻式共同节点不同。原文没有提供足以确认eMAR实现的提取代码；已核对正文和唯一补充文件，没有定位本研究开源仓库，不能把引用通用mimic-code当作本研究开源。当前差异只提供可检验的改进方向，不证明新颖性成立。\n\n'
    text+='[ADRENAL原始RCT](https://doi.org/10.1056/NEJMoa1705835)入组要求升压/强心药至少四小时且随机时仍在使用，另有机械通气、感染/SIRS及其他入排条件。本候选只借用初始支持后复评的理由，不是完整复制ADRENAL；其完整附录的细化排除本次未取得，保留UNKNOWN。\n\n'
    text+='[MIMIC eMAR官方说明](https://mimic.mit.edu/docs/iv/modules/hosp/emar.html)及[明细说明](https://mimic.mit.edu/docs/iv/modules/hosp/emar_detail.html)区分实际执行、due与formulary子行；本次未把父行应给剂量与子行实际剂量相加。官方说明系统于2011–2013实施，不能把本报告用于描述的2016边界误称为系统上线日期。患者anchor组先加t0年份差得到住院年份区间；具体历年存在区间不确定，不把患者anchor组直接当入院年。\n\n'
    if 'named_salt_revision' in b:
        rev=b['named_salt_revision']
        text+='## 一次有字段依据的解析修正\n\n修正保存UTC：`'+rev['utc']+'`。旧映射把许多药名明确为Hydrocortisone Na Succ.、执行状态Administered、子行正数mg，但route/product_description为空的事件全部记为未知。本次允许**明确注射盐型药名＋实际执行状态＋正数mg子行＋无冲突**作为系统性注射制剂给药证据，具体IV/IM仍未知；没有填造route或product字段，也没有更改4h/6h、目标人群或结局。研究药、冲突、仅医嘱/未执行及泛称激素仍不确定。原映射患者表和聚合结果保存在同一private缓存中。\n\n'
        text+='剂型依据：[DailyMed hydrocortisone sodium succinate](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=451d6d04-b7d9-47d9-ba71-43cc5b9b3054)；[methylprednisolone sodium succinate](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=77a8d96c-37df-4f3a-9757-b2d4131ff82b)。外部标签支持药物剂型知识，不证明本地厂家、具体途径或感染适应证。\n\n'
        if 'medication_features' in b:
            old=rev['prior_medication_features']; old=old[old.candidate_proxy]
            now=b['medication_features']; now=now[now.candidate_proxy]
            text+='|映射|代理候选约|六小时HC路径约|\n|---|---:|---:|\n'
            text+='|原保守剂型映射|{}|{}|\n'.format(count(len(old)),count(old.action_state.eq('HC_DOCUMENTED_WITHIN_6H').sum()))
            text+='|药名盐型证据补充后|{}|{}|\n\n'.format(count(len(now)),count(now.action_state.eq('HC_DOCUMENTED_WITHIN_6H').sum()))
    text+='## 已完成读取与成本\n\n|步骤|实际GB|秒|峰值MB|\n|---|---:|---:|---:|\n'
    for r in b['runs']:
        cost=('{:.3f}以上，未完成读取'.format(r['bytes_read_lower_bound']/1e9) if r.get('read_cost_type')=='INCOMPLETE_PASS_LOWER_BOUND' else '{:.3f}'.format(r.get('bytes_read',0)/1e9))
        text+='|{}|{}|{}|{}|\n'.format(r['stage'],cost,r.get('seconds',r.get('elapsed_s','')),r.get('peak_MB',''))
    text+='\n完整文件哈希、字段、读取行数、native键检查、源版本、协议、运行记录及解析事件留在本机private缓存；不含任何1.0混接。当前没有结局效应、PS或权重模型。\n'
    text+='\n首次节点的六项边界检查与实际给药十项检查通过；同时验证了映射修正前后候选、t0及代理资格完全一致。旧停药分析、冻结结果和暂停决定未改。新工作只创建一个入口、一份报告和一张聚合表，患者缓存仅在本机private中。\n'
    if 'decision_text' not in b:
        text+='\n## 当前阶段\n\n验证进行中；尚未宣布临床资格、因果识别或正式TTE通过。不得由技术人数推算有效样本量。\n'
    if (ROOT/'steroid_policy.py').exists() and (LOCAL/'HTE_sepsis'/'private'/'steroid_policy.pkl.gz').exists():
        policy=import_file('hc_policy_report',ROOT/'steroid_policy.py')
        text,table=policy.decorate(text,table)
        table.to_csv(TABLE,index=False,encoding='utf-8-sig')
    if (ROOT/'steroid_sequential.py').exists() and (LOCAL/'HTE_sepsis'/'private'/'steroid_sequential.pkl.gz').exists():
        sequential=import_file('hc_sequential_report',ROOT/'steroid_sequential.py')
        text,table=sequential.decorate(text,table)
        table.to_csv(TABLE,index=False,encoding='utf-8-sig')
    # Preserve the stored historical report suffix without newline conversion.
    REPORT.write_bytes(text.encode('utf-8'))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',required=True,choices=['prepare','emar','emar_detail','revise_med_mapping','med_features','cultures','culture_features','culture_assess','culture_support','culture_decide','labs','chart','clinical_features','assess','decide','report','policy_freeze','policy_build','policy_diagnose','policy_decide','policy_scope','policy_report','sequential_prepare','sequential_validate_source','sequential_emar','sequential_emar_detail','sequential_micro','sequential_resolve','sequential_candidates','sequential_risksets','sequential_coverage','sequential_checks','sequential_clinical_freeze','sequential_full_risksets','sequential_new_labs','sequential_new_chart','sequential_cached_history_features','sequential_analysis_freeze','sequential_build_clinical_features','sequential_support_diagnostics','sequential_control_followup_description','sequential_exploratory','sequential_precision_freeze','sequential_precision_primary','sequential_precision_strict','sequential_precision_merge','sequential_precision_summarize','sequential_finalize_clinical'])
    args=ap.parse_args()
    say({'python':sys.executable,'stage':args.stage,'protocol_signature':signature(),'outcome_models':0})
    if args.stage in ['emar','emar_detail']: scan(args.stage)
    elif args.stage=='prepare': prepare()
    elif args.stage=='med_features': med_features()
    elif args.stage=='revise_med_mapping': revise_med_mapping()
    elif args.stage=='cultures': scan_cultures()
    elif args.stage=='culture_features': culture_features()
    elif args.stage=='culture_assess': assess_cultures()
    elif args.stage=='culture_support': culture_clinical_support()
    elif args.stage=='culture_decide': decide_cultures()
    elif args.stage in ['labs','chart']: scan_clinical(args.stage)
    elif args.stage=='clinical_features': clinical_features()
    elif args.stage=='assess': assess()
    elif args.stage=='decide': decide()
    elif args.stage=='report': report()
    elif args.stage.startswith('sequential_'):
        sequential=import_file('hc_sequential_entry',ROOT/'steroid_sequential.py')
        stage=args.stage[len('sequential_'):]
        if stage in ['emar','emar_detail','micro']:sequential.scan(stage)
        else:getattr(sequential,stage)()
    elif args.stage.startswith('policy_'):
        policy=import_file('hc_policy_entry',ROOT/'steroid_policy.py')
        stage=args.stage[len('policy_'):]
        if stage=='report': policy.render(ROOT)
        else: getattr(policy,stage)()


if __name__=='__main__':
    main()
