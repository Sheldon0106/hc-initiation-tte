# -*- coding: utf-8 -*-
"""Earlier HC risk sets; qualification, fixed support and guarded exploratory ATT."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,hashlib,importlib.util,io,json,math,re,time
from pathlib import Path
from hc_paths import configured_path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
LOCAL = configured_path('legacy_workspace', '../restricted/legacy-workspace')
STATE = configured_path('private_dir', '../restricted/hc-private') / 'steroid_sequential.pkl.gz'
BASE_STATE = configured_path('private_dir', '../restricted/hc-private') / 'steroid_tte.pkl.gz'
CONFIG=dict(version='MIMIC-IV3.1 only',source='all adult observed NE; no future duration requirement',
    candidate_decision_horizon_h=24,medication_followup_extension_h=6,
    prior_GC='entire observed current admission before each index; uncertain separate',
    baseline_infection='actual antibacterial in prior24h plus diagnostic culture paired72h/24h, both strictly preindex',
    action='first actual systemic injectable-formulation-compatible HC; IV/IM unknown stays unknown',
    index='actual HC record; controls at identical elapsed time since own first observed ICU NE',
    comparator='not initiating at that index; subsequent treatment usual care, not six-hour deferral',
    estimand_candidate='treated-decision ATT; not population ATE',
    control_sampling='up to20 random eligible other-patient admissions per treated decision, seed521; descriptive field-projection panel only',
    outcome='current-admission hospital mortality; not accessed for contrasts in this phase',
    model_fitting=False,effect_estimation=False)
PROTOCOL='''## 较早HC启动：新来源与时间合同

本候选比较实际首次启动HC的决策与当时不启动、随后通常照护。目标首先限定在实际启动者所代表的决策人群（ATT），不宣称所有NE患者均应启动。候选索引处于首次可观察ICU NE起点后首24小时；这是本次独立保存的首日范围选择，不是指南硬性时限，不要求NE连续使用、ICU住院或存活满24小时。NE记录可短于4小时；源资格不再使用旧4h候选入口。

索引取实际HC记录时间T；未启动者在其自身首次观察NE后相同经过时长取T。资格和调整均只用T前信息；NE取左极限。两组当时均需有正数NE记录及已完成的实际抗菌药和诊断培养配对，院内更早实际GC及不确定GC分别处理。同秒混合激素不作为清楚的HC新启动。院外激素和真实休克起点仍未知。后续HC不改变当前分配；不预先删除后来启动或早死者。

当前不启动并不承诺至少等待1小时。本轮将记录随后15/30/60分钟的给药路径；若大量立即交叉，估计目标可能仅代表短暂推迟，须先评价临床意义。若以后需要至少等待1小时，必须另写该策略及相应删失，不把未来行为直接当基线。记录时刻不保证医生实际决定时刻；索引前15/30分钟信息截断属于待评估的时间误差问题。

先补齐源范围和资格。药物从入院至NE起点后30小时投影；培养从NE起点前96小时至后24小时投影，结果不参与资格。重复住院、重复对照和可先作对照后启动者均保留关联。本轮仅保存每个启动时点的可用风险集人数及固定随机抽取的字段检查面板，后者不是平衡匹配、不是最终估计样本，更不是新增独立患者。正式估计前还需调整集及支持诊断。

原始方法参照：[Li等，Balanced Risk Set Matching](https://doi.org/10.1198/016214501753208573)；借用只用截至各时点历史、允许对照以后治疗的设计原则，不声称原文已验证HC场景。首日范围也受已有激素研究24h分析启发，不能照搬其未来资格或24h存活限制。
'''

def base():
    s=importlib.util.spec_from_file_location('hc_original',ROOT/'steroid_tte.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for v in iter(lambda:f.read(1048576),b''):h.update(v)
    return h.hexdigest()
def signature():return hashlib.sha256((PROTOCOL+json.dumps(CONFIG,sort_keys=True)).encode('utf-8')).hexdigest()
def say(x):print(json.dumps(x,ensure_ascii=False,default=str),flush=True)
def load():
    b=pd.read_pickle(STATE);assert b['signature']==signature();assert b['models_fitted']==b['effects_estimated']==0;return b
def save(b):
    STATE.parent.mkdir(parents=True,exist_ok=True);pd.to_pickle(b,STATE,compression={'method':'gzip','compresslevel':1})
def row(b,sec,metric,n=None,d=None,value='',stratum='all',note=''):
    b['aggregates'].append(dict(section='seq_'+sec,metric=metric,exact_n=n,exact_d=d,value=value,stratum=stratum,note=note))
def parser(m):return m.import_file('hc_seq_parser',m.OLD/'withdrawal_association'/'supplement_analysis.py')

def prepare():
    if STATE.exists():load();say(dict(stage='sequential_prepare',reuse=True));return
    begin=time.perf_counter();m=base();f=m.load();assert sha(m.SOURCE)==f['source_cache_sha256'];src=pd.read_pickle(m.SOURCE)
    gate=m.import_file('hc_seq_gate',m.OLD/'clinical_tte_gate.py')
    ne=gate.normalize(src['events'][src['events'].itemid.eq(221906)],src['icustays'])
    ne=ne[ne.rate.gt(0)].copy();meta=src['adult_stays'].copy()
    ad=src['admissions'].copy();ad['admit']=m.sec(ad.admittime);ad['discharge']=m.sec(ad.dischtime);ad['death_time']=m.sec(ad.deathtime)
    meta=meta.merge(ad.drop(columns='subject_id'),on='hadm_id',validate='many_to_one');refs=meta.set_index('stay_id')
    rows=[]
    for stay,g in ne.groupby('stay_id',sort=False):
        if stay not in refs.index:continue
        c=refs.loc[stay];lo=max(c.icu_in,c.admit);hi=min(c.icu_out,c.discharge)
        gg=g[g.end.gt(lo)&g.start.lt(hi)]
        if not len(gg):continue
        t=max(float(gg.start.min()),lo)
        if np.isfinite(c.death_time) and c.death_time<=t:continue
        rows.append(dict(subject_id=int(c.subject_id),hadm_id=int(c.hadm_id),first_NE_stay=int(stay),ne0=t,
            age=c.age,gender=c.gender,first_careunit=c.first_careunit,admit=c.admit,discharge=c.discharge,
            death_time=c.death_time,hospital_expire_flag=c.hospital_expire_flag,
            ne_start_left_censored=bool(gg.start.min()<lo),anchor_year=c.anchor_year,anchor_year_group=c.anchor_year_group))
    q=pd.DataFrame(rows).sort_values(['ne0','first_NE_stay']).drop_duplicates('hadm_id').reset_index(drop=True)
    q['med_end']=np.minimum(q.ne0+30*3600,q.discharge)
    adm=ad[ad.subject_id.isin(q.subject_id)][['subject_id','hadm_id','admit','discharge']].copy()
    keep=['subject_id','hadm_id','stay_id','start','end','store','rate','dose','rateuom','patientweight','linkorderid','orderid','statusdescription']
    ne=ne[ne.hadm_id.isin(q.hadm_id)][keep].copy()
    plans={}
    for name,p,cols in [('emar',m.PATHS['emar'],m.COLS['emar']),('emar_detail',m.PATHS['emar_detail'],m.COLS['emar_detail']),('micro',m.CULTURE_PATH,m.CULTURE_COLS)]:
        st=p.stat();assert set(cols)<=set(pd.read_csv(p,nrows=0).columns)
        plans[name]=dict(path=str(p),bytes=st.st_size,mtime_ns=st.st_mtime_ns,columns=cols)
    b=dict(signature=signature(),protocol=PROTOCOL,config=CONFIG,frozen_utc=m.utc(),source=q,NE=ne,admission_link=adm,
        definitions=f['definitions'],plans=plans,aggregates=[],runs=[],models_fitted=0,effects_estimated=0,
        source_sha256=f['source_cache_sha256'],feasibility_sha256=sha(BASE_STATE),base_snapshots={},
        dependencies=f['dependencies'],stage='SOURCE_REBUILT_NO_FUTURE4H_REQUIREMENT')
    for name in ['steroid_tte.py','steroid_tte.md','steroid_tte_aggregates.csv']:
        b['base_snapshots'][name]=dict(sha256=sha(ROOT/name),text=(ROOT/name).read_text(encoding='utf-8-sig'))
    row(b,'source','全来源首次观察NE的成人医院住院',len(q));row(b,'source','独立患者',q.subject_id.nunique())
    row(b,'source','旧4h技术队列之外住院',int((~q.hadm_id.isin(f['candidates'].hadm_id)).sum()),len(q),note='只是新来源，不是新增合格治疗病例')
    row(b,'source','NE起点在ICU/入院观察边界被左截断',int(q.ne_start_left_censored.sum()),len(q),note='首次可观察，不称真实首次休克/给药')
    b['runs'].append(dict(stage='prepare',utc=m.utc(),seconds=round(time.perf_counter()-begin,2),bytes_read=0,peak_MB=m.peak_mb()))
    save(b);say(dict(stage='sequential_prepare',source_rounded=m.count(len(q)),large_raw_scans=0,models=0,planned_GB=round(sum(v['bytes'] for v in plans.values())/1e9,2)))

def validate_source():
    b=load();m=base()
    if 'ICU_boundaries_verified' in b:say(dict(stage='sequential_source_boundaries',reuse=True));return
    src=pd.read_pickle(m.SOURCE);bounds=src['adult_stays'][['stay_id','icu_in','icu_out']].set_index('stay_id')
    g=b['NE'].copy();before=len(g)
    g['source_start']=g.start;g['source_end']=g.end
    g['icu_lo']=g.stay_id.map(bounds.icu_in);g['icu_hi']=g.stay_id.map(bounds.icu_out)
    refs=b['source'].set_index('hadm_id')
    g['start']=np.maximum(g.start,np.maximum(g.icu_lo,g.hadm_id.map(refs.admit)))
    g['end']=np.minimum(g.end,np.minimum(g.icu_hi,g.hadm_id.map(refs.discharge)))
    changed=g.start.ne(g.source_start)|g.end.ne(g.source_end)
    g=g[g.start.notna()&g.end.gt(g.start)].copy()
    assert g.start.ge(g.icu_lo).all() and g.end.le(g.icu_hi).all()
    b['NE']=g;b['ICU_boundaries_verified']=dict(utc=m.utc(),reason='Explicit ICU observation boundary at every index; source record interval not ward evidence',source_intervals=before,clipped_intervals=int(changed.sum()),retained_intervals=len(g))
    row(b,'source','ICU/住院边界裁剪的NE区间',int(changed.sum()),before,note='源区间数不是患者数；不将ICU外延伸称ward给药')
    save(b);say(dict(stage='sequential_source_boundaries',passed=True,raw_scans=0))

def assign_missing(z,q,adm,time_col,hi_col=None,groups=None):
    """Full admission intervals, not candidate-only uniqueness."""
    if not z.hadm_id.isna().any():return z
    ag=groups if groups is not None else {s:g for s,g in adm.groupby('subject_id',sort=False)};target=set(q.hadm_id)
    for ix,r in z[z.hadm_id.isna()].iterrows():
        lo=r[time_col];hi=r[hi_col] if hi_col else lo
        if not np.isfinite(lo) or not np.isfinite(hi):continue
        a=ag.get(r.subject_id)
        if a is None:continue
        possible=a[a.admit.le(hi)&a.discharge.gt(lo)];full=possible[possible.admit.le(lo)&possible.discharge.gt(hi)]
        if len(possible)==1 and len(full)==1 and full.hadm_id.iloc[0] in target:
            z.at[ix,'hadm_id']=full.hadm_id.iloc[0];z.at[ix,'join_method']='unique_full_admission_interval'
    return z

def scan(name):
    b=load();m=base()
    if name in b:say(dict(stage='sequential_'+name,reuse=True));return
    if name=='emar_detail':assert 'emar' in b
    p=b['plans'][name];path=Path(p['path']);st=path.stat();assert st.st_size==p['bytes'] and st.st_mtime_ns==p['mtime_ns']
    q=b['source'];ref=q.set_index('hadm_id');subs=set(q.subject_id);adms=set(q.hadm_id)
    links=None if name=='emar_detail' else {s:g for s,g in b['admission_link'].groupby('subject_id',sort=False)}
    if name=='emar':
        supp=parser(m);gc=re.compile(supp.STEROID_PATTERN,re.I)
        abx=re.compile('|'.join(map(re.escape,b['definitions']['ANTIBIOTICS'])),re.I);exclude=re.compile(b['definitions']['ANTIBIOTIC_EXCLUDE_NAME'],re.I)
        capture=pd.Series(np.inf,index=ref.index);ids=None
    elif name=='emar_detail':ids=set(b['emar'].emar_id)
    begin=time.perf_counter();last=begin;parts=[];total=0;conflicts=0;unlinked=0
    reader=m.HashReader(path);stream=io.BufferedReader(reader,buffer_size=4*1024*1024)
    try:
        for raw in pd.read_csv(stream,usecols=p['columns'],dtype=str,chunksize=150000):
            total+=len(raw)
            if name=='emar_detail':z=raw[raw.emar_id.isin(ids)].copy()
            else:
                h=pd.to_numeric(raw.hadm_id,errors='coerce');s=pd.to_numeric(raw.subject_id,errors='coerce')
                z=raw[h.isin(adms)|(h.isna()&s.isin(subs))].copy()
                if len(z):
                    z['hadm_id']=h.loc[z.index];z['subject_id']=s.loc[z.index];z['join_method']='native_hadm'
                    if name=='emar':z['chart']=m.sec(z.charttime);z['store']=m.sec(z.storetime);z=assign_missing(z,q,b['admission_link'],'chart',groups=links)
                    else:
                        z['collection']=m.sec(z.charttime);z['collection_day']=m.sec(z.chartdate)
                        z['collection_lo']=z.collection.fillna(z.collection_day);z['collection_hi']=z.collection.where(z.collection.notna(),z.collection_day+86399)
                        z['result_store']=m.sec(z.storetime);z=assign_missing(z,q,b['admission_link'],'collection_lo','collection_hi',groups=links)
                    unlinked+=int(z.hadm_id.isna().sum());z=z[z.hadm_id.isin(adms)].copy()
                    good=z.subject_id.eq(z.hadm_id.map(ref.subject_id));conflicts+=int((~good).sum());z=z[good].copy()
                    if name=='emar':
                        z=z[z.chart.ge(z.hadm_id.map(ref.admit))&z.chart.le(z.hadm_id.map(ref.med_end))].copy()
                        co=z.groupby('hadm_id').chart.min();capture.loc[co.index]=np.minimum(capture.loc[co.index],co)
                        names=z.medication.fillna('');z['is_steroid_name']=names.str.contains(gc)
                        z['is_abx_name']=names.str.contains(abx)&~names.str.contains(exclude)
                        # Antibacterial acquisition covers every possible index's prior24h.
                        z=z[z.is_steroid_name|(z.is_abx_name&z.chart.ge(z.hadm_id.map(ref.ne0)-86400))].copy()
                    else:
                        lo=z.hadm_id.map(ref.ne0)-96*3600;hi=z.hadm_id.map(ref.ne0)+24*3600
                        z=z[(z.collection_hi.ge(lo)&z.collection_lo.lt(hi))|z.collection_lo.isna()].copy()
                    z['hadm_id']=z.hadm_id.astype('int64');z['subject_id']=z.subject_id.astype('int64')
            if len(z):parts.append(z)
            now=time.perf_counter()
            if now-last>=30:
                say(dict(stage='sequential_'+name,percent_bytes=round(100*reader.n/p['bytes'],1),seconds=round(now-begin),peak_MB=m.peak_mb()));last=now
        assert reader.n==p['bytes'];digest=reader.h.hexdigest()
    finally:stream.close()
    assert path.stat().st_size==p['bytes'] and path.stat().st_mtime_ns==p['mtime_ns'];assert conflicts==0
    z=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=p['columns']);z=z.drop_duplicates().reset_index(drop=True)
    if name=='micro':assert not z.microevent_id.duplicated().any()
    b[name]=z
    if name=='emar':b['capture_first']=capture.replace(np.inf,np.nan)
    p.update(sha256=digest,parsed_to_EOF=True,rows_traversed=total,rows_retained=len(z),unlinked_rows=unlinked,native_key_conflicts=conflicts)
    b['runs'].append(dict(stage=name,utc=m.utc(),seconds=round(time.perf_counter()-begin,2),bytes_read=p['bytes'],peak_MB=m.peak_mb()))
    save(b);say(dict(stage='sequential_'+name,completed=True,retained_rounded=m.count(len(z)),seconds=b['runs'][-1]['seconds'],peak_MB=m.peak_mb()))

def resolve():
    b=load();m=base();assert all(k in b for k in ['emar','emar_detail','micro'])
    if 'resolved_medications' in b:say(dict(stage='sequential_resolve',reuse=True));return
    begin=time.perf_counter();supp=parser(m);e=b['emar'];d=b['emar_detail'];headers=e.drop_duplicates('emar_id').set_index('emar_id')
    assert d.subject_id.eq(d.emar_id.map(headers.subject_id).astype(str)).all()
    assert pd.to_numeric(d.emar_seq).eq(pd.to_numeric(d.emar_id.map(headers.emar_seq))).all()
    conflicts=set(e.loc[e.duplicated('emar_id',keep=False),'emar_id']);groups={k:g for k,g in d.groupby('emar_id',sort=False)};rows=[];last=begin
    for i,ev in enumerate(e.drop_duplicates('emar_id').to_dict('records')):
        detail=groups.get(ev['emar_id']);adapted=dict(ev)
        if str(ev['event_txt']).strip().lower()=='confirmed administered':adapted['event_txt']='Administered'
        rr=m.resolve_gc(adapted,detail,supp,True) if ev['is_steroid_name'] else m.resolve_abx(ev,detail,supp,b)
        if ev['emar_id'] in conflicts:rr.update(status='UNCERTAIN',reason='conflicting_header')
        prod='' if detail is None else detail.product_description.fillna('').str.cat(sep=' | ')
        rec={k:ev[k] for k in ['subject_id','hadm_id','emar_id','chart','store','is_steroid_name','is_abx_name']}
        rec.update(rr);rec.update(hydrocortisone=bool(re.search(r'hydrocortis|solu.?cort|cortef',str(ev['medication']),re.I)),
            glucocorticoid=bool(ev['is_steroid_name'] and supp.drug_class(ev['medication'])=='glucocorticoid'),
            header_named_injectable_salt=m.named_injectable_salt(ev['medication']),
            injectable_hc_formulation=bool(re.search(r'hydrocortisone\s+(?:na|sodium)\s+succ',prod,re.I) and re.search(r'vial|syringe|\bmg\s+in',prod,re.I)))
        rows.append(rec)
        if time.perf_counter()-last>=30:
            say(dict(stage='sequential_resolve_drugs',percent=round(100*(i+1)/len(e),1),seconds=round(time.perf_counter()-begin),peak_MB=m.peak_mb()));last=time.perf_counter()
    r=pd.DataFrame(rows);r['actual_systemic']=r.status.isin(['ADMIN_DOCUMENTED','ADMIN_SYSTEMIC_FORMULATION_ROUTE_UNKNOWN'])
    iv={'IV','IVP','IV PUSH','IV DRIP','IVPB','INTRAVENOUS'}
    r['IV_confirmed']=r.route.map(lambda x:bool(set(str(x).split('|'))<=iv))&r.actual_systemic
    r['hc_action']=r.actual_systemic&r.hydrocortisone&(r.IV_confirmed|(r.route.eq('UNKNOWN')&(r.header_named_injectable_salt|r.injectable_hc_formulation)))
    b['resolved_medications']=r
    raw=b['micro'].copy();raw['kind']=[m.culture_kind(s,t) for s,t in zip(raw.spec_type_desc.fillna(''),raw.test_name.fillna(''))]
    specimens=[]
    for sid,g in raw.groupby('micro_specimen_id',sort=False):
        exact=sorted(g.collection.dropna().unique());days=set(g.collection_day.dropna().unique());conflict=g.hadm_id.nunique()!=1 or g.subject_id.nunique()!=1
        if len(exact)==1 and (not days or days=={math.floor(exact[0]/86400)*86400}):lo=hi=float(exact[0]);state='EXACT_COLLECTION_TIME'
        elif len(exact)==0 and len(days)==1:lo=float(next(iter(days)));hi=lo+86399;state='DATE_ONLY_INTERVAL'
        else:lo=hi=np.nan;state='UNKNOWN_OR_CONFLICTING_COLLECTION_TIME'
        specimens.append(dict(hadm_id=int(g.hadm_id.iloc[0]),subject_id=int(g.subject_id.iloc[0]),specimen=sid,lo=lo,hi=hi,state=state,
            diagnostic=bool(g.kind.eq('DIAGNOSTIC_BACTERIAL_CULTURE').any()),key_conflict=conflict))
    b['specimens']=pd.DataFrame(specimens);b['tests']=m.medication_tests()+m.culture_tests()
    row(b,'capture','NE起点前已存在任意eMAR记录的住院',int(b['capture_first'].lt(b['source'].set_index('hadm_id').ne0).sum()),len(b['source']),
        note='仅表示记录机会，不证明完整院内激素捕获；基线抗菌药另外逐索引验证')
    row(b,'capture','投影内有明确实际系统性GC的住院',r.loc[r.glucocorticoid&r.actual_systemic,'hadm_id'].nunique(),len(b['source']),note='入院至首次观察NE后30h，非首日入组人数')
    row(b,'capture','投影内有不确定GC记录的住院',r.loc[r.glucocorticoid&r.status.eq('UNCERTAIN'),'hadm_id'].nunique(),len(b['source']),note='索引前不确定GC阻止记录初用资格；索引后不反向排除')
    for state,g in b['specimens'].groupby('state'):
        row(b,'capture','培养标本采集时间精度',len(g),len(specimens),stratum=state,note='按标本折叠，非独立住院')
    b['runs'].append(dict(stage='resolve',utc=m.utc(),seconds=round(time.perf_counter()-begin,2),bytes_read=0,peak_MB=m.peak_mb()))
    save(b);say(dict(stage='sequential_resolve',medication_events_rounded=m.count(len(r)),specimens_rounded=m.count(len(specimens)),tests=len(b['tests']),effects=0))

def context(b):
    r=b['resolved_medications'];gc=r[r.glucocorticoid]
    actual=gc[gc.actual_systemic].groupby('hadm_id').chart.min().to_dict()
    uncertain=gc[gc.status.eq('UNCERTAIN')].groupby('hadm_id').chart.min().to_dict()
    aa=r[r.is_abx_name&r.actual_systemic]
    ag={int(h):g.sort_values('chart')[['chart','store']].to_numpy(dtype=float) for h,g in aa.groupby('hadm_id')}
    ds=b['specimens'];ds=ds[ds.diagnostic&~ds.key_conflict]
    dg={int(h):g[['lo','hi']].to_numpy(dtype=float) for h,g in ds.groupby('hadm_id')}
    return actual,uncertain,ag,dg

def qualify(h,t,ctx,m,is_case=False):
    actual,uncertain,ag,dg=ctx
    if actual.get(h,np.inf)<t or (not is_case and actual.get(h,np.inf)==t):return 'PRIOR_OR_CURRENT_SYSTEMIC_GC',False
    if uncertain.get(h,np.inf)<=t:return 'UNCERTAIN_GC_HISTORY_OR_TIE',False
    a=ag.get(h)
    if a is None:return 'NO_ACCEPTED_PRE24H_ABX_RECORD',False
    if isinstance(a,pd.DataFrame):a=a[['chart','store']].to_numpy(dtype=float)
    a=a[(a[:,0]<t)&(a[:,0]>=t-86400)]
    if not len(a):return 'NO_ACCEPTED_PRE24H_ABX_RECORD',False
    g=dg.get(h)
    if g is None:return 'NO_DIAGNOSTIC_CULTURE_IN_PROJECTION',False
    if isinstance(g,pd.DataFrame):g=g[['lo','hi']].to_numpy(dtype=float)
    at=a[:,0];ss=[m.culture_pair_interval(lo,hi,t,at) for lo,hi in g]
    if 2 in ss:
        strict=a[np.isfinite(a[:,1])&(a[:,1]<t),0]
        return 'QUALIFIED_RECORD_PROXY',any(m.culture_pair_interval(lo,hi,t,strict)==2 for lo,hi in g)
    if 1 in ss:return 'POSSIBLE_CULTURE_DATE_INTERVAL',False
    return 'NO_CONFIRMED_PREINDEX_PAIR',False

def active_ne(g,t,gate):
    z=g[(g.start.to_numpy()<t)&(g.end.to_numpy()>=t)]
    if 'icu_lo' in z:z=z[z.icu_lo.le(t)&z.icu_hi.gt(t)]
    if not len(z):return False,np.nan,np.nan,'NO_ACTIVE_POSITIVE_INTERVAL'
    if z.stay_id.nunique()!=1:return False,np.nan,np.nan,'MULTIPLE_ICU_INTERVAL_CONFLICT'
    stay=int(z.stay_id.iloc[0]);dose,flag,_=gate.State(g[g.stay_id.eq(stay)]).at(t,left=True)
    return True,stay,dose,flag

def candidates():
    b=load();m=base();assert 'resolved_medications' in b and 'ICU_boundaries_verified' in b
    if 'HC_candidates' in b:say(dict(stage='sequential_candidates',reuse=True));return
    begin=time.perf_counter();q=b['source'].set_index('hadm_id');r=b['resolved_medications'];ctx=context(b)
    gate=m.import_file('hc_seq_state',m.OLD/'clinical_tte_gate.py');ng={int(h):g for h,g in b['NE'].groupby('hadm_id')}
    first=r[r.hc_action].sort_values('chart').drop_duplicates('hadm_id');rows=[]
    for c in first.itertuples():
        h=int(c.hadm_id);s=q.loc[h];t=float(c.chart)
        state='';active=False;stay=np.nan;dose=np.nan;flag='NOT_EVALUATED';strict=False
        if t<s.ne0:state='FIRST_HC_BEFORE_FIRST_OBSERVED_NE'
        elif t>s.ne0+86400:state='FIRST_HC_AFTER_DAY1'
        elif t>=s.discharge or (np.isfinite(s.death_time) and t>=s.death_time):state='HOSPITAL_TIME_CONFLICT'
        else:
            active,stay,dose,flag=active_ne(ng[h],t,gate)
            if not active:state='NE_NOT_CONFIRMED_ACTIVE_BEFORE_HC'
            else:state,strict=qualify(h,t,ctx,m,True)
        # A same-time non-HC actual systemic GC makes the initiation version mixed.
        mixed=r[r.hadm_id.eq(h)&r.glucocorticoid&r.actual_systemic&~r.hc_action&r.chart.eq(t)]
        if state=='QUALIFIED_RECORD_PROXY' and len(mixed):state='SAME_TIME_OTHER_GC_VERSION'
        rows.append(dict(subject_id=int(s.subject_id),hadm_id=h,stay_id=stay,ne0=s.ne0,index=t,elapsed_h=(t-s.ne0)/3600,
            state=state,NE_dose=dose,NE_state=flag,strict_abx_pair=strict,IV_confirmed=bool(c.IV_confirmed),dose_mg=c.dose_mg))
    out=pd.DataFrame(rows);b['HC_candidates']=out
    b['aggregates']=[x for x in b['aggregates'] if x['section'] not in ['seq_HC_flow','seq_HC_timing']]
    row(b,'HC_flow','投影内首次HC实际记录住院',len(out),len(q))
    for state,z in out.groupby('state'):row(b,'HC_flow','首次HC资格／记录状态',len(z),len(out),stratum=state)
    ok=out[out.state.eq('QUALIFIED_RECORD_PROXY')]
    row(b,'HC_flow','首日且NE仍在用、此前记录代理符合的新HC启动',len(ok),len(q),note='未通过混杂、临床用途或最终支持诊断；不是已验证TTE病例')
    row(b,'HC_flow','合格启动独立患者',ok.subject_id.nunique(),note='重复住院保留患者关联')
    row(b,'HC_flow','抗菌药另要求索引前入库的确定配对',int(ok.strict_abx_pair.sum()),len(ok))
    row(b,'HC_flow','明确IV途径的启动',int(ok.IV_confirmed.sum()),len(ok),note='其余为注射剂型兼容、具体IV/IM未知')
    for dose,g in ok.groupby('dose_mg',dropna=False):
        row(b,'HC_dose','首次启动单次实际记录剂量',len(g),len(ok),stratum=str(dose)+'mg',note='单次记录不代表200mg/day疗程、持续时间或适应证')
    for lo,hi in [(0,1),(1,2),(2,4),(4,6),(6,12),(12,24)]:
        row(b,'HC_timing','合格启动距首次观察NE的时段',int((ok.elapsed_h.gt(lo)&ok.elapsed_h.le(hi)).sum()),len(ok),stratum='('+str(lo)+','+str(hi)+']h')
    b['runs'].append(dict(stage='candidates',utc=m.utc(),seconds=round(time.perf_counter()-begin,2),bytes_read=0,peak_MB=m.peak_mb()))
    save(b);say(dict(stage='sequential_candidates',first_HC_rounded=m.count(len(out)),qualified_proxy_rounded=m.count(len(ok)),models=0,effects=0))

def risksets():
    b=load();m=base()
    if 'riskset_panel' in b:say(dict(stage='sequential_risksets',reuse=True));return
    begin=time.perf_counter();ctx=context(b);q=b['source'].set_index('hadm_id');ne=b['NE'];gate=m.import_file('hc_seq_riskstate',m.OLD/'clinical_tte_gate.py')
    cases=b['HC_candidates'];cases=cases[cases.state.eq('QUALIFIED_RECORD_PROXY')].sort_values(['elapsed_h','index','hadm_id']).reset_index(drop=True)
    firsthc=b['resolved_medications'];firsthc=firsthc[firsthc.hc_action].groupby('hadm_id').chart.min().to_dict()
    ng={int(h):g for h,g in ne.groupby('hadm_id')};rng=np.random.RandomState(521);panel=[];summary=[]
    ne0=ne.hadm_id.map(q.ne0).to_numpy();a=ne.start.to_numpy()-ne0;e=ne.end.to_numpy()-ne0;harray=ne.hadm_id.to_numpy();last=begin
    for caseid,c in enumerate(cases.itertuples()):
        delta=c.index-c.ne0;at=ne0+delta
        active_mask=(a<delta)&(e>=delta)&(ne.icu_lo.to_numpy()<=at)&(ne.icu_hi.to_numpy()>at)
        pairs=ne.loc[active_mask,['hadm_id','stay_id']].drop_duplicates();counts=pairs.groupby('hadm_id').size()
        possible=counts[counts.eq(1)].index.to_numpy();pool=[]
        for h in possible:
            s=q.loc[h];t=s.ne0+delta
            if s.subject_id==c.subject_id or t>=s.discharge or (np.isfinite(s.death_time) and s.death_time<=t):continue
            if ctx[0].get(int(h),np.inf)<=t or ctx[1].get(int(h),np.inf)<=t:continue
            state,strict=qualify(int(h),t,ctx,m,False)
            if state=='QUALIFIED_RECORD_PROXY':pool.append(int(h))
        # Sampling only prepares a bounded clinical-field panel, not final matching.
        sampled=list(rng.choice(pool,size=min(20,len(pool)),replace=False)) if pool else []
        panel.append(dict(case_id=caseid,subject_id=c.subject_id,hadm_id=c.hadm_id,stay_id=c.stay_id,index=c.index,NE_dose=c.NE_dose,treated=1,sampling_probability=1.,pool_n=len(pool),elapsed_h=c.elapsed_h))
        for h in sampled:
            s=q.loc[h];t=s.ne0+delta;active,stay,dose,flag=active_ne(ng[h],t,gate)
            assert active
            panel.append(dict(case_id=caseid,subject_id=s.subject_id,hadm_id=h,stay_id=stay,index=t,NE_dose=dose,treated=0,sampling_probability=len(sampled)/len(pool),pool_n=len(pool),elapsed_h=c.elapsed_h))
        summary.append(dict(case_id=caseid,hadm_id=c.hadm_id,subject_id=c.subject_id,index=c.index,pool_n=len(pool),sampled_n=len(sampled)))
        if time.perf_counter()-last>=30:
            say(dict(stage='sequential_risksets',percent=round(100*(caseid+1)/len(cases),1),seconds=round(time.perf_counter()-begin)));last=time.perf_counter()
    z=pd.DataFrame(panel);s=pd.DataFrame(summary);b['riskset_panel']=z;b['riskset_summary']=s
    row(b,'riskset','启动时点有记录资格对照的病例',int(s.pool_n.gt(0).sum()),len(s),note='资格支持，不是混杂平衡或positivity已通过')
    row(b,'riskset','启动时点至少20个记录资格对照的病例',int(s.pool_n.ge(20).sum()),len(s))
    row(b,'riskset','风险集字段抽样面板行',len(z),note='重复决策/对照，不是独立患者或最终匹配样本')
    row(b,'riskset','抽样面板独立患者',z.subject_id.nunique())
    co=z[z.treated.eq(0)].copy();co['nextHC']=co.hadm_id.map(firsthc);dt=(co.nextHC-co['index'])/60
    for minute in [15,30,60]:row(b,'crossover','抽样对照随后检出HC记录',int((dt.gt(0)&dt.le(minute)).sum()),len(co),stratum=str(minute)+'min',note='只是记录路径；没有未来给药记录不等于确定未治疗；后续HC未用于筛对照')
    b['runs'].append(dict(stage='risksets',utc=m.utc(),seconds=round(time.perf_counter()-begin,2),bytes_read=0,peak_MB=m.peak_mb()))
    save(b);say(dict(stage='sequential_risksets',cases_rounded=m.count(len(s)),with_controls_rounded=m.count(s.pool_n.gt(0).sum()),panel_unique_patients_rounded=m.count(z.subject_id.nunique()),models=0,effects=0))

def coverage():
    """Existing projected measurements only; absence can be out of old scope."""
    b=load();m=base();f=m.load();z=b['riskset_panel'];old=f['medication_features'].set_index('hadm_id')
    b['aggregates']=[r for r in b['aggregates'] if r['section']!='seq_existing_coverage']
    b['existing_coverage_rows']=[]
    for arm,g in z.groupby('treated'):
        label='HC_START' if arm else 'SAMPLED_NOT_YET_HC';n=len(g)
        keys=set(f['clinical_features'].hadm_id)
        row(b,'existing_coverage','索引住院在旧clinical投影key内',int(g.hadm_id.isin(keys).sum()),n,stratum=label,
            note='字段抽样面板行，不是独立患者；key存在也不证明该索引回看窗口完整')
        for kind,frame,variables,ages in [('bedside',f['chart'],['MAP_arterial','MAP_noninvasive','HR','RR','SpO2','GCS_eye','GCS_verbal','GCS_motor','code_status'],m.CHART_HOURS),
                                         ('lab',f['labs'],['lactate','creatinine','platelet','bicarbonate'],m.LAB_HOURS)]:
            if kind=='bedside':
                frame=frame.copy();frame['stay_id']=pd.to_numeric(frame.stay_id,errors='coerce')
            groups={h:a for h,a in frame[frame.variable.isin(variables)].groupby('hadm_id')}
            counts={};strict={}
            for c in g.itertuples():
                a=groups.get(c.hadm_id)
                if a is None:continue
                present=a.valuenum.notna()
                if kind=='bedside':present=present|(a.variable.eq('code_status')&a.value.notna())
                sub=a[a.chart.lt(c.index)&a.chart.ge(c.index-a.variable.map(ages)*3600)&present]
                if kind=='bedside':sub=sub[sub.stay_id.eq(c.stay_id)|sub.variable.eq('code_status')]
                for var in sub.variable.unique():counts[var]=counts.get(var,0)+1
                sub=sub[sub.store.notna()&sub.store.lt(c.index)]
                for var in sub.variable.unique():strict[var]=strict.get(var,0)+1
            for var in variables:
                row(b,'existing_coverage','旧投影中索引前有非空值记录',counts.get(var,0),n,stratum=label+' / '+var+' / reported',
                    note='只表示目前取得非空记录，未在此重新验证临床数值范围；未取得者可能因旧key/窗口缺失，不能全称未测量')
                row(b,'existing_coverage','旧投影中索引前测量且已入库',strict.get(var,0),n,stratum=label+' / '+var+' / strict',
                    note='storetime为入库时钟，不能自动当医生首次可见；不按覆盖选择时钟')
    b['runs'].append(dict(stage='cached_clinical_coverage',utc=m.utc(),bytes_read=0,peak_MB=m.peak_mb()))
    b['next_required']='Define the analysis sampling/matching scope and project all required pretreatment intervals once; the20-control panel is for coverage only. No model/outcome until this measured-history basis is available.'
    save(b);say(dict(stage='sequential_existing_coverage',completed=True,raw_scans=0,models=0,effects=0))

def checks():
    """Targeted source/time verification; never inspect treatment-outcome contrasts."""
    b=load();m=base();q=b['source'].set_index('hadm_id');r=b['resolved_medications'];ctx=context(b)
    checks_done=[]
    assert b['ICU_boundaries_verified']
    assert b['NE'].start.ge(b['NE'].icu_lo).all() and b['NE'].end.le(b['NE'].icu_hi).all()
    checks_done.append('NE_intervals_clipped_to_observed_ICU')
    cases=b['HC_candidates'];ok=cases[cases.state.eq('QUALIFIED_RECORD_PROXY')]
    assert not cases.hadm_id.duplicated().any()
    assert ok['index'].gt(ok.ne0).all() and ok['index'].le(ok.ne0+86400).all()
    checks_done.append('first_HC_per_admission_and_day1_index')
    for c in ok.itertuples():
        assert qualify(int(c.hadm_id),c.index,ctx,m,True)[0]=='QUALIFIED_RECORD_PROXY'
        assert not len(r[r.hadm_id.eq(c.hadm_id)&r.glucocorticoid&r.actual_systemic&r.chart.lt(c.index)])
    checks_done.append('case_qualification_recomputed_using_preindex_events')
    z=b['riskset_panel'];s=b['riskset_summary'].set_index('case_id')
    assert np.allclose(z['index']-z.hadm_id.map(q.ne0),z.elapsed_h*3600)
    assert z['index'].lt(z.hadm_id.map(q.discharge)).all()
    assert z.sampling_probability.gt(0).all() and z.sampling_probability.le(1).all()
    checks_done.append('equal_NE_elapsed_time_and_hospital_observation')
    ng={h:g[['stay_id','start','end','icu_lo','icu_hi']].to_numpy(dtype=float) for h,g in b['NE'].groupby('hadm_id')}
    for c in z.itertuples():
        a=ng[c.hadm_id]
        a=a[(a[:,1]<c.index)&(a[:,2]>=c.index)&(a[:,3]<=c.index)&(a[:,4]>c.index)]
        assert len(a) and set(a[:,0])=={c.stay_id}
    checks_done.append('every_panel_index_has_same_ICU_NE_left_limit')
    treated=z[z.treated.eq(1)].set_index('case_id')
    assert not treated.index.duplicated().any() and len(treated)==len(ok)
    co=z[z.treated.eq(0)]
    assert co.subject_id.ne(co.case_id.map(s.subject_id)).all()
    assert ((co.case_id.value_counts()<=20)).all()
    for c in co.itertuples():assert qualify(int(c.hadm_id),c.index,ctx,m,False)[0]=='QUALIFIED_RECORD_PROXY'
    checks_done.append('controls_preindex_GC_free_record_proxy_and_other_patient')
    assert all(p['parsed_to_EOF'] for p in b['plans'].values())
    assert all(p['native_key_conflicts']==0 for k,p in b['plans'].items())
    assert sha(BASE_STATE)==b['feasibility_sha256']
    assert sha(m.SOURCE)==b['source_sha256']
    checks_done.append('raw_EOF_native_keys_and_readonly_dependencies')
    f=m.load();old=f['candidates'].sort_values('t0').drop_duplicates('hadm_id').set_index('hadm_id')
    b['aggregates']=[x for x in b['aggregates'] if x['section']!='seq_source_comparison']
    row(b,'source_comparison','新合格启动处于旧4h技术来源之外',int((~ok.hadm_id.isin(old.index)).sum()),len(ok),
        note='此处只比较key；不意味着旧临床资格错误')
    earlier=ok.hadm_id.map(old.t0)
    row(b,'source_comparison','新合格启动早于同住院旧4h索引',int((ok['index']<earlier).sum()),len(ok),
        note='旧4h索引按后来达4h的episode定义，可晚于首次观察NE后4h；旧来源外没有旧索引')
    row(b,'source_comparison','新合格启动在首次观察NE后4h以内',int(ok.elapsed_h.le(4).sum()),len(ok))
    row(b,'source_comparison','新合格启动在首次观察NE后12至24h',int((ok.elapsed_h.gt(12)&ok.elapsed_h.le(24)).sum()),len(ok),
        note='首日新使用者问题的既定范围；不是原4h＋6h自然期限政策的扩样')
    b['checks_completed']=dict(utc=m.utc(),items=checks_done,models_fitted=0,effects_estimated=0,
        independent_synthetic_boundary_groups=6,numpy_qualification_equivalence_scenarios=1000,
        independent_review='hc_estimator_design; production functions, synthetic data only')
    save(b);say(dict(stage='sequential_checks',passed=len(checks_done),models=0,effects=0))

def public_rows(b):
    m=base();out=[]
    for r in b['aggregates']:
        out.append(dict(section=r['section'],metric=r['metric'],stratum=r['stratum'],n='' if r['exact_n'] is None else m.count(r['exact_n']),
            denominator='' if r['exact_d'] is None else m.count(r['exact_d']),percent='' if r['exact_n'] is None or r['exact_d'] is None else m.pct(r['exact_n'],r['exact_d']),
            value=r['value'],note=r['note'],evidence_state=('POST_PRIMARY_FIXED_SENSITIVITY_NOT_REPLACEMENT_PRIMARY' if r['section'].startswith('seq_methods_') else 'DESCRIPTIVE_FOLLOWUP_RECORD_PATH_NOT_ELIGIBILITY_OR_EFFECT' if r['section']=='seq_control_paths' else 'EXPLORATORY_WEIGHTED_COMPARISON_NOT_VALIDATED_CAUSAL_EFFECT' if r['section'] in ['seq_effects','seq_bootstrap'] or r['section'].startswith('seq_precision_') else 'PRETREATMENT_MEASUREMENT_OR_SUPPORT_NO_EFFECT' if r['section'].startswith(('seq_clinical_','seq_new_','seq_support','seq_balance','seq_local_support','seq_control_paths')) else 'EARLIER_SOURCE_QUALIFICATION_NOT_EFFECT')))
    return pd.DataFrame(out)

def preserved_manuscript_section(text):
    """Keep the curated manuscript when rebuilding the technical report."""
    start='<!-- HC_MANUSCRIPT_START -->';end='<!-- HC_MANUSCRIPT_END -->'
    if start not in text and end not in text:return ''
    if text.count(start)!=1 or text.count(end)!=1:
        raise ValueError('AMBIGUOUS_MANUSCRIPT_MARKERS')
    lo=text.index(start);hi=text.index(end)
    if hi<lo:raise ValueError('REVERSED_MANUSCRIPT_MARKERS')
    return text[lo:hi+len(end)].rstrip()+'\n\n'


def decorate(text,table):
    if not STATE.exists():return text,table
    report_path=ROOT/'steroid_tte.md'
    manuscript=preserved_manuscript_section(report_path.read_text(encoding='utf-8-sig')) if report_path.exists() else ''
    b=load()
    if 'clinical_publication_base' in b:
        prior=b['clinical_publication_base']
        a=public_rows(b).iloc[prior['aggregate_count']:].copy()
        return manuscript+clinical_report_section(b)+prior['report_text'],pd.concat([prior['table'],a],ignore_index=True)
    b=load();m=base();a=public_rows(b);intro='# 较早HC启动：新来源重建与风险集\n\n'
    if 'HC_candidates' in b:
        q=b['HC_candidates'];ok=q[q.state.eq('QUALIFIED_RECORD_PROXY')]
        intro+='已从约'+m.count(len(b['source']))+'次成人NE住院重建，未要求活至或连续用药满4h。首日记录资格代理符合的HC启动约'+m.count(len(ok))+'次；这是新使用者候选，不是治疗效果或临床适应证已核验。\n\n'
        early=int(ok.elapsed_h.le(4).sum())
        intro+='其中约'+m.count(early)+'次索引位于首次观察NE后4h以内。它们不属于旧“到4h仍未用GC”的目标人群，不能回填到原自然期限政策，也不能把新旧人数当成同一方案扩样。\n\n'
        intro+='独立比对显示，新合格启动绝大多数仍来自原4h技术住院；真正旧技术来源之外的合格启动处于1–10隐私抑制范围。约250主要来自将资格重新对齐至实际启动前，以及使用已保存的首日索引范围；不是大量发现旧入口外患者，也不是证明原期限政策已恢复可行。\n\n'
        if 'riskset_summary' in b:
            s=b['riskset_summary'];p=b['riskset_panel']
            intro+='约'+m.count(int(s.pool_n.gt(0).sum()))+'次启动有同NE经过时长、当时仍在ICU使用NE且此前记录代理合格的对照。固定随机字段面板涵盖约'+m.count(p.subject_id.nunique())+'位患者；重复对照／决策保留关联。该面板没有匹配、加权或平衡，不能当作最终独立样本或ESS。\n\n'
        if 'existing_coverage_rows' in b:
            intro+='**本阶段决定：继续这个HC新启动候选的治疗前信息构建；尚不进入效应估计。**旧临床投影按旧人群和旧时点形成，不能用其中缺值否定新来源。下表区分key范围与已经取得的索引前记录；缺口需要按新风险集定向投影，不能用插补把“尚未提取”当成临床未测量。\n\n'
        intro+='这一步解决的是入口与感染／给药时间的工程问题。院外激素、感染性休克的临床确认、HC用途及治疗前病情的可比性仍需单独评价。实际HC注射剂型证据也不等于明确IV途径或完成200mg/day疗程。原自然期限政策的暂停结论继续保留。\n\n'
    else:intro+='新来源规则已先保存；实际药物与培养范围补充进行中，尚无新策略支持结论。\n\n'
    intro+=PROTOCOL+'\n\n'
    intro+='执行口径补充：对照在索引同秒有任何系统性GC记录均不满足“尚未启动”；索引后HC或其他GC均归于通常照护，不按后来给药删人。本轮并未定义持续不治疗策略。\n\n'
    if 'existing_coverage_rows' in b:
        intro+='## 下一步的明确边界\n\n只为本次新索引补治疗前测量：MAP/HR/RR/SpO₂回看1h，其他设备和神经状态4h，乳酸与血气6h，其他基础检验24h，治疗限制从本次住院起查至索引。床旁限定同次ICU；同一住院多个索引分别as-of，先合并读取区间。正式抽样／匹配方案须在读取前明确，并一次投影覆盖其需要的全部候选区间，避免只补20对照面板后再次扫描。实验室的采样与入库时间均须在索引前，床旁两种记录时钟分别保留。原始CSV字段投影仍需顺序读取约18.4GB lab及41.9GB chart，预计约30min；本轮未执行这两张新扫描，不把旧范围缺口记成无记录。先取得这些基础后才能决定调整与估计。\n\n'
    for sec,g in a.groupby('section',sort=False):
        intro+='### '+sec+'\n\n|检查|分层|约n|约分母|%|值|\n|---|---|---:|---:|---:|---|\n'
        for r in g.to_dict('records'):intro+='|'+'|'.join(str(r[k]) for k in ['metric','stratum','n','denominator','percent','value'])+'|\n'
        intro+='\n'
    intro+='## 本轮读取成本\n\n|步骤|GB|秒|峰值MB|\n|---|---:|---:|---:|\n'
    for r in b['runs']:intro+='|{}|{:.3f}|{}|{}|\n'.format(r['stage'],r.get('bytes_read',0)/1e9,r.get('seconds',''),r.get('peak_MB',''))
    if 'checks_completed' in b:
        c=b['checks_completed'];intro+='\n记录资格与时间核对'+str(len(c['items']))+'项、独立合成边界6组、NumPy判定一致性1000组通过；这些是实现检查，不是临床或因果验证。\n'
    if 'coverage_independent_verification' in b:intro+='\n旧投影覆盖另以独立筛选表达式复核32个分组／变量／时钟组合，计数一致；reported检验记录不等于结果在索引时已可用。\n'
    intro+='\n规则保存UTC：`'+b['frozen_utc']+'`；签名：`'+b['signature']+'`。本轮模型与效应均为0。原缓存、旧自然期限政策和暂停决定不改，患者明细仅在本机private。计数1–10抑制，其余取整到十位；百分比按实际分母且保护小单元。\n\n## 保留的前阶段结果\n\n'
    return manuscript+intro+text,pd.concat([table,a],ignore_index=True)

CLINICAL_CONFIG = dict(
    phase='complete earlier-HC riskset pretreatment measurement projection',
    population='all existing eligible risksets, not only the random20 coverage panel',
    sampling='retain every eligible control membership for fixed outcome-blind support; no new sampling',
    upper='strictly before own index, no post-index baseline',
    lab_clock='primary chart and store both before index; chart-only is record-availability audit',
    bedside_clock='primary chart before index; strict store-before-index sensitivity',
    windows_h=dict(vitals=1,device_neurologic_temperature=4,lactate_bloodgas=6,other_labs=24),
    limits='code and intubation status: current hospital admission to index; not future CMO',
    causal_scope='first HC initiation now vs not at this decision, later usual care, treated-decision ATT',
    models='not fitted until pretreatment clinical support has been inspected',
    no_outcome_contrasts=True)

def clinical_freeze():
    b=load();m=base()
    if 'clinical_phase' in b:
        assert b['clinical_phase']['config']==CLINICAL_CONFIG
        say(dict(stage='clinical_freeze',reuse=True));return
    plans={}
    for name in ['labs','chart']:
        p=m.PATHS[name];s=p.stat();assert set(m.COLS[name])<=set(pd.read_csv(p,nrows=0).columns)
        plans[name]=dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns,columns=m.COLS[name],
            mapping=m.LAB_MAP if name=='labs' else m.CHART_MAP,
            hours=m.LAB_HOURS if name=='labs' else m.CHART_HOURS,
            filter='Full riskset admission/ICU envelope; selected itemids; exact variable window and strict preindex checked again per index',
            estimate_seconds=1200,estimate_peak_MB=2000)
    b['clinical_phase']=dict(config=CLINICAL_CONFIG,frozen_utc=m.utc(),plans=plans,
        config_sha256=hashlib.sha256(json.dumps(CLINICAL_CONFIG,sort_keys=True).encode()).hexdigest(),
        prior_script_sha=sha(ROOT/'steroid_sequential.py'),prior_state_sha=sha(STATE),
        no_outcome_contrasts=True,models_fitted=0,effects_estimated=0)
    save(b);say(dict(stage='clinical_freeze',GB=round(sum(x['bytes'] for x in plans.values())/1e9,3),models=0,effects=0))

def full_risksets():
    b=load();m=base();assert 'clinical_phase' in b
    if 'full_risksets' in b:say(dict(stage='full_risksets',reuse=True));return
    begin=time.perf_counter();ctx=context(b);q=b['source'].set_index('hadm_id');ne=b['NE']
    cases=b['HC_candidates'];cases=cases[cases.state.eq('QUALIFIED_RECORD_PROXY')].sort_values(['elapsed_h','index','hadm_id']).reset_index(drop=True)
    ne0=ne.hadm_id.map(q.ne0).to_numpy();a=ne.start.to_numpy()-ne0;e=ne.end.to_numpy()-ne0
    l=ne.icu_lo.to_numpy();u=ne.icu_hi.to_numpy();parts=[];last=begin
    for caseid,c in enumerate(cases.itertuples()):
        delta=c.index-c.ne0;at=ne0+delta
        pairs=ne.loc[(a<delta)&(e>=delta)&(l<=at)&(u>at),['hadm_id','stay_id']].drop_duplicates()
        counts=pairs.groupby('hadm_id').size();pairs=pairs[pairs.hadm_id.isin(counts[counts.eq(1)].index)].set_index('hadm_id')
        pool=[]
        for h in pairs.index:
            s=q.loc[h];t=s.ne0+delta
            if s.subject_id==c.subject_id or t>=s.discharge or (np.isfinite(s.death_time) and s.death_time<=t):continue
            if ctx[0].get(int(h),np.inf)<=t or ctx[1].get(int(h),np.inf)<=t:continue
            state,strict=qualify(int(h),t,ctx,m,False)
            if state=='QUALIFIED_RECORD_PROXY':pool.append(int(h))
        original=b['riskset_summary'].set_index('case_id').loc[caseid]
        assert len(pool)==original.pool_n,'Full pool differs from saved riskset definition'
        zz=q.loc[pool,['subject_id','ne0']].reset_index()
        zz['stay_id']=zz.hadm_id.map(pairs.stay_id);zz['index']=zz.ne0+delta;zz=zz.drop(columns='ne0')
        zz['treated']=0
        case=pd.DataFrame([dict(subject_id=int(c.subject_id),hadm_id=int(c.hadm_id),stay_id=int(c.stay_id),index=c.index,treated=1)])
        zz=pd.concat([case,zz],ignore_index=True);zz['case_id']=caseid;zz['pool_n']=len(pool);zz['elapsed_h']=delta/3600
        parts.append(zz)
        if time.perf_counter()-last>=30:
            say(dict(stage='full_risksets',percent=round(100*(caseid+1)/len(cases),1),seconds=round(time.perf_counter()-begin),peak_MB=m.peak_mb()));last=time.perf_counter()
    z=pd.concat(parts,ignore_index=True)
    for k in ['subject_id','hadm_id','stay_id','case_id','pool_n','treated']:z[k]=z[k].astype('int64')
    assert not z.duplicated(['case_id','hadm_id']).any()
    assert z.treated.sum()==len(cases)
    # Exact preservation of the originally saved random coverage panel.
    chk=b['riskset_panel'].merge(z,on=['case_id','hadm_id','subject_id','stay_id','index','treated','pool_n','elapsed_h'],how='left',indicator=True)
    assert chk._merge.eq('both').all()
    env=z.groupby('hadm_id')['index'].agg(['min','max']).join(q[['subject_id','admit','discharge']])
    stayenv=z.groupby('stay_id')['index'].agg(['min','max'])
    b['full_risksets']=z;b['clinical_envelope']=env;b['clinical_stay_envelope']=stayenv
    row(b,'clinical_scope','完整风险集索引行',len(z),note='同患者重复时点/重复对照；不是独立人数或ESS')
    row(b,'clinical_scope','完整风险集独立患者',z.subject_id.nunique())
    row(b,'clinical_scope','完整风险集医院住院',z.hadm_id.nunique())
    b['runs'].append(dict(stage='full_risksets',utc=m.utc(),seconds=round(time.perf_counter()-begin,2),bytes_read=0,peak_MB=m.peak_mb()))
    save(b);say(dict(stage='full_risksets',completed=True,index_rows_rounded=m.count(len(z)),unique_patients_rounded=m.count(z.subject_id.nunique()),effects=0))

def scan_new_clinical(name):
    """One sequential file read; bounded chunks, full required riskset envelope."""
    assert name in ['labs','chart'];b=load();m=base();key='clinical_'+name
    if key in b:say(dict(stage=key,reuse=True));return
    assert 'full_risksets' in b
    p=b['clinical_phase']['plans'][name];path=Path(p['path']);st=path.stat()
    assert st.st_size==p['bytes'] and st.st_mtime_ns==p['mtime_ns']
    ids={int(i):v for v,ii in p['mapping'].items() for i in ii};env=b['clinical_envelope'];se=b['clinical_stay_envelope']
    adms=set(env.index);subs=set(env.subject_id);adm=b['admission_link'];groups={s:g for s,g in adm.groupby('subject_id',sort=False)}
    candidates=env.reset_index();begin=time.perf_counter();last=begin;parts=[];total=0;conflicts=0;unlinked=0
    join_recovered=0;outside_native_interval=0;missing_code_stay=0;stay_conflicts=0
    if name=='chart':
        sc=pd.read_pickle(m.SOURCE);sr=sc['icustays'].set_index('stay_id')[['hadm_id','subject_id']];del sc
    f=m.HashReader(path);stream=io.BufferedReader(f,buffer_size=4*1024*1024)
    try:
        for raw in pd.read_csv(stream,usecols=p['columns'],dtype=str,chunksize=150000):
            total+=len(raw);item=pd.to_numeric(raw.itemid,errors='coerce');s=pd.to_numeric(raw.subject_id,errors='coerce')
            z=raw[item.isin(ids)&s.isin(subs)].copy()
            if len(z):
                z['subject_id']=s.loc[z.index].astype('int64');z['itemid']=item.loc[z.index].astype('int64')
                z['hadm_id']=pd.to_numeric(z.hadm_id,errors='coerce');z=z[z.hadm_id.isna()|z.hadm_id.isin(adms)].copy()
                z['chart']=m.sec(z.charttime);z['store']=m.sec(z.storetime);z['join_method']='native_hadm'
                if name=='labs':
                    z=assign_missing(z,candidates,adm,'chart',groups=groups)
                    unlinked+=int(z.hadm_id.isna().sum());join_recovered+=int(z.join_method.ne('native_hadm').sum())
                z=z[z.hadm_id.isin(adms)].copy();good=z.subject_id.eq(z.hadm_id.map(env.subject_id))
                conflicts+=int((~good).sum());z=z[good].copy();z['variable']=z.itemid.map(ids)
                valid_hospital=z.chart.ge(z.hadm_id.map(env.admit))&z.chart.lt(z.hadm_id.map(env.discharge))
                outside_native_interval+=int((~valid_hospital).sum())
                if name=='labs':
                    lo=np.maximum(z.hadm_id.map(env.admit),z.hadm_id.map(env['min'])-z.variable.map(p['hours'])*3600)
                    hi=z.hadm_id.map(env['max'])
                    ok=valid_hospital&z.chart.ge(lo)&z.chart.lt(hi)
                    keep=['labevent_id','subject_id','hadm_id','specimen_id','itemid','chart','store','variable','valuenum','valueuom','join_method']
                else:
                    z['stay_id']=pd.to_numeric(z.stay_id,errors='coerce')
                    code=z.variable.isin(['code_status','intubation_status'])
                    valid_native=z.stay_id.isna()|(z.stay_id.map(sr.hadm_id).eq(z.hadm_id)&z.stay_id.map(sr.subject_id).eq(z.subject_id))
                    stay_conflicts+=int((~valid_native).sum())
                    missing_code_stay+=int((code&z.stay_id.isna()).sum())
                    lo=np.where(code,z.hadm_id.map(env.admit),z.stay_id.map(se['min'])-z.variable.map(p['hours'])*3600)
                    hi=np.where(code,z.hadm_id.map(env['max']),z.stay_id.map(se['max']))
                    ok=valid_hospital&valid_native&z.chart.ge(lo)&z.chart.lt(hi)&(code|z.stay_id.isin(se.index))
                    keep=['subject_id','hadm_id','stay_id','itemid','chart','store','variable','value','valuenum','valueuom','join_method']
                z=z.loc[ok,keep].copy();z['hadm_id']=z.hadm_id.astype('int64');z['valuenum']=pd.to_numeric(z.valuenum,errors='coerce')
                if name=='chart':
                    z['stay_id']=z.stay_id.fillna(-1).astype('int64')
                    # Numeric variables need no duplicated free-form value string.
                    cats=['code_status','intubation_status','oxygen_device','ventilator_mode','CRRT_mode','IABP_evidence','GCS_verbal']
                    z.loc[~z.variable.isin(cats),'value']=''
                for c in ['variable','valueuom','join_method']+(['value'] if name=='chart' else []):z[c]=z[c].astype('category')
                if len(z):parts.append(z)
            now=time.perf_counter()
            if now-last>=30:
                say(dict(stage=key,percent_bytes=round(100*f.n/p['bytes'],1),seconds=round(now-begin),peak_MB=m.peak_mb()));last=now
        assert f.n==p['bytes'];digest=f.h.hexdigest()
    finally:stream.close()
    st=path.stat();assert st.st_size==p['bytes'] and st.st_mtime_ns==p['mtime_ns'];assert conflicts==0,'Native key conflict'
    z=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
    before=len(z);z=z.drop_duplicates().reset_index(drop=True)
    for c in ['variable','valueuom','join_method']+(['value'] if name=='chart' else []):z[c]=z[c].astype('category')
    b[key]=z;p.update(sha256=digest,rows_traversed=total,rows_retained=len(z),duplicates_removed=before-len(z),parsed_to_EOF=True,
        missing_hadm_unlinked=unlinked,missing_hadm_recovered=join_recovered,native_key_conflicts=conflicts,outside_native_hospital_interval=outside_native_interval,
        native_stay_key_conflicts=stay_conflicts,code_missing_stay=missing_code_stay)
    b['runs'].append(dict(stage=key,utc=m.utc(),seconds=round(time.perf_counter()-begin,2),bytes_read=p['bytes'],peak_MB=m.peak_mb()))
    b['stage']='CLINICAL_FULL_SCOPE_PROJECTION_IN_PROGRESS_NO_EFFECTS';save(b)
    say(dict(stage=key,completed=True,retained_rounded=m.count(len(z)),seconds=b['runs'][-1]['seconds'],peak_MB=m.peak_mb(),effects=0))

def ne_values_at(g,t):
    """Vector left-limit states; no inferred off values outside intervals."""
    t=np.asarray(t,dtype=float);a=g.start.to_numpy();e=g.end.to_numpy();d=g.dose.to_numpy(dtype=float)
    chain=g.linkorderid.fillna(-1).to_numpy();w=pd.to_numeric(g.patientweight,errors='coerce').to_numpy(dtype=float)
    mask=(a[:,None]<t[None,:])&(e[:,None]>=t[None,:]);n=mask.sum(axis=0)
    ix=mask.argmax(axis=0);v=d[ix].copy();ww=w[ix].copy();flags=np.full(len(t),'NO_ACTIVE_RECORD',dtype=object)
    valid=(n>0)&~(mask&(~np.isfinite(d[:,None])|(chain[:,None]==-1))).any(axis=0)
    same=((~mask)|((chain[:,None]==chain[ix][None,:])&np.isclose(d[:,None],v[None,:],rtol=1e-7,atol=1e-9))).all(axis=0)
    valid&=same;v[~valid]=np.nan
    wsame=((~mask)|np.isclose(w[:,None],ww[None,:],rtol=1e-7,atol=1e-9)).all(axis=0)
    ww[~valid|~wsame|~np.isfinite(ww)|(ww<20)|(ww>300)]=np.nan
    flags[(n>0)&~valid]='ACTIVE_UNRESOLVED';flags[valid]='ACTIVE'
    # Historical explicit cessation is not propagated: inactive history stays missing.
    # This avoids interpreting old bag-end or uncertain observation gaps as dose zero.
    return v,ww,flags

def cached_history_features():
    b=load();m=base();assert 'full_risksets' in b
    if 'history_features' in b:say(dict(stage='history_features',reuse=True));return
    begin=time.perf_counter();z=b['full_risksets'];src=pd.read_pickle(m.SOURCE);q=b['source'].set_index('hadm_id')
    meta=src['adult_stays'].set_index('stay_id');x=pd.DataFrame(index=z.index)
    for col in ['age','gender','anchor_year_group','ne_start_left_censored']:
        x[col]=z.hadm_id.map(q[col])
    shifted_year=pd.to_datetime(z['index'],unit='s').dt.year.to_numpy()
    offset=shifted_year-z.hadm_id.map(q.anchor_year).to_numpy()
    yr=x.anchor_year_group.astype(str).str.extract(r'(\d{4})\D+(\d{4})')
    x['calendar_lower']=pd.to_numeric(yr[0],errors='coerce').to_numpy()+offset
    x['calendar_upper']=pd.to_numeric(yr[1],errors='coerce').to_numpy()+offset
    x['calendar_era']='UNKNOWN'
    x.loc[x.calendar_upper.le(2013),'calendar_era']='THROUGH_2013'
    x.loc[x.calendar_lower.ge(2014)&x.calendar_upper.le(2019),'calendar_era']='2014_THROUGH_2019'
    x.loc[x.calendar_lower.ge(2020),'calendar_era']='2020_ONWARD'
    x.loc[x.calendar_lower.le(2013)&x.calendar_upper.ge(2014),'calendar_era']='CROSSES_2013_2014'
    x.loc[x.calendar_lower.le(2019)&x.calendar_upper.ge(2020),'calendar_era']='CROSSES_2019_2020'
    x['ICU_type']=z.stay_id.map(meta.first_careunit);x['ICU_hours']=(z['index']-z.stay_id.map(meta.icu_in))/3600
    x['hospital_hours']=(z['index']-z.hadm_id.map(q.admit))/3600
    for col in ['NE_dose','NE_dose_1h','NE_dose_4h','weight','NE_prior1h_peak','NE_prior1h_covered_min']:x[col]=np.nan
    x['NE_state']='UNKNOWN';ng={s:g for s,g in b['NE'].groupby('stay_id',sort=False)}
    groups=list(z.groupby('stay_id',sort=False));last=begin
    for j,(s,idx) in enumerate(groups):
        g=ng.get(s)
        if g is None:continue
        t=idx['index'].to_numpy();g=g[g.start.lt(t.max())&g.end.ge(t.min()-14400)].copy()
        dose,weight,flag=ne_values_at(g,t)
        x.loc[idx.index,'NE_dose']=dose;x.loc[idx.index,'weight']=weight;x.loc[idx.index,'NE_state']=flag
        for h in [1,4]:x.loc[idx.index,'NE_dose_'+str(h)+'h']=ne_values_at(g,t-h*3600)[0]
        # Union duration of known infusion states within previous hour, exact boundaries.
        nodes=np.unique(np.r_[g.start.to_numpy(),g.end.to_numpy()]);mid=(nodes[:-1]+nodes[1:])/2
        vv,_,_=ne_values_at(g,mid);finite=np.isfinite(vv)
        aa=nodes[:-1][finite];ee=nodes[1:][finite];dd=vv[finite]
        overlap=np.maximum(0,np.minimum(ee[:,None],t[None,:])-np.maximum(aa[:,None],t[None,:]-3600))
        x.loc[idx.index,'NE_prior1h_covered_min']=overlap.sum(axis=0)/60
        peak=np.max(np.where(overlap>0,dd[:,None],-np.inf),axis=0) if len(dd) else np.full(len(t),np.nan)
        peak[~np.isfinite(peak)]=np.nan;x.loc[idx.index,'NE_prior1h_peak']=peak
        if time.perf_counter()-last>=30:
            say(dict(stage='history_features',percent=round(100*(j+1)/len(groups),1),seconds=round(time.perf_counter()-begin),peak_MB=m.peak_mb()));last=time.perf_counter()
    x['NE_change_1h']=x.NE_dose-x.NE_dose_1h;x['NE_change_4h']=x.NE_dose-x.NE_dose_4h
    gate=m.import_file('hc_seq_history_gate',m.OLD/'clinical_tte_gate.py');norm=gate.normalize(src['events'],src['icustays'])
    molecules={'VP':[222315],'EPI':[221289,229617],'PHENYLEPHRINE':[221749,229630,229631,229632],'DOPAMINE':[221662]}
    for label,ids in molecules.items():
        x[label+'_record']='NO_PRIOR_RECORD'
        gg={s:g for s,g in norm[norm.itemid.isin(ids)].groupby('stay_id',sort=False)}
        for s,idx in groups:
            g=gg.get(s)
            if g is None:continue
            tt=idx['index'].to_numpy();aa=g.start.to_numpy();ee=g.end.to_numpy();rr=g.rate.to_numpy()
            active=(aa[:,None]<tt[None,:])&(ee[:,None]>=tt[None,:]);prior=(aa[:,None]<tt[None,:]).any(axis=0)
            val=np.full(len(tt),'NO_PRIOR_RECORD',dtype=object);val[prior]='PRIOR_RECORD_CURRENT_UNKNOWN'
            val[(active&np.isfinite(rr[:,None])&(rr[:,None]>0)).any(axis=0)]='ACTIVE_RECORDED'
            val[(active&~np.isfinite(rr[:,None])).any(axis=0)]='ACTIVE_DOSE_UNKNOWN'
            x.loc[idx.index,label+'_record']=val
    p=m.RAW/'hosp'/'services.csv'/'services.csv';x['service']='UNKNOWN'
    if p.exists():
        ss=pd.read_csv(p,usecols=['hadm_id','transfertime','curr_service']);ss=ss[ss.hadm_id.isin(z.hadm_id)].copy();ss['t']=m.sec(ss.transfertime)
        sg={h:g.sort_values('t') for h,g in ss.groupby('hadm_id',sort=False)}
        for h,idx in z.groupby('hadm_id',sort=False):
            g=sg.get(h)
            if g is None:continue
            loc=np.searchsorted(g.t.to_numpy(),idx['index'].to_numpy(),side='left')-1;good=loc>=0
            x.loc[idx.index[good],'service']=g.curr_service.to_numpy()[loc[good]]
        b['clinical_phase']['services']=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),columns=['hadm_id','transfertime','curr_service'])
    # Check actual-case dose against the previous independently reconstructed value.
    cases=z[z.treated.eq(1)];old=b['HC_candidates'].set_index('hadm_id').NE_dose
    assert np.allclose(x.loc[cases.index,'NE_dose'],cases.hadm_id.map(old),equal_nan=True)
    for col in x.select_dtypes(include=['float']).columns:x[col]=x[col].astype('float32')
    for col in x.select_dtypes(include=['object']).columns:x[col]=x[col].astype('category')
    b['history_features']=x;b['runs'].append(dict(stage='history_features',utc=m.utc(),seconds=round(time.perf_counter()-begin,2),bytes_read=p.stat().st_size if p.exists() else 0,peak_MB=m.peak_mb()))
    save(b);say(dict(stage='history_features',completed=True,case_dose_equivalence=True,effects=0))

# This fragment is incorporated into the existing sequential entry point.
# It does not write data, change eligibility, or read outcomes.
import heapq

HC_CF_LABS = [
    'lactate', 'pH_bloodgas', 'pCO2_bloodgas', 'creatinine', 'platelet',
    'bilirubin', 'wbc', 'hemoglobin', 'bicarbonate', 'bun', 'glucose',
    'sodium', 'potassium',
]
HC_CF_CHART = [
    'MAP_arterial', 'MAP_noninvasive', 'HR', 'RR', 'SpO2',
    'temperature_F', 'temperature_C', 'GCS_eye', 'GCS_verbal', 'GCS_motor',
    'RASS', 'oxygen_device', 'ventilator_mode', 'FiO2', 'PEEP',
    'CRRT_mode', 'code_status', 'intubation_status',
]
HC_CF_CATEGORICAL = set(['oxygen_device', 'ventilator_mode', 'CRRT_mode',
                         'code_status', 'intubation_status'])
HC_CF_RANGES = {
    'lactate':(0,50), 'pH_bloodgas':(6,8.5), 'pCO2_bloodgas':(1,250),
    'creatinine':(0,40), 'platelet':(0,3000), 'bilirubin':(0,100),
    'wbc':(0,500), 'hemoglobin':(0,30), 'bicarbonate':(0,80), 'bun':(0,300),
    'glucose':(0,2000), 'sodium':(80,220), 'potassium':(1,15),
    'MAP_arterial':(10,250), 'MAP_noninvasive':(10,250), 'HR':(10,300),
    'RR':(1,80), 'SpO2':(1,100), 'temperature_F':(20,45),
    'temperature_C':(20,45), 'GCS_eye':(1,4), 'GCS_verbal':(1,5),
    'GCS_motor':(1,6), 'RASS':(-5,4), 'FiO2':(21,100), 'PEEP':(0,40),
}
HC_CF_UNITS = {
    'lactate':{'mmol/l'}, 'pH_bloodgas':{'units',''},
    'pCO2_bloodgas':{'mmhg'}, 'creatinine':{'mg/dl'},
    'platelet':{'k/ul','10^3/ul','10^9/l'}, 'bilirubin':{'mg/dl'},
    'wbc':{'k/ul','10^3/ul','10^9/l'}, 'hemoglobin':{'g/dl'},
    'bicarbonate':{'meq/l','mmol/l'}, 'bun':{'mg/dl'}, 'glucose':{'mg/dl'},
    'sodium':{'meq/l','mmol/l'}, 'potassium':{'meq/l','mmol/l'},
    'MAP_arterial':{'mmhg'}, 'MAP_noninvasive':{'mmhg'}, 'HR':{'bpm'},
    'RR':{'insp/min','breaths/min'}, 'SpO2':{'%'},
    'temperature_F':{'f','°f','�f','degf'},
    'temperature_C':{'c','°c','�c','degc'},
    'GCS_eye':{'','score'}, 'GCS_verbal':{'','score'}, 'GCS_motor':{'','score'},
    'RASS':{'','score'}, 'FiO2':{'','%'}, 'PEEP':{'cmh2o'},
}


def hc_cf_normalize(z, variable):
    """Unit/range validation precedes as-of selection; invalid latest rows remain unknown."""
    catmap = {}
    if variable in HC_CF_CATEGORICAL:
        txt = z['value'].astype(object).fillna('').astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
        unique = sorted(set(txt[txt.ne('')]))
        catmap = dict((s, i + 1) for i, s in enumerate(unique))
        values = txt.map(catmap).to_numpy(dtype=float)
        valid = np.isfinite(values)
        return values, valid, catmap
    values = pd.to_numeric(z.valuenum, errors='coerce').to_numpy(dtype=float)
    u = z.valueuom.astype(object).fillna('').astype(str).str.lower().str.replace(' ', '', regex=False)
    u = u.str.replace('µ','u',regex=False).str.replace('μ','u',regex=False)
    valid = u.isin(HC_CF_UNITS[variable]).to_numpy() & np.isfinite(values)
    if variable == 'temperature_F':
        values = (values - 32.) * (5. / 9.)
    if variable == 'FiO2':
        # Item 223835 records inspired oxygen; <=1 fractions and >=21 percentages
        # have disjoint plausible domains. Intermediate 1..21 remains invalid.
        frac = (values >= .21) & (values <= 1.)
        values[frac] *= 100.
    lo, hi = HC_CF_RANGES[variable]
    valid &= (values >= lo) & (values <= hi)
    if variable in ['GCS_eye','GCS_verbal','GCS_motor','RASS']:
        valid &= values == np.round(values)
    if variable == 'GCS_verbal' and 'value' in z:
        # A verbal value entered as untestable because of intubation is not
        # treated as an ordinary spontaneous verbal response score.
        untestable = z.value.astype(object).fillna('').astype(str).str.lower().str.contains(
            'ett|endotrache|intubat',regex=True).to_numpy()
        valid &= ~untestable
    return values, valid, catmap


def hc_cf_asof(chart, store, value, valid, queries, lower, strict):
    """Two-clock as-of lookup. Both chart and release must precede index strictly.

    All currently available entries at the latest measurement timestamp are
    considered. Discordant values or an invalid value in that bucket yield
    unknown; later-entered corrections never reach backward to earlier indices.
    Flags: 0 no eligible row, 1 usable, 2 invalid latest row, 3 conflicting latest
    rows. No carry-forward beyond the specified lower bound.
    """
    n = len(queries)
    ans = np.full(n, np.nan, dtype='float32')
    age = np.full(n, np.nan, dtype='float32')
    chosen_chart = np.full(n, np.nan)
    flag = np.zeros(n, dtype='uint8')
    if not len(chart) or not n:
        return ans, age, flag, chosen_chart
    ready = np.maximum(chart, store) if strict else chart.copy()
    if strict:
        ready[~np.isfinite(store)] = np.inf
    ready[~np.isfinite(chart)] = np.inf
    order = np.argsort(ready, kind='mergesort')
    qo = np.argsort(queries, kind='mergesort')
    heap, buckets, j = [], {}, 0
    for qi in qo:
        t = queries[qi]
        if not np.isfinite(t):
            continue
        while j < len(order) and ready[order[j]] < t:
            k = order[j]; ch = float(chart[k])
            if ch not in buckets:
                buckets[ch] = [set(), False]
                heapq.heappush(heap, -ch)
            bucket = buckets[ch]
            if valid[k]:
                bucket[0].add(float(value[k]))
            else:
                bucket[1] = True
            j += 1
        # Monotonic queries and admission/window lower bounds allow expired
        # timestamps to be discarded. Do not remove a bucket before all its
        # late records become ready; old late rows remain below the bound.
        while heap and -heap[0] < lower[qi]:
            heapq.heappop(heap)
        if not heap:
            continue
        ch = -heap[0]
        bucket = buckets[ch]
        chosen_chart[qi] = ch
        age[qi] = (t - ch) / 3600.
        if len(bucket[0]) > 1:
            flag[qi] = 3
        elif bucket[1] or not bucket[0]:
            flag[qi] = 2
        else:
            flag[qi] = 1
            ans[qi] = next(iter(bucket[0]))
    return ans, age, flag, chosen_chart


def hc_cf_decode(series, mapping):
    reverse = dict((v, k) for k, v in mapping.items())
    return series.map(reverse).astype(object).fillna('').astype(str).str.lower()


def hc_cf_status_derived(frame, categories):
    # Separate dimensions: a DNR record alone does not resolve intubation or
    # comfort-only intent. Only an explicit full-code record supplies zeroes.
    text = hc_cf_decode(frame['code_status'], categories.get('code_status', {}))
    full = text.eq('full code')
    dnr = text.str.contains('dnr|dnar|do not resuscitate|not attempt resuscitation', regex=True)
    dni = text.str.contains('dni|do not intubate', regex=True)
    cmo = text.str.contains('comfort', regex=False)
    for name, pos in [('recorded_DNR',dnr),('recorded_DNI',dni),('recorded_CMO',cmo)]:
        v = np.full(len(frame), np.nan, dtype='float32')
        v[full.to_numpy()] = 0.; v[pos.to_numpy()] = 1.
        frame[name] = v
    frame['recorded_full_code'] = np.where(frame.code_status.notna(), full, np.nan).astype('float32')
    frame['code_status_unknown'] = frame.code_status.isna().astype('uint8')
    itext = hc_cf_decode(frame['intubation_status'], categories.get('intubation_status', {}))
    ipos = itext.str.contains('dni|do not intubate|no intubation', regex=True)
    frame['intubation_limit_recorded'] = np.where(ipos, 1., np.nan).astype('float32')
    # These are recorded device states, not proof of ventilation or absence of
    # treatment. For example, a tracheostomy tube does not prove invasive support.
    oxy = hc_cf_decode(frame['oxygen_device'], categories.get('oxygen_device', {}))
    resp = np.full(len(frame), np.nan, dtype='float32')
    resp[oxy.ne('').to_numpy()] = 7.  # other recorded device
    resp[oxy.eq('none').to_numpy()] = 0.
    resp[oxy.isin(['nasal cannula','non-rebreather','medium conc mask','venti mask',
                  'face tent','oxymizer','aerosol-cool','high flow neb']).to_numpy()] = 1.
    resp[oxy.eq('high flow nasal cannula').to_numpy()] = 2.
    resp[oxy.isin(['bipap mask','cpap mask']).to_numpy()] = 3.
    resp[oxy.eq('endotracheal tube').to_numpy()] = 4.
    resp[oxy.isin(['tracheostomy tube','trach mask','t-piece']).to_numpy()] = 5.
    frame['oxygen_device_group'] = resp
    ctext = hc_cf_decode(frame['CRRT_mode'], categories.get('CRRT_mode', {})).str.strip()
    cr = np.full(len(frame), np.nan, dtype='float32')
    cr[ctext.isin(['cvvh','cvvhd','cvvhdf']).to_numpy()] = 1.
    cr[ctext.isin(['none','off','no','not in use','not applicable','discontinued','stopped']).to_numpy()] = 0.
    frame['CRRT_recorded'] = cr


def hc_cf_features(b, m, information_lag_minutes=0):
    """Return index-aligned features, two clocks, and exact private diagnostics.

    Does not mutate the bundle or view outcomes. The caller stores results and
    applies public small-cell rules. Primary labs use both clocks; primary
    bedside measurements use chart time. Strict sensitivity adds store<T for
    bedside records without changing age windows or cohort membership.
    """
    z = b['full_risksets'].reset_index(drop=True)
    n = len(z); admission = b['source'].set_index('hadm_id').admit
    admit = z.hadm_id.map(admission).to_numpy(dtype=float)
    times = z['index'].to_numpy(dtype=float)
    lag_seconds = float(information_lag_minutes) * 60.
    if not np.isfinite(lag_seconds) or lag_seconds < 0:
        raise ValueError('INFORMATION_LAG_MUST_BE_NONNEGATIVE')
    cutoffs = times - lag_seconds
    frames = dict(primary={}, strict={})
    selected = dict(primary={}, strict={})
    categories, diagnostics = {}, []
    group_hadm = dict((h, np.asarray(ix, dtype=int)) for h, ix in z.groupby('hadm_id',sort=False).indices.items())
    group_stay = dict((h, np.asarray(ix, dtype=int)) for h, ix in z.groupby(['hadm_id','stay_id'],sort=False).indices.items())
    for source, variables, ages in [('labs',HC_CF_LABS,m.LAB_HOURS),('chart',HC_CF_CHART,m.CHART_HOURS)]:
        d = b['clinical_'+source]
        for variable in variables:
            v = d[d.variable.eq(variable)].copy()
            values, good, cats = hc_cf_normalize(v,variable)
            categories[variable] = cats
            v['_value'] = values; v['_valid'] = good
            # Whole-hospital code and intubation limits; all other bedside
            # observations must be from the index ICU stay.
            is_code = variable in ['code_status','intubation_status']
            keys = ['hadm_id'] if source == 'labs' or is_code else ['hadm_id','stay_id']
            lookup = group_hadm if len(keys) == 1 else group_stay
            lo = admit.copy() if is_code else np.maximum(admit,times-ages[variable]*3600.)
            out = dict((clock,[np.full(n,np.nan,dtype='float32'),np.full(n,np.nan,dtype='float32'),
                              np.zeros(n,dtype='uint8'),np.full(n,np.nan)]) for clock in ['reported','strict'])
            for key, g in v.groupby(keys, sort=False):
                ix = lookup.get(key)
                if ix is None:
                    continue
                chart = g.chart.to_numpy(dtype=float); store = g.store.to_numpy(dtype=float)
                value = g._value.to_numpy(dtype=float); valid = g._valid.to_numpy(dtype=bool)
                for clock in ['reported','strict']:
                    got = hc_cf_asof(chart,store,value,valid,cutoffs[ix],lo[ix],clock=='strict')
                    # Keep age referenced to actual index and keep the original
                    # lower lookback bound. A lag never extends a lookback.
                    if lag_seconds:
                        got[1][:] += lag_seconds / 3600.
                    for j in range(4):out[clock][j][ix] = got[j]
            for clock in ['primary','strict']:
                mode = 'strict' if source == 'labs' or clock == 'strict' else 'reported'
                got = out[mode]
                frames[clock][variable] = got[0]
                frames[clock][variable+'__age_h'] = got[1]
                frames[clock][variable+'__flag'] = got[2]
                if variable in ['MAP_arterial','MAP_noninvasive','temperature_F','temperature_C',
                                'GCS_eye','GCS_verbal','GCS_motor']:
                    selected[clock][variable] = got[3]
            for arm in [0,1]:
                mask = z.treated.to_numpy() == arm
                for clock in ['reported','strict']:
                    got = out[clock]
                    diagnostics.append(dict(variable=variable,source=source,clock=clock,treated=arm,
                        n=int(mask.sum()),valid_n=int((mask & (got[2]==1)).sum()),
                        no_record_n=int((mask & (got[2]==0)).sum()),
                        invalid_n=int((mask & (got[2]==2)).sum()),
                        conflict_n=int((mask & (got[2]==3)).sum())))
            if hasattr(m,'say'):
                m.say(dict(stage='sequential_clinical_features',variable=variable,completed=True))
    for clock in ['primary','strict']:
        f = pd.DataFrame(frames[clock],index=z.index)
        # Use latest pre-index MAP measurement. At a shared time, arterial MAP
        # is preferred by prespecified measurement-source priority. Both source
        # values are also retained for audit and sensitivity.
        ac = selected[clock]['MAP_arterial']; nc = selected[clock]['MAP_noninvasive']
        choose_a = np.isfinite(ac) & (~np.isfinite(nc) | (ac >= nc))
        f['MAP'] = np.where(choose_a,f.MAP_arterial,f.MAP_noninvasive).astype('float32')
        f['MAP__age_h'] = np.where(choose_a,f.MAP_arterial__age_h,f.MAP_noninvasive__age_h).astype('float32')
        f['MAP_source'] = np.where(choose_a,1,np.where(np.isfinite(nc),2,np.nan)).astype('float32')
        fc = selected[clock]['temperature_F']; cc = selected[clock]['temperature_C']
        choose_c = np.isfinite(cc) & (~np.isfinite(fc) | (cc >= fc))
        f['temperature_C_combined'] = np.where(choose_c,f.temperature_C,f.temperature_F).astype('float32')
        f['temperature_C_combined__age_h'] = np.where(choose_c,f.temperature_C__age_h,f.temperature_F__age_h).astype('float32')
        e=selected[clock]['GCS_eye']; ve=selected[clock]['GCS_verbal']; mo=selected[clock]['GCS_motor']
        same = np.isfinite(e) & (e == ve) & (e == mo)
        complete = f[['GCS_eye','GCS_verbal','GCS_motor']].notna().all(axis=1).to_numpy()
        # This is a recorded simultaneous score, not an unsedated neurologic
        # assessment. Components and RASS/device state remain separate.
        f['GCS_simultaneous'] = np.where(same & complete,f.GCS_eye+f.GCS_verbal+f.GCS_motor,np.nan).astype('float32')
        f['GCS_simultaneous__age_h'] = np.where(same & complete,f.GCS_eye__age_h,np.nan).astype('float32')
        hc_cf_status_derived(f,categories)
        frames[clock] = f.copy()  # consolidate after controlled column creation
    contract = dict(labs=HC_CF_LABS,chart=HC_CF_CHART,lab_hours=dict((k,m.LAB_HOURS[k]) for k in HC_CF_LABS),
        chart_hours=dict((k,m.CHART_HOURS[k]) for k in HC_CF_CHART if k not in ['code_status','intubation_status']),
        code_window='current hospital admission through strictly before index',
        primary='labs chart and store<index; bedside chart<index; current ICU except code/intubation limits',
        strict='all chart and store<index; same age bounds and same indices',
        same_time='conflicting available values become unknown; no future correction leaks backward',
        latest_invalid='latest invalid or conflicting measurement remains unknown, no fallback to older valid value',
        MAP='latest source; arterial prioritized only at equal timestamp; both source channels retained',
        GCS='sum only simultaneous complete components; intubated/ETT verbal entries unknown; recorded score does not establish unsedated mental state',
        information_lag_minutes=float(information_lag_minutes),
        lag_scope='clinical measurements and charted limitations only; eligibility and medication/history features stay at actual index',
        flags={0:'no eligible row',1:'usable',2:'invalid latest row',3:'conflicting latest rows'},
        categorical_mappings=categories,range_checks=HC_CF_RANGES,unit_checks=HC_CF_UNITS)
    return dict(primary=frames['primary'],strict=frames['strict'],coverage=pd.DataFrame(diagnostics),contract=contract)


def hc_cf_tests():
    # Compact raw projections use categorical columns. Missing values must not
    # attempt to insert an unseen empty-string category during normalization.
    cat = pd.DataFrame(dict(valuenum=[2.,3.],
        valueuom=pd.Categorical(['mmol/L',None]),value=pd.Categorical(['2',None])))
    values, valid, _ = hc_cf_normalize(cat,'lactate')
    assert np.array_equal(valid,np.array([True,False])) and values[0]==2.
    text = pd.DataFrame(dict(valuenum=[np.nan,np.nan],
        valueuom=pd.Categorical([None,None]),value=pd.Categorical(['CVVH',None])))
    values, valid, categories = hc_cf_normalize(text,'CRRT_mode')
    assert valid[0] and not valid[1] and categories=={'CVVH':1}
    gcs = pd.DataFrame(dict(valuenum=[1.,5.],
        valueuom=pd.Categorical([None,None]),value=pd.Categorical(['No Response-ETT','Oriented'])))
    values, valid, _ = hc_cf_normalize(gcs,'GCS_verbal')
    assert np.array_equal(valid,np.array([False,True]))
    # A late correction cannot alter an earlier index. Chart at index is never
    # baseline. No history beyond max age; a recent invalid record is not zero.
    c=np.array([0.,100.,100.,200.]);s=np.array([1.,110.,250.,200.]);v=np.array([60.,70.,90.,80.]);valid=np.ones(4,dtype=bool)
    a=hc_cf_asof(c,s,v,valid,np.array([150.,200.,260.]),np.array([-1.,-1.,-1.]),True)
    assert np.array_equal(a[0],np.array([70.,70.,80.]))
    a=hc_cf_asof(c[:3],s[:3],v[:3],valid[:3],np.array([150.,260.]),np.array([-1.,-1.]),True)
    assert a[0][0]==70 and np.isnan(a[0][1]) and a[2][1]==3
    a=hc_cf_asof(c[:2],s[:2],v[:2],np.array([True,False]),np.array([150.]),np.array([-1.]),False)
    assert np.isnan(a[0][0]) and a[2][0]==2
    a=hc_cf_asof(c[:2],s[:2],v[:2],valid[:2],np.array([150.,500.]),np.array([100.,450.]),False)
    assert a[0][0]==70 and a[2][1]==0
    a=hc_cf_asof(np.array([100.]),np.array([np.nan]),np.array([70.]),np.array([True]),np.array([150.]),np.array([0.]),True)
    assert a[2][0]==0
    # Random brute-force agreement includes ties, unavailable stores, invalid
    # values, arbitrary index order, and increasing window bounds per group.
    rng=np.random.RandomState(913)
    for strict in [False,True]:
        for _ in range(100):
            c=rng.randint(0,30,35).astype(float);s=c+rng.randint(-2,20,35)
            s[rng.random(35)<.1]=np.nan;v=rng.randint(1,6,35).astype(float);valid=rng.random(35)>.1
            q=rng.randint(0,60,20).astype(float);lo=q-15
            got=hc_cf_asof(c,s,v,valid,q,lo,strict)
            for i,t in enumerate(q):
                mask=(c<t)&(c>=lo[i])
                if strict:mask&=np.isfinite(s)&(s<t)
                if not mask.any():assert got[2][i]==0;continue
                latest=c[mask].max();ix=mask&(c==latest);vals=set(v[ix&valid])
                expected=3 if len(vals)>1 else 2 if (~valid[ix]).any() or not vals else 1
                assert got[2][i]==expected
                if expected==1:assert got[0][i]==next(iter(vals))
    return dict(two_clock_synthetic=True,random_bruteforce_groups=200,
                categorical_backed_normalization=True,no_outcomes=True)


ANALYSIS_CONFIG = dict(
    estimand='ATT among current record-qualified first HC initiation decisions; subsequent usual care',
    exposure='actual systemic injectable-formulation-compatible HC now versus not at that index',
    score='fixed L2 conditional logistic single-event risksets, full control pool, sum-loss penalty=1',
    scaling='refit medians and scales with equal case/pool mass within uniform risksets; explicit missing states',
    numerical=['age','log_NE_dose','NE_change_1h','NE_prior1h_covered_min','log_NE_prior1h_peak',
        'ICU_hours','log_hospital_hours','MAP','HR','RR','SpO2','lactate','creatinine','platelet',
        'bicarbonate','pH_bloodgas','bilirubin','GCS_simultaneous','RASS','FiO2','PEEP','temperature_C_combined','wbc','weight'],
    categorical=['gender','calendar_era','ICU_group','service_group','ne_start_left_censored',
        'VP_record','EPI_record','other_pressor_record','oxygen_device_group','CRRT_recorded',
        'recorded_DNR','recorded_DNI','recorded_CMO','intubation_limit_recorded','code_status_unknown'],
    local_description=dict(NE_dose_ratio=2,MAP_difference_mmHg=15,lactate_difference_mmolL=2,
        purpose='description only, not clinical interchangeability or exclusion calipers'),
    clock_scenarios=['primary_bedside_chart_lab_released','strict_bedside_and_lab_released'],
    NE_clock='left-limit recorded infusion, no salt/base conversion, not asserted real-time visibility',
    categorical_unknown='unknown is separate; no record is not absence of treatment',
    weights='case weight1; controls exp(score) normalized to1 within each case riskset; no truncation/ATO',
    outcome='hospital_expire_flag in complete index admission; alive discharge0; ICU exit not censoring',
    bootstrap=dict(repetitions=100,seed=521,unit='source patient including all admissions and riskset roles',
        refit='preprocessing and score every replicate',interval='percentile, preliminary100 only'),
    support_review='joint causal measurement basis, observed value/missingness balance, local severity comparisons and patient concentration; no effect-driven selection',
    frozen_before_outcome_contrasts=True)

def analysis_freeze():
    b=load();m=base()
    if 'analysis_phase' in b:
        assert b['analysis_phase']['config']==ANALYSIS_CONFIG
        say(dict(stage='analysis_freeze',reuse=True));return
    b['analysis_phase']=dict(config=ANALYSIS_CONFIG,frozen_utc=m.utc(),
        signature=hashlib.sha256(json.dumps(ANALYSIS_CONFIG,sort_keys=True).encode()).hexdigest(),
        models_fitted=0,effects_estimated=0,decision='PENDING_CLINICAL_MEASUREMENT_AND_SUPPORT',
        rationale='Pretreatment circulation intensity/trajectory, recorded perfusion/organ/respiratory state, care setting, calendar interval and separate treatment limitations; no final diagnoses/future clinical states')
    save(b);say(dict(stage='analysis_freeze',before_outcome_contrasts=True,primary_covariates=len(ANALYSIS_CONFIG['numerical'])+len(ANALYSIS_CONFIG['categorical'])))

def build_clinical_features():
    b=load();m=base();assert 'analysis_phase' in b
    if 'new_clinical_features' in b:say(dict(stage='build_clinical_features',reuse=True));return
    begin=time.perf_counter();result=hc_cf_features(b,m);b['new_clinical_features']=result
    b['clinical_phase']['tests']=hc_cf_tests()
    cov=result['coverage']
    for r in cov.itertuples():
        row(b,'new_coverage',r.variable,int(r.valid_n),int(r.n),stratum=('HC' if r.treated else 'control_index')+' / '+r.clock,
            note='完整新风险集；control是可重复索引行，不是独立患者；reported实验室不是结果已可用')
    for clock in ['primary','strict']:
        x=result[clock];z=b['full_risksets']
        for arm in [0,1]:
            mask=z.treated.eq(arm)
            for variable in ['MAP','GCS_simultaneous']:
                row(b,'new_coverage',variable,int(x.loc[mask,variable].notna().sum()),int(mask.sum()),stratum=('HC' if arm else 'control_index')+' / '+clock)
    b['runs'].append(dict(stage='new_clinical_features',utc=m.utc(),seconds=round(time.perf_counter()-begin,2),bytes_read=0,peak_MB=m.peak_mb()))
    b['stage']='COMPLETE_NEW_PREINDEX_CLINICAL_FEATURES_NO_EFFECTS';save(b)
    say(dict(stage='build_clinical_features',completed=True,peak_MB=m.peak_mb(),effects=0))

def analysis_frame(b,clock):
    z=b['full_risksets'];x=pd.concat([z[['case_id','subject_id','hadm_id','treated']],b['history_features'],b['new_clinical_features'][clock]],axis=1)
    assert len(x)==len(z) and x.index.equals(z.index)
    x['log_NE_dose']=np.log(x.NE_dose.where(x.NE_dose.gt(0)))
    x['log_NE_prior1h_peak']=np.log(x.NE_prior1h_peak.where(x.NE_prior1h_peak.gt(0)))
    x['log_hospital_hours']=np.log1p(x.hospital_hours.where(x.hospital_hours.ge(0)))
    icu=x.ICU_type.astype(object).fillna('').astype(str)
    x['ICU_group']='OTHER_RECORDED';x.loc[icu.eq(''),'ICU_group']=np.nan
    x.loc[icu.str.contains('Surgical|Trauma',case=False),'ICU_group']='SURGICAL'
    x.loc[icu.str.contains('Medical',case=False),'ICU_group']='MEDICAL'
    x.loc[icu.str.contains('Medical',case=False)&icu.str.contains('Surgical',case=False),'ICU_group']='MIXED_MED_SURG'
    x.loc[icu.str.contains('Cardiac|Coronary',case=False),'ICU_group']='CARDIAC'
    x.loc[icu.str.contains('Neuro',case=False),'ICU_group']='NEURO'
    sv=x.service.astype(object).fillna('').astype(str)
    x['service_group']='OTHER_RECORDED';x.loc[sv.isin(['','UNKNOWN']),'service_group']=np.nan
    x.loc[sv.isin(['MED','OMED']),'service_group']='MEDICAL'
    x.loc[sv.eq('CMED'),'service_group']='CARDIAC'
    x.loc[sv.str.contains('SURG|ORTHO|GU|GYN|ENT',regex=True),'service_group']='SURGICAL'
    x.loc[sv.isin(['NMED','NSURG']),'service_group']='NEURO'
    ph=x.PHENYLEPHRINE_record.astype(object).astype(str);do=x.DOPAMINE_record.astype(object).astype(str)
    x['other_pressor_record']='NO_PRIOR_TARGET_RECORD'
    x.loc[ph.ne('NO_PRIOR_RECORD')|do.ne('NO_PRIOR_RECORD'),'other_pressor_record']='PRIOR_RECORD_CURRENT_UNKNOWN'
    x.loc[ph.eq('ACTIVE_RECORDED')|do.eq('ACTIVE_RECORDED'),'other_pressor_record']='ACTIVE_RECORDED'
    x.loc[ph.eq('ACTIVE_DOSE_UNKNOWN')|do.eq('ACTIVE_DOSE_UNKNOWN'),'other_pressor_record']='ACTIVE_DOSE_UNKNOWN'
    return x

# -*- coding: utf-8 -*-
"""Pure risk-set ATT functions; no file I/O and no clinical data are loaded.

Each stratum has one actual initiator and its complete eligible control pool.
The initiation score is a custom fixed-L2 conditional-logistic implementation,
not a reproduction of a published HC estimator. Later treatment is usual care.

For subject-cluster bootstrap counts m, stratum r is repeated m_case times;
its case remains one candidate, whereas every control occurs m_control times:
  loss = sum_r m_case * [log(exp(eta_case) + sum_j m_j exp(eta_j)) - eta_case]
         + 0.5 * l2 * ||beta||^2.
Do not multiply the case candidate in that denominator by m_case. This is
replication of single-event strata, not a multiple-event/tied Cox likelihood.

ATT controls are normalized separately in each stratum. Each initiating
decision has equal mass; a larger pool does not receive greater outcome mass.
The caller must first establish time-zero, source eligibility, covariate clocks,
outcome ascertainment and clinically interpretable measured support. Balance
and ESS do not establish absence of unmeasured confounding.

Sources for principles, not validation of this custom complete implementation:
Li et al. (2001), doi:10.1198/016214501753208573 (risk-set design);
Field and Welsh (2007), doi:10.1111/j.1467-9868.2007.00593.x (cluster bootstrap).
"""
import hashlib
import json
import time
import warnings
import numpy as np
import pandas as pd


class EstimatorFailure(ValueError):
    """A failed fit/replicate is explicit and must not be silently omitted."""


def _codes(values):
    codes, levels = pd.factorize(np.asarray(values), sort=False)
    if (codes < 0).any():
        raise EstimatorFailure('MISSING_CLUSTER_OR_STRATUM_IDENTIFIER')
    return codes.astype(np.int32), np.asarray(levels)


def make_plan(frame, numeric, categorical=(), case_col='case_id',
              patient_col='subject_id', treated_col='treated',
              admission_col='hadm_id', diagnostic_numeric=()):
    """Cache outcome-free arrays, categories and numeric sort orders once.

    Row order is changed internally to contiguous strata. All vectors passed to
    downstream functions must have that order; use plan['order'] to align them.
    Original identifiers stay in the caller's protected process.
    """
    if len(frame) == 0:
        raise EstimatorFailure('EMPTY_RISKSET_FRAME')
    numeric, categorical = list(numeric), list(categorical)
    diagnostic_numeric = [k for k in diagnostic_numeric if k not in numeric]
    if len(set(numeric + categorical)) != len(numeric + categorical):
        raise EstimatorFailure('DUPLICATE_FEATURE_NAME')
    gc, groups = _codes(frame[case_col].to_numpy())
    order = np.argsort(gc, kind='stable')
    gc = gc[order]
    starts = np.r_[0, np.flatnonzero(np.diff(gc)) + 1].astype(np.int64)
    ends = np.r_[starts[1:], len(frame)].astype(np.int64)
    y0 = pd.to_numeric(frame[treated_col], errors='coerce').to_numpy()[order]
    if not np.isin(y0, [0, 1]).all():
        raise EstimatorFailure('INVALID_TREATED_FLAG')
    treated = y0.astype(bool)
    if not np.all(np.add.reduceat(treated.astype(int), starts) == 1):
        raise EstimatorFailure('STRATUM_MUST_HAVE_EXACTLY_ONE_CASE')
    if (ends - starts <= 1).any():
        raise EstimatorFailure('INITIAL_STRATUM_HAS_NO_CONTROL')
    pc, patients = _codes(frame[patient_col].to_numpy()[order])
    case_rows = np.flatnonzero(treated)
    if (pc[~treated] == pc[case_rows][gc[~treated]]).any():
        raise EstimatorFailure('SAME_ORIGINAL_PATIENT_AS_OWN_CONTROL')
    if admission_col is not None and admission_col in frame:
        keys = pd.DataFrame({'g': gc, 'a': frame[admission_col].to_numpy()[order]})
        if keys.duplicated().any():
            raise EstimatorFailure('DUPLICATE_ADMISSION_IN_A_STRATUM')
    nv, sorts = {}, {}
    for name in numeric + diagnostic_numeric:
        v = pd.to_numeric(frame[name], errors='coerce').to_numpy(dtype=float)[order]
        v[~np.isfinite(v)] = np.nan
        nv[name] = v.astype(np.float32)
        valid = np.flatnonzero(np.isfinite(v))
        sorts[name] = valid[np.argsort(v[valid], kind='stable')].astype(np.int32)
    cats, levels = {}, {}
    for name in categorical:
        vals = frame[name].to_numpy()[order]
        vals = np.asarray(['__MISSING__' if pd.isna(v) else str(v) for v in vals])
        lev = sorted(set(vals) | {'__MISSING__'})
        # Fixed levels across bootstrap replicates; do not discover levels from Y.
        levels[name] = lev
        mapping = {v: j for j, v in enumerate(lev)}
        cats[name] = np.asarray([mapping[v] for v in vals], dtype=np.int32)
    result = dict(order=order, group=gc, group_ids=groups, starts=starts, ends=ends,
                treated=treated, case_rows=case_rows, patient=pc,
                patient_ids=patients, numeric=numeric, categorical=categorical,
                diagnostic_numeric=diagnostic_numeric,
                numeric_values=nv, numeric_sorts=sorts, category_codes=cats,
                category_levels=levels, n=len(frame), n_groups=len(starts))
    digest = hashlib.sha256(json.dumps(dict(numeric=numeric, categorical=categorical,
        diagnostic_numeric=diagnostic_numeric,
        levels=levels, patients=[str(v) for v in patients],
        groups=[str(v) for v in groups]), sort_keys=True).encode('utf-8'))
    for a in [order, gc, treated, pc] + [nv[k] for k in numeric+diagnostic_numeric] + [cats[k] for k in categorical]:
        digest.update(memoryview(np.ascontiguousarray(a)).cast('B'))
    result['plan_sha256'] = digest.hexdigest()
    return result


def multiplicities(plan, patient_counts=None):
    """Build exact original-patient replication counts without expanding rows."""
    if patient_counts is None:
        counts = np.ones(len(plan['patient_ids']), dtype=float)
    else:
        counts = np.asarray(patient_counts, dtype=float)
    if counts.shape != (len(plan['patient_ids']),) or not np.isfinite(counts).all():
        raise EstimatorFailure('INVALID_PATIENT_MULTIPLICITY_SHAPE')
    if (counts < 0).any() or not np.equal(counts, np.floor(counts)).all():
        raise EstimatorFailure('PATIENT_MULTIPLICITY_MUST_BE_NONNEGATIVE_INTEGER')
    row_counts = counts[plan['patient']]
    case_counts = row_counts[plan['case_rows']]
    if not case_counts.sum():
        raise EstimatorFailure('NO_BOOTSTRAP_CASES')
    control_counts = np.where(plan['treated'], 0., row_counts)
    pool_counts = np.add.reduceat(control_counts, plan['starts'])
    if ((case_counts > 0) & (pool_counts <= 0)).any():
        raise EstimatorFailure('EMPTY_BOOTSTRAP_CONTROL_POOL')
    # Case multiplicity repeats its complete stratum, not the case candidate.
    likelihood_counts = np.where(plan['treated'], 1., row_counts)
    return dict(patient=counts, row=row_counts, case=case_counts,
                controls=control_counts, pool=pool_counts,
                likelihood=likelihood_counts)


def preprocessing_weights(plan, mult):
    """Half of each stratum's mass to its case, half across its full pool."""
    g = plan['group']; active = mult['case'][g]
    den = mult['pool'][g]
    w = np.zeros(plan['n'], dtype=float)
    co = ~plan['treated']
    w[co] = np.divide(.5 * active[co] * mult['controls'][co], den[co],
                      out=np.zeros(co.sum(), dtype=float), where=den[co] > 0)
    w[plan['treated']] = .5 * mult['case']
    if not np.isclose(w.sum(), mult['case'].sum()):
        raise EstimatorFailure('PREPROCESSING_MASS_MISMATCH')
    return w / w.sum()


def _weighted_median_sorted(values, sorted_rows, weights):
    if not len(sorted_rows):
        return 0., True
    ww = weights[sorted_rows]
    total = ww.sum()
    if not total:
        return 0., True
    j = np.searchsorted(np.cumsum(ww), .5 * total, side='left')
    return float(values[sorted_rows[min(j, len(sorted_rows) - 1)]]), False


def preprocess(plan, mult):
    """Refit medians/scales under current patient bootstrap; no future data/Y.

    Missing indicators are always explicit. Numerical zero after imputation is
    an internal model value, never an assertion of a normal clinical finding.
    Weighted reference mass is uniform over initiating decisions, not pool rows.
    Dense float32 design avoids a full n_rows x n_features float64 allocation.
    """
    w = preprocessing_weights(plan, mult)
    width = 2 * len(plan['numeric']) + sum(
        max(0, len(plan['category_levels'][k]) - 1) for k in plan['categorical'])
    if width == 0:
        raise EstimatorFailure('NO_ADJUSTMENT_FEATURES')
    x = np.empty((plan['n'], width), dtype=np.float32)
    names, meta = [], {}; j = 0
    for name in plan['numeric']:
        v = plan['numeric_values'][name]
        missing = ~np.isfinite(v)
        med, all_missing = _weighted_median_sorted(v, plan['numeric_sorts'][name], w)
        a = np.where(missing, med, v).astype(float)
        mean = float(np.dot(w, a))
        sd = float(np.sqrt(max(0., np.dot(w, (a - mean) ** 2))))
        sd = sd if sd > 1.e-12 else 1.
        x[:, j] = (a - mean) / sd
        names.append(name); j += 1
        a = missing.astype(float); mm = float(np.dot(w, a))
        ms = float(np.sqrt(max(0., mm * (1. - mm))))
        ms = ms if ms > 1.e-12 else 1.
        x[:, j] = (a - mm) / ms
        names.append(name + '__MISSING'); j += 1
        meta[name] = dict(median=med, mean=mean, scale=sd,
                          missing_mean=mm, missing_scale=ms,
                          all_missing_in_weighted_sample=all_missing)
    for name in plan['categorical']:
        # Drop the fixed missing reference; levels do not change across bootstrap.
        lev = plan['category_levels'][name]
        ref = lev.index('__MISSING__')
        for code, label in enumerate(lev):
            if code == ref:
                continue
            x[:, j] = plan['category_codes'][name] == code
            names.append(name + '=' + label); j += 1
    assert j == width
    return x, dict(numeric=meta, feature_names=names,
                   category_levels=plan['category_levels'],
                   reference='equal_case_and_pool_mass_within_uniform_risksets')


def _matvec(x, beta, chunk_rows=65536):
    out = np.empty(len(x), dtype=float)
    for lo in range(0, len(x), chunk_rows):
        hi = min(lo + chunk_rows, len(x))
        out[lo:hi] = np.asarray(x[lo:hi], dtype=float).dot(beta)
    return out


def _rmatvec(x, v, chunk_rows=65536):
    out = np.zeros(x.shape[1], dtype=float)
    for lo in range(0, len(x), chunk_rows):
        hi = min(lo + chunk_rows, len(x))
        out += np.asarray(x[lo:hi], dtype=float).T.dot(v[lo:hi])
    return out


def conditional_loss_gradient(beta, x, plan, mult, l2=1.):
    """Stable grouped softmax and analytic gradient, including cluster counts."""
    eta = _matvec(x, np.asarray(beta, dtype=float))
    lm = np.full(plan['n'], -np.inf)
    positive = mult['likelihood'] > 0
    lm[positive] = np.log(mult['likelihood'][positive])
    a = eta + lm; g = plan['group']; starts = plan['starts']
    maximum = np.maximum.reduceat(a, starts)
    ex = np.exp(a - maximum[g])
    normal = np.add.reduceat(ex, starts)
    logden = maximum + np.log(normal)
    cm = mult['case']
    loss = np.dot(cm, logden - eta[plan['case_rows']]) + .5 * l2 * np.dot(beta, beta)
    residual = cm[g] * ex / normal[g]
    residual[plan['case_rows']] -= cm
    gradient = _rmatvec(x, residual) + l2 * beta
    return float(loss), gradient


def fit_score(plan, patient_counts=None, l2=1., max_iter=600,
              tolerance=1.e-6, require_convergence=True):
    """Fit one fixed conditional score, without outcomes, tuning or row splits."""
    started = time.perf_counter()
    from scipy.optimize import minimize
    if not np.isfinite(l2) or l2 <= 0:
        raise EstimatorFailure('FIXED_L2_MUST_BE_POSITIVE')
    mult = multiplicities(plan, patient_counts)
    x, prep = preprocess(plan, mult)
    # Same float32-rounded inputs, cast once rather than in every gradient chunk.
    # Objective and gradient equality verified; no statistical setting changes.
    x = np.asarray(x, dtype=np.float64)
    # Bootstrap-zero likelihood rows contribute exactly zero. Preprocessing
    # remains on the full plan, and final eta/weights retain the full plan.
    opt_x,opt_plan,opt_mult=x,plan,mult
    active=np.flatnonzero(mult['case']>0)
    keep=np.flatnonzero((mult['case'][plan['group']]>0)&(mult['likelihood']>0))
    if len(keep)<plan['n']:
        group=np.searchsorted(active,plan['group'][keep])
        starts=np.r_[0,np.flatnonzero(np.diff(group))+1]
        case_rows=np.searchsorted(keep,plan['case_rows'][active])
        assert np.array_equal(keep[case_rows],plan['case_rows'][active])
        opt_x=x[keep]
        opt_plan=dict(n=len(keep),group=group,starts=starts,case_rows=case_rows)
        opt_mult=dict(likelihood=mult['likelihood'][keep],case=mult['case'][active])
    prep_finished = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        opt = minimize(conditional_loss_gradient, np.zeros(x.shape[1]),
                       args=(opt_x, opt_plan, opt_mult, l2), jac=True, method='L-BFGS-B',
                       options=dict(maxiter=int(max_iter), ftol=1.e-11,
                                    gtol=float(tolerance), maxls=40))
    info = dict(success=bool(opt.success), status=int(opt.status),
                message=str(opt.message), iterations=int(opt.nit),
                function_evaluations=int(opt.nfev),
                maximum_abs_gradient=float(np.max(np.abs(opt.jac))),
                warnings=[str(z.message) for z in caught],
                fixed_sumloss_l2=float(l2), preprocessing=prep,
                preprocessing_seconds=prep_finished-started,
                optimization_seconds=time.perf_counter()-prep_finished,
                design_matrix_MB=x.nbytes/1048576.)
    info['optimization_rows']=len(opt_x)
    if not opt.success and require_convergence:
        raise EstimatorFailure('CONDITIONAL_SCORE_OPTIMIZATION_FAILED: ' + str(opt.message))
    eta = _matvec(x, opt.x)
    info['seconds'] = time.perf_counter()-started
    return dict(beta=opt.x, linear_predictor=eta, mult=mult,
                fit=info, preprocessing=prep)


def att_weights(plan, linear_predictor, mult=None):
    """Each case has unit mass; its controls divide one unit by relative score.

    With bootstrap multiplicities, each control row's mass includes its original
    patient's multiplicity, and each stratum has its case multiplicity. There
    is no overlap-weighting factor applied to cases and no future HC censoring.
    """
    if mult is None:
        mult = multiplicities(plan)
    eta = np.asarray(linear_predictor, dtype=float)
    if eta.shape != (plan['n'],) or not np.isfinite(eta).all():
        raise EstimatorFailure('INVALID_INITIATION_SCORE')
    control = ~plan['treated']; g = plan['group']; starts = plan['starts']
    a = np.full(plan['n'], -np.inf)
    valid = control & (mult['controls'] > 0)
    a[valid] = eta[valid] + np.log(mult['controls'][valid])
    maximum = np.maximum.reduceat(a, starts)
    safe_max = np.where(np.isfinite(maximum), maximum, 0.)
    ex = np.zeros(plan['n'], dtype=float)
    ex[valid] = np.exp(a[valid] - safe_max[g[valid]])
    den = np.add.reduceat(ex, starts)
    if ((mult['case'] > 0) & (den <= 0)).any():
        raise EstimatorFailure('EMPTY_WEIGHTED_CONTROL_POOL')
    within = np.divide(ex, den[g], out=np.zeros_like(ex), where=den[g] > 0)
    wc = mult['case'][g] * within
    wt = np.zeros(plan['n'], dtype=float); wt[plan['case_rows']] = mult['case']
    if not (np.isclose(wc.sum(), wt.sum()) and
            np.allclose(np.add.reduceat(wc, starts), mult['case'])):
        raise EstimatorFailure('ATT_STRATUM_MASS_MISMATCH')
    # Baseline comparison is each stratum's uniform replicated control pool.
    ref = np.divide(mult['controls'], mult['pool'][g],
                    out=np.zeros(plan['n']), where=mult['pool'][g] > 0)
    ref *= mult['case'][g]
    return dict(treated=wt, control=wc, control_reference=ref,
                within_control=within, mult=mult)


def _ess(mass):
    mass = np.asarray(mass, dtype=float); mass = mass[mass > 0]
    if not len(mass):
        return np.nan
    return float(mass.sum() ** 2 / np.dot(mass, mass))


def _concentration(mass):
    mass = np.asarray(mass, dtype=float); mass = mass[mass > 0]
    if not len(mass):
        return dict(ess=np.nan, maximum_share=np.nan, top5_share=np.nan,
                    top10_share=np.nan, nonzero_units=0)
    ordered = np.sort(mass)[::-1]; total = mass.sum()
    return dict(ess=_ess(mass), maximum_share=float(ordered[0] / total),
                top5_share=float(ordered[:5].sum() / total),
                top10_share=float(ordered[:10].sum() / total),
                nonzero_units=int(len(mass)))


def concentration_diagnostics(plan, weights):
    """Patient masses include all admissions and roles; no identifiers returned."""
    pc = plan['patient']; size = len(plan['patient_ids'])
    summary = {}
    for arm in ['treated', 'control']:
        summary[arm + '_patient'] = _concentration(np.bincount(
            pc, weights=weights[arm], minlength=size))
        summary[arm + '_decision_row'] = _concentration(weights[arm])
    within = weights['within_control']; rows = []
    for j, (lo, hi) in enumerate(zip(plan['starts'], plan['ends'])):
        if weights['mult']['case'][j] <= 0:
            continue
        w = within[lo:hi]; on = w > 0
        patient_mass = np.bincount(pc[lo:hi], weights=w, minlength=size)
        patient_mass = patient_mass[patient_mass > 0]
        rows.append(dict(case_id=plan['group_ids'][j],
                         case_multiplicity=int(weights['mult']['case'][j]),
                         control_row_ess=_ess(w), control_patient_ess=_ess(patient_mass),
                         max_control_row_share=float(w.max()),
                         max_control_patient_share=float(patient_mass.max()),
                         available_control_rows=int(on.sum()),
                         available_control_patients=int(len(patient_mass))))
    return summary, pd.DataFrame(rows)


def _weighted_summary(v, w):
    valid = np.isfinite(v) & (w > 0)
    total = w[valid].sum()
    if not total:
        return np.nan, np.nan, 0.
    vv, ww = v[valid], w[valid]
    mean = float(np.dot(ww, vv) / total)
    variance = float(np.dot(ww, (vv - mean) ** 2) / total)
    coverage = float(total / w.sum()) if w.sum() else np.nan
    return mean, variance, coverage


def _smd(difference, sd):
    if not np.isfinite(difference) or not np.isfinite(sd):
        return np.nan
    if sd > 1.e-12:
        return float(difference / sd)
    return 0. if abs(difference) <= 1.e-12 else float(np.sign(difference) * np.inf)


def balance_diagnostics(plan, weights):
    """Observed-value and missing-state balance; fixed unweighted-reference SD.

    'Before' pools are risk-set uniform, not raw rows dominated by large pools.
    Cases retain their ATT mass. Missing values are never treated as normal.
    """
    rows = []; wt, wr, wc = [weights[k] for k in
                            ['treated', 'control_reference', 'control']]
    def add(name, kind, v):
        a, va, ca = _weighted_summary(v, wt)
        b, vb, cb = _weighted_summary(v, wr)
        c, _, cc = _weighted_summary(v, wc)
        sd = np.sqrt(.5 * (va + vb)) if np.isfinite(va + vb) else np.nan
        rows.append(dict(variable=name, kind=kind, treated_mean=a,
                         reference_control_mean=b, weighted_control_mean=c,
                         smd_before=_smd(a-b, sd), smd_after=_smd(a-c, sd),
                         treated_coverage=ca, reference_control_coverage=cb,
                         weighted_control_coverage=cc, reference_sd=float(sd)))
    for name in plan['numeric'] + plan.get('diagnostic_numeric', []):
        v = plan['numeric_values'][name].astype(float)
        add(name, 'observed_value', v)
        add(name, 'missing_state', (~np.isfinite(v)).astype(float))
    for name in plan['categorical']:
        for code, level in enumerate(plan['category_levels'][name]):
            add(name + '=' + level, 'category',
                (plan['category_codes'][name] == code).astype(float))
    return pd.DataFrame(rows)


def within_set_balance(plan, weights, variables):
    """Per-case differences expose cancellation hidden by global balance.

    Standardization uses the fixed global before-weighting pooled SD. This is
    a discrepancy diagnostic, not an estimated within-single-case variance.
    """
    all_balance = balance_diagnostics(plan, weights)
    refsd = all_balance[all_balance.kind.eq('observed_value')].set_index('variable').reference_sd
    rows = []
    for j, (lo, hi) in enumerate(zip(plan['starts'], plan['ends'])):
        if weights['mult']['case'][j] <= 0:
            continue
        case_row = plan['case_rows'][j]
        for name in variables:
            v = plan['numeric_values'][name]
            mean, _, coverage = _weighted_summary(v[lo:hi], weights['within_control'][lo:hi])
            case_value = float(v[case_row])
            rows.append(dict(case_id=plan['group_ids'][j], variable=name,
                             case_observed=bool(np.isfinite(case_value)),
                             weighted_control_coverage=coverage,
                             standardized_difference=_smd(case_value-mean, float(refsd[name]))))
    return pd.DataFrame(rows)


def local_severity_overlap(plan, weights, severity_edges):
    """Descriptive fixed state-cell/range support; not a positivity proof.

    severity_edges maps numeric names to predeclared cut-points, including outer
    +/- infinity. Missingness has a separate cell. The caller must not search
    cut-points to obtain better-looking overlap or select on future outcomes.
    """
    names = list(severity_edges); codes = []
    for name in names:
        v = plan['numeric_values'][name]
        edges = np.asarray(severity_edges[name], dtype=float)
        if len(edges) < 2 or not np.all(np.diff(edges) > 0):
            raise EstimatorFailure('INVALID_FIXED_SEVERITY_EDGES')
        code = np.searchsorted(edges, v, side='right') - 1
        code[~np.isfinite(v)] = -1
        codes.append(code)
    state = np.column_stack(codes); rows = []
    for j, (lo, hi) in enumerate(zip(plan['starts'], plan['ends'])):
        if weights['mult']['case'][j] <= 0:
            continue
        case_row = plan['case_rows'][j]; w = weights['within_control'][lo:hi]
        same = np.all(state[lo:hi] == state[case_row], axis=1) & (w > 0)
        rec = dict(case_id=plan['group_ids'][j],
                   case_severity_missing=bool((state[case_row] < 0).any()),
                   same_state_control_rows=int(same.sum()),
                   same_state_control_mass=float(w[same].sum()))
        for name in names:
            v = plan['numeric_values'][name]; on = (w > 0) & np.isfinite(v[lo:hi])
            current = float(v[case_row]); observed = v[lo:hi][on]
            rec[name + '_outside_control_range'] = (bool(current < observed.min() or current > observed.max())
                if np.isfinite(current) and len(observed) else None)
        rows.append(rec)
    return pd.DataFrame(rows)


def local_similarity_diagnostics(plan, weights, ne_name='NE_dose', map_name='MAP',
                                 lactate_name='lactate', ne_fold=2.,
                                 map_difference=15., lactate_difference=2.):
    """One fixed, descriptive similarity scale; never a selection rule.

    Main description requires observed, positive interpretable NE doses within
    a factor of 2 AND MAP within 15 mmHg. Lactate within 2 mmol/L is reported
    separately among measured values. Missing information is not declared to be
    clinical disagreement or normality. These are researcher-chosen coarse
    descriptions, not validated interchangeability limits or matching calipers.
    Caller must first reconcile NE units and lactate mmol/L; no salt conversion.
    No case is dropped or downweighted by this diagnostic.
    """
    if ne_fold <= 1 or map_difference <= 0 or lactate_difference <= 0:
        raise EstimatorFailure('INVALID_FIXED_SIMILARITY_DESCRIPTION')
    names = [ne_name, map_name, lactate_name]
    if any(name not in plan['numeric_values'] for name in names):
        raise EstimatorFailure('MISSING_DECLARED_LOCAL_SIMILARITY_VARIABLE')
    ne, mp, la = [plan['numeric_values'][k] for k in names]
    rows = []
    for j, (lo, hi) in enumerate(zip(plan['starts'], plan['ends'])):
        if weights['mult']['case'][j] <= 0:
            continue
        rr = plan['case_rows'][j]; w = weights['within_control'][lo:hi]
        on = w > 0
        ne_observed = on & np.isfinite(ne[lo:hi]) & (ne[lo:hi] > 0)
        map_observed = on & np.isfinite(mp[lo:hi])
        case_ne_ok = bool(np.isfinite(ne[rr]) and ne[rr] > 0)
        case_map_ok = bool(np.isfinite(mp[rr]))
        both_observed = ne_observed & map_observed
        close = np.zeros(hi-lo, dtype=bool)
        if case_ne_ok and case_map_ok:
            ratio = np.ones(hi-lo, dtype=float)
            np.divide(ne[lo:hi], ne[rr], out=ratio, where=ne_observed)
            close = both_observed & (ratio >= 1./ne_fold) & (ratio <= ne_fold)
            close &= np.abs(mp[lo:hi]-mp[rr]) <= map_difference
        lactate_observed = on & np.isfinite(la[lo:hi])
        case_lactate_ok = bool(np.isfinite(la[rr]))
        compatible_lactate = (lactate_observed &
            (np.abs(la[lo:hi]-la[rr]) <= lactate_difference)) if case_lactate_ok else np.zeros(hi-lo,dtype=bool)
        rec = dict(case_id=plan['group_ids'][j],
            case_NE_observed_positive=case_ne_ok, case_MAP_observed=case_map_ok,
            case_lactate_observed=case_lactate_ok,
            control_NE_MAP_observed_mass=float(w[both_observed].sum()),
            control_lactate_observed_mass=float(w[lactate_observed].sum()),
            close_NE_MAP_control_rows=(int(close.sum()) if case_ne_ok and case_map_ok else np.nan),
            close_NE_MAP_control_patients=(int(np.unique(plan['patient'][lo:hi][close]).size)
                if case_ne_ok and case_map_ok else np.nan),
            close_NE_MAP_control_mass=(float(w[close].sum()) if case_ne_ok and case_map_ok else np.nan),
            close_lactate_mass_among_observed=(float(w[compatible_lactate].sum()/w[lactate_observed].sum())
                if case_lactate_ok and w[lactate_observed].sum() > 0 else np.nan),
            close_NE_MAP_plus_lactate_mass=(float(w[close & compatible_lactate].sum())
                if case_ne_ok and case_map_ok and case_lactate_ok else np.nan),
            descriptive_NE_fold=float(ne_fold), descriptive_MAP_difference_mmHg=float(map_difference),
            descriptive_lactate_difference_mmolL=float(lactate_difference))
        rows.append(rec)
    return pd.DataFrame(rows)


def binary_risks(plan, weights, outcome_original_order):
    """Generic complete terminal binary endpoint, without ordinary censoring.

    For hospital mortality the caller supplies verified admission expire flags:
    alive discharge is Y=0, not loss to follow-up. ICU exit does not end follow-up.
    Unknown outcomes raise; they are never filled as alive or silently dropped.
    This helper must not be called with clinical outcomes before support review.
    """
    y = np.asarray(outcome_original_order, dtype=float)[plan['order']]
    used = (weights['treated'] + weights['control']) > 0
    if not np.isin(y[used], [0, 1]).all():
        raise EstimatorFailure('UNKNOWN_OR_INVALID_TERMINAL_OUTCOME')
    mass = weights['treated'].sum()
    r1 = float(np.dot(weights['treated'][used], y[used]) / mass)
    r0 = float(np.dot(weights['control'][used], y[used]) / mass)
    return dict(risk_treated=r1, risk_control=r0, risk_difference=r1-r0,
                risk_ratio=(r1/r0 if r0 > 0 else np.nan),
                risk_ratio_defined=bool(r0 > 0))


def draw_patient_counts(plan, rng, source_patient_ids=None):
    """Resample original independent patients, keeping all admissions/roles."""
    population = plan['patient_ids'] if source_patient_ids is None else pd.unique(source_patient_ids)
    population = np.asarray(population)
    if not len(population) or pd.isna(population).any():
        raise EstimatorFailure('INVALID_BOOTSTRAP_SOURCE_POPULATION')
    where = pd.Index(population).get_indexer(plan['patient_ids'])
    if (where < 0).any():
        raise EstimatorFailure('ANALYSIS_PATIENT_ABSENT_FROM_BOOTSTRAP_SOURCE')
    draws = rng.multinomial(len(population), np.full(len(population), 1./len(population)))
    return draws[where]


def bootstrap_att(plan, outcome_original_order, repetitions=100, seed=521,
                  source_patient_ids=None, l2=1., progress=None,
                  checkpoint_callback=None, checkpoint_every=10,
                  resume_state=None, analysis_signature=None):
    """Pure bootstrap; every replicate refits preprocessing and initiation score.

    Failure and undefined-RR counts remain explicit; there is no outcome-based
    acceptance, seed search or silent replacement of failed replicates.
    Raw source tables and frozen clinical features are never reread here.
    checkpoint_callback receives a pure payload; the caller may put it into ONE
    existing canonical private bundle rather than new checkpoint files. Resume
    preserves the RNG state and cannot change data/model/population/seed or the
    optional clinical signature. Extending 100 to 500 repetitions is allowed.
    """
    started = time.perf_counter()
    population = plan['patient_ids'] if source_patient_ids is None else pd.unique(source_patient_ids)
    spec = dict(estimator_version='conditional_riskset_ATT_v1', seed=int(seed),
                fixed_sumloss_l2=float(l2), plan_sha256=plan['plan_sha256'],
                source_patient_sha256=hashlib.sha256(json.dumps([str(v) for v in population]).encode('utf-8')).hexdigest(),
                analysis_signature=analysis_signature)
    rng = np.random.RandomState(seed); results = []; previous_seconds = 0.
    if resume_state is not None:
        if resume_state['specification'] != spec:
            raise EstimatorFailure('BOOTSTRAP_RESUME_SPECIFICATION_CHANGED')
        results = [dict(rec) for rec in resume_state['results']]
        if resume_state['next_replicate'] != len(results) or len(results) > repetitions:
            raise EstimatorFailure('INVALID_BOOTSTRAP_RESUME_POSITION')
        rng.set_state(resume_state['rng_state'])
        previous_seconds = float(resume_state.get('total_seconds', 0.))
    def state():
        return dict(specification=spec, results=[dict(rec) for rec in results],
                    next_replicate=len(results), rng_state=rng.get_state(),
                    total_seconds=previous_seconds+time.perf_counter()-started)
    for r in range(len(results), int(repetitions)):
        replicate_started = time.perf_counter()
        rec = dict(replicate=r, success=False)
        try:
            counts = draw_patient_counts(plan, rng, source_patient_ids)
            fit = fit_score(plan, counts, l2=l2)
            ww = att_weights(plan, fit['linear_predictor'], fit['mult'])
            rec.update(binary_risks(plan, ww, outcome_original_order))
            rec.update(success=True, fit_iterations=fit['fit']['iterations'],
                       maximum_abs_gradient=fit['fit']['maximum_abs_gradient'],
                       warnings=fit['fit']['warnings'],
                       fit_seconds=fit['fit']['seconds'],
                       preprocessing_seconds=fit['fit']['preprocessing_seconds'],
                       optimization_seconds=fit['fit']['optimization_seconds'],
                       initiating_decision_multiplicity=float(fit['mult']['case'].sum()))
        except (EstimatorFailure, FloatingPointError, RuntimeError) as exc:
            rec.update(error_type=type(exc).__name__, error=str(exc))
        rec['seconds'] = time.perf_counter()-replicate_started
        results.append(rec)
        if progress is not None:
            progress(rec)
        if checkpoint_callback is not None and len(results) % max(1,int(checkpoint_every)) == 0:
            checkpoint_callback(state())
    last_state = state()
    if checkpoint_callback is not None:
        checkpoint_callback(last_state)
    table = pd.DataFrame(results)
    table.attrs['checkpoint_state'] = last_state
    return table


def synthetic_tests():
    """Small fabricated data only; no file I/O, patients or clinical outcomes."""
    rows = []
    # Patients 1/2 initiate in distinct strata and are controls in another stratum.
    records = [(10, 1, 1, 1.2), (10, 2, 0, .5), (10, 3, 0, np.nan),
               (20, 2, 1, .8), (20, 1, 0, 1.0), (20, 4, 0, .2),
               (30, 5, 1, .6), (30, 3, 0, .7), (30, 4, 0, .3)]
    for group, patient, treated, value in records:
        rows.append(dict(case_id=group, subject_id=patient, hadm_id=patient,
                         treated=treated, dose=value, MAP=60+10*patient,
                         unit='MED' if patient % 2 else 'SURG'))
    frame = pd.DataFrame(rows)
    plan = make_plan(frame, ['dose', 'MAP'], ['unit'])
    mult = multiplicities(plan)
    pw = preprocessing_weights(plan, mult)
    assert np.allclose(np.add.reduceat(pw, plan['starts']), 1./3)
    assert np.allclose(pw[plan['case_rows']], 1./6)
    x, _ = preprocess(plan, mult); beta = np.linspace(-.2, .3, x.shape[1])
    loss, grad = conditional_loss_gradient(beta, x, plan, mult)
    delta = 1.e-5; approx = []
    for j in range(len(beta)):
        d = np.zeros(len(beta)); d[j] = delta
        approx.append((conditional_loss_gradient(beta+d,x,plan,mult)[0] -
                       conditional_loss_gradient(beta-d,x,plan,mult)[0])/(2*delta))
    assert np.allclose(grad, approx, atol=1.e-6, rtol=1.e-5)
    counts = np.asarray([2, 1, 3, 2, 1])  # aligned to this plan's patient levels
    bm = multiplicities(plan, counts)
    loss, grad = conditional_loss_gradient(beta, x, plan, bm)
    expanded_loss = .5*np.dot(beta,beta); expanded_grad = beta.copy()
    for j,(lo,hi) in enumerate(zip(plan['starts'],plan['ends'])):
        case = plan['case_rows'][j]
        idx = [case]
        for rr in range(lo,hi):
            if rr != case:
                idx.extend([rr]*int(bm['row'][rr]))
        xx = x[np.asarray(idx)].astype(float); eta=xx.dot(beta)
        a=eta.max(); prob=np.exp(eta-a); den=prob.sum(); prob/=den
        for copy in range(int(bm['case'][j])):
            expanded_loss += a+np.log(den)-eta[0]
            expanded_grad += xx.T.dot(prob)-xx[0]
    assert np.allclose(loss,expanded_loss) and np.allclose(grad,expanded_grad)
    ww = att_weights(plan, _matvec(x,beta), bm)
    assert np.allclose(np.add.reduceat(ww['control'],plan['starts']),bm['case'])
    assert np.isclose(ww['treated'].sum(),ww['control'].sum())
    fit = fit_score(plan)
    assert fit['fit']['success'] and np.isfinite(fit['linear_predictor']).all()
    ww0 = att_weights(plan, fit['linear_predictor'])
    summary, local = concentration_diagnostics(plan, ww0)
    assert len(local)==3 and 0 < summary['control_patient']['ess'] <= 5
    bal=balance_diagnostics(plan,ww0)
    assert set(bal.kind)=={'observed_value','missing_state','category'}
    assert len(within_set_balance(plan,ww0,['dose','MAP']))==6
    assert len(local_severity_overlap(plan,ww0,{'dose':[-np.inf,.5,np.inf],
                                               'MAP':[-np.inf,65,np.inf]}))==3
    example=frame.copy();example['lactate']=[2.,2.5,np.nan,4.,3.,2.,2.,np.nan,1.]
    lp=make_plan(example,['dose','MAP','lactate'],['unit'])
    lw=att_weights(lp,np.zeros(lp['n']))
    ls=local_similarity_diagnostics(lp,lw,ne_name='dose')
    assert len(ls)==3 and ls.close_NE_MAP_control_mass.notna().all()
    # Synthetic binary outcomes check mass accounting, not a clinical result.
    y=np.asarray([0,1,0,1,0,1,0,0,1],dtype=float)
    risks=binary_risks(plan,ww0,y)
    assert 0 <= risks['risk_treated'] <= 1 and 0 <= risks['risk_control'] <= 1
    bad=y.copy();bad[0]=np.nan
    try:binary_risks(plan,ww0,bad);raise AssertionError('Unknown outcome silently accepted')
    except EstimatorFailure:pass
    # Removing all controls from an active case set must fail, not drop that set.
    empty=np.ones(len(plan['patient_ids']));empty[1:3]=0
    try:multiplicities(plan,empty);raise AssertionError('Empty pool silently dropped')
    except EstimatorFailure:pass
    draws=draw_patient_counts(plan,np.random.RandomState(17),np.r_[plan['patient_ids'],99])
    assert len(draws)==len(plan['patient_ids']) and (draws>=0).all()
    # Checkpoint/resume has identical draws, estimates and failures to one run.
    snapshots=[]
    first=bootstrap_att(plan,y,repetitions=3,seed=717,
                        checkpoint_callback=snapshots.append,checkpoint_every=2)
    resumed=bootstrap_att(plan,y,repetitions=6,seed=717,
                          resume_state=first.attrs['checkpoint_state'])
    once=bootstrap_att(plan,y,repetitions=6,seed=717)
    keep=[k for k in once if k not in ['seconds','fit_seconds','preprocessing_seconds','optimization_seconds']]
    pd.testing.assert_frame_equal(resumed[keep],once[keep],check_exact=False,atol=1.e-9,rtol=1.e-9)
    assert snapshots and snapshots[-1]['next_replicate']==3
    try:
        bootstrap_att(plan,y,repetitions=6,seed=718,resume_state=first.attrs['checkpoint_state'])
        raise AssertionError('Changed seed incorrectly accepted on resume')
    except EstimatorFailure:pass
    # Pure diagnostic variables do not change the initiation-score design.
    extra=frame.copy();extra['raw_NE']=.2
    extra_plan=make_plan(extra,['dose','MAP'],['unit'],diagnostic_numeric=['raw_NE'])
    extra_x,_=preprocess(extra_plan,multiplicities(extra_plan))
    assert extra_x.shape==x.shape and 'raw_NE' not in fit['preprocessing']['feature_names']
    return ['riskset_uniform_preprocessing','analytic_gradient',
            'patient_bootstrap_equals_explicit_replication','ATT_stratum_mass',
            'fixed_score_convergence','patient_not_row_concentration',
            'observed_and_missing_balance','within_set_and_local_state_checks',
            'terminal_binary_risk_accounting','unknown_outcome_not_alive',
            'empty_bootstrap_pool_not_dropped','whole_patient_source_bootstrap',
            'fixed_local_similarity_not_selection','checkpoint_exact_resume',
            'resume_rejects_changed_specification','diagnostic_variable_not_in_score']


def support_diagnostics():
    b=load();m=base();assert 'new_clinical_features' in b and 'history_features' in b
    assert b['analysis_phase']['config']==ANALYSIS_CONFIG
    b.setdefault('support',{})
    for clock in ['primary','strict']:
        if clock in b['support']:say(dict(stage='support',clock=clock,reuse=True));continue
        begin=time.perf_counter();frame=analysis_frame(b,clock)
        plan=make_plan(frame,ANALYSIS_CONFIG['numerical'],ANALYSIS_CONFIG['categorical'],diagnostic_numeric=['NE_dose'])
        say(dict(stage='support',clock=clock,question='fixed pretreatment initiation score and full-pool ATT support; no outcomes',rows_rounded=m.count(plan['n'])))
        fit=fit_score(plan,l2=1.);ww=att_weights(plan,fit['linear_predictor'],fit['mult'])
        bal=balance_diagnostics(plan,ww);con,per=concentration_diagnostics(plan,ww)
        within=within_set_balance(plan,ww,['NE_dose','MAP','lactate','NE_change_1h'])
        local=local_similarity_diagnostics(plan,ww)
        b['support'][clock]=dict(beta=fit['beta'],linear_predictor=fit['linear_predictor'],fit=fit['fit'],
            plan_sha256=plan['plan_sha256'],balance=bal,concentration=con,per_case=per,within=within,local=local)
        b['analysis_phase']['models_fitted']+=1
        for arm in ['treated','control']:
            d=con[arm+'_patient']
            for key in ['ess','maximum_share','top5_share','top10_share']:
                row(b,'support',key,value='{:.4f}'.format(d[key]),stratum=clock+' / '+arm+'_patient',note='合并同一患者全部住院/索引质量；不是person-hour ESS')
        abs_smd=bal.smd_after.abs()
        row(b,'support','最大可计算绝对SMD',value='{:.4f}'.format(abs_smd.max()),stratum=clock,note='全变量/缺失状态；全部私有诊断保留；单个数值不证明识别')
        row(b,'support','无法计算SMD的变量/状态行',value=str(int(abs_smd.isna().sum())),stratum=clock,note='不可比较的测量基础单列；无穷大SMD不被删除')
        for var in ['NE_dose','NE_change_1h','MAP','HR','lactate','creatinine','platelet','FiO2','ICU_hours']:
            rr=bal[(bal.variable==var)&(bal.kind=='observed_value')]
            if len(rr):
                r=rr.iloc[0];nn=int(np.isfinite(plan['numeric_values'][var][plan['treated']]).sum())
                if nn>10:row(b,'balance',var,value='SMD {:.3f} -> {:.3f}; HC coverage {}%; weighted control {:.1%}'.format(r.smd_before,r.smd_after,m.pct(nn,int(plan['treated'].sum())),r.weighted_control_coverage),stratum=clock)
        evaluable=local.case_NE_observed_positive&local.case_MAP_observed
        row(b,'local_support','HC时点NE与MAP均可评价',int(evaluable.sum()),len(local),stratum=clock)
        row(b,'local_support','可评价HC时点有相似NE/MAP记录对照',int((evaluable&local.close_NE_MAP_control_patients.gt(0)).sum()),int(evaluable.sum()),stratum=clock,
            note='固定NE比<=2、MAP差<=15；粗粒度描述，不是纳排或临床互换性验证')
        for col in ['control_patient_ess','max_control_patient_share']:
            row(b,'local_support',col,value='P10 {:.3f}; median {:.3f}; P90 {:.3f}'.format(*per[col].quantile([.1,.5,.9]).tolist()),stratum=clock)
        row(b,'local_support','相似NE/MAP对照的加权质量',value='P10 {:.3f}; median {:.3f}; P90 {:.3f}'.format(*local.loc[evaluable,'close_NE_MAP_control_mass'].quantile([.1,.5,.9]).tolist()),stratum=clock)
        b['runs'].append(dict(stage='support_'+clock,utc=m.utc(),seconds=round(time.perf_counter()-begin,2),bytes_read=0,peak_MB=m.peak_mb()))
        b['stage']='FIXED_CLINICAL_SUPPORT_DIAGNOSTICS_COMPLETE_NO_EFFECTS';save(b)
        say(dict(stage='support',clock=clock,completed=True,fit_seconds=round(fit['fit']['seconds'],1),
            maximum_abs_SMD=round(abs_smd.max(),4),control_patient_ESS=round(con['control_patient']['ess'],1),
            max_control_patient_share=round(con['control_patient']['maximum_share'],4),peak_MB=m.peak_mb(),effects=0))
        del frame,plan,fit,ww

def control_followup_description():
    """Describe later recorded care, never select on it or relabel baseline."""
    b=load();m=base();assert all(k in b.get('support',{}) for k in ['primary','strict'])
    if 'control_followup' in b and 'complement_patients' in b['control_followup']:
        say(dict(stage='control_followup',reuse=True));return
    b['aggregates']=[r for r in b['aggregates'] if r['section']!='seq_control_paths']
    z=b['full_risksets'];q=b['source'].set_index('hadm_id');r=b['resolved_medications']
    masks={'HC':r.hc_action,
           'other_GC':r.glucocorticoid&r.actual_systemic&~r.hc_action,
           'uncertain_GC':r.glucocorticoid&r.status.eq('UNCERTAIN')}
    times={k:z.hadm_id.map(r.loc[v].groupby('hadm_id').chart.min()).to_numpy() for k,v in masks.items()}
    t=z['index'].to_numpy();end=z.hadm_id.map(q.med_end).to_numpy()
    descriptions=[]
    for clock in ['primary','strict']:
        frame=analysis_frame(b,clock)
        plan=make_plan(frame,ANALYSIS_CONFIG['numerical'],ANALYSIS_CONFIG['categorical'],diagnostic_numeric=['NE_dose'])
        assert plan['plan_sha256']==b['support'][clock]['plan_sha256']
        ww=att_weights(plan,b['support'][clock]['linear_predictor'])
        for minute in [15,30,60]:
            horizon=t+60*minute
            state=np.full(len(z),'NO_HC_RECORD_IN_PROJECTED_WINDOW',dtype=object)
            state[end<horizon]='PROJECTED_WINDOW_ENDS_EARLY'
            for k,label in [('uncertain_GC','UNCERTAIN_GC_RECORD'),('other_GC','OTHER_SYSTEMIC_GC_RECORD'),('HC','HC_RECORD')]:
                seen=np.isfinite(times[k])&(times[k]>t)&(times[k]<=horizon)&(times[k]<=end)
                state[seen]=label
            for label in ['HC_RECORD','OTHER_SYSTEMIC_GC_RECORD','UNCERTAIN_GC_RECORD','PROJECTED_WINDOW_ENDS_EARLY','NO_HC_RECORD_IN_PROJECTED_WINDOW']:
                selected=state[plan['order']]==label
                patients=z.loc[(state==label)&z.treated.eq(0),'subject_id'].nunique()
                complement=z.loc[(state!=label)&z.treated.eq(0),'subject_id'].nunique()
                for scheme,key in [('uniform_pool','control_reference'),('ATT','control')]:
                    mass=float(ww[key][selected].sum()/ww[key].sum())
                    descriptions.append(dict(clock=clock,minute=minute,state=label,scheme=scheme,mass=mass,unique_patients=int(patients),complement_patients=int(complement)))
                    # Tiny positive cells are not disclosed through weighted percentages.
                    value=('1–10位患者相关单元或补集抑制' if 0<patients<=10 or 0<complement<=10 else '{:.2%}'.format(mass))
                    row(b,'control_paths',label,value=value,stratum=clock+' / '+scheme+' / '+str(minute)+'min',
                        note='对照风险集质量占比；窗口内任一HC优先，其次其他实际GC、不确定GC、投影提前结束、未检出；不是首个后续事件。未来路径不用于纳排或删失，未检出不是确定未治疗')
        del frame,plan,ww
    b['control_followup']=pd.DataFrame(descriptions);save(b)
    say(dict(stage='control_followup',completed=True,no_outcome_contrasts=True))


def exploratory():
    b=load();m=base()
    assert b['analysis_phase']['decision']=='GO_EXPLORATORY_FIXED_ATT','Outcome contrasts require documented measured-support decision'
    assert b['analysis_phase']['config']==ANALYSIS_CONFIG
    assert b['analysis_phase']['signature']==hashlib.sha256(json.dumps(ANALYSIS_CONFIG,sort_keys=True).encode()).hexdigest()
    for clock in ['primary','strict']:
        assert clock in b.get('support',{}),'Both fixed outcome-blind diagnostic scenarios must be complete'
        checkframe=analysis_frame(b,clock)
        checkplan=make_plan(checkframe,ANALYSIS_CONFIG['numerical'],ANALYSIS_CONFIG['categorical'],diagnostic_numeric=['NE_dose'])
        assert checkplan['plan_sha256']==b['support'][clock]['plan_sha256']
        del checkframe,checkplan
    b.setdefault('estimates',{});b.setdefault('bootstrap',{})
    for clock in ['primary','strict']:
        begin=time.perf_counter();frame=analysis_frame(b,clock)
        plan=make_plan(frame,ANALYSIS_CONFIG['numerical'],ANALYSIS_CONFIG['categorical'],diagnostic_numeric=['NE_dose'])
        assert plan['plan_sha256']==b['support'][clock]['plan_sha256']
        outcome=frame.hadm_id.map(b['source'].set_index('hadm_id').hospital_expire_flag).to_numpy()
        if clock not in b['estimates']:
            ww=att_weights(plan,b['support'][clock]['linear_predictor']);est=binary_risks(plan,ww,outcome)
            raw=att_weights(plan,np.zeros(plan['n']));est['riskset_uniform_unadjusted']=binary_risks(plan,raw,outcome)
            b['estimates'][clock]=est;b['analysis_phase']['effects_estimated']+=1;save(b)
            say(dict(stage='exploratory',clock=clock,point_estimate_saved=True,note='not displayed until all fixed scenarios are complete'))
        state=b['bootstrap'].get(clock)
        if state is not None and state['next_replicate']>=100:continue
        def progress(rec):
            say(dict(stage='patient_bootstrap',clock=clock,replicate_completed=rec['replicate']+1,success=rec['success'],seconds=round(rec['seconds'],1)))
        def checkpoint(state):
            b['bootstrap'][clock]=state;save(b)
        bootstrap_att(plan,outcome,repetitions=100,seed=521,source_patient_ids=b['source'].subject_id.unique(),
            progress=progress,checkpoint_callback=checkpoint,checkpoint_every=10,resume_state=state,
            analysis_signature=b['analysis_phase']['signature'])
        b['runs'].append(dict(stage='bootstrap_'+clock,utc=m.utc(),seconds=round(time.perf_counter()-begin,2),bytes_read=0,peak_MB=m.peak_mb()))
        b['stage']='EXPLORATORY_HC_ATT_ESTIMATES_PENDING_PRECISION_AND_CLINICAL_REVIEW';save(b)
        del frame,plan

def new_labs():scan_new_clinical('labs')
def new_chart():scan_new_clinical('chart')

PRECISION_CONFIG = dict(
    target_repetitions=500, seed=521, prefixes=[100,250,500],
    rationale='numerical precision extension after viewing the initial100; no scientific design revision',
    preserve='same source patients, RNG continuation, data, preprocessing, score, point estimates and all initial100 draws',
    endpoint_mc='pointwise 95% binomial order-statistic interval for each bootstrap percentile endpoint',
    RD_mc_reference=0.01,
    RD_mc_reference_meaning='maximum absolute distance from percentile endpoint to its MC bounds; one percentage point is an investigator numerical reporting reference, not a clinical threshold',
    stopping='fixed500 per clock; report instability and failures without automatic expansion, replacement draws or outcome-based stopping',
    workers='two independent read-only-cache clock workers; separate atomic temporary checkpoints; merge only after both finish')

def precision_path(clock):
    assert clock in ['primary','strict']
    return STATE.parent/('steroid_precision_'+clock+'.pkl')

def precision_input_hashes(b):
    z=b['full_risksets']
    membership=z[['case_id','subject_id','hadm_id','stay_id','index','treated']]
    y=z.hadm_id.map(b['source'].set_index('hadm_id').hospital_expire_flag).to_numpy(dtype='float64')
    assert np.isfinite(y).all() and np.isin(y,[0.,1.]).all()
    return dict(membership=hashlib.sha256(pd.util.hash_pandas_object(membership,index=True).values.tobytes()).hexdigest(),
                hospital_outcome=hashlib.sha256(y.tobytes()).hexdigest())

def precision_freeze():
    import copy
    b=load();m=base()
    if 'precision_extension' in b:
        assert b['precision_extension']['config']==PRECISION_CONFIG
        say(dict(stage='precision_freeze',reuse=True));return
    assert b['analysis_phase']['config']==ANALYSIS_CONFIG
    assert b['analysis_phase']['decision']=='GO_EXPLORATORY_FIXED_ATT'
    for clock in ['primary','strict']:
        state=b['bootstrap'][clock]
        assert state['next_replicate']==100
        assert [r['replicate'] for r in state['results']]==list(range(100))
        assert not precision_path(clock).exists()
    files={}
    for name,expected in b['clinical_deployment']['outputs'].items():
        # Sources can already contain this additive extension; prior source bytes
        # are supplied by the staged deployment guard and retained below.
        if name.endswith(('.md','.csv')):assert sha(ROOT/name)==expected
        files[name]=dict(sha256=sha(ROOT/name),bytes=(ROOT/name).read_bytes())
    b['precision_extension']=dict(config=PRECISION_CONFIG,frozen_utc=m.utc(),
        signature=hashlib.sha256(json.dumps(PRECISION_CONFIG,sort_keys=True).encode()).hexdigest(),
        analysis_signature=b['analysis_phase']['signature'],
        initial_bootstrap=copy.deepcopy(b['bootstrap']),
        initial_estimates=copy.deepcopy(b['estimates']),
        initial_aggregates=copy.deepcopy(b['aggregates']),
        initial_result_status=b['analysis_phase']['result_status'],
        initial_files=files, input_hashes=precision_input_hashes(b), status='FIXED_500_NOT_STARTED')
    save(b)
    say(dict(stage='precision_freeze',target_per_clock=500,prior100_preserved=True,
             original_analysis_config_unchanged=True,new_source_reads=0))

def precision_worker(clock):
    """No writes to shared bundle. Each clock owns one temporary checkpoint."""
    b=load();m=base();px=b['precision_extension'];assert px['config']==PRECISION_CONFIG
    assert b['analysis_phase']['config']==ANALYSIS_CONFIG
    assert b['analysis_phase']['signature']==px['analysis_signature']
    assert b['estimates']==px['initial_estimates']
    assert precision_input_hashes(b)==px['input_hashes']
    path=precision_path(clock);state=pd.read_pickle(path) if path.exists() else b['bootstrap'][clock]
    initial=px['initial_bootstrap'][clock]
    assert state['specification']==initial['specification']
    assert state['results'][:100]==initial['results']
    if state['next_replicate']==500:
        say(dict(stage='precision_worker',clock=clock,reuse=True,completed=500));return
    begin=time.perf_counter();frame=analysis_frame(b,clock)
    plan=make_plan(frame,ANALYSIS_CONFIG['numerical'],ANALYSIS_CONFIG['categorical'],diagnostic_numeric=['NE_dose'])
    assert plan['plan_sha256']==b['support'][clock]['plan_sha256']
    outcome=frame.hadm_id.map(b['source'].set_index('hadm_id').hospital_expire_flag).to_numpy()
    source=b['source'].subject_id.unique();signature=b['analysis_phase']['signature']
    # Release unused cached projections before numerical work; all scientific
    # inputs remain in the immutable plan/outcome/source arrays.
    del frame,b
    import gc;gc.collect()
    say(dict(stage='precision_worker',clock=clock,start=state['next_replicate'],
             target=500,question='same patient bootstrap, endpoint numerical precision only'))
    def progress(rec):
        if (rec['replicate']+1)%10==0 or not rec['success']:
            say(dict(stage='precision_worker',clock=clock,completed=rec['replicate']+1,
                     success=rec['success'],seconds=round(rec['seconds'],2),peak_MB=m.peak_mb()))
    def checkpoint(value):
        value['precision_worker']=dict(clock=clock,extension_signature=px['signature'],
             seconds=round(time.perf_counter()-begin,2),peak_MB=m.peak_mb(),utc=m.utc())
        temporary=path.with_suffix('.writing')
        pd.to_pickle(value,temporary);os.replace(str(temporary),str(path))
    bootstrap_att(plan,outcome,repetitions=500,seed=521,source_patient_ids=source,
        progress=progress,checkpoint_callback=checkpoint,checkpoint_every=10,
        resume_state=state,analysis_signature=signature)
    say(dict(stage='precision_worker',clock=clock,completed=500,
             seconds=round(time.perf_counter()-begin,2),peak_MB=m.peak_mb()))

def precision_primary():precision_worker('primary')
def precision_strict():precision_worker('strict')

def precision_merge():
    b=load();m=base();px=b['precision_extension'];assert px['config']==PRECISION_CONFIG
    assert b['analysis_phase']['config']==ANALYSIS_CONFIG
    assert b['analysis_phase']['signature']==px['analysis_signature']
    assert px['analysis_signature']==hashlib.sha256(json.dumps(ANALYSIS_CONFIG,sort_keys=True).encode()).hexdigest()
    if px['status'] in ['BOTH_500_COMPLETE_PENDING_SUMMARY','COMPLETED_FIXED_500']:
        assert b['estimates']==px['initial_estimates'] and precision_input_hashes(b)==px['input_hashes']
        assert all(b['bootstrap'][k]['next_replicate']==500 for k in ['primary','strict'])
        say(dict(stage='precision_merge',reuse=True));return
    states={clock:pd.read_pickle(precision_path(clock)) for clock in ['primary','strict']}
    assert b['estimates']==px['initial_estimates']
    assert precision_input_hashes(b)==px['input_hashes']
    source_sha=hashlib.sha256(json.dumps([str(v) for v in b['source'].subject_id.unique()]).encode('utf-8')).hexdigest()
    for clock,state in states.items():
        old=px['initial_bootstrap'][clock]
        assert state['specification']==old['specification']
        assert state['specification']['source_patient_sha256']==source_sha
        assert state['results'][:100]==old['results']
        assert state['next_replicate']==500 and len(state['results'])==500
        assert [r['replicate'] for r in state['results']]==list(range(500))
        assert state['precision_worker']['extension_signature']==px['signature']
    a=states['primary'];c=states['strict']
    assert a['specification']['source_patient_sha256']==c['specification']['source_patient_sha256']
    assert a['rng_state'][0]==c['rng_state'][0] and np.array_equal(a['rng_state'][1],c['rng_state'][1])
    assert a['rng_state'][2:]==c['rng_state'][2:]
    for x,y in zip(a['results'],c['results']):
        if x['success'] and y['success']:assert np.isclose(x['risk_treated'],y['risk_treated'],atol=1e-12,rtol=0)
    b['bootstrap']=states
    px['status']='BOTH_500_COMPLETE_PENDING_SUMMARY';px['merged_utc']=m.utc()
    px['preservation_checks']=dict(initial100_records_identical=True,point_estimates_identical=True,
        original_analysis_config_identical=True,same_source_and_rng_both_clocks=True,
        same_treated_risk_common_successful_draws=True)
    b['analysis_phase']['recorded_bootstrap_refits']=1000
    for clock,state in states.items():
        info=state['precision_worker']
        b['runs'].append(dict(stage='precision_'+clock,utc=info['utc'],seconds=info['seconds'],
                             peak_MB=info['peak_MB'],bytes_read=0))
    save(b)
    # These two small worker files remain resumable until report verification;
    # they are not additional public results or a second source of truth.
    say(dict(stage='precision_merge',completed=True,all_initial_results_preserved=True))

def percentile_mc_interval(values,p,confidence=.95):
    """Pointwise order-statistic MC interval, not a patient uncertainty CI."""
    from scipy.stats import binom
    x=np.asarray(values,dtype=float);x=np.sort(x[np.isfinite(x)]);n=len(x)
    if not n:raise ValueError('No finite bootstrap values')
    assert 0<p<1 and 0<confidence<1
    tail=(1-confidence)/2
    lo=int(binom.ppf(tail,n,p));hi=int(binom.ppf(1-tail,n,p))+1
    q=float(np.quantile(x,p))
    lower=float(x[lo-1]) if lo>=1 else -np.inf
    upper=float(x[hi-1]) if hi<=n else np.inf
    return dict(p=p,n=n,quantile=q,lower_order=lo,upper_order=hi,
        mc_lower=lower,mc_upper=upper,maximum_mc_error=max(q-lower,upper-q),
        pointwise_confidence=confidence)

def precision_summary_tests():
    x=np.arange(1.,501.)
    a=percentile_mc_interval(x,.025);c=percentile_mc_interval(x,.975)
    assert (a['lower_order'],a['upper_order'],a['mc_lower'],a['mc_upper'])==(6,21,6.,21.)
    assert (c['lower_order'],c['upper_order'],c['mc_lower'],c['mc_upper'])==(480,495,480.,495.)
    short=percentile_mc_interval(np.arange(100.),.025)
    assert short['lower_order']==0 and np.isneginf(short['mc_lower'])
    tied=percentile_mc_interval(np.ones(500),.025)
    assert tied['mc_lower']==tied['mc_upper']==tied['quantile']==1.
    a2=percentile_mc_interval(np.r_[x,np.nan,np.inf],.025)
    assert a2==a
    return ['binomial_order_indices_500','unbounded_tail_not_clipped_100',
            'tied_values_retained','metric_specific_finite_denominator']

def precision_summarize():
    """Append precision results; retain all original100 result rows unchanged."""
    b=load();m=base();px=b['precision_extension']
    assert px['config']==PRECISION_CONFIG and b['estimates']==px['initial_estimates']
    assert precision_input_hashes(b)==px['input_hashes']
    tests=precision_summary_tests();summaries=[];mc=[];warnings=[]
    # Idempotent summaries use only dedicated new sections. Legacy results are
    # still present as the initial100 evidence, not overwritten or renamed.
    b['aggregates']=[r for r in b['aggregates'] if not r['section'].startswith('seq_precision_')]
    assert b['aggregates']==px['initial_aggregates']
    for clock in ['primary','strict']:
        state=b['bootstrap'][clock];assert state['next_replicate']==500
        assert state['results'][:100]==px['initial_bootstrap'][clock]['results']
        bs=pd.DataFrame(state['results'])
        assert bs.replicate.tolist()==list(range(500))
        good=bs[bs.success]
        warning_count=int(good.warnings.map(bool).sum())
        warnings.append(dict(clock=clock,attempts=500,successful=len(good),failed=500-len(good),
                             with_warnings=warning_count,undefined_RR=int((~np.isfinite(good.risk_ratio)).sum())))
        for key,value in warnings[-1].items():
            if key!='clock':row(b,'precision_bootstrap',key,value=str(value),stratum=clock,note='重抽样次数，不是患者计数；失败不补抽')
        for err,n in bs.loc[~bs.success].get('error_type',pd.Series(dtype=object)).value_counts().items():
            row(b,'precision_bootstrap','failure_'+str(err),value=str(int(n)),stratum=clock)
        for prefix in PRECISION_CONFIG['prefixes']:
            sub=bs.iloc[:prefix];ok=sub[sub.success]
            for key in ['risk_treated','risk_control','risk_difference','risk_ratio']:
                vals=pd.to_numeric(ok[key],errors='coerce').replace([np.inf,-np.inf],np.nan).dropna()
                assert len(vals)>0
                lo,hi=vals.quantile([.025,.975]).to_numpy()
                record=dict(clock=clock,attempts=prefix,successful=len(ok),metric=key,valid_n=len(vals),
                    point=b['estimates'][clock][key],lower=float(lo),upper=float(hi))
                summaries.append(record)
                row(b,'precision_effects',key,value='{:.9f}; percentile95% [{:.9f}, {:.9f}]'.format(record['point'],lo,hi),
                    stratum=clock+' / '+str(prefix)+' attempts',
                    note='同种子原重抽样前缀；有效次数{}；点估计未改变；探索性加权比较'.format(len(vals)))
                if prefix==500:
                    for p in [.025,.975]:
                        result=percentile_mc_interval(vals,p);result.update(clock=clock,metric=key);mc.append(result)
                        row(b,'precision_MC',key,value='endpoint={:.9f}; MC95% [{:.9f}, {:.9f}]; maximum_distance={:.9f}'.format(
                            result['quantile'],result['mc_lower'],result['mc_upper'],result['maximum_mc_error']),
                            stratum=clock+' / percentile '+str(p),
                            note='端点的pointwise模拟误差；不是临床效应CI，也不是全部端点联合95%；有效重复数{}'.format(len(vals)))
    px['summary']=pd.DataFrame(summaries);px['endpoint_mc']=pd.DataFrame(mc);px['bootstrap_counts']=warnings
    px['tests']=tests;px['status']='COMPLETED_FIXED_500';px['completed_utc']=m.utc()
    px['RD_mc_reference_met']={clock:bool(px['endpoint_mc'].loc[
        px['endpoint_mc'].clock.eq(clock)&px['endpoint_mc'].metric.eq('risk_difference'),'maximum_mc_error'].max()<=.01)
        for clock in ['primary','strict']}
    for clock,value in px['RD_mc_reference_met'].items():
        row(b,'precision_MC','RD_maximum_MC_distance_within_1pp',value=str(value),stratum=clock,
            note='研究者选定数值参考；不是临床意义/发表/因果识别门槛；未达也不按跨零与否自动扩算')
    b['analysis_phase']['post_estimation_checks'].update(
        unique_replicates_0_through_499=True,initial100_and_point_estimates_unchanged=True,
        complete_membership_outcome_source_hashes_checked=True,binomial_MC_summary_tests=tests)
    b['analysis_phase']['result_status']='COMPLETED_EXPLORATORY_WEIGHTED_ASSOCIATION_500_BOOTSTRAPS_PER_CLOCK'
    b['stage']='EXPLORATORY_HC_ATT_COMPLETED_500_PER_FIXED_CLOCK';save(b)
    say(dict(stage='precision_summarize',completed=True,bootstrap_counts=warnings,
             RD_mc_reference_met=px['RD_mc_reference_met'],original_results_unchanged=True))

def finalize_clinical():
    """Summarize all fixed results; no refitting or selection on effect direction."""
    b=load();m=base();assert b['analysis_phase']['decision']=='GO_EXPLORATORY_FIXED_ATT'
    primary=b['bootstrap']['primary'];strict=b['bootstrap']['strict']
    assert primary['specification']['source_patient_sha256']==strict['specification']['source_patient_sha256']
    assert primary['specification']['seed']==strict['specification']['seed']
    for state in [primary,strict]:
        assert [r['replicate'] for r in state['results']]==list(range(100))
    for a,c in zip(primary['results'],strict['results']):
        if a['success'] and c['success']:
            assert np.isclose(a['risk_treated'],c['risk_treated'],atol=1e-12,rtol=0)
    b['analysis_phase']['post_estimation_checks']={'same_source_patient_draws_both_clocks':True,
        'unique_replicates_0_through_99':True,'same_treated_risk_each_common_successful_draw':True,
        'independent_point_arithmetic_max_difference':4e-15,
        'hospital_endpoint_alignment_independently_checked':True}
    b['aggregates']=[r for r in b['aggregates'] if r['section'] not in ['seq_effects','seq_bootstrap','seq_balance_detail']]
    for clock in ['primary','strict']:
        for r in b['support'][clock]['balance'].itertuples():
            row(b,'balance_detail',r.variable,value='SMD_before={:.6f}; SMD_after={:.6f}'.format(r.smd_before,r.smd_after),stratum=clock+' / '+r.kind,
                note='完整固定变量/缺失状态平衡；device编码0无设备、1普通氧疗、2HFNC、3NIV面罩、4气管导管、5气切相关、7其他记录设备；不是临床功能已核验')
        bs=pd.DataFrame(b['bootstrap'][clock]['results'])
        assert len(bs)==100 and b['bootstrap'][clock]['next_replicate']==100
        good=bs[bs.success];point=b['estimates'][clock]
        for key in ['risk_treated','risk_control','risk_difference','risk_ratio']:
            vals=pd.to_numeric(good[key],errors='coerce').replace([np.inf,-np.inf],np.nan).dropna()
            ci=vals.quantile([.025,.975]).to_numpy() if len(vals) else [np.nan,np.nan]
            row(b,'effects',key,value='{:.6f}; percentile95% [{:.6f}, {:.6f}]'.format(point[key],*ci),stratum=clock,
                note='固定探索性ATT加权比较；初步100次患者bootstrap，不等于已验证治疗效果')
            row(b,'effects','riskset_uniform_unadjusted_'+key,value='{:.6f}'.format(point['riskset_uniform_unadjusted'][key]),stratum=clock,
                note='每个风险集均匀对照的未调整描述，不是全NE来源简单早晚分组')
        warn=good.warnings.map(lambda x:bool(x) if isinstance(x,list) else False) if 'warnings' in good else pd.Series(False,index=good.index)
        row(b,'bootstrap','completed_success_failed',value='{}/{}/{}'.format(len(bs),len(good),int((~bs.success).sum())),stratum=clock,note='重复次数，不是患者计数')
        row(b,'bootstrap','successful_replicates_with_warnings',value=str(int(warn.sum())),stratum=clock)
        row(b,'bootstrap','undefined_RR_among_success',value=str(int((~np.isfinite(good.risk_ratio)).sum())),stratum=clock)
        row(b,'bootstrap','checkpoint_cumulative_seconds',value='{:.2f}'.format(b['bootstrap'][clock]['total_seconds']),stratum=clock,
            note='按checkpoint累计的重抽样函数运行时间；不包含所有导入/报告时间')
        for error,n in bs.loc[~bs.success].get('error_type',pd.Series(dtype=object)).value_counts().items():
            row(b,'bootstrap','failure_'+str(error),value=str(int(n)),stratum=clock)
    b['analysis_phase']['finalized_utc']=m.utc()
    b['analysis_phase']['result_status']='COMPLETED_EXPLORATORY_WEIGHTED_ASSOCIATION_100_BOOTSTRAPS_PER_CLOCK'
    b['stage']='EXPLORATORY_HC_ATT_COMPLETED_100_PER_FIXED_CLOCK';save(b)
    say(dict(stage='finalize_clinical',completed=True,models_or_thresholds_searched=False))

def clinical_report_section(b):
    m=base();z=b['full_risksets'];case=z.treated.eq(1);n=int(case.sum());ap=b['analysis_phase']
    precision=b.get('precision_extension',{});complete500=precision.get('status')=='COMPLETED_FIXED_500'
    repetitions=500 if complete500 else 100
    text='# 当前HC启动比较：完整风险集、治疗前支持与探索性结果\n\n'
    text+='本阶段只推进首次可观察HC启动与当时尚未启动、随后通常照护的比较。约{}次启动、约{}位分析患者；约{}条风险集行包含重复时点和重复对照，不能作为独立样本量。原自然期限政策的暂停结论不变。\n\n'.format(m.count(n),m.count(z.subject_id.nunique()),m.count(len(z)))
    text+='**当前决定：{}。** {}\n\n'.format(ap.get('decision','PENDING'),ap.get('decision_reason','尚待无结局支持诊断。'))
    if ap.get('result_status','').startswith('COMPLETED_EXPLORATORY_WEIGHTED_ASSOCIATION_'):
        text+='**已完成实际探索性估计，两时钟各{}次患者bootstrap。** '.format(repetitions)
        for clock,label in [('primary','主口径'),('strict','严格时钟')]:
            e=b['estimates'][clock];v=pd.DataFrame(b['bootstrap'][clock]['results']);v=v[v.success]
            ci=100*v.risk_difference.quantile([.025,.975]).to_numpy()
            text+='{}院内死亡RD为{:.2f}个百分点，95%百分位区间[{:.2f}, {:.2f}]；'.format(label,100*e['risk_difference'],*ci)
        intervals=[pd.DataFrame(b['bootstrap'][k]['results']).query('success').risk_difference.quantile([.025,.975]).to_numpy() for k in ['primary','strict']]
        if all(lo<=0<=hi for lo,hi in intervals):
            text+='区间均包含0，既覆盖风险降低，也覆盖风险升高；结果不能作为等效、无效或确定获益的证明。'
        else:
            text+='区间按两个既定口径完整报告，不据区间方向选择分析或作治疗建议。'
        text+='观测支持通过的是本次探索性比较门槛，临床适应证与因果解释仍有下述限制。\n\n'
    text+='这次使用完整对照池，在全部所需索引范围一次投影。资格、既往GC、抗菌药/培养和NE左极限沿用已保存规则；没有按将来的HC、死亡、存活时长或治疗成功筛人。实际注射剂型记录不等于已确认IV途径或200mg/day疗程，感染评估代理也不等于充分复苏或已确诊感染性休克。\n\n'
    text+='## 实际补取与测量基础\n\n'
    text+='前一阶段已完成两张3.1 CSV的60.339GB顺序读取，以及约26MB服务表的读取，另复用已有输注缓存。本轮100至500次续算仅使用既有缓存，没有再次扫描原始表或下载新库。读取字段、过滤、SHA与时钟合同保存在既有private bundle；公开表只含聚合。\n\n'
    if 'new_clinical_features' in b:
        text+='|HC启动前变量|主口径覆盖|严格口径覆盖|\n|---|---:|---:|\n'
        for v in ['MAP','HR','RR','SpO2','lactate','pH_bloodgas','creatinine','platelet','bilirubin','GCS_simultaneous','FiO2','PEEP']:
            vals=[]
            for clock in ['primary','strict']:
                nn=int(b['new_clinical_features'][clock].loc[case,v].notna().sum());pc=m.pct(nn,n)
                vals.append(('{}%'.format(pc)) if pc != '' else '小单元抑制')
            text+='|{}|{}|{}|\n'.format(v,*vals)
        text+='\n检验主口径要求charttime与storetime均在索引前；逐条采样、结果发布及医生获知时间不能由字段名称直接确定。床旁主口径使用测量记录，严格敏感性另要求入库。storetime不是医生首次可见时间的直接证明。两口径保留相同病例与风险集，只改变测量/权重。NE仍使用记录中的输注左极限，未称实时可见；不进行未知盐基换算。治疗限制从本次住院起检查，DNR、DNI、CMO和未知分开。\n\n'
    text+='## 固定支持诊断\n\n'
    text+='自定义固定L2条件启动评分按病例风险集拟合，不是普通全人群PS，也不把相对评分称为分钟级启动概率。每个病例权重1，其全部对照按相对评分归一成总质量1，目标是启动者决策人群ATT。后续HC属于通常照护，不删失；不采用旧期限政策权重或ATO。\n\n'
    text+='调整集在结局比较前保存：年龄、性别、anchor推算的日历区间、ICU与服务类别、NE观察起点左截断、体重、入ICU/住院经过时间；当前NE剂量、前1h变化/峰值/记录覆盖，以及VP、EPI和其他升压药记录；MAP、HR、RR、SpO₂、乳酸、肌酐、血小板、碳酸氢盐、血气pH、胆红素、白细胞、GCS、RASS、FiO₂、PEEP、体温、氧疗设备、CRRT及分别记录的治疗限制状态。连续变量采用风险集等质量下中位数插补与显式缺失指标，分类缺失独立保留。固定sum-loss L2惩罚为1，不搜索参数、不截断权重。\n\n'
    if b.get('support'):
        text+='|时钟|患者级HC ESS|患者级对照ESS|最大对照患者份额|前5患者份额|最大绝对SMD|\n|---|---:|---:|---:|---:|---:|\n'
        for clock,d in b['support'].items():
            a=d['concentration']['treated_patient'];c=d['concentration']['control_patient'];mx=d['balance'].smd_after.abs().max()
            text+='|{}|{:.1f}|{:.1f}|{:.2%}|{:.2%}|{:.3f}|\n'.format(clock,a['ess'],c['ess'],c['maximum_share'],c['top5_share'],mx)
        text+='\nESS以患者全部住院/重复角色的聚合权重计算，另存每个病例风险集的局部集中和病情比较。NE剂量比≤2、MAP差≤15mmHg及已测乳酸差≤2mmol/L仅是固定的粗粒度支持描述，不是筛人规则或已验证的临床可互换界限。全部观测值、缺失状态和分类变量的诊断均保留，不以单个良好SMD宣称无未测混杂。\n\n'
        text+='|时钟|NE/MAP可评价比例|相似NE/MAP对照质量中位数|同时相近乳酸质量中位数|单风险集对照患者ESS中位数|\n|---|---:|---:|---:|---:|\n'
        for clock,d in b['support'].items():
            l=d['local'];ok=l.case_NE_observed_positive&l.case_MAP_observed
            text+='|{}|{}%|{:.1%}|{:.1%}|{:.1f}|\n'.format(clock,m.pct(int(ok.sum()),len(l)),l.loc[ok,'close_NE_MAP_control_mass'].median(),l.loc[ok&l.case_lactate_observed,'close_NE_MAP_plus_lactate_mass'].median(),d['per_case'].control_patient_ess.median())
        text+='\n相似对照质量的分母是每个病例风险集全部加权对照，乳酸列仅在病例三项均已测时汇总。总体平衡不等于每例都完成临床匹配；不因局部质量较少而事后删病例。平衡为拟合样本内诊断，不能当作独立预测验证或模型设定正确的证明。GCS同时完整分量及治疗限制记录仍不充分，严格时钟的床旁可评价比例明显下降。\n\n'
    if 'control_followup' in b:
        text+='## 对照后续记录路径\n\n|时钟|窗口|均匀风险集HC记录质量|ATT加权HC记录质量|\n|---|---:|---:|---:|\n'
        a=public_rows(b)
        for clock in ['primary','strict']:
            for minute in [15,30,60]:
                vals=[]
                for scheme in ['uniform_pool','ATT']:
                    rr=a[(a.section=='seq_control_paths')&(a.metric=='HC_RECORD')&(a.stratum==clock+' / '+scheme+' / '+str(minute)+'min')]
                    vals.append(rr.iloc[0]['value'])
                text+='|{}|{}min|{}|{}|\n'.format(clock,minute,*vals)
        text+='\n其他系统性GC、不确定给药及投影观察提前结束的路径另列聚合表；未检出HC不代表确认持续未治疗。上述未来记录只描述通常照护，不用于资格、分组或删失。当前对照不是强制等待60min，也不是持续不用激素。\n\n'
    if b.get('estimates'):
        text+='## 探索性医院死亡比较\n\n'
        text+='以下是本次固定设计的探索性加权比较；因果解释依赖所述时间、治疗版本和无未测混杂假设。存活出院为本次院内死亡未发生，转出ICU不终止住院随访；该结局受出院实践影响，不等同固定日全因死亡。\n\n'
        text+='|时钟|HC风险（%）|当时未启动风险（%）|RD（百分点）|RR|患者bootstrap完成/成功/失败|\n|---|---:|---:|---:|---:|---:|\n'
        for clock,e in b['estimates'].items():
            bs=pd.DataFrame(b.get('bootstrap',{}).get(clock,{}).get('results',[]));valid=bs[bs.success] if len(bs) else bs
            def interval(k):
                if not len(valid) or k not in valid:return ''
                v=valid[k].replace([np.inf,-np.inf],np.nan).dropna();scale=1 if k=='risk_ratio' else 100
                fmt=' [{:.3f}, {:.3f}]' if k=='risk_ratio' else ' [{:.2f}, {:.2f}]'
                return fmt.format(*(v.quantile([.025,.975]).to_numpy()*scale)) if len(v) else ''
            text+='|{}|{:.2f}{}|{:.2f}{}|{:.2f}{}|{:.3f}{}|{}/{}/{}|\n'.format(clock,100*e['risk_treated'],interval('risk_treated'),100*e['risk_control'],interval('risk_control'),100*e['risk_difference'],interval('risk_difference'),e['risk_ratio'],interval('risk_ratio'),len(bs),len(valid),len(bs)-len(valid))
        text+='\n本次固定{}次患者bootstrap，实际完成与失败见表。源患者的所有住院与病例/对照角色一起重抽，每次重新拟合插补、尺度和评分。失败、警告及未定义RR保留，区间使用各指标有限有效重复的2.5%与97.5%分位数。更多重复改善计算精度，不增加独立临床信息；未根据方向选择时钟或调整集。\n\n'.format(repetitions)
        text+='未调整比较采用同样风险集内均匀对照质量：\n\n|时钟|HC风险|对照风险|RD（百分点）|RR|\n|---|---:|---:|---:|---:|\n'
        for clock,e in b['estimates'].items():
            r=e['riskset_uniform_unadjusted']
            text+='|{}|{:.1%}|{:.1%}|{:.2f}|{:.3f}|\n'.format(clock,r['risk_treated'],r['risk_control'],100*r['risk_difference'],r['risk_ratio'])
        text+='\n该描述不替代固定加权比较，也不是将整个来源库按日后是否治疗分组。未调整与调整后的差异提示两组记录病情及治疗选择明显不同，不能将调整后接近零解释为所有适应证混杂已经消除。\n\n'
        for clock,state in b.get('bootstrap',{}).items():
            rows=state.get('results',[]);ok=[r for r in rows if r.get('success')]
            warning=sum(bool(r.get('warnings')) for r in ok);undefined=sum(not np.isfinite(r.get('risk_ratio',np.nan)) for r in ok)
            text+='{}：成功重复中{}次带数值警告，{}次RR未定义；失败原因按原日志保留，未替换种子或选择性补跑。\n\n'.format(clock,warning,undefined)
        if 'zero_mass_optimization' in ap.get('numerical_runtime_note',{}):
            text+='主口径曾为等价数值加速从第20次已保存断点续算；保留原随机状态，未持久化的重复按原抽样重新计算。该中断不是更换抽样或拟合失败。完整预处理与模型目标不变，仅省去零贡献行的重复矩阵计算，等价性以合成数据核对。\n\n'
        if complete500:
            text+='## 100至500次的区间精度复核\n\n'
            text+='本次扩算在初始100次结果已见后决定，属于数值精度完善。固定扩至500次，不根据区间跨零、效应方向或时钟表现提前停止。前100次记录及点估计逐项一致；原配置、源患者集合与顺序、风险集成员/索引及结局哈希通过校验。两时钟继续原种子521和各自原随机状态。旧100次公开结果行和报告原文保留在原缓存，聚合表原行不覆盖。\n\n'
            text+='|时钟|尝试次数|RD 95%区间（百分点）|RR 95%区间|成功/失败|\n|---|---:|---:|---:|---:|\n'
            for clock in ['primary','strict']:
                for prefix in PRECISION_CONFIG['prefixes']:
                    sub=precision['summary'];sub=sub[sub.clock.eq(clock)&sub.attempts.eq(prefix)].set_index('metric')
                    rd=sub.loc['risk_difference'];rr=sub.loc['risk_ratio']
                    text+='|{}|{}|[{:.2f}, {:.2f}]|[{:.3f}, {:.3f}]|{}/{}|\n'.format(clock,prefix,100*rd.lower,100*rd.upper,rr.lower,rr.upper,int(rd.successful),prefix-int(rd.successful))
            text+='\n前缀按最初100、250、500次尝试截取，并非重新选择种子或仅选成功次数。下表量化500次时RD区间端点的模拟误差；它围绕端点的数值不确定性，不是另一个临床效应区间。\n\n'
            text+='|时钟|端点|端点估计（百分点）|端点95% MC区间（百分点）|最大单侧MC距离（百分点）|\n|---|---|---:|---:|---:|\n'
            for r in precision['endpoint_mc'][precision['endpoint_mc'].metric.eq('risk_difference')].itertuples():
                text+='|{}|{}|{:.2f}|[{:.2f}, {:.2f}]|{:.2f}|\n'.format(r.clock,'下限' if r.p==.025 else '上限',100*r.quantile,100*r.mc_lower,100*r.mc_upper,100*r.maximum_mc_error)
            text+='\nMC区间由二项分布序数构造：对有效重复数N及分位p，l=Binom.ppf(0.025,N,p)、u=Binom.ppf(0.975,N,p)+1，使用从1编号的排序值[X(l),X(u)]；越界保留无穷，不裁至最小/最大值。这是各端点的pointwise 95%覆盖，不是多个端点联合95%。500个有效重复时，2.5%端点用第6–21排序值，97.5%端点用第480–495排序值。100次时一侧可能无界，因此不能以100与500看似接近宣称充分稳定。[SciPy二项分布定义与分位函数](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html)。\n\n'
            text+='在续算前保存的数值参考为：两个RD端点各自距MC区间最远边界均不超过1个百分点。这是报告精度参考，不是临床意义、发表或识别门槛。'
            for clock,met in precision['RD_mc_reference_met'].items():
                text+='{}：{}；'.format(clock,'达到' if met else '未达到')
            text+='无论是否达到，本轮固定停在500次；其他风险与RR的MC区间完整见聚合表。\n\n'
            text+='研究精度另以RD区间半宽描述：'
            for clock in ['primary','strict']:
                q=precision['summary'];q=q[q.clock.eq(clock)&q.attempts.eq(500)&q.metric.eq('risk_difference')].iloc[0]
                text+='{}为{:.2f}个百分点；'.format(clock,50*(q.upper-q.lower))
            text+='这是样本与分析下的估计不确定性，和上表端点MC误差不同。它不代表临床等效界值，也不因区间跨零就认定无效。\n\n'
            text+='**当前研究判断：可以继续整理这项固定问题的探索性研究结果，不按本轮方向修改分析。** 当前数据已支持实际给药、共同经过时间风险集和完整住院结局的可复现比较。临床解释仍取决于HC用途、真实决定时间、复苏状态和治疗限制等记录能否充分代表选择依据；500次重抽样不会补足这些临床信息。结果可供导师讨论研究价值与识别假设，不能据此推荐HC启动或停用。\n\n'
    else:text+='本阶段尚未估计分组死亡风险、RD或RR；不以未运行效果冒充阴性结果。\n\n'
    text+='## 解释边界与复现\n\n'
    text+='仍未直接观察HC的全部用途、院外GC、充分复苏、所有混合休克病因以及真实治疗决定时刻。当前临床状态/用药历史可以降低部分混杂，但缺失指示变量不能恢复未测量的临床事实。本设计比较首次记录启动选择，不能扩展成持续疗程、强制等待或所有NE患者的普遍政策。\n\n'
    text+='原则参考：[Li等的风险集设计](https://doi.org/10.1198/016214501753208573)、[Field与Welsh的聚类bootstrap](https://doi.org/10.1111/j.1467-9868.2007.00593.x)。本次条件评分与ATT组合是明确保存并测试的实现，并非原论文提供的HC完整管线。\n\n'
    text+='主入口仍为`steroid_tte.py`，新增阶段统一使用`sequential_`前缀；实现位于既有`steroid_sequential.py`。读取/模型计划在结局对照前保存，当前签名`{}`。计数1–10及其补集抑制，其他计数取整到十位；百分比和加权诊断依原始分母计算。\n\n'.format(ap['signature'])
    text+='## 本阶段读取与运行成本\n\n|步骤|实际读取GB|秒|进程峰值MB|\n|---|---:|---:|---:|\n'
    for r in b['runs']:
        if r['stage'] in ['full_risksets','clinical_labs','history_features','clinical_chart','new_clinical_features'] or r['stage'].startswith(('support_','bootstrap_','precision_')):
            oldstates=precision.get('initial_bootstrap',b.get('bootstrap',{}))
            seconds=oldstates.get(r['stage'][len('bootstrap_'):],{}).get('total_seconds',r.get('seconds','')) if r['stage'].startswith('bootstrap_') else r.get('seconds','')
            text+='|{}|{:.3f}|{}|{}|\n'.format(r['stage'],r.get('bytes_read',0)/1e9,round(seconds,2) if isinstance(seconds,(int,float)) else seconds,r.get('peak_MB',''))
    text+='\nbootstrap阶段列初始100次的断点累计函数时间；precision阶段列本次新增400次的worker时间。两precision分支并行，时长不能相加作为墙钟时间；内存是各进程峰值，不是新增临床样本。缓存读取不列作原始表扫描量。\n'
    text+='\n## 保留的既有结果与暂停决定\n\n以下保留较早阶段的报告原文；其中“尚未读取／估计”描述当时状态，当前结果以上文为准。初始100次区间另见上文及既有缓存。旧自然期限政策的暂停判断未改变。\n\n'
    return text

def main():
    import sys
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser();ap.add_argument('--stage',required=True,choices=['prepare','validate_source','emar','emar_detail','micro','resolve','candidates','risksets','coverage','checks','clinical_freeze','full_risksets','new_labs','new_chart','cached_history_features','analysis_freeze','build_clinical_features','support_diagnostics','control_followup_description','exploratory','precision_freeze','precision_primary','precision_strict','precision_merge','precision_summarize','finalize_clinical']);args=ap.parse_args()
    say(dict(stage='sequential_'+args.stage,python=sys.executable))
    if args.stage in ['emar','emar_detail','micro']:scan(args.stage)
    else:globals()[args.stage]()

if __name__=='__main__':main()
