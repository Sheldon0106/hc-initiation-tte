# -*- coding: utf-8 -*-
"""Outcome-blind HC evidence and native MIMIC-IV 3.1 eMAR -> POE linkage.

This module never refits a model or changes eligibility, indices, or outcomes.
Exact event keys remain in the existing private checkpoint. Public material is
aggregate-only; the small review pilot contains pseudonyms and relative times.
"""
import argparse
import hashlib
import io
import gzip
import importlib.util
import json
import os
import pickle
import re
import time
from pathlib import Path
from hc_paths import configured_path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = configured_path('mimic_dir', '../restricted/mimic-iv-3.1')
PRIVATE = configured_path('private_dir', '../restricted/hc-private')
POE_CACHE = PRIVATE/'hc_poe_linkage.pkl.gz'
CONFIG = dict(
    version='MIMIC-IV 3.1; no cross-version linkage',
    purpose='evidence review and order linkage; no new effect estimation',
    cohort='all frozen primary HC initiations; unchanged primary risksets',
    evidence='observed infection evaluation + current NE + actual injectable-compatible HC; not validated shock indication',
    clinical_intent='UNKNOWN unless direct source evidence supports a specific interpretation; not inferred from mortality or subsequent course',
    clock='chart<index; labs additionally store<index; bedside late-store separately flagged',
    windows='vitals 1h; respiratory/neuro/temperature 4h; lactate/bloodgas 6h; other labs 24h; code status current admission; NE trajectory 4h',
    selection='three treated initiation-time strata (0-4,4-12,12-24h); stable hash seed523 within stratum; one distinct same-riskset control each',
    sample_scope='six-case material-usability pilot, not prevalence or qualification validation',
    linkage='native emar_id -> poe_id, subject and admission consistency; no nearest-time or cross-version imputation',
    poe_status='final status is retrospective metadata, not necessarily known at index',
    preindex_orders='ordertime<index; equal, after, missing, duplicate, and key conflict separately retained',
    no_changes=['cohort','time_zero','treatment','weights','outcomes','primary_results','bootstrap'],
    clinical_review='structured researcher evidence assessment; no expert adjudication claimed')

EMAR_COLUMNS = ['subject_id','hadm_id','emar_id','emar_seq','poe_id','pharmacy_id',
                'charttime','medication','event_txt','scheduletime','storetime']
POE_COLUMNS = ['poe_id','poe_seq','subject_id','hadm_id','ordertime','order_type',
               'order_subtype','transaction_type','discontinue_of_poe_id',
               'discontinued_by_poe_id','order_status']


def module():
    spec = importlib.util.spec_from_file_location('hc_review_seq', ROOT/'steroid_sequential.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    def save(b):
        pending = m.STATE.with_name(m.STATE.name+'.writing')
        pd.to_pickle(b, pending, compression={'method':'gzip','compresslevel':1})
        os.replace(str(pending), str(m.STATE))
    m.save = save
    return m


def poe_module():
    """Keep new POE evidence separate from the concurrently reused HC cache."""
    m=module(); original_load=m.load
    def load():
        b=original_load()
        if POE_CACHE.exists():
            side=pd.read_pickle(POE_CACHE)
            assert side['base_frozen_signature']==frozen_signature(b)
            b['event_review']=side['event_review']
        check(b)
        return b
    def save(b):
        check(b)
        pending=POE_CACHE.with_name(POE_CACHE.name+'.writing')
        side=pd.read_pickle(POE_CACHE) if POE_CACHE.exists() else {}
        side.update(base_frozen_signature=frozen_signature(b),event_review=b['event_review'])
        pd.to_pickle(side,pending,compression={'method':'gzip','compresslevel':1})
        os.replace(str(pending),str(POE_CACHE))
    m.load=load;m.save=save
    return m


def sha(data):
    return hashlib.sha256(data).hexdigest()


def frozen_signature(b):
    # The effect values and resamples are never passed to evidence builders.
    h = hashlib.sha256()
    for key in ['estimates','bootstrap','precision_extension']:
        h.update(pickle.dumps(b[key], protocol=4))
    h.update(pd.util.hash_pandas_object(b['full_risksets'], index=True).values.tobytes())
    h.update(pd.util.hash_pandas_object(b['source'][['subject_id','hadm_id',
             'hospital_expire_flag','death_time','discharge']], index=True).values.tobytes())
    return h.hexdigest()


def check(b):
    e = b['event_review']
    assert e['config'] == CONFIG
    assert frozen_signature(b) == e['frozen_signature']
    assert sha((ROOT/'steroid_tte_aggregates.csv').read_bytes()) == e['aggregate_sha256']
    return e


def freeze():
    m=module(); b=m.load()
    if 'event_review' in b:
        check(b); m.say(dict(stage='event_review_freeze',reuse=True)); return
    p = b['full_risksets']; c = p[p.treated.eq(1)].copy()
    r=b['resolved_medications']; r=r[r.hc_action].copy()
    targets=c.merge(r[['subject_id','hadm_id','emar_id','chart','store','dose_mg',
                      'route','IV_confirmed','status','partial','detail_conflict']],
        left_on=['subject_id','hadm_id','index'],right_on=['subject_id','hadm_id','chart'],how='left')
    assert targets.emar_id.notna().all()
    b['event_review']=dict(config=CONFIG,frozen_utc=m.base().utc(),
        prior_results_seen=True,frozen_signature=frozen_signature(b),
        aggregate_sha256=sha((ROOT/'steroid_tte_aggregates.csv').read_bytes()),
        prior_report=(ROOT/'steroid_tte.md').read_bytes(),
        targets=targets,runs=[],poe_state='NOT_LOCATED_IN_CURRENT_3_1_DIRECTORY',
        sources=['https://mimic.mit.edu/docs/iv/modules/hosp/emar.html',
                 'https://mimic.mit.edu/docs/iv/modules/hosp/poe.html'])
    m.save(b); m.say(dict(stage='event_review_freeze',candidate_count=m.base().count(len(c)),
        estimated_scan_GB=round((RAW/'icu'/'emar.csv').stat().st_size/1e9,2),
        outcomes_used_for_evidence=False))


def secs(s):
    x=pd.to_datetime(s,errors='coerce',utc=True)
    # Preserve missing/out-of-range timestamps; never turn NaT into a valid time.
    v=x.values.astype('datetime64[ns]').astype('int64').astype(float)/1e9
    return pd.Series(v,index=s.index).where(x.notna())


def project_emar():
    m=module(); b=m.load(); e=check(b); base=m.base()
    if 'emar_links' in e:
        m.say(dict(stage='event_review_emar',reuse=True)); return
    path=RAW/'icu'/'emar.csv'; st=path.stat(); ids=set(e['targets'].emar_id.astype(str))
    assert set(EMAR_COLUMNS)<=set(pd.read_csv(path,nrows=0).columns)
    begin=time.perf_counter(); last=begin; pieces=[]; rows=0
    m.say(dict(stage='event_review_emar_start',bytes=st.st_size,usecols=EMAR_COLUMNS,
               filter='frozen first HC emar_ids only',chunksize=150000))
    for chunk in pd.read_csv(path,usecols=EMAR_COLUMNS,dtype=str,chunksize=150000):
        rows+=len(chunk); z=chunk[chunk.emar_id.isin(ids)]
        if len(z): pieces.append(z.copy())
        if time.perf_counter()-last>30:
            m.say(dict(stage='event_review_emar_progress',rows_read_rounded=base.count(rows),
                seconds=round(time.perf_counter()-begin),peak_MB=base.peak_mb()));last=time.perf_counter()
    assert path.stat().st_size==st.st_size and path.stat().st_mtime_ns==st.st_mtime_ns
    found=pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame(columns=EMAR_COLUMNS)
    found['chart']=secs(found.charttime); found['store']=secs(found.storetime)
    found['scheduled']=secs(found.scheduletime)
    e['emar_links']=found
    e['runs'].append(dict(stage='emar_link_projection',bytes_read=st.st_size,
        rows_read=rows,seconds=round(time.perf_counter()-begin,2),peak_MB=base.peak_mb(),
        path=str(path),mtime_ns=st.st_mtime_ns,projection_sha=sha(found.to_csv(index=False).encode())))
    check(b);m.save(b)
    m.say(dict(stage='event_review_emar_done',matched_events=base.count(found.emar_id.nunique()),
        target_events=base.count(len(ids)),seconds=e['runs'][-1]['seconds'],peak_MB=base.peak_mb()))


def link_state(target, rows):
    if not len(rows): return 'NOT_FOUND'
    if len(rows.drop_duplicates())>1: return 'DUPLICATE_OR_CONFLICT'
    r=rows.iloc[0]
    if str(r.subject_id)!=str(int(target.subject_id)): return 'SUBJECT_KEY_CONFLICT'
    if pd.notna(r.hadm_id) and str(r.hadm_id)!=str(int(target.hadm_id)): return 'ADMISSION_KEY_CONFLICT'
    if not np.isclose(r.chart,target['index'],rtol=0,atol=0): return 'ADMIN_TIME_CONFLICT'
    if pd.isna(r.poe_id) or not str(r.poe_id).strip(): return 'MISSING_POE_ID'
    return 'NATIVE_KEY_AVAILABLE_ADMISSION_UNKNOWN' if pd.isna(r.hadm_id) else 'NATIVE_KEY_AVAILABLE'


def poe_state(target, rows):
    if not len(rows): return 'POE_ID_NOT_FOUND'
    if len(rows.drop_duplicates())>1: return 'DUPLICATE_OR_CONFLICT'
    r=rows.iloc[0]
    if str(r.subject_id)!=str(int(target.subject_id)): return 'SUBJECT_KEY_CONFLICT'
    if pd.isna(r.hadm_id): return 'ADMISSION_KEY_UNKNOWN'
    if str(r.hadm_id)!=str(int(target.hadm_id)): return 'ADMISSION_KEY_CONFLICT'
    if not np.isfinite(r.order): return 'ORDER_TIME_UNKNOWN'
    if r.order<target['index']: return 'ORDER_BEFORE_ADMIN'
    if r.order==target['index']: return 'SAME_TIME_ORDER_ADMIN'
    return 'ORDER_AFTER_ADMIN'


def link_poe(path):
    m=poe_module();b=m.load();e=check(b);base=m.base();p=Path(path).resolve()
    if RAW.resolve() not in p.parents: raise ValueError('REQUIRE_SAME_VERSION_3_1_ROOT')
    assert p.name in ['poe.csv','poe.csv.gz']
    st=p.stat(); signature=(str(p),st.st_size,st.st_mtime_ns)
    if e.get('poe_file_signature')==signature:
        m.say(dict(stage='event_review_poe',reuse=True));return
    assert set(POE_COLUMNS)<=set(pd.read_csv(p,nrows=0).columns)
    ids=set(e['emar_links'].poe_id.dropna().astype(str));pieces=[];related=[];n=0;begin=time.perf_counter();last=begin
    m.say(dict(stage='event_review_poe_start',path=str(p),file_bytes=st.st_size,
        columns=POE_COLUMNS,filter='linked frozen HC first-dose poe_ids plus direct discontinue links; no other hospital-order projection',
        memory='chunked 150000 rows; current checkpoint plus projection'))
    for chunk in pd.read_csv(p,usecols=POE_COLUMNS,dtype=str,chunksize=150000):
        n+=len(chunk);z=chunk[chunk.poe_id.isin(ids)]
        if len(z):pieces.append(z.copy())
        rel=chunk[chunk.discontinue_of_poe_id.isin(ids)|chunk.discontinued_by_poe_id.isin(ids)]
        if len(rel):related.append(rel.copy())
        if time.perf_counter()-last>30:
            m.say(dict(stage='event_review_poe_progress',seconds=round(time.perf_counter()-begin),
                       rows_read_rounded=base.count(n)));last=time.perf_counter()
    assert p.stat().st_size==st.st_size and p.stat().st_mtime_ns==st.st_mtime_ns
    q=pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame(columns=POE_COLUMNS)
    q['order']=secs(q.ordertime);e['poe_projection']=q;e['poe_file_signature']=signature
    rel=pd.concat(related,ignore_index=True) if related else pd.DataFrame(columns=POE_COLUMNS)
    rel['order']=secs(rel.ordertime);e['poe_direct_links']=rel
    e['poe_state']='TARGETED_SAME_VERSION_PROJECTION_COMPLETE'
    e['runs'].append(dict(stage='poe',bytes_read=st.st_size,rows_read=n,seconds=round(time.perf_counter()-begin,2),
                         peak_MB=base.peak_mb(),projection_sha=sha(q.to_csv(index=False).encode())))
    check(b);m.save(b);m.say(dict(stage='event_review_poe_done',seconds=e['runs'][-1]['seconds'],
        peak_MB=e['runs'][-1]['peak_MB'],cache=str(POE_CACHE),base_checkpoint_unchanged=True))


def evidence(b,e):
    p=b['full_risksets'];ix=p.index[p.treated.eq(1)]
    c=p.loc[ix].copy().reset_index(drop=True)
    cf=b['new_clinical_features']['primary'].loc[ix].reset_index(drop=True)
    sf=b['new_clinical_features']['strict'].loc[ix].reset_index(drop=True)
    hf=b['history_features'].loc[ix].reset_index(drop=True)
    cand=b['HC_candidates'].set_index('hadm_id'); data=[]
    for j,t in c.iterrows():
        h=int(t.hadm_id);z=e['targets'][e['targets'].case_id.eq(t.case_id)]
        keys=e['emar_links'][e['emar_links'].emar_id.isin(z.emar_id)]
        states=[link_state(tt,e['emar_links'][e['emar_links'].emar_id.eq(tt.emar_id)]) for _,tt in z.iterrows()]
        lag=(keys.store-keys.chart)/60
        ne=b['NE']; ne=ne[ne.hadm_id.eq(h)&ne.start.lt(t['index'])&ne.end.ge(t['index'])&ne.rate.gt(0)]
        v=dict(case_id=int(t.case_id),subject_id=int(t.subject_id),hadm_id=h,index=t['index'],
            n_index_hc_events=len(z),NE_positive=bool(hf.loc[j,'NE_dose']>0),
            positive_NE_interval_recorded=bool(len(ne)),NE_state=str(hf.loc[j,'NE_state']),
            infection_pair_before_index=True,abx_pair_recorded_before_index=bool(cand.loc[h,'strict_abx_pair']),
            IV_confirmed=bool(z.IV_confirmed.all()),dose_mg=float(z.dose_mg.iloc[0]) if z.dose_mg.nunique()==1 else np.nan,
            emar_state='|'.join(sorted(set(states))),poe_id_available=bool(len(keys) and keys.poe_id.notna().all()),
            dose_conflict=bool(len(z)!=1 or z.dose_mg.nunique()!=1),
            documentation_lag_min=float(lag.iloc[0]) if len(lag)==1 else np.nan,
            scheduled_minus_admin_min=float((keys.scheduled-keys.chart).iloc[0]/60) if len(keys)==1 else np.nan,
            HC_indication='UNKNOWN',specific_IV_IM='IV_CONFIRMED' if z.IV_confirmed.all() else 'UNKNOWN',
            sufficient_resuscitation='UNKNOWN',outpatient_GC_history='UNKNOWN',
            common_clinical_choice='UNKNOWN',independent_expert_adjudication='NOT_PERFORMED')
        for name in ['MAP','lactate','GCS_simultaneous','recorded_DNR','recorded_DNI','recorded_CMO',
                     'recorded_full_code','code_status_unknown','intubation_limit_recorded']:
            v[name]=cf.loc[j,name];v[name+'_strict']=sf.loc[j,name]
        for name in ['NE_dose','NE_change_1h','NE_prior1h_covered_min','VP_record','EPI_record']:
            v[name]=hf.loc[j,name]
        if 'poe_projection' in e:
            kk=keys.poe_id.dropna().unique();q=e['poe_projection'];q=q[q.poe_id.isin(kk)]
            v['poe_time_state']=poe_state(t,q)
            v['linked_order_time']=float(q.order.iloc[0]) if len(q)==1 else np.nan
            v['admin_minus_order_min']=(t['index']-q.order.iloc[0])/60 if len(q)==1 else np.nan
            for name in ['MAP','lactate']:
                observed_age=cf.loc[j,name+'__age_h']
                v[name+'_measurement_after_order']=bool(t['index']-observed_age*3600>q.order.iloc[0]) if len(q)==1 and np.isfinite(observed_age) and np.isfinite(q.order.iloc[0]) else np.nan
        else:v['poe_time_state']='POE_SOURCE_NOT_YET_LOCAL';v['admin_minus_order_min']=np.nan
        data.append(v)
    return pd.DataFrame(data)


def hashed(row,role):
    return sha(('523|'+role+'|'+str(int(row.subject_id))+'|'+str(int(row.hadm_id))+'|'+str(float(row['index']))).encode())


def select_pilot(panel):
    # Only whitelisted pretreatment columns are supplied here, never outcomes.
    cases=panel[panel.treated.eq(1)].copy();cases['rank']=[hashed(r,'case') for _,r in cases.iterrows()]
    selected=[];used=set()
    for lo,hi in [(0,4),(4,12),(12,24)]:
        pool=cases[cases.elapsed_h.ge(lo)&cases.elapsed_h.le(hi)&~cases.subject_id.isin(used)]
        if lo>0:pool=pool[pool.elapsed_h.gt(lo)]
        if not len(pool):pool=cases[~cases.subject_id.isin(used)]
        row=pool.sort_values('rank').iloc[0].drop('rank');selected.append(row);used.add(row.subject_id)
    output=[]
    for j,row in enumerate(selected):
        row=row.copy();row['review_id']='H'+str(j+1);output.append(row)
        pool=panel[panel.case_id.eq(row.case_id)&panel.treated.eq(0)&~panel.subject_id.isin(used)].copy()
        pool['rank']=[hashed(r,'control') for _,r in pool.iterrows()]
        ctrl=pool.sort_values('rank').iloc[0].drop('rank');used.add(ctrl.subject_id)
        ctrl['review_id']='C'+str(j+1);output.append(ctrl)
    result=pd.DataFrame(output).reset_index(drop=True)
    assert len(result)==6 and result.subject_id.nunique()==6
    return result


def safe_category(variable,value):
    text=str(value).strip().lower()
    if variable=='code_status':
        if text=='full code':return 'recorded full code'
        terms=[]
        for lab,regex in [('DNR',r'dnr|dnar|do not resuscitate'),('DNI',r'dni|do not intubate'),('CMO',r'comfort')]:
            if re.search(regex,text):terms.append(lab)
        return '+'.join(terms) if terms else 'other structured status'
    if variable=='intubation_status':
        return 'recorded DNI' if re.search(r'dni|do not intubate|no intubation',text) else 'other structured intubation status'
    allowed={'oxygen_device':{'endotracheal tube','nasal cannula','high flow nasal cannula','bipap mask','cpap mask','none','tracheostomy tube','trach mask'},
             'CRRT_mode':{'cvvh','cvvhd','cvvhdf','none','off','no','stopped','discontinued'}}
    return text if text in allowed.get(variable,set()) else 'other structured category'


def build_pilot(b,e,m):
    base=m.base(); panel=b['full_risksets'];chosen=select_pilot(panel)
    e['pilot_key']=chosen.copy(); meta=[];timeline=[]
    for _,c in chosen.iterrows():
        h=int(c.hadm_id);s=int(c.stay_id);t=float(c['index']);rid=c.review_id
        match=panel.index[(panel.case_id==c.case_id)&(panel.hadm_id==h)&(panel['index']==t)]
        assert len(match)==1;i=match[0];hf=b['history_features'].loc[i];cf=b['new_clinical_features']['primary'].loc[i]
        row=dict(review_id=rid,index_action='HC first administration' if c.treated else 'not initiating at this index',
            NE_elapsed_h=round(c.elapsed_h,2),NE_dose=hf.NE_dose,NE_change_1h=hf.NE_change_1h,
            NE_prior1h_covered_min=hf.NE_prior1h_covered_min,age_band=str(int(hf.age)//10*10)+'s',
            ICU_type=hf.ICU_type,service=hf.service,
            recorded_infection_pair='present by frozen eligibility',
            HC_use='UNKNOWN',resuscitation_sufficient='UNKNOWN',both_choices_reasonable='UNKNOWN',
            researcher_assessment='recorded infection evaluation and active NE; HC indication not identified',
            reviewer_name='',review_date='',reviewer_judgment='',missing_information='')
        for n in ['MAP','lactate','GCS_simultaneous','recorded_DNR','recorded_DNI','recorded_CMO','recorded_full_code']:
            row[n]=cf[n]
        meta.append(row)
        def add(source,var,ch,st,val,unit='',scope='pretreatment'):
            assert ch<t or (scope=='index_action' and ch==t)
            timeline.append(dict(review_id=rid,source=source,variable=var,
                minutes_before_index=round((t-ch)/60,2),value=val,unit=unit,
                documented_before_index='yes' if np.isfinite(st) and st<t else 'no_or_unknown',
                scope=scope))
        for name,ages in [('chart',base.CHART_HOURS),('labs',base.LAB_HOURS)]:
            x=b['clinical_'+name];x=x[x.hadm_id.eq(h)&x.chart.lt(t)]
            if name=='chart':x=x[x.stay_id.eq(s)|x.variable.isin(['code_status','intubation_status'])]
            if name=='labs':x=x[x.store.lt(t)]
            admit=float(b['source'].loc[b['source'].hadm_id.eq(h),'admit'].iloc[0])
            for var,g in x.groupby('variable'):
                lower=admit if var in ['code_status','intubation_status'] else max(admit,t-float(ages[var])*3600)
                g=g[g.chart.ge(lower)].sort_values('chart')
                if var in ['code_status','intubation_status']:g=g.tail(12)
                for rr in g.itertuples():
                    if pd.notna(rr.valuenum):value=float(rr.valuenum)
                    elif name=='chart':value=safe_category(var,rr.value)
                    else:continue
                    add(name,var,float(rr.chart),float(rr.store),value,str(rr.valueuom) if pd.notna(rr.valueuom) else '')
        ne=b['NE'];ne=ne[ne.hadm_id.eq(h)&ne.stay_id.eq(s)&ne.start.lt(t)&ne.end.gt(t-4*3600)]
        for rr in ne.itertuples():
            add('inputevents','NE recorded rate',max(float(rr.start),t-4*3600),float(rr.store),float(rr.dose),'local normalized dose; salt/base unresolved')
        meds=b['resolved_medications'];meds=meds[meds.hadm_id.eq(h)&meds.chart.ge(t-24*3600)&meds.chart.lt(t)]
        for rr in meds.itertuples():
            if rr.is_abx_name and rr.actual_systemic:
                add('eMAR','actual antibacterial',float(rr.chart),float(rr.store),'recorded administration')
        if c.treated:
            z=e['targets'][e['targets'].case_id.eq(c.case_id)]
            for rr in z.itertuples():add('eMAR','HC first dose',t,float(rr.store),float(rr.dose_mg),'mg','index_action')
    meta=pd.DataFrame(meta);tl=pd.DataFrame(timeline).sort_values(['review_id','minutes_before_index'],ascending=[True,False])
    forbidden={'subject_id','hadm_id','stay_id','case_id','index','hospital_expire_flag','death_time','discharge','poe_id','emar_id'}
    assert not forbidden.intersection(meta.columns) and not forbidden.intersection(tl.columns)
    assert tl.minutes_before_index.ge(0).all()
    e['pilot_summary']=meta;e['pilot_timeline']=tl
    meta.to_csv(PRIVATE/'hc_review_cases.csv',index=False,encoding='utf-8-sig')
    tl.to_csv(PRIVATE/'hc_review_timeline.csv',index=False,encoding='utf-8-sig')
    e['pilot_tests']=dict(distinct_patients=True,outcome_columns_absent=True,
        postindex_clinical_rows_absent=True,raw_identifiers_absent=True,
        original_clinical_text_absent=True,patient_outcome_not_used_in_sampling=True,
        researcher_judgment_not_expert_adjudication=True)


def summary_stats(f,m):
    rows=[];n=len(f)
    def add(label,mask,scope='all frozen HC initiations'):
        k=int(pd.Series(mask).fillna(False).sum());rows.append(dict(metric=label,n_exact=k,d_exact=n,
            count=m.count(k),percent=m.pct(k,n),scope=scope))
    add('原始eMAR关联键和给药时刻一致',f.emar_state.eq('NATIVE_KEY_AVAILABLE'))
    add('有原生POE关联键',f.poe_id_available)
    add('索引首剂仅一条且剂量无冲突',~f.dose_conflict)
    add('索引前正速率NE区间记录',f.positive_NE_interval_recorded)
    add('左极限NE规范化剂量可判定且为正',f.NE_positive)
    add('NE重叠或剂量记录仍未解析',f.NE_state.eq('ACTIVE_UNRESOLVED'))
    add('索引前感染评估记录配对',f.infection_pair_before_index)
    add('抗菌药另要求索引前已入库仍满足配对',f.abx_pair_recorded_before_index)
    add('HC具体IV途径已确认',f.IV_confirmed)
    add('HC入库晚于给药记录',f.documentation_lag_min.gt(0))
    add('HC入库晚于给药超过15分钟',f.documentation_lag_min.gt(15))
    add('HC入库晚于给药超过30分钟',f.documentation_lag_min.gt(30))
    add('HC入库早于给药记录',f.documentation_lag_min.lt(0))
    add('HC入库时刻缺失',f.documentation_lag_min.isna())
    for key in ['recorded_DNR','recorded_DNI','recorded_CMO','recorded_full_code','code_status_unknown']:
        add(key,f[key].eq(1))
    add('HC实际用途仍未由结构化记录确认',f.HC_indication.eq('UNKNOWN'))
    return pd.DataFrame(rows)


def tests():
    t=pd.Series(dict(subject_id=11,hadm_id=22,index=100.))
    r=pd.DataFrame([dict(subject_id='11',hadm_id='22',chart=100.,poe_id='fake1',order=90.)])
    assert link_state(t,r)=='NATIVE_KEY_AVAILABLE'
    assert link_state(t,r.assign(hadm_id='23'))=='ADMISSION_KEY_CONFLICT'
    assert link_state(t,r.assign(poe_id=np.nan))=='MISSING_POE_ID'
    assert poe_state(t,r)=='ORDER_BEFORE_ADMIN'
    assert poe_state(t,r.assign(order=100.))=='SAME_TIME_ORDER_ADMIN'
    assert poe_state(t,r.assign(order=101.))=='ORDER_AFTER_ADMIN'
    assert poe_state(t,r.assign(hadm_id=np.nan))=='ADMISSION_KEY_UNKNOWN'
    assert poe_state(t,pd.concat([r,r.assign(order=99.)]))=='DUPLICATE_OR_CONFLICT'
    assert poe_state(t,r.iloc[:0])=='POE_ID_NOT_FOUND'
    return 9


def poe_clock_features(b,e,m):
    """Exact selected chart/store times; no float32-age boundary reconstruction.

    Retain the original administration-anchored lower lookback bound. Only the
    information cutoff changes to the verified linked order time. These are
    descriptive HC-case features, not a new order-time cohort or risk set.
    """
    base=m.base();panel=b['full_risksets'];ix=panel.index[panel.treated.eq(1)]
    z=panel.loc[ix,['case_id','subject_id','hadm_id','stay_id','index','elapsed_h']].reset_index(drop=True)
    ev=e['evidence'].set_index('case_id').loc[z.case_id]
    order=ev.linked_order_time.to_numpy(float).copy()
    okay=ev.poe_time_state.isin(['ORDER_BEFORE_ADMIN','SAME_TIME_ORDER_ADMIN']).to_numpy()
    times=z['index'].to_numpy();order[~okay]=np.nan
    admit=z.hadm_id.map(b['source'].set_index('hadm_id').admit).to_numpy(float)
    n=len(z);groups_h={k:np.asarray(v) for k,v in z.groupby('hadm_id',sort=False).indices.items()}
    groups_s={k:np.asarray(v) for k,v in z.groupby(['hadm_id','stay_id'],sort=False).indices.items()}
    outputs={c:{a:dict(value={},chart={},store={},flag={}) for a in ['administration','order_cutoff']} for c in ['primary','strict']}
    cats={}
    for source,variables,ages in [('labs',m.HC_CF_LABS,base.LAB_HOURS),('chart',m.HC_CF_CHART,base.CHART_HOURS)]:
        data=b['clinical_'+source]
        for variable in variables:
            v=data[data.variable.eq(variable)].copy()
            values,valid,cat=m.hc_cf_normalize(v,variable);cats[variable]=cat
            v['_val']=values;v['_valid']=valid
            hosp=source=='labs' or variable in ['code_status','intubation_status']
            keys=['hadm_id'] if hosp else ['hadm_id','stay_id'];lookup=groups_h if hosp else groups_s
            lower=admit if variable in ['code_status','intubation_status'] else np.maximum(admit,times-float(ages[variable])*3600.)
            for clock in outputs:
                for anchor in outputs[clock]:
                    a=outputs[clock][anchor]
                    for key in ['value','chart','store']:a[key][variable]=np.full(n,np.nan)
                    a['flag'][variable]=np.zeros(n,dtype='uint8')
            for key,g in v.groupby(keys,sort=False):
                positions=lookup.get(key)
                if positions is None:continue
                ch=g.chart.to_numpy(float);st=g.store.to_numpy(float)
                val=g._val.to_numpy(float);good=g._valid.to_numpy(bool)
                for clock in outputs:
                    strict=source=='labs' or clock=='strict'
                    for anchor,cut in [('administration',times),('order_cutoff',order)]:
                        got=m.hc_cf_asof(ch,st,val,good,cut[positions],lower[positions],strict)
                        out=outputs[clock][anchor]
                        out['value'][variable][positions]=got[0];out['chart'][variable][positions]=got[3]
                        out['flag'][variable][positions]=got[2]
                        for local,pos in enumerate(positions):
                            if got[2][local]!=1:continue
                            matching=(ch==got[3][local])&good&np.isfinite(st)&np.isclose(val,float(got[0][local]),rtol=1e-6,atol=1e-6)
                            if strict:matching&=st<cut[pos]
                            # Earliest entry of the selected recorded value is
                            # documentation evidence, not clinician visibility.
                            if matching.any():out['store'][variable][pos]=float(np.min(st[matching]))
    for clock in outputs:
        for anchor in outputs[clock]:
            out=outputs[clock][anchor]
            for combined,a,c in [('MAP','MAP_arterial','MAP_noninvasive'),('temperature_C_combined','temperature_C','temperature_F')]:
                ac=out['chart'][a];cc=out['chart'][c]
                use=np.isfinite(ac)&(~np.isfinite(cc)|(ac>=cc))
                for k in ['value','chart','store','flag']:out[k][combined]=np.where(use,out[k][a],out[k][c])
            chs=np.column_stack([out['chart'][v] for v in ['GCS_eye','GCS_verbal','GCS_motor']])
            vals=np.column_stack([out['value'][v] for v in ['GCS_eye','GCS_verbal','GCS_motor']])
            sts=np.column_stack([out['store'][v] for v in ['GCS_eye','GCS_verbal','GCS_motor']])
            good=np.isfinite(chs).all(1)&(chs==chs[:,[0]]).all(1)&np.isfinite(vals).all(1)
            out['value']['GCS_simultaneous']=np.where(good,vals.sum(1),np.nan)
            out['chart']['GCS_simultaneous']=np.where(good,chs[:,0],np.nan)
            out['store']['GCS_simultaneous']=np.where(good&np.isfinite(sts).all(1),sts.max(1),np.nan)
    numerical=[v for v in m.ANALYSIS_CONFIG['numerical'] if v in outputs['primary']['administration']['value']]
    rows=[];private=[]
    for clock in outputs:
        original=b['new_clinical_features'][clock].loc[ix].reset_index(drop=True)
        now=outputs[clock]['administration'];prior=outputs[clock]['order_cutoff']
        for v in numerical:
            a=now['value'][v];p=prior['value'][v];ch=now['chart'][v];st=now['store'][v]
            assert np.allclose(a,original[v].to_numpy(float),equal_nan=True,rtol=1e-6,atol=1e-6),v
            available=okay&np.isfinite(a);pre=okay&np.isfinite(p)
            after=available&(ch>order);at=available&(ch==order)
            late=available&(ch<order)&np.isfinite(st)&(st>=order)
            equal=available&pre&np.isclose(a,p,rtol=1e-6,atol=1e-6)
            assert np.all(ch[available]<times[available])
            assert np.all(prior['chart'][v][pre]<order[pre])
            row=dict(clock=clock,variable=v,verified_preorder_n=int(okay.sum()),
                administration_available_n=int(available.sum()),preorder_available_n=int(pre.sum()),
                sampled_after_order_n=int(after.sum()),sampled_same_second_n=int(at.sum()),
                sampled_before_but_documented_at_or_after_order_n=int(late.sum()),
                missing_documentation_time_n=int((available&~np.isfinite(st)).sum()),
                lost_at_order_cutoff_n=int((available&~pre).sum()),
                gained_at_order_cutoff_n=int((~available&pre).sum()),
                both_available_n=int((available&pre).sum()),value_changed_n=int((available&pre&~equal).sum()))
            rows.append(row)
            private.append(pd.DataFrame(dict(case_id=z.case_id,clock=clock,variable=v,
                linked_order=order,admin=times,admin_value=a,selected_chart=ch,selected_store=st,
                order_cutoff_value=p,order_cutoff_chart=prior['chart'][v],
                sampled_after_order=after,sampled_same_second=at,preorder_sample_late_entry=late)))
    return pd.DataFrame(rows),pd.concat(private,ignore_index=True),z,order,okay


def poe_review():
    m=poe_module();b=m.load();e=check(b);base=m.base();begin=time.perf_counter()
    assert 'poe_projection' in e
    signature=frozen_signature(b);source_stat=(m.STATE.stat().st_size,m.STATE.stat().st_mtime_ns)
    e['evidence']=evidence(b,e)
    # Synthetic exact-time checks: exclude equal-time observations and retain
    # a sample-versus-entry distinction without making visibility claims.
    assert tests()==9
    ch=np.array([90.,100.,110.]);st=np.array([105.,100.,111.]);val=np.array([1.,2.,3.])
    query=np.array([100.,120.]);lower=np.array([0.,0.])
    strict=m.hc_cf_asof(ch,st,val,np.ones(3,bool),query,lower,True)
    measured=m.hc_cf_asof(ch,st,val,np.ones(3,bool),query,lower,False)
    assert np.isnan(strict[0][0]) and strict[0][1]==3.
    assert measured[0][0]==1. and measured[3][0]==90.
    assert np.isnan(m.hc_cf_asof(ch,st,val,np.ones(3,bool),np.array([80.]),np.array([90.]),False)[0][0])
    table,private,z,orders,okay=poe_clock_features(b,e,m)
    for variable in ['MAP','lactate']:
        exact=private[private.clock.eq('primary')&private.variable.eq(variable)].set_index('case_id')
        available=np.isfinite(exact.linked_order)&np.isfinite(exact.admin_value)
        flags=pd.Series(np.where(available,exact.sampled_after_order.astype(float),np.nan),index=exact.index)
        e['evidence'][variable+'_measurement_after_order']=e['evidence'].case_id.map(flags)
    ev=e['evidence'].set_index('case_id').loc[z.case_id].reset_index(drop=True)
    ctx=m.context(b);admit=z.hadm_id.map(b['source'].set_index('hadm_id').admit).to_numpy(float)
    hf=b['history_features'].loc[b['full_risksets'].treated.eq(1)].reset_index(drop=True)
    rows=[];ng={int(s):g for s,g in b['NE'].groupby('stay_id',sort=False)}
    for j,t in z.iterrows():
        row=dict(case_id=int(t.case_id),order_time_usable=bool(okay[j]),
                 original_eligibility_unchanged=True)
        if okay[j]:
            o=float(orders[j]);a=float(t['index']);g=ng.get(int(t.stay_id))
            active=bool(g is not None and ((g.start<o)&(g.end>=o)&(g.rate>0)).any())
            qstate,_=m.qualify(int(t.hadm_id),o,ctx,base,is_case=True)
            row.update(order_before_current_admission=bool(o<admit[j]),
                order_before_first_observed_NE=bool(o<a-float(t.elapsed_h)*3600),
                order_before_current_ICU=bool(o<a-float(hf.loc[j,'ICU_hours'])*3600),
                active_NE_interval_at_order=active,infection_and_GC_record_state_at_order=qstate)
            if active:
                doses,_,flags=m.ne_values_at(g,np.array([o,a]))
                row.update(NE_order=float(doses[0]),NE_admin=float(doses[1]),NE_order_state=str(flags[0]),
                    NE_dose_changed=bool(not np.isclose(doses[0],doses[1],rtol=1e-7,atol=1e-9)) if np.isfinite(doses).all() else np.nan)
        rows.append(row)
    context=pd.DataFrame(rows)
    # Direct links are retrospective evidence only, never a baseline exclusion.
    rel=e.get('poe_direct_links',pd.DataFrame());links=[]
    for _,t in z.iterrows():
        targets=e['targets'][e['targets'].case_id.eq(t.case_id)]
        em=e['emar_links'][e['emar_links'].emar_id.isin(targets.emar_id)]
        keys=set(em.poe_id.dropna().astype(str));r=rel[(rel.discontinue_of_poe_id.isin(keys)|rel.discontinued_by_poe_id.isin(keys))].copy()
        if len(r):
            compatible=pd.to_numeric(r.subject_id,errors='coerce').eq(t.subject_id)&pd.to_numeric(r.hadm_id,errors='coerce').eq(t.hadm_id)
            successors=r[r.discontinue_of_poe_id.isin(keys)&compatible]
            predecessors=r[r.discontinued_by_poe_id.isin(keys)&compatible]
            links.append(dict(case_id=int(t.case_id),key_conflict=bool((~compatible).any()),
                successor_before_admin=bool(successors.order.lt(t['index']).any()),
                successor_at_admin=bool(successors.order.eq(t['index']).any()),
                successor_after_admin=bool(successors.order.gt(t['index']).any()),
                predecessor_present=bool(len(predecessors)),
                missing_successor_time=bool(successors.order.isna().any())))
    lag=ev.admin_minus_order_min;valid=ev.poe_time_state.isin(['ORDER_BEFORE_ADMIN','SAME_TIME_ORDER_ADMIN','ORDER_AFTER_ADMIN'])
    out=dict(contract=dict(population='all frozen actual HC initiations; no exclusions or effects',
        order='native eMAR-linked POE version; not necessarily earliest HC order or first clinical decision',
        denominator='all native-linked events for time summaries; verified before/same order for feature comparisons; available-at-admin subsets explicitly separate',
        lookback='original administration-anchored lookback held fixed; cutoff only moved to linked order; this is NOT an order-time emulation',
        clock='primary: chart for bedside and chart+store for labs; strict: chart+store for all; store is record entry, not guaranteed clinical availability',
        exact_time='raw selected timestamps are used; float32 measurement ages are not used to classify boundary seconds',
        order_status='Inactive and successor links are retrospective metadata, not proof of non-administration or baseline cancellation',
        history='existing candidate projections only; absent earlier evidence is not proof of absent treatment',
        unsupported='HC indication, first clinician decision, prehospital steroids and treatment intentions remain unobserved'),
        cohort_n=len(ev),native_key_status=ev.poe_time_state.value_counts().to_dict(),
        time_quantiles_minutes=lag[valid].quantile([.1,.25,.5,.75,.9,.95]).to_dict(),
        lag_thresholds={str(k):dict(n=int((valid&lag.gt(k)).sum()),denominator=int(valid.sum())) for k in [15,30,60,120]},
        order_metadata={v:e['poe_projection'][v].fillna('MISSING').value_counts().to_dict() for v in ['order_type','order_subtype','transaction_type','order_status']},
        feature_summary=table,feature_evidence=private,context=context,direct_link_summary=pd.DataFrame(links),
        tests=dict(native_link_synthetic=9,exact_time_cutoff_tests=3,
            frozen_numeric_features_reproduced_both_clocks=True,all_order_cutoff_observations_strictly_before_order=True,
            no_new_outcome_or_weight_estimation=True,base_frozen_signature_unchanged=signature==frozen_signature(b)))
    e['poe_analysis']=out
    e['runs'].append(dict(stage='poe_cached_time_review',seconds=round(time.perf_counter()-begin,2),
        bytes_read_raw=0,peak_MB=base.peak_mb()))
    assert source_stat==(m.STATE.stat().st_size,m.STATE.stat().st_mtime_ns)
    check(b);m.save(b)
    m.say(dict(stage='poe_cached_time_review_done',seconds=e['runs'][-1]['seconds'],
        candidate_events=base.count(len(ev)),feature_rows=len(table),cache=str(POE_CACHE),
        base_checkpoint_unchanged=True,effect_models_run=0))


def publish_poe():
    """Update only the canonical author-facing POE results section."""
    m=poe_module();b=m.load();e=check(b);base=m.base();a=e['poe_analysis']
    path=ROOT/'steroid_tte.md';before=path.read_bytes();text=before.decode('utf-8')
    newline='\r\n' if '\r\n' in text else '\n';text=text.replace('\r\n','\n')
    protected=[m.STATE,ROOT/'steroid_tte_aggregates.csv',ROOT/'steroid_results.pdf',
               ROOT/'steroid_results.png',PRIVATE/'phenotype_discovery_model.json']
    metadata={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in protected}
    def fraction(n,d):
        pct=base.pct(int(n),int(d))
        return ('抑制' if pct=='' else str(pct)+'%')+'（'+base.count(n)+'/'+base.count(d)+'）'
    def percent(n,d):
        pct=base.pct(int(n),int(d));return '抑制' if pct=='' else str(pct)+'%'
    n=a['cohort_n'];states=a['native_key_status'];q=a['time_quantiles_minutes']
    compatible=sum(states.get(k,0) for k in ['ORDER_BEFORE_ADMIN','SAME_TIME_ORDER_ADMIN','ORDER_AFTER_ADMIN'])
    lines=['<!-- HC_POE_REVIEW_START -->','### POE关联与医嘱前信息核查','',
        '**本轮结论：**原生关联支持现有实际HC首剂事件的可追溯性，同时揭示给药起点与关联医嘱起点之间不可忽略的间隔。现用给药前变量中，较多生命体征和部分检验是在医嘱之后取得；此前统一提前15/30分钟的敏感性不能替代逐例医嘱时间核查。当前死亡估计、风险集、权重和bootstrap均未改变。','',
        '#### 原生关联与时间差','',
        '来源为用户提供的同版MIMIC-IV 3.1 `'+e['poe_file_signature'][0]+'`。POE虽保存在本地icu子目录，仍按医院医嘱来源解释；表头、原生POE键、患者键及住院键均作核对。仅保留冻结首剂的关联行及直接撤回关联，不以近邻时间猜测匹配。','',
        '|核查项目|结果|','|---|---|',
        '|原生医嘱关联、患者及住院键可核实|'+fraction(compatible,n)+'|',
        '|医嘱早于给药|'+fraction(states.get('ORDER_BEFORE_ADMIN',0),n)+'|',
        '|医嘱与给药同秒|'+fraction(states.get('SAME_TIME_ORDER_ADMIN',0),n)+'|',
        '|医嘱晚于给药|'+fraction(states.get('ORDER_AFTER_ADMIN',0),n)+'|',
        '|未找到、键冲突或时间不可判定|'+fraction(n-compatible,n)+'|',
        '|给药减医嘱时间：中位数（四分位数）|'+f'{q[.5]:.2f}（{q[.25]:.2f}–{q[.75]:.2f}）分钟|',
        '|第90、第95百分位|'+f'{q[.9]:.2f}、{q[.95]:.2f}分钟|']
    for minutes,v in a['lag_thresholds'].items():lines.append('|间隔超过'+minutes+'分钟|'+fraction(v['n'],v['denominator'])+'|')
    lines+=['','时间差分位数使用所有键及时间可核实的首剂，保留负间隔；负间隔病例未被从原研究中排除。后续临床信息诊断仅限关联医嘱早于或等于给药的病例。计数按既定规则取整至10，1–10及可反推小样本的百分比抑制，百分比用内部准确分母计算。','',
        '全部关联医嘱类型为Medications，order_subtype均缺失。New约'+base.count(a['order_metadata']['transaction_type'].get('New',0))+'例、Change约'+base.count(a['order_metadata']['transaction_type'].get('Change',0))+'例。这是eMAR关联的医嘱版本，并未证明它是最早HC医嘱或医生首次决定时刻。全部最终状态为Inactive；已有实际给药证据，不能据该终态标为未执行或基线撤回。','',
        '#### 现用调整变量在医嘱前后的位置','',
        '下表逐变量报告。给药前及医嘱前可用率的分母为医嘱先于或同于给药、关联可靠的固定病例；“医嘱后测量”及“采样早但入库晚”的分母为该变量在给药时点可用的病例。采用原始秒级时间，不以float32测量年龄重构边界。','',
        '**重要口径：**医嘱前截断保留原来的“相对于给药”的回看下界，只剔除医嘱之后的信息。例如医嘱早于给药1小时以上时，原给药前1小时生命体征窗口可能完全没有医嘱前记录。因此下列下降不能解释为按医嘱重新起算1小时后仍然缺乏数据，也不是新医嘱起点队列的覆盖率。','']
    names={'primary':'主测量时钟：床旁按测量，检验同时要求入库','strict':'严格入库时钟：床旁及检验均要求入库'}
    for clock in ['primary','strict']:
        lines += ['**'+names[clock]+'**','',
            '|变量|给药时点可用率|现用值在医嘱后测量|现用值采样早于、但入库不早于医嘱|同一原窗口截断至医嘱前后的可用率|',
            '|---|---:|---:|---:|---:|']
        for r in a['feature_summary'][a['feature_summary'].clock.eq(clock)].itertuples():
            lines.append('|'+r.variable+'|'+percent(r.administration_available_n,r.verified_preorder_n)+'|'+
                percent(r.sampled_after_order_n,r.administration_available_n)+'|'+
                percent(r.sampled_before_but_documented_at_or_after_order_n,r.administration_available_n)+'|'+
                percent(r.preorder_available_n,r.verified_preorder_n)+'|')
        lines.append('')
    lines += ['医嘱后测量仍可能是给药前的真实临床状态，不自动等于HC给药后变量，也不能仅据时间顺序判定偏倚方向。它影响的是将当前比较解释为“开立HC医嘱时的治疗选择”的合理性。storetime仅代表记录入库属性，不被当作医生首次获知结果的确定时刻。','',
        '#### 医嘱时点的记录背景与直接撤回关联','',
        '|项目|结果|','|---|---|']
    c=a['context'];c=c[c.order_time_usable]
    for key,label in [('order_before_current_admission','关联医嘱早于本次入院'),
                      ('order_before_first_observed_NE','关联医嘱早于首次观察到NE'),
                      ('order_before_current_ICU','关联医嘱早于当前ICU入科'),
                      ('active_NE_interval_at_order','医嘱时点已有当前ICU正速率NE区间'),
                      ('NE_dose_changed','医嘱与给药节点NE剂量不同（仅两时点剂量可判定者）')]:
        lines.append('|'+label+'|'+fraction(int(c[key].fillna(False).sum()),int(c[key].notna().sum()))+'|')
    qstate=c.infection_and_GC_record_state_at_order
    lines.append('|缓存中医嘱前已满足原感染配对及既往激素记录条件|'+fraction(int(qstate.eq('QUALIFIED_RECORD_PROXY').sum()),len(c))+'|')
    rel=a['direct_link_summary']
    lines += ['',
        '上述是既有投影中的记录背景，没有据此重新筛选患者。医嘱前未找到正速率区间或感染配对，只记为现有记录不能确认；不等于患者没有接受相应治疗。未额外追查投影之外的更早病程。','',
        '约'+base.count(len(rel))+'例存在直接撤回关联记录；直接关联键未见患者/住院冲突。在已捕获的直接后继撤回记录中，给药前及给药同秒分别为'+base.count(int(rel.successor_before_admin.sum()))+'和'+base.count(int(rel.successor_at_admin.sum()))+'例，撤回发生在给药之后。这个结果不覆盖未关联的口头决定、所有更早医嘱版本或真实临床意图。','',
        '#### 对当前研究的影响及剩余工作','',
        '1. 已解决：首次给药与POE的原生键、住院一致性、实际关联医嘱时间、记录先后关系，以及当前数值来自医嘱前还是医嘱后。','2. 尚未解决：HC具体用途、最初临床决定时刻、院外激素史及未记录的床旁选择依据。Change事务及药物子类型缺失亦不支持把关联行称为首张处方。','3. 当前约250次首剂的逐例诊断不能自动代表全部同期对照。若比较目标改为医嘱启动，必须同时识别对照中已开医嘱但尚未给药者、开医嘱却从未给药者，并重建资格、风险集和权重；不能只把病例索引前移后接用原权重。','4. 若保持实际给药启动的主目标，应明确医嘱早已存在这一记录事实，并另行规定如何对两组对称处理医嘱至给药间的信息。不能仅对启动者截断，或把本表当成已经完成的效果敏感性分析。','5. 本轮不以数据缺口自动否定TTE，也不因关联成功宣布因果识别成立。原主结果保持探索性解释，不据本次诊断改变效应方向或筛选患者。','',
        '官方语义：[eMAR](https://mimic.mit.edu/docs/iv/modules/hosp/emar.html)与[POE](https://mimic.mit.edu/docs/iv/modules/hosp/poe.html)。ordertime为医嘱录入/开立时刻；最终状态及撤回关系按其实际时序解释。','',
        '#### 复现与读取成本','',
        '入口：`steroid_event_review.py link-poe --poe-path <同版POE路径>` → `poe-review` → `poe-publish`。新POE结果仅保存在私有`hc_poe_linkage.pkl.gz`，不回写共享`steroid_sequential.pkl.gz`。`poe-review`仅使用缓存且不拟合模型；旧`build/publish`阶段用于早期材料，不用于发布本次POE结果。']
    scan=[x for x in e['runs'] if x['stage']=='poe'][-1];review=[x for x in e['runs'] if x['stage']=='poe_cached_time_review'][-1]
    lines += ['',f"POE单次扫描{scan['bytes_read']/1e9:.2f}GB，{scan['seconds']:.2f}秒，峰值工作集{scan['peak_MB']/1024:.2f}GiB；随后缓存诊断{review['seconds']:.2f}秒。原有9项关联测试及3项精确时间边界断言通过，主/严格时钟的冻结数值逐项复现。原始表、主分析缓存、模型、死亡结果、bootstrap、聚合CSV及图表未改变。",'',
        '研究伦理/数据授权说明、作者信息和注册情况仍由作者提供。内部规则保存不称为公开预注册。','<!-- HC_POE_REVIEW_END -->','']
    block='\n'.join(lines)
    if '<!-- HC_POE_REVIEW_START -->' in text:
        start=text.index('<!-- HC_POE_REVIEW_START -->');stop=text.index('<!-- HC_POE_REVIEW_END -->',start)+len('<!-- HC_POE_REVIEW_END -->')
        while stop<len(text) and text[stop]=='\n':stop+=1
    else:
        start=text.index('### POE到位后唯一需要补的核查')
        stop=text.index('### 实际代码与证据位置',start)
    updated=text[:start]+block+'\n'+text[stop:]
    assert updated[:start]==text[:start] and updated.endswith(text[stop:])
    after=updated.replace('\n',newline).encode('utf-8')
    if before==after:
        m.say(dict(stage='poe_publish',reuse=True));return
    assert path.read_bytes()==before,'REPORT_CHANGED_DURING_POE_PUBLICATION'
    previous=e.setdefault('poe_report_history',[])
    if not any(x['sha256']==sha(before) for x in previous):previous.append(dict(sha256=sha(before),bytes=before))
    e['poe_executed_source']=Path(__file__).read_bytes()
    e['poe_publication']=dict(report_path=str(path),report_sha256=sha(after),code_sha256=sha(e['poe_executed_source']),
        previous_report_sha256=sha(before),updated_section_only=True,utc=base.utc())
    m.save(b);path.write_bytes(after)
    assert all(metadata[str(p)]==(p.stat().st_size,p.stat().st_mtime_ns) for p in protected)
    m.say(dict(stage='poe_publish_done',report=str(path),only_POE_section_updated=True,base_checkpoint_unchanged=True))


CONTROL_ORDER_CONFIG = dict(
    purpose='describe prior HC orders in all unchanged treated/control riskset memberships',
    cohort='frozen full_risksets; no exclusions, new weights or outcomes',
    capture='all HC-named eMAR rows during full candidate hospitalizations, then native POE and direct discontinue links',
    identity='injectable salt name or previously resolved hc_action; ambiguous HC separate',
    cutoff='order strictly before index; discontinue at or before index not pending; same-second order separate',
    missing='no matched HC order is NOT confirmed order-naive; orders without eMAR require prescriptions/pharmacy',
    future='later eMAR can identify medication attached to an earlier order, not preindex clinical information or live availability',
    comparison='patient counts, repeated membership counts and fixed original ATT mass reported separately',
    effects='none; no refit, no new outcome contrasts')


def order_flags(order, stop, index):
    prior=order<index
    return prior, prior&(~np.isfinite(stop)|(stop>index)), order==index


def control_orders():
    m=poe_module();b=m.load();e=check(b);base=m.base();panel=b['full_risksets']
    if 'control_orders' not in e:
        e['control_orders']=dict(config=CONTROL_ORDER_CONFIG,prior_results_seen=True,
            frozen_utc=base.utc(),runs=[],source_before=Path(__file__).read_bytes())
        m.save(b)
    ext=e['control_orders'];assert ext['config']==CONTROL_ORDER_CONFIG
    p1,p2,p3=order_flags(np.array([9.,10.,11.,9.,9.]),np.array([np.nan,np.nan,np.nan,10.,11.]),10.)
    assert p1.tolist()==[True,False,False,True,True]
    assert p2.tolist()==[True,False,False,False,True]
    assert p3.tolist()==[False,True,False,False,False]
    ref=b['source'].set_index('hadm_id');adms=set(panel.hadm_id);subs=set(panel.subject_id)
    if 'emar' not in ext:
        path=RAW/'icu'/'emar.csv';stat=path.stat();start=time.perf_counter();last=start;parts=[];unknown=[];rows=0
        m.say(dict(stage='control_order_emar_start',bytes=stat.st_size,columns=EMAR_COLUMNS,
            filter='HC-name rows in unchanged riskset hospitalizations; whole admission; missing native admission keys retained separately'))
        for chunk in pd.read_csv(path,usecols=EMAR_COLUMNS,dtype=str,chunksize=150000):
            rows+=len(chunk);h=pd.to_numeric(chunk.hadm_id,errors='coerce');s=pd.to_numeric(chunk.subject_id,errors='coerce')
            name=chunk.medication.fillna('').str.contains(r'hydrocortis|solu.?cort|cortef',case=False,regex=True)
            z=chunk[name&h.isin(adms)].copy()
            if len(z):
                z['hadm_id']=h.loc[z.index].astype('int64');z['subject_id']=s.loc[z.index].astype('int64')
                z['chart']=secs(z.charttime);z['store']=secs(z.storetime)
                assert z.subject_id.eq(z.hadm_id.map(ref.subject_id)).all()
                z=z[z.chart.ge(z.hadm_id.map(ref.admit))&z.chart.le(z.hadm_id.map(ref.discharge))]
                if len(z):parts.append(z)
            u=chunk[name&h.isna()&s.isin(subs)]
            if len(u):unknown.append(u.copy())
            if time.perf_counter()-last>30:
                m.say(dict(stage='control_order_emar_progress',seconds=round(time.perf_counter()-start),rows_read_rounded=base.count(rows)));last=time.perf_counter()
        assert (path.stat().st_size,path.stat().st_mtime_ns)==(stat.st_size,stat.st_mtime_ns)
        z=pd.concat(parts,ignore_index=True).drop_duplicates()
        ext['emar_key_conflicts']=z[z.emar_id.duplicated(False)].copy();z=z[~z.emar_id.duplicated(False)].copy()
        known=set(b['resolved_medications'].loc[b['resolved_medications'].hc_action,'emar_id'])
        z['injectable_compatible']=z.medication.map(base.named_injectable_salt)|z.emar_id.isin(known)
        ext['emar']=z;ext['missing_native_hadm']=pd.concat(unknown,ignore_index=True) if unknown else pd.DataFrame(columns=EMAR_COLUMNS)
        ext['runs'].append(dict(stage='control_order_emar',path=str(path),bytes_read=stat.st_size,mtime_ns=stat.st_mtime_ns,
            seconds=round(time.perf_counter()-start,2),rows_read=rows,peak_MB=base.peak_mb(),projection_sha=sha(z.to_csv(index=False).encode())))
        m.save(b);m.say(dict(stage='control_order_emar_done',retained=base.count(len(z)),seconds=ext['runs'][-1]['seconds']))
    if 'poe' not in ext:
        path=Path(e['poe_file_signature'][0]);stat=path.stat();ids=set(ext['emar'].poe_id.dropna().astype(str))
        start=time.perf_counter();last=start;parts=[];related=[];rows=0
        m.say(dict(stage='control_order_poe_start',bytes=stat.st_size,columns=POE_COLUMNS,
            filter='native HC order identifiers across frozen riskset admissions plus direct discontinue links'))
        for chunk in pd.read_csv(path,usecols=POE_COLUMNS,dtype=str,chunksize=150000):
            rows+=len(chunk);z=chunk[chunk.poe_id.isin(ids)]
            if len(z):parts.append(z.copy())
            rel=chunk[chunk.discontinue_of_poe_id.isin(ids)|chunk.discontinued_by_poe_id.isin(ids)]
            if len(rel):related.append(rel.copy())
            if time.perf_counter()-last>30:
                m.say(dict(stage='control_order_poe_progress',seconds=round(time.perf_counter()-start),rows_read_rounded=base.count(rows)));last=time.perf_counter()
        assert (path.stat().st_size,path.stat().st_mtime_ns)==(stat.st_size,stat.st_mtime_ns)
        q=pd.concat(parts,ignore_index=True).drop_duplicates();q['order']=secs(q.ordertime)
        ext['poe']=q;ext['direct_links']=pd.concat(related,ignore_index=True).drop_duplicates() if related else pd.DataFrame(columns=POE_COLUMNS)
        ext['runs'].append(dict(stage='control_order_poe',path=str(path),bytes_read=stat.st_size,mtime_ns=stat.st_mtime_ns,
            seconds=round(time.perf_counter()-start,2),rows_read=rows,peak_MB=base.peak_mb(),projection_sha=sha(q.to_csv(index=False).encode())))
        m.save(b);m.say(dict(stage='control_order_poe_done',retained=base.count(len(q)),seconds=ext['runs'][-1]['seconds']))
    if 'summary' in ext:
        m.say(dict(stage='control_orders',reuse=True));return
    start=time.perf_counter();raw=ext['emar'];q=ext['poe'];rel=ext['direct_links'].copy();rel['order']=secs(rel.ordertime)
    bad=set(q.loc[q.poe_id.duplicated(False),'poe_id']);q=q[~q.poe_id.isin(bad)].set_index('poe_id')
    orders=[];fail=[]
    for key,g in raw.dropna(subset=['poe_id']).groupby('poe_id',sort=False):
        if key not in q.index:
            fail.append(dict(poe_id=key,reason='MISSING_OR_DUPLICATE'));continue
        r=q.loc[key];h=int(g.hadm_id.iloc[0]);s=int(g.subject_id.iloc[0])
        if g.hadm_id.nunique()!=1 or g.subject_id.nunique()!=1 or str(r.subject_id)!=str(s) or str(r.hadm_id)!=str(h) or not np.isfinite(r.order):
            fail.append(dict(poe_id=key,reason='KEY_OR_TIME_CONFLICT'));continue
        successors=rel[rel.discontinue_of_poe_id.eq(key)&pd.to_numeric(rel.hadm_id,errors='coerce').eq(h)&pd.to_numeric(rel.subject_id,errors='coerce').eq(s)]
        stops=successors.order.dropna()
        stop=float(stops.min()) if len(stops) else np.nan
        orders.append(dict(poe_id=key,hadm_id=h,subject_id=s,order=float(r.order),stop=stop,
            temporal_conflict=bool(np.isfinite(stop) and stop<r.order),
            within_admission=bool(ref.loc[h,'admit']<=r.order<=ref.loc[h,'discharge']),
            injectable_compatible=bool(g.injectable_compatible.any()),first_emar=float(g.chart.min()),first_store=float(g.store.min()),
            emar_not_given_or_delayed=bool(g.event_txt.str.lower().isin(['not given','delayed']).any())))
    orders=pd.DataFrame(orders);ext['order_evidence']=orders;ext['order_failures']=pd.DataFrame(fail)
    flags=pd.DataFrame(False,index=panel.index,columns=['prior_injectable_order','prior_injectable_not_discontinued',
        'prior_ambiguous_HC_order','same_second_injectable_order','prior_order_link_known_only_from_later_emar','order_conflict'])
    groups={int(h):g for h,g in orders.groupby('hadm_id')}
    for h,ix in panel.groupby('hadm_id',sort=False).groups.items():
        g=groups.get(int(h));t=panel.loc[ix,'index'].to_numpy(float)
        if g is None:continue
        for r in g.itertuples():
            if r.temporal_conflict or not r.within_admission:
                flags.loc[ix,'order_conflict']|=r.order<t;continue
            prior,pending,tie=order_flags(r.order,r.stop,t)
            if r.injectable_compatible:
                flags.loc[ix,'prior_injectable_order']|=prior
                flags.loc[ix,'prior_injectable_not_discontinued']|=pending
                flags.loc[ix,'same_second_injectable_order']|=tie
                flags.loc[ix,'prior_order_link_known_only_from_later_emar']|=prior&(r.first_emar>=t)
            else:flags.loc[ix,'prior_ambiguous_HC_order']|=prior
    # Reconstruct unchanged original weights from stored predictors, without fitting.
    weighted={}
    for clock in ['primary','strict']:
        frame=m.analysis_frame(b,clock);plan=m.make_plan(frame,m.ANALYSIS_CONFIG['numerical'],m.ANALYSIS_CONFIG['categorical'],diagnostic_numeric=['NE_dose'])
        assert plan['plan_sha256']==b['support'][clock]['plan_sha256']
        ww=m.att_weights(plan,b['support'][clock]['linear_predictor'])
        back=np.empty(plan['n']);back[plan['order']]=ww['control'];weighted[clock]=back
    summary=[]
    for arm,mask in [('HC',panel.treated.eq(1)),('control',panel.treated.eq(0))]:
        for col in flags:
            hit=flags[col]&mask
            row=dict(arm=arm,metric=col,memberships=int(hit.sum()),membership_denominator=int(mask.sum()),
                patients_ever=int(panel.loc[hit,'subject_id'].nunique()),patient_denominator=int(panel.loc[mask,'subject_id'].nunique()),
                risksets_any=int(panel.loc[hit,'case_id'].nunique()),riskset_denominator=int(panel.loc[mask,'case_id'].nunique()))
            for clock,w in weighted.items():row[clock+'_fixed_weight_share']=float(w[hit].sum()/w[mask].sum()) if arm=='control' else np.nan
            summary.append(row)
    ext['flags']=flags;ext['summary']=pd.DataFrame(summary)
    ext['unknown_control_memberships']=int((panel.treated.eq(0)&~flags.prior_injectable_order).sum())
    ext['checks']=dict(native_keys_checked=True,no_models_fitted=True,no_outcomes_compared=True,
        original_weights_reproduced=True,synthetic_boundary_cases=5,negative_order_state_not_imputed=True)
    ext['runs'].append(dict(stage='control_order_cached_summary',seconds=round(time.perf_counter()-start,2),bytes_read=0,peak_MB=base.peak_mb()))
    ext['executed_source']=Path(__file__).read_bytes();check(b);m.save(b)
    public=[]
    for r in ext['summary'].itertuples():
        if r.metric not in ['prior_injectable_order','prior_injectable_not_discontinued','prior_ambiguous_HC_order']:continue
        public.append(dict(arm=r.arm,metric=r.metric,memberships=base.count(r.memberships),percent=base.pct(r.memberships,r.membership_denominator),
            patients_ever=base.count(r.patients_ever),primary_weight_percent=round(100*r.primary_fixed_weight_share,2) if np.isfinite(r.primary_fixed_weight_share) else None))
    m.say(dict(stage='control_order_done',summary=public,effects=0,main_cache_unchanged=True))


def control_publish():
    m=poe_module();b=m.load();e=check(b);base=m.base();x=e['control_orders'];s=x['summary']
    path=ROOT/'steroid_tte.md';before=path.read_bytes();old=before.decode('utf-8');newline='\r\n' if '\r\n' in old else '\n';old=old.replace('\r\n','\n')
    def fmt(n,d):
        p=base.pct(n,d)
        return base.count(n)+' / '+base.count(d)+('（比例抑制）' if p=='' else '（'+str(p)+'%）')
    def share(r,clock):
        if base.pct(r.memberships,r.membership_denominator)=='':return '抑制'
        v=getattr(r,clock+'_fixed_weight_share');return f'{100*v:.2f}%' if np.isfinite(v) else '不适用'
    labels={'prior_injectable_order':'此前有注射剂型兼容HC医嘱',
        'prior_injectable_not_discontinued':'上述医嘱中至少一张截至索引未捕获停止记录',
        'prior_ambiguous_HC_order':'此前有其他或剂型未确定HC医嘱',
        'same_second_injectable_order':'注射剂型兼容HC医嘱与索引同秒',
        'prior_order_link_known_only_from_later_emar':'至少一张此前HC医嘱的关联药名仅见于索引同秒或以后eMAR事件',
        'order_conflict':'医嘱时间或住院边界冲突'}
    lines=['<!-- HC_CONTROL_ORDER_START -->','### 同期未启动对照中的既往HC医嘱','',
        '**本次推进：**在原风险集的全部住院中补取HC eMAR原生医嘱关联，覆盖本次住院全过程，并据POE时间判断索引前是否已有医嘱。保留实际首剂为t0；没有重筛患者、增加评分变量、重估死亡或增加bootstrap。这个核查描述实际给药前的路径，不将未知临床意图作成标签。','',
        '|组别与记录状态|索引记录数/组内记录数|曾出现该状态的独立患者/组内患者|含该状态的风险集/全部风险集|原主分析对照权重质量|原严格时钟对照权重质量|',
        '|---|---:|---:|---:|---:|---:|']
    for r in s.itertuples():
        if r.metric not in ['prior_injectable_order','prior_injectable_not_discontinued','prior_ambiguous_HC_order']:continue
        lines.append('|'+('HC启动者' if r.arm=='HC' else '同期未启动对照')+'：'+labels[r.metric]+'|'+fmt(r.memberships,r.membership_denominator)+'|'+fmt(r.patients_ever,r.patient_denominator)+'|'+fmt(r.risksets_any,r.riskset_denominator)+'|'+share(r,'primary')+'|'+share(r,'strict')+'|')
    lines+=['',
        '同一患者可在多个索引时点成为对照，因此索引记录不是独立样本；“曾出现”比例也不是任意时点的患病率。表中状态可以重叠。权重质量直接使用原模型的固定预测值及原归一化方式计算，主/严格两种口径全部展示，未重新拟合或据结果挑选模型。人数按10取整，小单元及可反推比例抑制。','',
        '#### 捕获范围和不能直接推出的结论','',
        '1. 阳性证据来自同版原生eMAR→POE键，且逐一验证患者、住院及医嘱时间。注射剂型兼容由明确HC注射盐型药名或既有实际HC解析支持；其他剂型或不明确药名另列，不自动当作静脉HC策略。','2. “截至索引未捕获停止记录”不等于肯定有效医嘱。直接撤回时间不晚于索引者不计入该类；最终Inactive不用于反推基线状态。口头取消、暂停或未关联变更仍可能漏记。','3. 部分药名映射来自索引之后的eMAR。它能帮助确认一张已经存在的历史医嘱所指药物，但不证明我们在实时系统中当时就能知道该映射；更不能据后续是否成功给药排除患者。','4. 当前3.1目录未找到prescriptions或pharmacy。没有任何eMAR记录的HC医嘱可能不可见，因此阴性仅表示本次关联未检出，不能编码成确定“未开医嘱”。此缺口也意味着阳性比例是这套捕获规则下的描述，不能当作完整医嘱发生率。','',
        '|额外关联核查|结果|','|---|---|',
        '|eMAR事件键冲突行|'+base.count(len(x['emar_key_conflicts']))+'|',
        '|候选患者中缺失原生住院键的HC eMAR行（单列，未强配）|'+base.count(len(x['missing_native_hadm']))+'|',
        '|HC医嘱键缺失、重复或键/时间冲突|'+base.count(len(x['order_failures']))+'|']
    for r in s[s.metric.isin(['same_second_injectable_order','prior_order_link_known_only_from_later_emar','order_conflict'])].itertuples():
        lines.append('|'+r.arm+'：'+labels[r.metric]+'|'+fmt(r.memberships,r.membership_denominator)+'|')
    lines+=['','#### 对稿件与下一步的决定','',
        '保留“当前实际给药启动与同期尚未启动、随后通常照护”的主问题。医嘱之后但给药之前的测量并不自动是HC治疗后变量；它可以反映给药时病情，但不能代表开立医嘱前的状态。对照中已有医嘱者也不自动算作暴露错分，因为当前策略按实际给药定义。','',
        '暂不把不完整的“是否已有医嘱”直接加入启动评分，也不只保留已经开医嘱的对照。前者可能把捕获不足变成已知阴性，后者会改变对照人群并需要重新评估支持，不能直接沿用原权重。原给药起点、ATT范围、风险集和500次区间均保留。','',
        '唯一优先的补充是在同版prescriptions或pharmacy可用后，补齐未执行/无eMAR医嘱的药物身份与时间。先在同一风险集完成两组一致的医嘱状态核查，再决定是否有条件进行一个限定的order-aware分析。这个追加分析的任何模型变更都属见过主结果后的补充，不能重新称为完全预设。POE本身没有具体药名，继续扫描相同POE不能填补这一身份缺口。','',
        '当前可以继续完善稿件和结果解释，不因残余未知无限暂停，也不把这一步成功等同于一区录用把握已经提高。eICU后续首先检验能否复现同一个实际首剂问题；若只有药物订单而无实际给药证据，应作为不同暴露代理单列。','',
        '#### 三篇近邻研究的可借鉴处理','',
        '|原始研究|已核实做法|本研究采取的做法与增量|','|---|---|',
        '|[Rajendran 2025，Nature Communications](https://link.springer.com/article/10.1038/s41467-025-59643-z)|主暴露窗为ICU前10h至后24h，后续用药者仍保留原分组；协变量取首24h最差值，另有克隆删失及未测混杂敏感性。研究价值同时来自轨迹分层和多库验证。|保留通常照护，承认既有时间偏倚分析先例；不复制其宽窗来扩大当前队列。当前首剂风险集与医嘱到给药记录核查提供另一个具体观察视角。|',
        '|[Toppin 2026，ICM Experimental](https://link.springer.com/article/10.1186/s40635-026-00972-8)|主分析涵盖休克前12h至ICU出院激素，要求ICU至少24h；24h敏感性缩小治疗组、对照不变；作者报告感染来源、剂量时机和残余混杂限制。|保留我们现有的索引前资格和后来可治疗对照，不为了复制已发表做法引入未来资格。临床分层价值需要自己的独立证据。|',
        '|[Ahn 2026，Scientific Reports](https://www.nature.com/articles/s41598-026-56521-6)|KoSS临床登记分析实际给药时间、首剂时NE等效剂量与乳酸，并做时间变化Cox；可排除长期和其他用途激素。|实际给药节点有临床研究先例；其用途信息强于本地结构化代理。我们应清楚表述所观察的记录人群，而非宣称全面更严谨。|','',
        '在此次核实的三篇正文及Toppin补充材料范围内，未见逐例POE至给药间隔及调整变量相对医嘱位置的报告；这不是全球首创检索。有限测量改进可以增强可信度，独立复现和临床分层增量更可能改变稿件的发表价值。','',
        '#### 复现','',
        '现有入口：`steroid_event_review.py control-orders`，随后`control-publish`。原始表只各分块扫描一次，已存在投影则复用。新增关联留在既有私有`hc_poe_linkage.pkl.gz`，旧共享分析缓存和分型材料不改写。']
    for r in x['runs']:
        lines.append('- '+r['stage']+'：'+str(r['seconds'])+'秒；原始读取'+f"{r['bytes_read']/1e9:.2f}"+'GB；峰值工作集'+f"{r['peak_MB']/1024:.2f}"+'GiB。')
    lines+=['','<!-- HC_CONTROL_ORDER_END -->','']
    block='\n'.join(lines)
    if '<!-- HC_CONTROL_ORDER_START -->' in old:
        start=old.index('<!-- HC_CONTROL_ORDER_START -->');end=old.index('<!-- HC_CONTROL_ORDER_END -->',start)+len('<!-- HC_CONTROL_ORDER_END -->')
        while end<len(old) and old[end]=='\n':end+=1
    else:start=end=old.index('### 实际代码与证据位置')
    out=(old[:start]+block+'\n'+old[end:]).replace('\n',newline).encode('utf-8')
    if out==before:m.say(dict(stage='control_publish',reuse=True));return
    assert path.read_bytes()==before
    history=e.setdefault('poe_report_history',[])
    if not any(y['sha256']==sha(before) for y in history):history.append(dict(sha256=sha(before),bytes=before))
    x['publication']=dict(report_hash=sha(out),before_hash=sha(before),code_hash=sha(Path(__file__).read_bytes()),only_control_section_added=True)
    x['executed_source']=Path(__file__).read_bytes();m.save(b);path.write_bytes(out)
    m.say(dict(stage='control_publish_done',report=str(path),frozen_effects_unchanged=True))


def build():
    m=module();b=m.load();e=check(b);begin=time.perf_counter()
    e['evidence']=evidence(b,e);e['summary']=summary_stats(e['evidence'],m.base())
    build_pilot(b,e,m);e['synthetic_tests_passed']=tests()
    e['runs'].append(dict(stage='cached_evidence_and_blinded_pilot',bytes_read_raw=0,
        seconds=round(time.perf_counter()-begin,2),peak_MB=m.base().peak_mb()))
    check(b);m.save(b)
    m.say(dict(stage='cached_evidence_done',candidate_count=m.base().count(len(e['evidence'])),
        pilot_cases=6,tests=e['synthetic_tests_passed'],poe_state=e['poe_state'],
        summary=e['summary'][['metric','count','percent']].to_dict('records')))


def publication_text(e):
    s=e['summary'].set_index('metric'); rows=[]
    selected=['原始eMAR关联键和给药时刻一致','有原生POE关联键','索引首剂仅一条且剂量无冲突',
        '索引前正速率NE区间记录','NE重叠或剂量记录仍未解析','HC具体IV途径已确认',
        'HC入库晚于给药记录','HC入库晚于给药超过15分钟','recorded_full_code',
        'recorded_DNR','recorded_DNI','recorded_CMO','code_status_unknown']
    names={'recorded_full_code':'索引前明确full code记录','recorded_DNR':'索引前明确DNR记录',
        'recorded_DNI':'索引前明确DNI记录','recorded_CMO':'索引前明确CMO记录','code_status_unknown':'索引前code-status未知'}
    for name in selected:
        r=s.loc[name];pct='抑制' if r.percent=='' else str(r.percent)+'%'
        if name=='recorded_full_code':pct='互补抑制'
        rows.append('|'+names.get(name,name)+'|'+str(r['count'])+'|'+pct+'|')
    run=[r for r in e['runs'] if r['stage']=='emar_link_projection'][-1]
    poe_intro='本地本轮尚未找到3.1 POE文件。用户只需补同版`hosp/poe.csv`或`hosp/poe.csv.gz`。'
    if 'poe_projection' in e:
        poe_intro='已完成同版POE的首剂定向关联。以下状态保留所有冻结候选，尚未以医嘱时间重建队列或估计：'
        counts=e['evidence'].poe_time_state.value_counts()
        def disclosed(n):
            return '1–10（抑制）' if 0<int(n)<=10 else str(int(np.floor(int(n)/10.+.5))*10)
        poe_intro+='；'.join(str(k)+'='+disclosed(v) for k,v in counts.items())+'。'
    return '''## 作者讨论：研究者证据判读与POE关联

本轮按研究者可复核的结构化证据推进写作，不将临床合作者作为计算或成稿前置条件。研究对象仍是有感染评估记录、正在接受NE的首次HC启动节点。既有分析描述这一记录人群中的当前启动比较；“用于感染性休克的辅助治疗”“复苏充分”“两种选择在临床上可互换”分别需要额外证据，当前不生成这些确定标签。没有把DNR、DNI、CMO与未知合并，也没有使用死亡或后续疗程来推断用途。

### 已完成的实证核查

以下分母均为当前约250次HC启动。人数按既有十位取整和小单元抑制规则展示，比例按原始分母计算；计数取整相同不表示逐行相等。

|核查事项|记录数|比例|
|---|---:|---:|
'''+ '\n'.join(rows)+'''

所有候选首剂均在原始eMAR中找到了唯一对应事件和原生POE关联键，患者、住院及给药时刻一致。少量NE规范化剂量仍标为ACTIVE_UNRESOLVED；资格依据当前正速率输注区间，剂量不确定按原分析的缺失状态保留。本轮没有排除这些病例或重算权重。

HC入库延迟的中位数为0分钟，第75及90百分位均为1分钟；记录入库过程较快并不能确定医生作出决定的时刻。具体IV/IM途径仍未知，当前药名与实际执行支持注射剂型兼容的首剂。没有DNI/CMO阳性记录表示在这套索引前记录中的未检出，不能解释为治疗限制不存在。HC适应证、院外激素、充分复苏及混合休克病因均不由当前字段直接确认。

### 六例材料及研究者判读规则

已用固定、不依赖结局的规则选取3次HC启动及各自同期风险集中的3名不同患者。HC按NE经过时间0–4、4–12、12–24小时分层，以种子523的稳定哈希顺序选择；每层不足时按同一排序补入。对照来自同一风险集，排除已选患者后以固定排序选取，不按结局、权重或疗效选择。材料只使用相对时间、假名、结构化数值和允许的状态类别，隐藏患者键、绝对日期、死亡、后续停药及治疗后结果。

病例表：[hc_review_cases.csv](C:/Users/黄晓东/Documents/ChatGPT/TTE/HTE_sepsis/private/hc_review_cases.csv)。相应时间线：[hc_review_timeline.csv](C:/Users/黄晓东/Documents/ChatGPT/TTE/HTE_sepsis/private/hc_review_timeline.csv)。关联原键仅留在既有私有缓存中。两份材料仅限授权本地审阅，不作为公开源数据。它们用于检查材料是否能支持有效判读，不用于估计全队列用途比例或宣布临床资格通过。

判读依次回答：当时有哪些可直接观察的循环与感染评估证据；是否存在HC其他用途的直接证据；当前记录能否支持讨论HC首剂选择；如果无法确定，缺失的是哪一项信息。数值、时间及来源属于观察证据；感染评估与NE并存属于研究资格代理；具体适应证和临床可选择性保留为研究者解释或未知。病例表中的自动说明是结构化证据摘要，人工判断栏仍为空，不声称已经完成独立临床专家核验。研究者可补充有依据的判断，不必等待外部临床人员。

时间线沿用原临床回看窗口。检验要求采样/记录和入库均早于索引，床旁记录按测量时刻截断，晚录状态单列；首次HC执行本身显示在t0。治疗后的生理、给药和结局均未进入材料。入库状态是记录属性，并不等于医生首次可见时间。

### POE到位后唯一需要补的核查

'''+poe_intro+'''接口按原生`emar_id → poe_id`关联，并核对患者及住院键；同秒、晚于给药、缺失及冲突分别报告，不做近邻时间猜测。POE中的最终状态和撤回链接作为回顾性元数据；只有发生时间明确早于索引的事件才可讨论其基线含义。

根据[官方eMAR说明](https://mimic.mit.edu/docs/iv/modules/hosp/emar.html)，charttime对应给药、storetime对应记录，poe_id用于关联医嘱。[官方POE说明](https://mimic.mit.edu/docs/iv/modules/hosp/poe.html)将ordertime定义为医嘱时间。该时间可用于量化首剂与医嘱的间隔，并检查现有MAP、乳酸等是否在下医嘱后测得；它不自动代表医生首次决策时刻，也不保证包含HC适应证。即使关联完成，不能将病例的医嘱时间直接替换原索引并接用旧对照或权重。

可运行入口为`steroid_event_review.py link-poe --poe-path <同版POE路径>`，随后运行`build`更新私有证据。该接口没有效应估计或bootstrap路径。当前主分析及全部方法敏感性保持原值，新增工作不用于搜索更有利结果。

本轮eMAR只读投影扫描'''+str(round(run['bytes_read']/1e9,2))+'''GB，耗时'''+str(run['seconds'])+'''秒，扫描阶段峰值工作集约'''+str(round(run['peak_MB']/1024,2))+'''GiB；仅保留冻结首剂的键及相关时刻。没有重新扫描检验、生命体征或eMAR_detail。新增9项合成关联边界检查通过；六例输出通过标识符、结局字段及治疗后记录排除检查。原模型、估计、500次bootstrap、聚合CSV及结果图保持不变。

研究伦理/数据授权说明、作者信息和注册情况仍由作者提供。研究者的内部规则保存不称为公开预注册；稿件中的临床解释由作者负责，统计平衡和自动证据摘要均不替代这些解释。

'''


def publish():
    m=module();b=m.load();e=check(b)
    assert 'evidence' in e and e['synthetic_tests_passed']==9
    current=(ROOT/'steroid_tte.md').read_bytes()
    source=Path(__file__).read_bytes()
    data_signature=sha(e['evidence'].to_csv(index=False).encode('utf-8'))
    if ('published' in e and sha(current)==e['published']['report_sha256']
        and sha(source)==e['published']['code_sha256']
        and data_signature==e['published'].get('evidence_sha256')):
        m.say(dict(stage='event_review_publish',reuse=True));return
    assert (current==e['prior_report'] or sha(current)==e.get('published',{}).get('report_sha256')),'USER_REPORT_CHANGE_REQUIRES_MERGE'
    if current!=e['prior_report']:e.setdefault('prior_generated_report_bytes',[]).append(current)
    marker=b'<!-- HC_MANUSCRIPT_END -->'
    front,tail=e['prior_report'].split(marker,1);nl='\r\n' if b'\r\n' in front else '\n'
    text=front.decode('utf-8').replace('\r\n','\n')
    old='当前无需再建数据平台；临床核验聚焦本节后文的两个决策问题。'
    assert text.count(old)==1
    text=text.replace(old,'当前无需再建数据平台；研究者证据判读及POE医嘱关联见本节后文，不等待外部临床合作者。')
    old='当前住院、当前ICU和NE正剂量左极限均有记录。'
    assert text.count(old)==1
    text=text.replace(old,'当前住院、当前ICU和索引前仍覆盖该节点的正速率NE区间有记录；少量规范化剂量未解析者按缺失状态保留。')
    old='当前比较是对“现在启动、以后通常照护”的点治疗选择进行探索性仿真。是否具有足够明确、可执行的临床含义，需要临床合作者评阅；不能以记录时点的精细程度代替这项判断。尤其不能从本表推导“应在NE后多少小时给药”或“应给予200mg/day”的政策结论。'
    new='当前比较是对“现在启动、以后通常照护”的点治疗选择进行探索性仿真。研究者依据记录和已核实文献评价其含义，目标范围限定于实际首剂启动者所代表的状态。该问题不规定必须等待的时长或200mg/day疗程，临床解释不以获得外部专家核验作为工作前置条件。'
    assert text.count(old)==1;text=text.replace(old,new)
    start=text.index('## 作者讨论：投稿前真正需要解决的内容')
    stop=text.index('### 实际代码与证据位置',start)
    text=text[:start]+publication_text(e)+text[stop:]
    abstract='''### 摘要

**目的：**评价NE治疗首日、已有感染评估记录的成人ICU患者中，当前首次启动HC与同期尚未启动、随后接受通常照护的院内死亡差异。

**方法：**使用MIMIC-IV 3.1构建探索性目标试验仿真。在每次首次实际HC给药时，以各患者自首次观察NE后的相同经过时间确定同期风险集。资格与调整均使用索引前记录，对照可在后续接受HC。采用条件启动评分将对照标准化至启动者的决策分布，报告启动者ATT对应的加权风险、风险差及风险比，以500次患者整群bootstrap计算95%区间。

**结果：**约250次HC启动及其风险集涉及约4,080位患者。调整后最大绝对SMD为0.056，对照患者ESS为637.0。HC启动者与标准化对照的院内死亡风险分别为53.82%和56.28%；风险差为−2.47个百分点（95%区间−7.72至2.87），风险比为0.956（0.863–1.050）。严格入库时钟的风险差为−2.32个百分点（−7.61至3.09）。提前15/30分钟截断临床记录及固定回归增强的点估计相对主分析变化不超过0.90个百分点，三项补充区间均包含获益与伤害。

**结论：**在实际HC启动者所代表的记录人群中，当前启动与同期尚未启动的加权院内死亡风险接近，估计区间仍容纳有临床意义的差异。该研究提供首剂节点的独立观察证据，解释范围取决于HC用途、真实决定时间及未测量病情差异。

'''
    title_end='**接受去甲肾上腺素且有感染评估记录的ICU患者中氢化可的松首次启动与院内死亡：探索性目标试验仿真**\n\n'
    assert text.count(title_end)==1;text=text.replace(title_end,title_end+abstract)
    old='两组均须在ICU内有索引前正剂量NE输注，且过去24小时有实际抗菌药记录'
    assert text.count(old)==1
    text=text.replace(old,'两组均须在ICU内有从索引前延续至该节点的正速率NE输注区间，且过去24小时有实际抗菌药记录')
    anchor='#### 观测平衡及权重分布\n\n'
    paragraph='''#### 首剂记录与用途证据

全部HC首剂均与原始eMAR事件的患者、住院和给药时刻一致，并具有原生POE关联键。32.1%的首剂记录在给药时刻后入库，延迟中位数为0分钟，第90百分位为1分钟。索引前code-status未知占72.3%。这些记录支持首次实际给药及NE治疗中感染评估的研究背景；HC具体用途与IV/IM途径分别保留为未确认信息。

'''
    assert text.count(anchor)==1;text=text.replace(anchor,paragraph+anchor)
    out=text.replace('\n',nl).encode('utf-8')+marker+tail
    assert out.split(marker,1)[1]==tail
    for number in ['53.82','56.28','−2.47','−7.72','2.87','0.956','0.863','1.050','−2.32','637.0','675.7']:
        assert number in text
    source=Path(__file__).read_bytes();dest=ROOT/'steroid_event_review.py'
    if dest.exists() and dest.read_bytes()!=source:
        e.setdefault('prior_code_bytes',[]).append(dest.read_bytes())
    dest.write_bytes(source)
    (ROOT/'steroid_tte.md').write_bytes(out)
    e['published']=dict(report_sha256=sha(out),code_sha256=sha(source),evidence_sha256=data_signature,utc=m.base().utc(),
        original_history_unchanged=True,original_aggregate_unchanged=True,
        frozen_effects_and_bootstrap_unchanged=True)
    e['executed_source']=source
    # Keep the existing canonical-report guard current for the earlier module.
    b['methods_extension']['published']['sha256']['steroid_tte.md']=sha(out)
    check(b);m.save(b)
    m.say(dict(stage='event_review_publish',report=str(ROOT/'steroid_tte.md'),
        code=str(dest),prior_results_preserved=True,prior_report_in_private_checkpoint=True))


RX_COLUMNS=['subject_id','hadm_id','pharmacy_id','poe_id','poe_seq','starttime','stoptime',
    'drug_type','drug','formulary_drug_cd','gsn','ndc','prod_strength','form_rx','dose_val_rx',
    'dose_unit_rx','doses_per_24_hrs','route']
RX_GC=re.compile(r'hydrocortis|methylpred|predniso|prednisone|dexameth|betameth|(?<!fludro)cortisone|solu.?cort|solu.?med|decadron|cortef|medrol|deltasone',re.I)
RX_HC=re.compile(r'hydrocortis|solu.?cort|cortef',re.I)
RX_CONFIG=dict(
    purpose='same-version prescriptions extension in unchanged HC riskset admissions',
    identity='native pharmacy/POE and patient/admission keys; no nearest-time match',
    prescription='prescribed route/start/stop; NOT proof of administered route or execution',
    time='POE ordertime for preindex order evidence; prescribed start/stop shown separately',
    discontinuation='only native timed POE D/C link used as preindex withdrawal evidence; final Inactive is not cancellation',
    emar_capture='cached HC-name eMAR across full riskset admissions; unmatched is no matching record in this projection, not proof never administered',
    models='none; reuse fixed original ATT weights only for evidence coverage',
    outcomes='not used in linkage, definitions, or grouping')


def stream_projection(path,columns,callback,base,label):
    stat=path.stat();reader=base.HashReader(path);buffer=io.BufferedReader(reader,buffer_size=4*1024*1024)
    stream=gzip.GzipFile(fileobj=buffer) if path.name.endswith('.gz') else buffer
    start=time.perf_counter();last=start;rows=0
    try:
        for z in pd.read_csv(stream,usecols=columns,dtype=str,chunksize=100000):
            callback(z);rows+=len(z)
            if time.perf_counter()-last>30:
                print(json.dumps(dict(stage=label,rows_read_rounded=base.count(rows),seconds=round(time.perf_counter()-start),peak_MB=base.peak_mb())),flush=True);last=time.perf_counter()
        assert reader.n==stat.st_size;digest=reader.h.hexdigest()
    finally:stream.close();buffer.close()
    assert (stat.st_size,stat.st_mtime_ns)==(path.stat().st_size,path.stat().st_mtime_ns)
    return dict(stage=label,path=str(path),signature=(str(path),stat.st_size,stat.st_mtime_ns),bytes_read=stat.st_size,
        sha256=digest,rows_read=rows,seconds=round(time.perf_counter()-start,2),peak_MB=base.peak_mb(),columns=columns,parsed_to_EOF=True)


def prescription_project():
    m=poe_module();b=m.load();e=check(b);base=m.base()
    x=e.setdefault('prescriptions',dict(config=RX_CONFIG,runs=[],prior_results_seen=True))
    assert x['config']==RX_CONFIG
    path=RAW/'hosp'/'prescriptions.csv';signature=(str(path),path.stat().st_size,path.stat().st_mtime_ns)
    if 'records' in x:
        assert tuple(x['runs'][0]['signature'])==signature
        m.say(dict(stage='prescription_project',reuse=True));return
    assert set(RX_COLUMNS)<=set(pd.read_csv(path,nrows=0).columns)
    adms=set(b['full_risksets'].hadm_id);ref=b['source'].set_index('hadm_id').subject_id
    native=set(e['emar_links'].pharmacy_id.dropna().astype(str));pieces=[];stats=dict(admission_key_conflicts=0)
    m.say(dict(stage='prescription_project_start',bytes=signature[1],columns=RX_COLUMNS,
        filter='GC-name prescriptions in unchanged riskset hospitalizations and exact native first-dose pharmacy keys',chunk_rows=100000))
    def consume(z):
        h=pd.to_numeric(z.hadm_id,errors='coerce');z=z[h.isin(adms)].copy()
        if not len(z):return
        name=z.drug.fillna('');z=z[name.str.contains(RX_GC)|z.pharmacy_id.isin(native)].copy()
        if not len(z):return
        z['hadm_id']=pd.to_numeric(z.hadm_id).astype('int64');z['subject_id']=pd.to_numeric(z.subject_id).astype('int64')
        ok=z.subject_id.eq(z.hadm_id.map(ref));stats['admission_key_conflicts']+=int((~ok).sum())
        pieces.append(z[ok])
    run=stream_projection(path,RX_COLUMNS,consume,base,'prescription_projection')
    q=pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame(columns=RX_COLUMNS)
    stats['exact_duplicate_rows']=int(q.duplicated().sum());q=q.drop_duplicates().copy()
    x['raw_projection']=q;x['quality']=stats;x['runs'].append(run);m.save(b)
    q['start']=secs(q.starttime);q['stop_rx']=secs(q.stoptime)
    q['is_HC']=q.drug.str.contains(RX_HC,na=False);q['is_GC']=q.drug.str.contains(RX_GC,na=False)
    x['records']=q;x['source_current']=Path(__file__).read_bytes()
    check(b);m.save(b)
    m.say(dict(stage='prescription_projection_done',retained_rows=base.count(len(q)),seconds=run['seconds'],effects=0))


def prescription_poe():
    m=poe_module();b=m.load();e=check(b);base=m.base();x=e['prescriptions'];assert x['config']==RX_CONFIG
    if 'poe' in x:m.say(dict(stage='prescription_poe',reuse=True));return
    ids=set(x['records'].loc[x['records'].is_GC,'poe_id'].dropna())|set(e['control_orders']['order_evidence'].poe_id)
    old=e['control_orders']['poe'];missing=ids-set(old.poe_id)
    path=Path(e['poe_file_signature'][0]);parts=[];links=[]
    m.say(dict(stage='prescription_poe_start',bytes=path.stat().st_size,columns=POE_COLUMNS,
        filter='new steroid prescription POE keys plus native direct discontinue links; complements previously cached eMAR-linked orders',
        new_keys_rounded=base.count(len(missing))))
    def consume(z):
        q=z[z.poe_id.isin(missing)]
        if len(q):parts.append(q.copy())
        q=z[z.discontinue_of_poe_id.isin(ids)|z.discontinued_by_poe_id.isin(ids)]
        if len(q):links.append(q.copy())
    run=stream_projection(path,POE_COLUMNS,consume,base,'prescription_poe_projection')
    q=pd.concat([old]+parts,ignore_index=True).drop_duplicates(subset=POE_COLUMNS)
    q['order']=secs(q.ordertime)
    x['poe']=q;x['direct_links']=pd.concat(links,ignore_index=True).drop_duplicates() if links else pd.DataFrame(columns=POE_COLUMNS)
    x['direct_links']['order']=secs(x['direct_links'].ordertime)
    x['runs'].append(run);x['source_current']=Path(__file__).read_bytes();check(b);m.save(b)
    m.say(dict(stage='prescription_poe_done',seconds=run['seconds'],effects=0))


def rx_route(s):
    s=str(s).strip().upper()
    if s.startswith('IV') or s in ['INTRAVENOUS','VENOUS']:return 'IV_prescribed'
    if s in ['IM','INTRAMUSCULAR']:return 'IM_prescribed'
    if s in ['PO','PO/NG','ORAL','NG','NGT','OG','GT','PEG','PO/NGT']:return 'enteral_prescribed'
    if re.search(r'TOP|RECT|OPHTH|OTIC|INHAL|INTRATHEC',s) or s in ['PR','OU','OS','OD','IT']:return 'other_nonIV_route'
    return 'route_unknown'


def prescription_review():
    m=poe_module();b=m.load();e=check(b);base=m.base();x=e['prescriptions'];assert x['config']==RX_CONFIG
    if 'summary' in x:m.say(dict(stage='prescription_review',reuse=True));return
    start=time.perf_counter();panel=b['full_risksets'];ref=b['source'].set_index('hadm_id')
    rx=x['records'].copy();rx['route_class']=rx.route.map(rx_route);hc=rx[rx.is_HC].copy()
    first=[];em=e['emar_links'].set_index('emar_id')
    for _,t in e['targets'].iterrows():
        row=em.loc[t.emar_id];same=hc[hc.hadm_id.eq(t.hadm_id)&hc.subject_id.eq(t.subject_id)]
        native=same[same.pharmacy_id.eq(row.pharmacy_id)&same.poe_id.eq(row.poe_id)]
        mode='pharmacy_and_POE'
        if not len(native):
            native=same[same.poe_id.eq(row.poe_id)];mode='POE_only' if len(native) else 'NO_MATCH'
        routes=set(native.route_class)
        route=next(iter(routes)) if len(routes)==1 else 'route_conflict' if len(routes)>1 else 'UNKNOWN'
        within=native.start.le(t['index'])&(native.stop_rx.isna()|native.stop_rx.ge(t['index']))
        first.append(dict(case_id=t.case_id,emar_id=t.emar_id,mode=mode,route=route,
            n_prescription_components=len(native),within_any_prescribed_interval=bool(within.any()),
            starts_after_admin=bool(len(native) and native.start.gt(t['index']).all()),
            prescribed_time_unknown=bool(len(native) and native.start.isna().any())))
    first=pd.DataFrame(first);x['first_dose_route_evidence']=first
    q=x['poe'];bad=set(q.loc[q.poe_id.duplicated(False),'poe_id']);q=q[~q.poe_id.isin(bad)].set_index('poe_id')
    raw=e['control_orders']['emar'];old=e['control_orders']['order_evidence'];rel=x['direct_links']
    rxgroups={k:g for k,g in hc.dropna(subset=['poe_id']).groupby('poe_id')}
    emgroups={k:g for k,g in raw.dropna(subset=['poe_id']).groupby('poe_id')}
    oldgroups=old.set_index('poe_id');keys=set(rxgroups)|set(emgroups);orders=[];fail=[]
    for key in keys:
        rr=rxgroups.get(key);ee=emgroups.get(key)
        identity=rr if rr is not None else ee
        h=int(identity.hadm_id.iloc[0]);sub=int(identity.subject_id.iloc[0])
        if key not in q.index:
            fail.append(dict(poe_id=key,reason='missing_or_duplicate_POE'));continue
        r=q.loc[key]
        if identity.hadm_id.nunique()!=1 or identity.subject_id.nunique()!=1 or str(r.subject_id)!=str(sub) or str(r.hadm_id)!=str(h) or not np.isfinite(r.order):
            fail.append(dict(poe_id=key,reason='identity_or_time_conflict'));continue
        if ee is not None and (not ee.hadm_id.eq(h).all() or not ee.subject_id.eq(sub).all()):
            fail.append(dict(poe_id=key,reason='emar_identity_conflict'));continue
        stops=rel[rel.discontinue_of_poe_id.eq(key)&pd.to_numeric(rel.hadm_id,errors='coerce').eq(h)&pd.to_numeric(rel.subject_id,errors='coerce').eq(sub)].order.dropna()
        stop=float(stops.min()) if len(stops) else np.nan
        routes=set(rr.route_class) if rr is not None else set()
        route=next(iter(routes)) if len(routes)==1 else 'route_conflict' if len(routes)>1 else 'route_unknown'
        old_injectable=bool(key in oldgroups.index and oldgroups.loc[key,'injectable_compatible'])
        orders.append(dict(poe_id=key,hadm_id=h,subject_id=sub,order=float(r.order),stop=stop,route=route,
            injectable_compatible=route in ['IV_prescribed','IM_prescribed'] or old_injectable,
            prescription_present=rr is not None,emar_present=ee is not None,known_old=key in oldgroups.index,
            emar_before_first_order=bool(ee is not None and ee.chart.lt(r.order).any()),
            first_emar=ee.chart.min() if ee is not None else np.nan,
            temporal_conflict=bool(np.isfinite(stop) and stop<r.order),within_admission=bool(ref.loc[h,'admit']<=r.order<=ref.loc[h,'discharge']),
            prescribed_start=rr.start.min() if rr is not None else np.nan,
            prescribed_stop=rr.stop_rx.max() if rr is not None else np.nan))
    orders=pd.DataFrame(orders);x['orders']=orders;x['failures']=pd.DataFrame(fail)
    columns=['prior_injectable_order','prior_IV_order','prior_IV_no_native_discontinue',
        'prior_IV_without_matched_emar','prior_IV_newly_visible_order','prior_IV_recorded_stop_before_index',
        'prior_ambiguous_order','same_second_IV_order','order_conflict']
    flags=pd.DataFrame(False,index=panel.index,columns=columns);groups={h:g for h,g in orders.groupby('hadm_id')}
    for h,ix in panel.groupby('hadm_id',sort=False).groups.items():
        g=groups.get(h)
        if g is None:continue
        t=panel.loc[ix,'index'].to_numpy(float)
        for r in g.itertuples():
            prior,pending,tie=order_flags(r.order,r.stop,t)
            if r.temporal_conflict or not r.within_admission:
                flags.loc[ix,'order_conflict']|=prior;continue
            if r.injectable_compatible:flags.loc[ix,'prior_injectable_order']|=prior
            if r.route=='IV_prescribed':
                flags.loc[ix,'prior_IV_order']|=prior
                flags.loc[ix,'prior_IV_no_native_discontinue']|=pending
                flags.loc[ix,'same_second_IV_order']|=tie
                flags.loc[ix,'prior_IV_recorded_stop_before_index']|=prior&np.isfinite(r.prescribed_stop)&(r.prescribed_stop<=t)
                if not r.emar_present:flags.loc[ix,'prior_IV_without_matched_emar']|=prior
                if not r.known_old:flags.loc[ix,'prior_IV_newly_visible_order']|=prior
            elif r.route in ['route_unknown','route_conflict']:
                flags.loc[ix,'prior_ambiguous_order']|=prior
    oldflags=e['control_orders']['flags']
    assert ((~oldflags.prior_injectable_order)|flags.prior_injectable_order).all()
    flags['newly_detected_prior_injectable']=flags.prior_injectable_order&~oldflags.prior_injectable_order
    weighted={}
    for clock in ['primary','strict']:
        frame=m.analysis_frame(b,clock);plan=m.make_plan(frame,m.ANALYSIS_CONFIG['numerical'],m.ANALYSIS_CONFIG['categorical'],diagnostic_numeric=['NE_dose'])
        assert plan['plan_sha256']==b['support'][clock]['plan_sha256']
        w=m.att_weights(plan,b['support'][clock]['linear_predictor']);back=np.empty(plan['n']);back[plan['order']]=w['control'];weighted[clock]=back
    summary=[]
    for arm,mask in [('HC',panel.treated.eq(1)),('control',panel.treated.eq(0))]:
        for col in flags:
            hit=flags[col]&mask
            row=dict(arm=arm,metric=col,n=int(hit.sum()),d=int(mask.sum()),patients=int(panel.loc[hit,'subject_id'].nunique()),
                patient_denominator=int(panel.loc[mask,'subject_id'].nunique()),risksets=int(panel.loc[hit,'case_id'].nunique()))
            for clock,w in weighted.items():row[clock+'_share']=float(w[hit].sum()/w[mask].sum()) if arm=='control' else np.nan
            summary.append(row)
    x['flags']=flags;x['summary']=pd.DataFrame(summary)
    x['order_quality']=dict(HC_prescription_rows=len(hc),HC_prescription_orders=hc.poe_id.nunique(),
        HC_missing_POE_rows=int(hc.poe_id.isna().sum()),parsed_orders=len(orders),POE_failures=len(fail),
        IV_orders=int(orders.route.eq('IV_prescribed').sum()),IV_without_matched_emar=int((orders.route.eq('IV_prescribed')&~orders.emar_present).sum()),
        prescribed_stop_before_start=int(hc.stop_rx.lt(hc.start).sum()),prescribed_start_missing=int(hc.start.isna().sum()))
    x['tests']=dict(original_weight_predictions_reproduced=True,no_outcomes_compared=True,no_model_refits=True,old_order_positives_retained=True,
        no_prescription_labeled_actual_administration=True)
    x['runs'].append(dict(stage='prescription_cached_review',seconds=round(time.perf_counter()-start,2),bytes_read=0,peak_MB=base.peak_mb()))
    x['source_current']=Path(__file__).read_bytes();check(b);m.save(b)
    m.say(dict(stage='prescription_review_done',first_dose_routes={k:base.count(v) for k,v in first.route.value_counts().items()},
        summary=[dict(arm=r.arm,metric=r.metric,n=base.count(r.n),d=base.count(r.d),primary_share=round(r.primary_share*100,2) if np.isfinite(r.primary_share) else None)
        for r in x['summary'][x['summary'].metric.isin(['prior_IV_order','prior_IV_without_matched_emar','newly_detected_prior_injectable'])].itertuples()],effects=0))


def prescription_validate():
    m=poe_module();b=m.load();e=check(b);x=e['prescriptions'];base=m.base()
    for run in x['runs']:
        if 'signature' not in run:continue
        f=Path(run['signature'][0]);assert tuple(run['signature'])==(str(f),f.stat().st_size,f.stat().st_mtime_ns)
    # Explicit missing time, exact boundary, route and final-status semantics.
    dt=secs(pd.Series(['2150-01-01 00:00:00',None,'2999-01-01 00:00:00']))
    assert np.isfinite(dt.iloc[0]) and dt.iloc[1:].isna().all()
    assert rx_route('IV DRIP')=='IV_prescribed' and rx_route('IM')=='IM_prescribed' and rx_route('TOP')=='other_nonIV_route'
    assert order_flags(9.,10.,10.)==(True,False,False)
    assert order_flags(10.,np.nan,10.)==(False,False,True)
    em=e['emar_links'].set_index('emar_id');rx=x['records'];targets=e['targets'].set_index('emar_id');out=[]
    for t in x['first_dose_route_evidence'].itertuples():
        if t.route!='UNKNOWN':continue
        source=em.loc[t.emar_id];target=targets.loc[t.emar_id]
        g=rx[rx.hadm_id.eq(target.hadm_id)&rx.subject_id.eq(target.subject_id)&rx.pharmacy_id.eq(source.pharmacy_id)]
        out.append(dict(emar_id=t.emar_id,reason='same_pharmacy_different_POE_requires_version_resolution' if len(g) else 'no_same_admission_native_prescription_match',
            pharmacy_only_drug_names=sorted(set(g.drug.dropna()))))
    x['unmatched_first_dose_evidence']=pd.DataFrame(out)
    x['execution_notes']=['Initial prescription read reached conversion but the legacy NaT-to-int conversion failed before checkpointing. Re-read once after explicit missing/out-of-range timestamp handling; no frozen results changed.']
    x['validation']=dict(synthetic_checks=8,raw_signatures_unchanged=True,old_positive_orders_retained=True,
        weights_not_refit=True,clinical_outcomes_not_compared=True)
    x['source_current']=Path(__file__).read_bytes();check(b);m.save(b)
    m.say(dict(stage='prescription_validate',checks=8,unmatched_reasons={k:base.count(v) for k,v in x['unmatched_first_dose_evidence'].reason.value_counts().items()}))


def prescription_publish():
    m=poe_module();b=m.load();e=check(b);base=m.base();x=e['prescriptions'];assert 'validation' in x
    path=ROOT/'steroid_tte.md';before=path.read_bytes();old=before.decode('utf-8');newline='\r\n' if '\r\n' in old else '\n';old=old.replace('\r\n','\n')
    def fmt(n,den=None):
        out=base.count(n)
        if den is not None:
            pct=base.pct(n,den);out+=' / '+base.count(den)+('（比例抑制）' if pct=='' else '（'+str(pct)+'%）')
        return out
    def weight(r,clock):return '不适用' if r.arm=='HC' else '抑制' if base.pct(r.n,r.d)=='' else f'{100*getattr(r,clock+"_share"):.2f}%'
    f=x['first_dose_route_evidence'];n=len(f)
    lines=['<!-- HC_RX_EXTENSION_START -->','### 处方补齐后的HC给药途径与既往医嘱核查','',
        '本节更新前文“prescriptions尚未取得”的历史状态。已使用同版3.1处方、POE及既有eMAR投影；原患者、首剂时点、死亡结果、模型及bootstrap均未改变。新增的是处方来源证据，不将处方当作实际给药。','',
        '|首剂核查|记录数/分母|','|---|---:|',
        '|实际首剂关联到明确IV处方|'+fmt(f.route.eq('IV_prescribed').sum(),n)+'|',
        '|具体处方途径仍未知|'+fmt(f.route.eq('UNKNOWN').sum(),n)+'|',
        '|同时满足原生pharmacy与POE关联|'+fmt(f['mode'].eq('pharmacy_and_POE').sum(),n)+'|',
        '|仅原生POE匹配（单列）|'+fmt(f['mode'].eq('POE_only').sum(),n)+'|','',
        'IV是关联处方的途径；实际执行由原eMAR及剂量证据支持，两者结合提高注射治疗的可解释性，但不把处方途径称为独立记录的实际执行途径。未匹配病例保留原资格；部分有相同pharmacy而不同POE版本的记录，需解析版本关系，未用时间邻近强配。','',
        '|组别与索引前状态|索引记录数/组内分母|涉及独立患者/组内患者|主时钟固定对照权重质量|严格时钟固定对照权重质量|','|---|---:|---:|---:|---:|']
    labels={'prior_injectable_order':'注射剂型兼容HC医嘱（与前次同口径）','prior_IV_order':'明确IV HC医嘱',
        'prior_IV_no_native_discontinue':'IV医嘱且截至索引无原生撤回记录','prior_IV_without_matched_emar':'IV医嘱但全住院缓存无匹配HC eMAR',
        'newly_detected_prior_injectable':'本次新检出的注射剂型兼容既往医嘱'}
    for r in x['summary'].itertuples():
        if r.metric not in labels:continue
        lines.append('|'+('HC启动' if r.arm=='HC' else '同期未启动')+'：'+labels[r.metric]+'|'+fmt(r.n,r.d)+'|'+fmt(r.patients,r.patient_denominator)+'|'+weight(r,'primary')+'|'+weight(r,'strict')+'|')
    oldrow=e['control_orders']['summary'];oldrow=oldrow[(oldrow.arm=='control')&(oldrow.metric=='prior_injectable_order')].iloc[0]
    newrow=x['summary'][(x['summary'].arm=='control')&(x['summary'].metric=='prior_injectable_order')].iloc[0]
    lines+=['',f'同口径的既往注射剂型兼容HC医嘱，其主分析对照权重质量由{100*oldrow.primary_fixed_weight_share:.2f}%增至{100*newrow.primary_share:.2f}%；严格口径由{100*oldrow.strict_fixed_weight_share:.2f}%增至{100*newrow.strict_share:.2f}%。这是固定旧权重下的来源覆盖变化，不是新效果、权重改善或独立样本增加。','',
        '同一患者可贡献多个索引记录；计数按10取整，小单元及可反推比例抑制。“全住院无匹配HC eMAR”仅指已有HC药名投影，不能推断从未实际给药；用全住院记录检查来源捕获，不据未来给药与否改变基线资格。','',
        '|处方与原生键质量|结果|','|---|---:|']
    for key,label in [('HC_prescription_rows','HC处方成分行'),('HC_prescription_orders','HC处方POE键'),('HC_missing_POE_rows','HC处方缺少POE键'),
        ('POE_failures','键或时间无法确认的订单'),('IV_without_matched_emar','IV医嘱无匹配HC eMAR（整个住院）'),('prescribed_stop_before_start','处方停止早于计划开始的行')]:
        lines.append('|'+label+'|'+fmt(x['order_quality'][key])+'|')
    lines+=['',
        '处方starttime/stoptime是计划用药时间；本次用POE ordertime确定医嘱是否早于索引，只用有时间的原生撤回链判断已记录撤回。最终Inactive与计划停止都不自动等于当时取消，停止早于开始的异常保留并单列。','',
        '**对研究的实际影响：**部分HC给药病例已有IV处方互证，原对照中已有HC医嘱者的捕获更完整。首剂与关联医嘱相隔约50分钟的既有发现仍然成立，不能把给药时点自动称为作出开药决定的时点。本次没有重新估计死亡，也没有将“已开药”直接塞进原评分模型；它可能代表治疗选择过程，需先明确是在比较开药决定还是实际首剂交付。','',
        '后续若仍回答实际首剂问题，需对两组对称规定医嘱状态与信息截止的敏感性规则。若改为开立HC医嘱，则构成另一个明确目标，须同时纳入未执行医嘱并重新建立风险集；不能仅把启动者t0前移。未知适应证、院外激素与治疗限制不会由处方关联自动消失。','',
        '原始处方成功扫描'+f"{x['runs'][0]['bytes_read']/1e9:.2f}GB，{x['runs'][0]['seconds']:.2f}秒；新增POE关联扫描{ x['runs'][1]['bytes_read']/1e9:.2f}GB，{x['runs'][1]['seconds']:.2f}秒。首遍处方读取在时间转换时因NaT失败，修复后额外重读一次；该次未生成估计结果。8项针对性校验通过，旧权重数值复现，未拟合模型。" ,'',
        '入口：`steroid_event_review.py prescription-project` → `prescription-poe` → `prescription-review` → `prescription-validate` → `prescription-publish`。结果与原始投影存于既有私有`hc_poe_linkage.pkl.gz`，旧版本代码及本节前的报告内容保留在同一缓存。','',
        '官方字段依据：[prescriptions](https://mimic.mit.edu/docs/iv/modules/hosp/prescriptions.html)、[POE](https://mimic.mit.edu/docs/iv/modules/hosp/poe.html)。','<!-- HC_RX_EXTENSION_END -->','']
    block='\n'.join(lines)
    if '<!-- HC_RX_EXTENSION_START -->' in old:
        start=old.index('<!-- HC_RX_EXTENSION_START -->');end=old.index('<!-- HC_RX_EXTENSION_END -->',start)+len('<!-- HC_RX_EXTENSION_END -->')
        while end<len(old) and old[end]=='\n':end+=1
    else:start=end=old.index('<!-- HC_EICU_SOURCE_START -->')
    after=(old[:start]+block+'\n'+old[end:]).replace('\n',newline).encode('utf-8')
    assert path.read_bytes()==before
    if after!=before:
        history=x.setdefault('report_history',[])
        if not any(v['sha256']==sha(before) for v in history):history.append(dict(sha256=sha(before),bytes=before))
        x['publication']=dict(report_sha256=sha(after),code_sha256=sha(Path(__file__).read_bytes()),updated_section_only=True)
        check(b);m.save(b);path.write_bytes(after)
    m.say(dict(stage='prescription_publish',report=str(path),effects=0))


ORDER_PROXY_CONFIG=dict(
    purpose='outcome-blind measurement bridge for a possible eICU order-based supplementary analysis',
    denominator='existing MIMIC riskset hospitalizations only; not a population validation sample or a new external cohort',
    order_sample='first native-linked IV HC prescription order in the admission, chosen before reading subsequent execution',
    qualification='original NE first-day, current NE interval and strictly prior infection/GC record rules evaluated again at order time',
    execution='resolved actual injectable-compatible HC within the existing medication projection; native POE linkage and any same-admission HC shown separately',
    windows_minutes=[30,60,120,360],
    short_observation='retain admissions ending before each window; report capture-short state, do not turn it into confirmed non-administration',
    first_dose_clock='all frozen actual initiations, linked POE version and unique native prescription start; negative and unknown lags retained',
    recentered_covariates='same clinical age windows ending at order, no widened windows; existing cache envelope completeness explicit',
    decision='no acceptance threshold or clock selected for better death results; no equality of order and actual administration assumed',
    no_changes=['main cohort','original indices','weights','death estimates','bootstrap'],
    prior_main_results_seen=True)


def order_proxy_window(order,admin,end,minutes):
    """Source-evidence categories, not diagnostic truth or future eligibility."""
    if not np.isfinite(order):return 'UNKNOWN_ORDER_TIME'
    if np.isfinite(admin) and admin<order:return 'ADMIN_BEFORE_ORDER'
    limit=order+float(minutes)*60
    if np.isfinite(admin) and order<=admin<=limit and admin<=end:return 'CONFIRMED_ADMIN_IN_WINDOW'
    if not np.isfinite(end) or end<limit:return 'CAPTURE_ENDS_BEFORE_WINDOW'
    return 'NO_CONFIRMED_ADMIN_IN_COMPLETE_WINDOW'


def order_proxy():
    m=poe_module();b=m.load();e=check(b);base=m.base();start=time.perf_counter()
    if 'order_proxy_bridge' not in e:
        e['order_proxy_bridge']=dict(config=ORDER_PROXY_CONFIG,frozen_utc=base.utc(),source_at_freeze=Path(__file__).read_bytes())
        m.save(b)
    d=e['order_proxy_bridge'];assert d['config']==ORDER_PROXY_CONFIG
    if 'summary' in d:m.say(dict(stage='order_proxy',reuse=True));return
    m.say(dict(stage='order_proxy_start',raw_bytes=0,question='order-to-resolved-administration and eligibility at each clock; no death contrasts'))
    x=e['prescriptions'];ref=b['source'].set_index('hadm_id');ctx=m.context(b)
    gate=base.import_file('hc_order_bridge_gate',base.OLD/'clinical_tte_gate.py')
    ng={int(h):g for h,g in b['NE'].groupby('hadm_id',sort=False)}
    actual=b['resolved_medications'];actual=actual[actual.hc_action].copy()
    em=e['control_orders']['emar'];actual['poe_id']=actual.emar_id.map(em.set_index('emar_id').poe_id)
    ag={int(h):g for h,g in actual.groupby('hadm_id')}
    rg={key:g for key,g in x['records'][x['records'].is_HC].dropna(subset=['poe_id']).groupby('poe_id')}
    er=e['emar_links'].set_index('emar_id');summary=[]
    def add(scope,metric,n=None,den=None,value=None):summary.append(dict(scope=scope,metric=metric,n=n,den=den,value=value))
    def qualification(h,t):
        s=ref.loc[h];stay=np.nan;state='UNKNOWN_TIME';strict=False
        if not np.isfinite(t):return state,stay,strict
        if t<s.ne0:return 'BEFORE_FIRST_OBSERVED_NE',stay,strict
        if t>s.ne0+86400:return 'AFTER_NE_DAY1',stay,strict
        if t<s.admit or t>=s.discharge or (np.isfinite(s.death_time) and t>=s.death_time):return 'HOSPITAL_TIME_CONFLICT',stay,strict
        active,stay,_,_=m.active_ne(ng[h],t,gate)
        if not active:return 'NE_NOT_CONFIRMED_ACTIVE',stay,strict
        state,strict=m.qualify(h,t,ctx,base,True)
        r=b['resolved_medications'];mixed=r[r.hadm_id.eq(h)&r.glucocorticoid&r.actual_systemic&~r.hc_action&r.chart.eq(t)]
        if state=='QUALIFIED_RECORD_PROXY' and len(mixed):state='SAME_TIME_OTHER_GC_VERSION'
        return state,stay,strict
    def unique_start(key,pharmacy=None):
        r=rg.get(key)
        if r is None:return np.nan,'NO_NATIVE_RX'
        if pharmacy is not None:
            exact=r[r.pharmacy_id.eq(pharmacy)]
            if len(exact):r=exact
        t=r.start.dropna().unique()
        return (float(t[0]),'UNIQUE') if len(t)==1 and r.start.notna().all() else (np.nan,'AMBIGUOUS_OR_MISSING')
    # Forward denominator contains nonexecuted orders and is not selected by delivery.
    orders=x['orders'];first=orders[orders.route.eq('IV_prescribed')&orders.within_admission].sort_values(['order','poe_id']).drop_duplicates('hadm_id').copy()
    rows=[]
    for t in first.itertuples():
        h=int(t.hadm_id);o=float(t.order);s=ref.loc[h];g=ag.get(h,actual.iloc[:0])
        future=g[g.chart.ge(o)];native=future[future.poe_id.eq(t.poe_id)]
        any_time=float(future.chart.min()) if len(future) else np.nan
        native_time=float(native.chart.min()) if len(native) else np.nan
        state,stay,strict=qualification(h,o);planned,pstate=unique_start(t.poe_id)
        end=float(min(s.med_end,s.discharge))
        rr=dict(hadm_id=h,subject_id=int(s.subject_id),poe_id=t.poe_id,index=o,stay_id=stay,
            qualification=state,strict_infection_pair=strict,planned=planned,planned_state=pstate,
            actual_any=any_time,actual_native=native_time,prior_actual_HC=bool(g.chart.lt(o).any()),
            capture_end=end,native_stop=t.stop,temporal_order_conflict=t.temporal_conflict,
            prescribed_start_before_order=bool(np.isfinite(planned) and planned<o))
        for minutes in ORDER_PROXY_CONFIG['windows_minutes']:
            rr['any_'+str(minutes)]=order_proxy_window(o,any_time,end,minutes)
            rr['native_'+str(minutes)]=order_proxy_window(o,native_time,end,minutes)
        rows.append(rr)
    first=pd.DataFrame(rows);qualified=first[first.qualification.eq('QUALIFIED_RECORD_PROXY')].copy()
    add('forward','first_IV_order_admissions_in_cached_risksets',len(first))
    for state,g in first.groupby('qualification'):add('qualification',state,len(g),len(first))
    for scope,f in [('first_IV_all',first),('qualified_first_IV',qualified)]:
        add(scope,'patients',f.subject_id.nunique());add(scope,'orders',len(f))
        for kind in ['any','native']:
            for minutes in ORDER_PROXY_CONFIG['windows_minutes']:
                for state,g in f.groupby(kind+'_'+str(minutes)):
                    add(scope,kind+'_'+str(minutes)+'_'+state,len(g),len(f))
        add(scope,'planned_start_before_order',f.prescribed_start_before_order.sum(),len(f))
        add(scope,'native_discontinue_within_6h',((f.native_stop>=f['index'])&(f.native_stop<=f['index']+21600)).sum(),len(f))
        lag=(f.actual_any-f['index'])/60
        for quant in [.25,.5,.75,.9]:add(scope,'actual_any_lag_P'+str(int(quant*100)),value=float(lag.dropna().quantile(quant)))
    # Reverse view: all actual cases, including non-IV/unknown prescription routes.
    linked=e['evidence'].set_index('case_id');case=[]
    for t in e['targets'].itertuples():
        info=er.loc[t.emar_id];o=float(linked.loc[t.case_id,'linked_order_time']);h=int(t.hadm_id)
        planned,pstate=unique_start(info.poe_id,info.pharmacy_id)
        state,stay,strict=qualification(h,o)
        case.append(dict(case_id=t.case_id,hadm_id=h,subject_id=int(t.subject_id),emar_id=t.emar_id,
            index=o,administration=float(t.index),planned=planned,planned_state=pstate,stay_id=stay,
            qualification=state,strict_infection_pair=strict,lag_order_min=(t.index-o)/60,
            lag_planned_min=(t.index-planned)/60,order_before_admin=bool(np.isfinite(o) and o<=t.index)))
    cases=pd.DataFrame(case)
    for field in ['lag_order_min','lag_planned_min']:
        v=cases[field];add('actual_cases',field+'_available',v.notna().sum(),len(v))
        add('actual_cases',field+'_negative',v.lt(0).sum(),len(v))
        for minutes in ORDER_PROXY_CONFIG['windows_minutes']:
            add('actual_cases',field+'_absolute_within_'+str(minutes),v.abs().le(minutes).sum(),len(v))
        for quant in [.25,.5,.75,.9]:add('actual_cases',field+'_P'+str(int(quant*100)),value=float(v.dropna().quantile(quant)))
    for state,g in cases.groupby('qualification'):add('actual_cases_qualification_at_order',state,len(g),len(cases))
    # A proper order-time lookback differs from truncating an administration-anchored window.
    request=[]
    for population,f in [('qualified_first_IV',qualified),('actual_cases_at_order',cases[cases.order_before_admin&cases.stay_id.notna()])]:
        z=f[['subject_id','hadm_id','stay_id','index']].copy();z['population']=population;z['treated']=1;request.append(z)
    req=pd.concat(request,ignore_index=True);req.stay_id=req.stay_id.astype('int64')
    temporary=dict(full_risksets=req,source=b['source'],clinical_labs=b['clinical_labs'],clinical_chart=b['clinical_chart'])
    feature=m.hc_cf_features(temporary,base)
    coverage=[]
    variables={'MAP':('chart','MAP_arterial'),'HR':('chart','HR'),'RR':('chart','RR'),
        'lactate':('labs','lactate'),'creatinine':('labs','creatinine'),'platelet':('labs','platelet')}
    for variable,(source,native) in variables.items():
        hours=(base.LAB_HOURS if source=='labs' else base.CHART_HOURS)[native]
        env=b['clinical_envelope'] if source=='labs' else b['clinical_stay_envelope']
        keys=req.hadm_id if source=='labs' else req.stay_id
        amin=req.hadm_id.map(ref.admit);available_lo=np.maximum(amin,keys.map(env['min'])-hours*3600)
        desired_lo=np.maximum(amin,req['index']-hours*3600)
        complete=(desired_lo>=available_lo)&req['index'].le(keys.map(env['max']))
        for population,ix in req.groupby('population').groups.items():
            for clock in ['primary','strict']:
                valid=feature[clock][variable].notna();full=complete.loc[ix]
                coverage.append(dict(population=population,variable=variable,clock=clock,n=len(ix),
                    observed_valid=int(valid.loc[ix].sum()),full_window_cached=int(full.sum()),
                    valid_full_window=int((valid.loc[ix]&full).sum()),
                    not_fully_projected=int((~full).sum())))
    # Small, meaningful boundary checks, not outcome or model-selection tests.
    assert order_proxy_window(0,60,500,1)=='CONFIRMED_ADMIN_IN_WINDOW'
    assert order_proxy_window(0,-1,500,1)=='ADMIN_BEFORE_ORDER'
    assert order_proxy_window(0,np.nan,30,1)=='CAPTURE_ENDS_BEFORE_WINDOW'
    assert order_proxy_window(0,np.nan,60,1)=='NO_CONFIRMED_ADMIN_IN_COMPLETE_WINDOW'
    assert order_proxy_window(0,61,500,1)=='NO_CONFIRMED_ADMIN_IN_COMPLETE_WINDOW'
    assert first.hadm_id.is_unique and cases.case_id.is_unique
    for minutes in ORDER_PROXY_CONFIG['windows_minutes']:
        pos=qualified['native_'+str(minutes)].eq('CONFIRMED_ADMIN_IN_WINDOW')
        assert qualified.loc[pos,'any_'+str(minutes)].eq('CONFIRMED_ADMIN_IN_WINDOW').all()
    d.update(summary=pd.DataFrame(summary),first_orders=first,actual_cases=cases,
        feature_requests=req,order_features=feature,order_feature_coverage=pd.DataFrame(coverage),
        checks=dict(window_tests=5,first_order_chosen_without_future_execution=True,no_outcome_contrasts=True,
            native_admin_implies_same_admission_admin=True,cache_envelopes_checked=True,main_results_unchanged=True),
        seconds=round(time.perf_counter()-start,2),source_current=Path(__file__).read_bytes())
    check(b);m.save(b)
    m.say(dict(stage='order_proxy_done',first_IV_orders=base.count(len(first)),qualified_at_order=base.count(len(qualified)),
        actual_cases=base.count(len(cases)),seconds=d['seconds'],raw_bytes=0,effects=0))


def order_proxy_publish():
    m=poe_module();b=m.load();e=check(b);d=e['order_proxy_bridge'];base=m.base()
    assert d['config']==ORDER_PROXY_CONFIG and 'checks' in d
    path=ROOT/'steroid_tte.md';before=path.read_bytes();old=before.decode('utf-8');newline='\r\n' if '\r\n' in old else '\n';old=old.replace('\r\n','\n')
    s=d['summary'];coverage=d['order_feature_coverage'];cases=d['actual_cases'];req=d['feature_requests']
    def result(scope,metric):
        z=s[s.scope.eq(scope)&s.metric.eq(metric)]
        return z.iloc[0] if len(z) else None
    def n(scope,metric):
        r=result(scope,metric);return int(r.n) if r is not None else 0
    def pct(a,total):
        v=base.pct(a,total);return '小单元或互补抑制' if v=='' else str(v)+'%'
    def fmt(a,total=None):
        return base.count(a) if total is None else base.count(a)+' / '+base.count(total)+'（'+pct(a,total)+'）'
    total=n('qualified_first_IV','orders')
    paired=[];used=req[req.population.eq('actual_cases_at_order')]
    caseids=cases.loc[cases.hadm_id.isin(used.hadm_id),'case_id']
    prior=e['poe_analysis']['feature_evidence']
    for clock in ['primary','strict']:
        for variable in ['MAP','lactate','creatinine']:
            q=prior[prior.clock.eq(clock)&prior.variable.eq(variable)&prior.case_id.isin(caseids)]
            r=coverage[coverage.population.eq('actual_cases_at_order')&coverage.clock.eq(clock)&coverage.variable.eq(variable)].iloc[0]
            assert len(q)==r.n
            paired.append(dict(clock=clock,variable=variable,n=len(q),
                original_admin_window_truncated=int(q.order_cutoff_value.notna().sum()),
                order_anchored_window_observed=int(r.observed_valid),full_window_cached=int(r.full_window_cached)))
    d['paired_window_comparison']=pd.DataFrame(paired)
    lines=['<!-- HC_ORDER_PROXY_START -->','### MIMIC实测医嘱启动代理，并确定eICU补充路线','',
        '**当前建议：继续医嘱定义的多中心补充分析准备，保留MIMIC实际首剂为主分析。** 两种来源不要求完全一致，但需分别说明干预对象和计时方式。以下实测没有查看新的组间死亡效果，没有用最佳结果选择窗口。','',
        '范围仅为既有风险集住院，不是全部MIMIC或eICU人群。先在这些住院中选首次原生POE关联的IV HC医嘱，再用医嘱之前的资料重新判断NE首日、当前输注、感染评估配对及既往激素记录；没有以随后给药、存活或治疗成功作为入组条件。原始主结果已经看过，本项是新增测量核查，不能称为完全前瞻预设。','',
        '|核查|结果|','|---|---:|',
        '|现有缓存住院中的首次IV HC医嘱|'+fmt(n('forward','first_IV_order_admissions_in_cached_risksets'))+'|',
        '|医嘱时符合原记录资格的首次IV HC医嘱|'+fmt(total)+'|',
        '|原实际启动者在关联医嘱时也符合记录资格|'+fmt(n('actual_cases_qualification_at_order','QUALIFIED_RECORD_PROXY'),len(cases))+'|','',
        '第二行不是新的全库治疗组：原始处方投影只覆盖已有风险集住院。该样本适合检验医嘱与给药的对应关系；若另估计医嘱启动效应，必须在其来源人群重新构建两组风险集，不能将这批记录接上旧首剂权重。','',
        '|医嘱后观察窗|同次住院内有解析确认的HC给药记录 / 全部符合记录资格医嘱|','|---|---:|']
    for minutes in ORDER_PROXY_CONFIG['windows_minutes']:
        lines.append('|'+str(minutes)+'分钟|'+fmt(n('qualified_first_IV','any_'+str(minutes)+'_CONFIRMED_ADMIN_IN_WINDOW'),total)+'|')
    lines+=['',
        '上表保留未执行、未在观察窗内执行及观察提前结束的医嘱作为分母；没有把剩余部分全部解释为未给药。原生POE对应的执行与同住院其他医嘱下的HC执行分别保存于私有证据；表中采用后者的总覆盖，不把两者混称为订单链完全一致。实际执行依据既有eMAR和剂量解析，明确Not Given等记录不算执行。住院提前结束、未捕获、医嘱变更均可能解释未确认部分。本地捕获不是外部金标准，也不能将本地执行比例直接外推至eICU。','',
        '|原约250次实际HC启动：时间代理|结果|','|---|---:|',
        '|实际给药减关联POE医嘱时间，中位数|'+f"{result('actual_cases','lag_order_min_P50').value:.1f}分钟"+'|',
        '|POE医嘱与实际给药相差不超过60分钟|'+fmt(n('actual_cases','lag_order_min_absolute_within_60'),len(cases))+'|',
        '|唯一原生处方计划开始时间可解析|'+fmt(n('actual_cases','lag_planned_min_available'),len(cases))+'|',
        '|处方计划开始与实际给药相差不超过60分钟|'+fmt(n('actual_cases','lag_planned_min_absolute_within_60'),len(cases))+'|',
        '|实际给药早于处方计划开始时间|'+fmt(n('actual_cases','lag_planned_min_negative'),len(cases))+'|','',
        '计划开始时间也并非无误差的替代：即使某个总体时间指标更接近实际首剂，也不据此选择看起来更好的时钟。eICU的drugStartOffset与MIMIC的处方starttime不是已经证明完全相同的字段，本地对应结果只能帮助解释风险，不能证明跨院记录语义相同。','',
        '#### 医嘱时点可获得的病情资料','',
        '前一轮“医嘱前截断”保持给药锚定的原窗口，只衡量原基线有多少信息出现在医嘱之后。它不是完整的医嘱时点可用率。本轮将同样长度的窗口整体移到医嘱前：床旁指标仍为1小时、乳酸6小时、其他列示实验室24小时，不延长窗口、不加入医嘱后信息。','',
        '|同一实际启动病例子集中的MAP（主测量时钟）|已有缓存中可用比例|','|---|---:|']
    pr=d['paired_window_comparison'];r=pr[pr.clock.eq('primary')&pr.variable.eq('MAP')].iloc[0]
    lines+=['|保留给药锚定窗口，仅提前截止|'+pct(r.original_admin_window_truncated,r.n)+'|',
        '|相同1小时回看窗重新锚定医嘱|'+pct(r.order_anchored_window_observed,r.n)+'|','',
        '|首次IV医嘱且记录资格成立者|主测量时钟覆盖|严格入库时钟覆盖|','|---|---:|---:|']
    for variable in ['MAP','lactate','creatinine']:
        q=coverage[coverage.population.eq('qualified_first_IV')&coverage.variable.eq(variable)].set_index('clock')
        lines.append('|'+variable+'|'+pct(q.loc['primary','observed_valid'],q.loc['primary','n'])+'|'+pct(q.loc['strict','observed_valid'],q.loc['strict','n'])+'|')
    lines+=['',
        '覆盖统计是已有缓存中可检出的资料，不要求完整病例入组。严格时钟额外要求床旁记录已入库，两种口径均报告；storetime并不自动代表医生首次看到数值。每个请求窗口已与原提取边界比较，超出边界标为“尚未完整提取”，不能记作源表无记录；边界证据与数值保存在同一私有缓存。这里的改善来自对齐到正确的研究节点，不是改善死亡效果或已经解决混杂。','',
        '#### 执行路径及backup','',
        '1. **主稿保留实际首剂问题。** 使用处方和POE结果明确医嘱先于给药的过程；不改写已冻结的MIMIC估计。\n'
        '2. **优先尝试eICU医嘱启动补充。** 可以使用记录中的HC医嘱作为治疗启动代理，不要求等待另一个实际首剂数据库。它回答“现在记录启动HC治疗、之后通常照护”的补充问题；资格、变量和随访一起锚定医嘱，后来接受HC者仍可作为此前对照。最终取消状态不用于未来筛选。需要明确的感染记录和病情调整，不以院内结局反推用途。\n'
        '3. **若不能建立对齐的医嘱比较，保留备选。** 以MIMIC实际首剂稿件加本项测量敏感性为完整主线；eICU只承担HC医嘱、NE轨迹及可迁移性的描述性证据，不强行合并效应或把异定义分析称为同协议外部验证。','',
        '本地eICU目前只有7张已读取来源。下一步最小补充为同版`admissionDx.csv`与`diagnosis.csv`：先核查时间戳可支持的感染/休克记录。它们本身不等于原MIMIC抗菌药—培养配对，也不能用最终出院诊断倒填基线。若该关能建立可解释的资格，再按确切字段补治疗前lab和vitals；不要求一次下载全库。当前药物缓存可继续复用。','',
        '入口：`steroid_event_review.py order-proxy` → `order-proxy-publish`。本轮从缓存完成，原始大表读取0字节，未拟合新模型或增加bootstrap。5项时间边界测试、首医嘱独立性、原生给药对应及缓存边界检查通过；旧协议、结果和历史文件保留。','',
        '来源：[eICU medication官方说明](https://eicu.mit.edu/eicutables/medication/)、[MIMIC prescriptions官方说明](https://mimic.mit.edu/docs/iv/modules/hosp/prescriptions.html)。','<!-- HC_ORDER_PROXY_END -->','']
    block='\n'.join(lines)
    if '<!-- HC_ORDER_PROXY_START -->' in old:
        start=old.index('<!-- HC_ORDER_PROXY_START -->');end=old.index('<!-- HC_ORDER_PROXY_END -->',start)+len('<!-- HC_ORDER_PROXY_END -->')
        while end<len(old) and old[end]=='\n':end+=1
    else:start=end=old.index('### 实际代码与证据位置')
    after=(old[:start]+block+'\n'+old[end:]).replace('\n',newline).encode('utf-8');assert path.read_bytes()==before
    if after!=before:
        hist=d.setdefault('report_history',[])
        if not any(r['sha256']==sha(before) for r in hist):hist.append(dict(sha256=sha(before),bytes=before))
        d['publication']=dict(report_sha256=sha(after),code_sha256=sha(Path(__file__).read_bytes()))
        check(b);m.save(b);path.write_bytes(after)
    m.say(dict(stage='order_proxy_publish',report=str(path),effects=0,backup='MIMIC primary plus eICU descriptive source evidence'))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','project-emar','build','link-poe','poe-review','poe-publish','control-orders','control-publish','prescription-project','prescription-poe','prescription-review','prescription-validate','prescription-publish','order-proxy','order-proxy-publish','test','publish'])
    p.add_argument('--poe-path');a=p.parse_args()
    if a.stage=='freeze':freeze()
    elif a.stage=='project-emar':project_emar()
    elif a.stage=='build':build()
    elif a.stage=='publish':publish()
    elif a.stage=='poe-review':poe_review()
    elif a.stage=='poe-publish':publish_poe()
    elif a.stage=='control-orders':control_orders()
    elif a.stage=='control-publish':control_publish()
    elif a.stage=='prescription-project':prescription_project()
    elif a.stage=='prescription-poe':prescription_poe()
    elif a.stage=='prescription-review':prescription_review()
    elif a.stage=='prescription-validate':prescription_validate()
    elif a.stage=='prescription-publish':prescription_publish()
    elif a.stage=='order-proxy':order_proxy()
    elif a.stage=='order-proxy-publish':order_proxy_publish()
    elif a.stage=='link-poe':
        if not a.poe_path:p.error('--poe-path is required')
        link_poe(a.poe_path)
    else:print(json.dumps(dict(synthetic_tests_passed=tests())))


if __name__=='__main__':main()
