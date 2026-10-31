# -*- coding: utf-8 -*-
"""Bounded eICU source feasibility for the existing HC initiation question.

Python 3.8, CPU. Raw sources are read-only. Source checks precede any estimate.
Only aggregate material is published in the canonical HC report. The clinical
extension includes fixed assignment diagnostics and explicitly configured
exploratory order-based associations; frozen MIMIC estimates are never changed.
"""
import argparse
import gzip
import hashlib
import importlib.util
import io
import json
import os
import re
import time
import warnings
from pathlib import Path
from hc_paths import configured_path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = configured_path('eicu_dir', '../restricted/eicu-crd-2.0')
PRIVATE = configured_path('private_dir', '../restricted/hc-private')
STATE = PRIVATE/'hc_eicu_feasibility.pkl.gz'
CONFIG = dict(
    database='eICU-CRD v2.0; independent keys, never joined to MIMIC',
    question='source feasibility for actual HC initiation during NE; no new effect estimation',
    population='all recorded adult ICU stays as a source denominator, NOT NE/HC-qualified patients',
    age='numeric age >=18 or documented >89; missing/unparseable age separate',
    hospital_clock='ICU offset minus hospitaladmitoffset; never sort surrogate stay IDs as time',
    hospital_outcome='one consistent observed Alive/Expired across ICU rows of one hospital stay; missing/conflict distinct',
    outcome_caveat='status is this hospital discharge, not fixed-day or post-transfer mortality',
    fixed_coverage_minutes=[0,360,1440],
    coverage_interpretation='source timing around ICU admission only, not HC time zero or new eligibility',
    history='admissionDrug is recorded admission medication history, not proof of new ICU administration or complete absence',
    micro='collapse same unit/collection offset/site for timing only; no organism/susceptibility used as baseline; result availability unknown',
    careplan='entered offset and definite/probable/possible assessment; no discharge-active or future-response variables',
    absent='unobserved source != no disease/no treatment; no source coverage threshold selected',
    disclosure='counts rounded to10; cells1-10 and their reconstructable percentages suppressed',
    changes='MIMIC results, risksets, weights, phenotype files unchanged')
COLS = {
    'patient':['patientunitstayid','patienthealthsystemstayid','uniquepid','hospitalid','wardid','age','gender',
        'hospitaladmitoffset','hospitaldischargeoffset','hospitaldischargestatus','hospitaldischargelocation',
        'hospitaldischargeyear','unitdischargeoffset','unitdischargestatus','unittype','unitadmitsource',
        'unitvisitnumber','unitstaytype','admissionweight'],
    'hospital':['hospitalid','numbedscategory','teachingstatus','region'],
    'admissiondrug':['admissiondrugid','patientunitstayid','drugoffset','drugenteredoffset','drugname',
        'drugdosage','drugunit','drugadmitfrequency','rxincluded','writtenineicu'],
    'microlab':['microlabid','patientunitstayid','culturetakenoffset','culturesite'],
    'careplaninfectiousdisease':['cplinfectid','patientunitstayid','cplinfectdiseaseoffset',
        'infectdiseasesite','infectdiseaseassessment','treatment']}
COLS.update({
    'medication':['medicationid','patientunitstayid','drugorderoffset','drugstartoffset',
        'drugivadmixture','drugordercancelled','drugname','drughiclseqno','dosage','routeadmin',
        'frequency','loadingdose','prn','drugstopoffset','gtc'],
    'infusiondrug':['infusiondrugid','patientunitstayid','infusionoffset','drugname',
        'drugrate','infusionrate','drugamount','volumeoffluid','patientweight'],
    'admissiondx':['admissiondxid','patientunitstayid','admitdxenteredoffset','admitdxpath','admitdxname'],
    'diagnosis':['diagnosisid','patientunitstayid','diagnosisoffset','diagnosisstring','icd9code']})
EXPECTED=list(COLS)
GC = re.compile(r'hydrocortis|methylpred|predniso|prednisone|dexameth|betameth|(?<!fludro)cortisone|solu.?cort|solu.?med|decadron|cortef|medrol|deltasone',re.I)
HC = re.compile(r'hydrocortis|solu.?cort|cortef',re.I)
TOPICAL = re.compile(r'cream|ointment|ophth|otic|topical|rectal|suppositor|lotion|enema|foam|gel\b',re.I)
NE = re.compile(r'norepineph|noradren|levophed',re.I)
DRUG_CONFIG = dict(
    purpose='NE/HC source observability, not a qualified treatment comparison',
    names='HC/other glucocorticoids and norepinephrine or Levophed; code-only identities remain unknown',
    clock='offsets mapped to hospital elapsed minutes using each unit hospitaladmitoffset',
    medication='orders and prescribed start; cancellation has no cancellation timestamp; no actual dose inferred',
    infusion='nursing infusion records; offset is recorded entry, not proven start/stop; missing not zero',
    dose='parse explicit units in name only; mcg/min uses row weight or admission weight; no assumed concentration or salt conversion',
    coverage='whole adult source, then NE source and first HC IV-order record; no outcome-based filtering',
    technical_window_minutes=1440,
    recent_NE_minutes=[30,60,120],
    new_user='known earlier same-hospital steroid evidence flagged, no-record never proves steroid naive',
    outcome='only completeness; no mortality by exposure, no models')
# DRUG_CONFIG records the initial name-based pass. The subsequent, separately
# recorded code_resolution rule extends drug identity without overwriting it.


def drug_project():
    b=load();assert b['summary'] is not None
    d=b.setdefault('drug',dict(config=DRUG_CONFIG,runs=[],source_before=b['source_current']))
    assert d['config']==DRUG_CONFIG
    p=b['patient'];ids=set(p.patientunitstayid);adult=set(p.loc[p.adult,'patientunitstayid'])
    for name,timecol in [('medication','drugorderoffset'),('infusiondrug','infusionoffset')]:
        if name in d:
            info=b['inputs'][name];f=locate(name)
            assert tuple(info['signature'])==(str(f),f.stat().st_size,f.stat().st_mtime_ns)
            continue
        parts=[];keys=[];seen=set();named=set();stats=dict(rows=0,unlinked_rows=0,missing_name_rows=0,missing_time_rows=0)
        names={}
        def consume(z):
            stats['rows']+=len(z);key=pd.to_numeric(z.patientunitstayid,errors='coerce')
            keys.append(pd.to_numeric(z[name+'id'],errors='raise').to_numpy('int64'))
            stats['unlinked_rows']+=int((~key.isin(ids)).sum());z=z[key.isin(ids)].copy()
            z['patientunitstayid']=key.loc[z.index].astype('int64');nm=lower(z.drugname)
            seen.update(z.patientunitstayid);named.update(z.loc[nm.ne(''),'patientunitstayid'])
            stats['missing_name_rows']+=int(nm.eq('').sum())
            stats['missing_time_rows']+=int(pd.to_numeric(z[timecol],errors='coerce').isna().sum())
            hit=nm.str.contains(GC)|nm.str.contains(NE)
            q=z[hit].copy()
            if len(q):
                q['is_HC']=q.drugname.str.contains(HC,na=False);q['is_GC']=q.drugname.str.contains(GC,na=False)
                q['is_NE']=q.drugname.str.contains(NE,na=False);q['explicit_local_name']=q.drugname.str.contains(TOPICAL,na=False)
                parts.append(q)
                for k,v in q.drugname.value_counts().items():names[k]=names.get(k,0)+int(v)
        changed=read_chunks(b,name,consume)
        if not changed:raise ValueError('DRUG_INPUT_MISSING_OR_INCOMPLETE_CACHE: '+name)
        kk=np.concatenate(keys);stats['duplicate_primary_keys']=int(len(kk)-len(np.unique(kk)))
        d[name]=dict(records=pd.concat(parts,ignore_index=True),stats=stats,units_any=sorted(seen),units_named=sorted(named),drug_names=names)
        save(b)
        say(dict(stage='eicu_drug_project',table=name,retained_rows=base().count(len(d[name]['records'])),effects=0))
    b['available_now']={n:str(locate(n)) if locate(n) else None for n in EXPECTED}
    d['source_current']=Path(__file__).read_bytes();save(b)


def drug_code_project():
    """One bounded reread to resolve code-only rows after missing names were observed."""
    b=load();d=b['drug'];m=base()
    if 'code_resolution' in d:say(dict(stage='eicu_drug_codes',reuse=True));return
    original=d['medication']['records'];codes=set(original.drughiclseqno.dropna().astype(str));parts=[]
    path=locate('medication');st=path.stat();start=time.perf_counter();last=start;rows=0
    say(dict(stage='eicu_drug_codes_start',bytes=st.st_size,columns=COLS['medication'],
        filter='HICL codes occurring in named NE/GC records; validate all named rows for code conflicts, then recover blank-name rows'))
    reader=m.HashReader(path);buffer=io.BufferedReader(reader,buffer_size=4*1024*1024)
    stream=gzip.GzipFile(fileobj=buffer) if path.name.endswith('.gz') else buffer
    try:
        for z in pd.read_csv(stream,usecols=COLS['medication'],dtype=str,chunksize=100000):
            rows+=len(z);q=z[z.drughiclseqno.isin(codes)]
            if len(q):parts.append(q.copy())
            if time.perf_counter()-last>30:say(dict(stage='eicu_drug_codes_progress',seconds=round(time.perf_counter()-start),rows=m.count(rows)));last=time.perf_counter()
        assert reader.n==st.st_size;digest=reader.h.hexdigest()
    finally:stream.close();buffer.close()
    assert digest==b['inputs']['medication']['sha256']
    assert (st.st_size,st.st_mtime_ns)==(path.stat().st_size,path.stat().st_mtime_ns)
    q=pd.concat(parts,ignore_index=True);q['patientunitstayid']=pd.to_numeric(q.patientunitstayid,errors='raise').astype('int64')
    mapping={};evidence=[]
    for code,g in q.groupby('drughiclseqno'):
        named=lower(g.drugname);named=named[named.ne('')]
        kinds={label:bool(len(named) and named.str.contains(pattern).all()) for label,pattern in [('is_HC',HC),('is_GC',GC),('is_NE',NE)]}
        mapping[code]=kinds;evidence.append(dict(code=code,n_named=len(named),names=sorted(set(named)),**kinds,
            blank_name_rows=int(lower(g.drugname).eq('').sum())))
    extra=q[lower(q.drugname).eq('')].copy()
    for label in ['is_HC','is_GC','is_NE']:extra[label]=extra.drughiclseqno.map(lambda c:mapping[c][label])
    extra=extra[extra.is_GC|extra.is_NE].copy();extra['explicit_local_name']=False;extra['identity_source']='unambiguous_same_table_HICL'
    original=original.copy();original['identity_source']='explicit_name'
    extra=extra[~extra.medicationid.isin(original.medicationid)]
    assert extra.patientunitstayid.isin(b['patient'].patientunitstayid).all()
    if 'name_only_stage' not in d:
        d['name_only_stage']={k:d[k] for k in ['summary','technical_candidates','site_summary','concentration_review','validation','source_current'] if k in d}
    d['medication']['records']=pd.concat([original,extra],ignore_index=True)
    d['code_resolution']=dict(rule='only blank-name rows with a HICL code whose every nonblank source-table name matches the drug class; conflicting codes remain UNKNOWN',
        evidence=pd.DataFrame(evidence),added_rows=len(extra),added_HC_rows=int(extra.is_HC.sum()),
        added_NE_rows=int(extra.is_NE.sum()),rows_read=rows,seconds=time.perf_counter()-start,bytes_read=st.st_size,sha256=digest,
        source_current=Path(__file__).read_bytes())
    # Preserve the earlier diagnostic summaries, then recompute from the fuller projection.
    for key in ['summary','technical_candidates','site_summary','concentration_review','validation','medication_resolved','NE_records']:
        d.pop(key,None)
    save(b);say(dict(stage='eicu_drug_codes_done',added_rows=m.count(len(extra)),added_HC_rows=m.count(extra.is_HC.sum()),
        added_NE_rows=m.count(extra.is_NE.sum()),effects=0))


def route_class(s):
    v=str(s).strip().upper()
    if v in ['IV','IVP','IVPB','IV PUSH','PERIPH IV','INTRAVENOU','INTRAVENOUS','IV (INJECTION)'] or v.startswith('IV - '):return 'IV_prescribed'
    if v in ['IM','INTRAMUSCULAR']:return 'IM_prescribed'
    if v in ['PO','ORAL','NG','NGT','G TUBE','PEG'] or v.startswith('PO/'):return 'enteral_prescribed'
    if re.search(r'INTRATHEC|TOPICAL|RECTAL|OPHTH|OTIC|INHAL',v) or v=='IT':return 'other_nonIV_route'
    return 'route_unknown'


def rate_unit(name):
    tokens=re.findall(r'\(([^()]*)\)',str(name).lower())
    for tok in tokens[::-1]:
        t=tok.replace(' ','').replace('ug','mcg').replace('hour','hr')
        if t in ['mcg/kg/min','mcg/min','mcg/kg/hr','mcg/hr','mg/kg/min','mg/min','mg/hr','ml/hr']:return t
    return 'unknown'


def normalize_ne(rate,unit,weight):
    if not np.isfinite(rate):return np.nan
    if unit=='mcg/kg/min':return rate
    if unit=='mcg/kg/hr':return rate/60
    if unit=='mg/kg/min':return rate*1000
    if not np.isfinite(weight) or weight<=0:return np.nan
    return {'mcg/min':rate/weight,'mcg/hr':rate/weight/60,
        'mg/min':rate*1000/weight,'mg/hr':rate*1000/weight/60}.get(unit,np.nan)


def drug_review():
    b=load();d=b['drug'];assert d['config']==DRUG_CONFIG
    if 'summary' in d:say(dict(stage='eicu_drug_review',reuse=True));return
    begin=time.perf_counter();p=b['patient'];ref=p.set_index('patientunitstayid');adult=p[p.adult];adult_ids=set(adult.patientunitstayid)
    med=d['medication']['records'].copy();inf=d['infusiondrug']['records'].copy()
    def times(q,cols):
        for col in cols:
            q[col]=pd.to_numeric(q[col],errors='coerce');q[col+'_hospital']=q[col]-q.patientunitstayid.map(ref.hospitaladmitoffset)
        for col in ['patienthealthsystemstayid','hospitalid','uniquepid','adult','unitdischargeoffset','hospitaladmitoffset','hospitaldischargeoffset']:
            q[col]=q.patientunitstayid.map(ref[col])
    times(med,['drugorderoffset','drugstartoffset','drugstopoffset']);times(inf,['infusionoffset'])
    med['route_class']=med.routeadmin.map(route_class)
    med['cancelled_final']=lower(med.drugordercancelled).eq('yes')
    med['prn_record']=lower(med.prn).eq('yes')
    med['order_in_hospital']=med.drugorderoffset_hospital.ge(0)&med.drugorderoffset.le(med.hospitaldischargeoffset)
    med['order_in_ICU']=med.drugorderoffset.ge(0)&med.drugorderoffset.le(med.unitdischargeoffset)
    inf['rate_numeric']=pd.to_numeric(inf.drugrate,errors='coerce');inf['infusion_numeric']=pd.to_numeric(inf.infusionrate,errors='coerce')
    inf['row_weight']=pd.to_numeric(inf.patientweight,errors='coerce');inf['admission_weight']=inf.patientunitstayid.map(ref.admissionweight)
    inf['weight_used']=inf.row_weight.where(inf.row_weight.gt(0),inf.admission_weight.where(inf.admission_weight.gt(0)))
    inf['unit']=inf.drugname.map(rate_unit)
    inf['positive_record']=inf.rate_numeric.gt(0)|inf.infusion_numeric.gt(0)
    # Volume-only cumulative rows are not rates. No absent record becomes a stop.
    volume_only=inf.drugname.str.contains(r'volume',case=False,na=False)&inf.unit.eq('unknown')
    inf.loc[volume_only,'positive_record']=False
    inf['in_ICU']=inf.infusionoffset.ge(0)&inf.infusionoffset.le(inf.unitdischargeoffset)&inf.infusionoffset.le(inf.hospitaldischargeoffset)
    ne=inf[inf.is_NE&inf.adult&inf.in_ICU].copy()
    ne['dose_mcgkgmin']=[normalize_ne(r.rate_numeric,r.unit,r.weight_used) for r in ne.itertuples()]
    ne['concentration_fields_positive']=pd.to_numeric(ne.drugamount,errors='coerce').gt(0)&pd.to_numeric(ne.volumeoffluid,errors='coerce').gt(0)
    pos=ne[ne.positive_record];ne_units=set(pos.patientunitstayid);ne_hosp=set(pos.patienthealthsystemstayid)
    hc=med[med.is_HC&med.adult];hiv=hc[hc.route_class.eq('IV_prescribed')&hc.order_in_hospital].copy()
    first_hiv=hiv.sort_values(['drugorderoffset_hospital','medicationid']).drop_duplicates('patienthealthsystemstayid')
    first_ne=pos.groupby('patienthealthsystemstayid').infusionoffset_hospital.min()
    first_hiv['first_NE_hospital']=first_hiv.patienthealthsystemstayid.map(first_ne)
    first_hiv['elapsed_NE_min']=first_hiv.drugorderoffset_hospital-first_hiv.first_NE_hospital
    cand=first_hiv[first_hiv.order_in_ICU&first_hiv.elapsed_NE_min.between(0,1440,inclusive='both')].copy()
    groups={k:g.sort_values('infusionoffset') for k,g in ne.groupby('patientunitstayid')}
    gcmed=med[med.is_GC&med.order_in_hospital&~med.route_class.eq('other_nonIV_route')]
    gm={h:g for h,g in gcmed.groupby('patienthealthsystemstayid')}
    # Any explicit earlier steroid record is evidence; missing history is not a negative test.
    history=b['tables']['admissiondrug']['steroid_records'].copy()
    history['patienthealthsystemstayid']=history.patientunitstayid.map(ref.patienthealthsystemstayid)
    history['drug_hospital']=history.drugoffset-history.patientunitstayid.map(ref.hospitaladmitoffset)
    history['entry_hospital']=history.drugenteredoffset-history.patientunitstayid.map(ref.hospitaladmitoffset)
    gh={h:g for h,g in history.groupby('patienthealthsystemstayid')}
    micro=b['tables']['microlab']['records'].copy();micro['hospital']=micro.patientunitstayid.map(ref.patienthealthsystemstayid)
    micro['when']=micro.culturetakenoffset-micro.patientunitstayid.map(ref.hospitaladmitoffset)
    mm={h:g.when.to_numpy(float) for h,g in micro.groupby('hospital')}
    care=b['tables']['careplaninfectiousdisease']['records'].copy();care=care[care.supported_infection_assessment]
    care['hospital']=care.patientunitstayid.map(ref.patienthealthsystemstayid);care['when']=care.cplinfectdiseaseoffset-care.patientunitstayid.map(ref.hospitaladmitoffset)
    cc={h:g.when.to_numpy(float) for h,g in care.groupby('hospital')}
    evidence=[]
    for t in cand.itertuples():
        g=groups.get(t.patientunitstayid);prior=g[g.infusionoffset<t.drugorderoffset]
        last=prior[prior.infusionoffset.eq(prior.infusionoffset.max())] if len(prior) else prior
        age=float(t.drugorderoffset-last.infusionoffset.iloc[0]) if len(last) else np.nan
        explicit_zero=last.rate_numeric.eq(0)|last.infusion_numeric.eq(0)
        positive=bool(len(last) and last.positive_record.all())
        conflict=bool(len(last) and explicit_zero.any() and last.positive_record.any())
        if conflict:positive=False
        x=gm.get(t.patienthealthsystemstayid)
        prior_gc=bool(x is not None and x.drugorderoffset_hospital.lt(t.drugorderoffset_hospital).any())
        y=gh.get(t.patienthealthsystemstayid)
        prior_history=bool(y is not None and ((y.drug_hospital<t.drugorderoffset_hospital)&(y.entry_hospital<t.drugorderoffset_hospital)&~y.explicit_topical_or_local_name).any())
        cult=mm.get(t.patienthealthsystemstayid,np.array([]));assess=cc.get(t.patienthealthsystemstayid,np.array([]))
        r=dict(medicationid=t.medicationid,patientunitstayid=t.patientunitstayid,hospital=t.patienthealthsystemstayid,
            subject=t.uniquepid,site=t.hospitalid,index=t.drugorderoffset_hospital,last_NE_age_min=age,
            last_NE_positive=positive,last_NE_conflict=conflict,known_prior_GC_order=prior_gc,known_prior_GC_history=prior_history,
            culture_before=bool((cult<t.drugorderoffset_hospital).any()),infection_assessment_before=bool((assess<t.drugorderoffset_hospital).any()),
            cancellation_final=t.cancelled_final,prn=t.prn_record,
            HC_start_before_order=t.drugstartoffset<t.drugorderoffset,NE_dose_available=bool(positive and last.dose_mcgkgmin.notna().all()),
            same_second_NE=bool(g.infusionoffset.eq(t.drugorderoffset).any()))
        for minutes in DRUG_CONFIG['recent_NE_minutes']:r['recent_positive_NE_'+str(minutes)]=bool(positive and age<=minutes)
        evidence.append(r)
    ev=pd.DataFrame(evidence);rows=[]
    def add(section,metric,n,den=None,value=None):rows.append(dict(section=section,metric=metric,n=n,d=den,value=value))
    for name in ['medication','infusiondrug']:
        units=set(d[name]['units_named'])&adult_ids
        add('coverage',name+'：成人ICU有药名记录',len(units),len(adult))
        add('coverage',name+'：有记录的医院',p[p.patientunitstayid.isin(units)].hospitalid.nunique(),adult.hospitalid.nunique())
    add('NE','成人ICU内有正速率NE记录的ICU记录',len(ne_units),len(adult))
    add('NE','上述独立住院',len(ne_hosp));add('NE','上述独立患者',pos.uniquepid.nunique());add('NE','上述医院',pos.hospitalid.nunique())
    add('NE','NE正速率记录行',len(pos),len(ne));add('NE','正速率行剂量可按明确单位归一',pos.dose_mcgkgmin.gt(0).sum(),len(pos))
    add('NE','仅容量速率或单位未知的正速率行',pos.unit.isin(['ml/hr','unknown']).sum(),len(pos))
    add('NE','容量/未知单位正速率行有浓度字段（未自动转换）',(pos.unit.isin(['ml/hr','unknown'])&pos.concentration_fields_positive).sum(),len(pos))
    add('NE','正速率行有当前记录体重',pos.row_weight.gt(0).sum(),len(pos))
    add('NE','使用入ICU体重而非当前记录体重',(pos.row_weight.le(0)|pos.row_weight.isna()).sum(),len(pos))
    ne['zero_record']=ne.rate_numeric.eq(0)
    unique_time=ne.groupby(['patientunitstayid','infusionoffset']).agg(rows=('infusiondrugid','size'),positive=('positive_record','any'),zero=('zero_record','any'),maxdose=('dose_mcgkgmin','max'),mindose=('dose_mcgkgmin','min'))
    add('NE','同ICU同刻多条记录',unique_time.rows.gt(1).sum(),len(unique_time))
    add('NE','同ICU同刻零与正速率并存',(unique_time.positive&unique_time.zero).sum(),len(unique_time))
    add('NE','同刻已归一剂量不一致',(unique_time.maxdose-unique_time.mindose).abs().gt(.001).sum(),len(unique_time))
    gaps=pos.drop_duplicates(['patientunitstayid','infusionoffset']).sort_values(['patientunitstayid','infusionoffset']).groupby('patientunitstayid').infusionoffset.diff().dropna()
    for q in [.5,.9,.95]:add('NE','NE正速率记录间隔P'+str(int(q*100))+'分钟',None,value=float(gaps.quantile(q)))
    add('HC','成人HC医嘱记录',len(hc));add('HC','有HC医嘱的独立住院',hc.patienthealthsystemstayid.nunique())
    add('HC','医院内IV HC医嘱涉及住院',hiv.patienthealthsystemstayid.nunique());add('HC','医院内IV HC医嘱涉及医院',hiv.hospitalid.nunique())
    hi=inf[inf.is_HC&inf.adult];add('HC','HC输注记录行',len(hi));add('HC','HC输注记录涉及住院',hi.patienthealthsystemstayid.nunique())
    add('HC','HC正速率输注涉及住院',hi.loc[hi.positive_record&hi.in_ICU,'patienthealthsystemstayid'].nunique())
    for k,g in hc.groupby('route_class'):add('HC','HC医嘱途径：'+k,len(g),len(hc))
    add('HC','HC医嘱最终取消标志为Yes',hc.cancelled_final.sum(),len(hc))
    add('HC','HC医嘱计划开始早于开立',hc.drugstartoffset.lt(hc.drugorderoffset).sum(),len(hc))
    add('HC','HC医嘱计划停止早于开始',hc.drugstopoffset.lt(hc.drugstartoffset).sum(),len(hc))
    add('candidate','首次IV HC医嘱在首次观察正速率NE之后0–24h且开于ICU',len(ev))
    add('candidate','上述独立患者',ev.subject.nunique());add('candidate','上述医院',ev.site.nunique())
    for minutes in DRUG_CONFIG['recent_NE_minutes']:add('candidate','医嘱前'+str(minutes)+'分钟内最近NE记录为正',ev['recent_positive_NE_'+str(minutes)].sum(),len(ev))
    for col in ['NE_dose_available','known_prior_GC_order','known_prior_GC_history','culture_before','infection_assessment_before','cancellation_final','HC_start_before_order','last_NE_conflict','same_second_NE']:
        add('candidate',col,ev[col].sum(),len(ev))
    clear=~ev.known_prior_GC_order&~ev.known_prior_GC_history
    add('candidate','未检出更早GC记录且60分钟内NE正记录（不是资格通过）',(clear&ev.recent_positive_NE_60).sum(),len(ev))
    add('candidate','上述记录并有医嘱前培养或感染评估',(clear&ev.recent_positive_NE_60&(ev.culture_before|ev.infection_assessment_before)).sum(),len(ev))
    units=set(ev.patientunitstayid);history_units=set(b['tables']['admissiondrug']['coverage'].index[b['tables']['admissiondrug']['coverage'].row_first.notna()])
    add('candidate','候选所在ICU有入院药史记录',len(units&history_units),len(units))
    h=b['hospital_summary'].set_index('hospital_stay')
    # Do not inspect death proportions by exposure. Only endpoint completeness is used here.
    add('candidate','候选住院院内结局可判定',ev.hospital.map(h.outcome).isin(['alive','expired']).sum(),len(ev))
    sites=pd.DataFrame({'adult_units':adult.groupby('hospitalid').size(),
        'NE_units':pos.groupby('hospitalid').patientunitstayid.nunique(),
        'HC_order_hospitals':hiv.groupby('hospitalid').patienthealthsystemstayid.nunique(),
        'technical_candidates':ev.groupby('site').size()}).fillna(0)
    add('site','技术候选前5医院占比',None,value=float(sites.technical_candidates.nlargest(5).sum()/len(ev)) if len(ev) else np.nan)
    d.update(summary=pd.DataFrame(rows),medication_resolved=med,NE_records=ne,technical_candidates=ev,site_summary=sites,
        tests=dict(no_effect_estimation=True,no_new_weights=True,no_candidate_called_eligible=True,no_missing_filled_zero=True),
        review_seconds=time.perf_counter()-begin,source_current=Path(__file__).read_bytes())
    save(b)
    out=[]
    for r in d['summary'][d['summary'].section.isin(['HC','candidate'])].itertuples():
        out.append(dict(metric=r.metric,n=base().count(r.n),denominator=base().count(r.d) if pd.notna(r.d) else None))
    say(dict(stage='eicu_drug_review_done',summary=out,effects=0))


def named_final_concentration(name):
    s=str(name).lower()
    q=re.search(r'(\d+(?:\.\d+)?)\s*mg\s*(?:in|/)\s*(\d+(?:\.\d+)?)\s*ml',s)
    if not q:return np.nan
    mg,ml=map(float,q.groups())
    # Restrict to an explicit final infusion bag, not a concentrated ampoule.
    if mg<=0 or ml<50 or not re.search(r'\bns\b|\bd5w\b|dextrose|infusion|drip',s):return np.nan
    return mg*1000/ml


def concentration_review():
    b=load();d=b['drug']
    if 'concentration_review' in d:say(dict(stage='eicu_concentration_review',reuse=True));return
    started=time.perf_counter();last_tick=started;processed=0
    c=dict(rule='explicit NE final-bag mg/mL in same-ICU medication name, order+start before infusion record and stop after; unique concentration across temporally compatible orders',
        no_default='no default concentration; no ampoule dilution inference; no salt conversion',
        cancellation='cancelled orders not assumed historically inactive; presence makes proxy unavailable',
        interpretation='dose reconstruction proxy, not a validated native infusion-order linkage; no new eligibility or effect estimation')
    m=d['medication_resolved'];med=m[m.is_NE].copy();med['concentration']=med.drugname.map(named_final_concentration)
    ne=d['NE_records'].copy();ne['dose_concentration_proxy']=np.nan;ne['concentration_state']='not_required_or_not_volume_rate'
    groups={u:g for u,g in med.groupby('patientunitstayid')}
    select=ne.positive_record&ne.unit.eq('ml/hr')
    for u,ix in ne[select].groupby('patientunitstayid').groups.items():
        processed+=1
        if time.perf_counter()-last_tick>30:
            say(dict(stage='eicu_concentration_progress',seconds=round(time.perf_counter()-started),unit_groups=base().count(processed)));last_tick=time.perf_counter()
        g=groups.get(u)
        if g is None:ne.loc[ix,'concentration_state']='no_NE_order_in_unit';continue
        t=ne.loc[ix,'infusionoffset'].to_numpy(float)
        lower_bound=np.maximum(g.drugorderoffset.to_numpy(float),g.drugstartoffset.to_numpy(float))
        upper_bound=g.drugstopoffset.to_numpy(float)
        active=(t[:,None]>=lower_bound)&(t[:,None]<upper_bound)
        conc=g.concentration.to_numpy(float);known=np.isfinite(conc)
        incomplete=(active&~known).any(1)|(active&g.cancelled_final.to_numpy(bool)).any(1)
        any_active=active.any(1);n_known=(active&known).sum(1)
        lo=np.where(active&known,conc,np.inf).min(1);hi=np.where(active&known,conc,-np.inf).max(1)
        ok=any_active&~incomplete&(n_known>0)&np.isclose(lo,hi)
        state=np.where(~any_active,'no_temporally_compatible_order',np.where(incomplete,'unknown_recipe_or_cancellation_time',np.where(ok,'unique_explicit_recipe','conflicting_recipe')))
        ne.loc[ix,'concentration_state']=state
        rate=ne.loc[ix,'rate_numeric'].to_numpy(float)
        alternate=ne.loc[ix,'infusion_numeric'].to_numpy(float)
        rate=np.where(np.isfinite(rate),rate,alternate)
        disagree=np.isfinite(rate)&np.isfinite(alternate)&(np.abs(rate-alternate)>np.maximum(.01,np.abs(rate)*.01))
        w=ne.loc[ix,'weight_used'].to_numpy(float)
        valid=ok&~disagree&np.isfinite(w)&(w>0)&(rate>0)
        value=np.full(len(ix),np.nan);value[valid]=rate[valid]*lo[valid]/60/w[valid]
        ne.loc[ix,'dose_concentration_proxy']=value
        ne.loc[np.asarray(ix)[ok&disagree],'concentration_state']='rate_fields_conflict'
    ev=d['technical_candidates'].copy();groups={u:g for u,g in ne.groupby('patientunitstayid')};ref=b['patient'].set_index('patientunitstayid')
    availability=[]
    for t in ev.itertuples():
        q=groups[t.patientunitstayid];index=t.index+ref.loc[t.patientunitstayid,'hospitaladmitoffset']
        q=q[q.infusionoffset<index];last=q[q.infusionoffset.eq(q.infusionoffset.max())]
        ok=bool(t.last_NE_positive and len(last) and last.dose_mcgkgmin.combine_first(last.dose_concentration_proxy).notna().all())
        availability.append(ok)
    ev['NE_dose_with_concentration_proxy']=availability
    pos=ne.positive_record
    summary=dict(positive_rows=int(pos.sum()),direct_dose_rows=int((pos&ne.dose_mcgkgmin.gt(0)).sum()),
        additional_proxy_rows=int((pos&ne.dose_mcgkgmin.isna()&ne.dose_concentration_proxy.gt(0)).sum()),
        technical_candidates=len(ev),candidates_direct_dose=int(ev.NE_dose_available.sum()),
        candidates_with_proxy=int(ev.NE_dose_with_concentration_proxy.sum()))
    d['concentration_review']=dict(config=c,summary=summary,row_states=ne.loc[select,'concentration_state'].value_counts().to_dict(),
        source_current=Path(__file__).read_bytes(),seconds=time.perf_counter()-started)
    d['NE_records']=ne;d['technical_candidates']=ev;save(b)
    say(dict(stage='eicu_concentration_review_done',summary={k:base().count(v) for k,v in summary.items()},effects=0))


def drug_validate():
    b=load();d=b['drug'];ne=d['NE_records'];ev=d['technical_candidates']
    assert normalize_ne(8,'mcg/min',80)==.1
    assert normalize_ne(.48,'mg/hr',80)==.1
    assert normalize_ne(6,'mcg/kg/hr',np.nan)==.1
    assert np.isnan(normalize_ne(3,'ml/hr',80))
    assert named_final_concentration('NOREPINEPHRINE 8 MG in 250mL NS')==32
    assert np.isnan(named_final_concentration('NOREPINEPHRINE 4 mg/4 mL Inj'))
    assert np.isnan(named_final_concentration('NOREPINEPHRINE BITARTRATE 1 MG/ML IV : 4 ML'))
    assert route_class('INTRATHECA')=='other_nonIV_route'
    assert route_class('IV Push')=='IV_prescribed'
    assert normalize_ne(0,'mcg/kg/min',80)==0
    assert ev.loc[ev.recent_positive_NE_60,'last_NE_age_min'].gt(0).all()
    assert ev.loc[ev.recent_positive_NE_60,'last_NE_age_min'].le(60).all()
    assert ev.hospital.is_unique and ev['index'].notna().all()
    for name,info in b['inputs'].items():
        f=locate(name);assert tuple(info['signature'])==(str(f),f.stat().st_size,f.stat().st_mtime_ns)
    # Independent vectorized source counts, not the candidate loop.
    assert int(ne.loc[ne.positive_record,'patienthealthsystemstayid'].nunique())==int(d['summary'].loc[d['summary'].metric.eq('上述独立住院'),'n'].iloc[0])
    assert (ne.dose_concentration_proxy.notna()<=ne.unit.eq('ml/hr')).all()
    med=d['medication']['records'];assert med.medicationid.is_unique
    code=d.get('code_resolution')
    if code is not None:
        extra=med[med.identity_source.eq('unambiguous_same_table_HICL')]
        assert lower(extra.drugname).eq('').all()
        evidence=code['evidence'].set_index('code')
        for label in ['is_HC','is_GC','is_NE']:
            assert extra[label].eq(extra.drughiclseqno.map(evidence[label])).all()
        assert code['sha256']==b['inputs']['medication']['sha256']
    d['validation']=dict(synthetic_checks=10,unique_hospital_candidate=True,preindex_NE_only=True,input_signatures_unchanged=True,
        independently_recomputed_NE_hospital_count=True,unique_medication_events=True,
        recovered_codes_match_all_named_source_evidence=code is not None,no_outcomes_compared=True,no_models_fitted=True)
    d['source_current']=Path(__file__).read_bytes();save(b);say(dict(stage='eicu_drug_validate',passed=True))


def drug_publish():
    b=load();d=b['drug'];assert 'validation' in d;m=base();s=d['summary'];ev=d['technical_candidates']
    path=ROOT/'steroid_tte.md';before=path.read_bytes();old=before.decode('utf-8');newline='\r\n' if '\r\n' in old else '\n';old=old.replace('\r\n','\n')
    def fmt(n,den=None):
        out=m.count(n)
        if den is not None:
            p=m.pct(n,den);v='<0.1' if p==0 and n>0 else '>99.9' if p==100 and n<den else str(p)
            out+=' / '+m.count(den)+('（比例抑制）' if p=='' else '（'+v+'%）')
        return out
    def metric(k):
        r=s[s.metric.eq(k)].iloc[0];return fmt(r.n,r.d if pd.notna(r.d) else None)
    lines=['<!-- HC_EICU_DRUG_START -->','### eICU两张药物表到位后的实际验证','',
        '本节更新前文等待medication和infusionDrug的状态，两表已完整读取。当前结果支持NE与HC医嘱的技术关联；实际HC首剂与MIMIC同协议的外部验证尚未建立。所有计数为独立源表核查，不与MIMIC患者键拼接，也没有估计新死亡效果。','']
    if 'code_resolution' in d:
        code=d['code_resolution'];stats=d['medication']['stats'];prior=d['name_only_stage']['technical_candidates']
        identity=ev.medicationid.map(d['medication_resolved'].set_index('medicationid').identity_source)
        lines+=['|药物身份补查：同一技术筛查规则|结果|','|---|---:|',
            '|整张medication表药名为空|'+fmt(stats['missing_name_rows'],stats['rows'])+'|',
            '|经同表具名记录一致性核查补回的HC编码记录行|'+fmt(code['added_HC_rows'])+'|',
            '|仅按药名识别时的技术候选住院|'+fmt(len(prior))+'|',
            '|补入可解析编码后的技术候选住院|'+fmt(len(ev))+'|',
            '|当前候选首次HC医嘱依靠编码识别|'+fmt(identity.eq('unambiguous_same_table_HICL').sum(),len(ev))+'|','',
            '药名为空不等于没有药物。仅当同一HICL编码在整张源表中的所有非空药名均属于相同药物类别时，才补回其空药名记录；存在跨药物冲突的编码未用于补回。未替空药名编造药名、剂量配方或给药途径。该证据是源表内编码一致性，尚不是独立药物词典或实际给药核验；药名识别与编码识别来源分别保留。初轮仅药名结果存于同一私有缓存供敏感性比较，当前汇总采用补查后的投影。','']
    lines+=['|来源与NE轨迹|实际结果|','|---|---:|']
    for k in ['medication：成人ICU有药名记录','infusiondrug：成人ICU有药名记录','成人ICU内有正速率NE记录的ICU记录','上述独立住院','上述独立患者','上述医院',
              '正速率行剂量可按明确单位归一','仅容量速率或单位未知的正速率行','同ICU同刻零与正速率并存']:
        lines.append('|'+k+'|'+metric(k)+'|')
    lines+=['|NE正速率记录间隔中位数 / P90|'+str(int(s.loc[s.metric.eq('NE正速率记录间隔P50分钟'),'value'].iloc[0]))+' / '+str(int(s.loc[s.metric.eq('NE正速率记录间隔P90分钟'),'value'].iloc[0]))+'分钟|','',
        'infusionOffset是输注记录录入偏移；连续记录的中位间隔为60分钟，并非分钟级连续监测。零与正速率同刻并存、体重缺失或单位不明分别保留，缺记录不推断停药。mcg/min换算使用同条体重，缺失时使用入ICU体重并标记；盐基不再换算，解释为源记录的NE剂量。','',
        '|HC记录与技术候选|实际结果|','|---|---:|']
    for k in ['成人HC医嘱记录','有HC医嘱的独立住院','医院内IV HC医嘱涉及住院','医院内IV HC医嘱涉及医院','HC输注记录涉及住院',
        '首次IV HC医嘱在首次观察正速率NE之后0–24h且开于ICU','上述独立患者','医嘱前30分钟内最近NE记录为正','医嘱前60分钟内最近NE记录为正','医嘱前120分钟内最近NE记录为正']:
        if k=='上述独立患者':
            r=s[(s.section=='candidate')&(s.metric==k)].iloc[0];lines.append('|技术候选独立患者|'+fmt(r.n)+'|')
        else:lines.append('|'+k+'|'+metric(k)+'|')
    c=d['concentration_review']['summary']
    lines+=['|技术候选涉及医院|'+fmt(ev.site.nunique())+'|',
        '|技术候选：最近NE记录的剂量可直接归一|'+fmt(c['candidates_direct_dose'],len(ev))+'|',
        '|加上唯一输注配方匹配后的剂量可解释上界|'+fmt(c['candidates_with_proxy'],len(ev))+'|',
        '|最近NE在60分钟内且直接或配方剂量可解析|'+fmt((ev.recent_positive_NE_60&ev.NE_dose_with_concentration_proxy).sum(),len(ev))+'|',
        '|技术候选前5医院占比|'+f"{100*s.loc[s.metric.eq('技术候选前5医院占比'),'value'].iloc[0]:.1f}%"+'|','',
        '浓度补充仅用同ICU、已有医嘱和计划输注时段内、明确写出的最终输注袋浓度。多个同时兼容医嘱浓度一致才能重建；未知浓度、取消时间不明、泵速冲突均不转换。4mg/4mL等浓缩注射剂不当作输注袋，未采用默认浓度。重建值单列为处方匹配代理，缺乏输注—医嘱原生事件键，不等于已独立验证真实泵速剂量。此步骤额外补回约'+m.count(c['additional_proxy_rows'])+'条记录的剂量代理，未增加患者或改变临床阈值。','',
        '技术候选按本次住院首次IV HC医嘱筛查，包含最终取消、可能补录及计划开始早于医嘱者；没有按之后是否执行、持续使用或存活筛人。30/60/120分钟仅描述最近NE记录年龄，不是根据结果选择的入组阈值。过去24小时窗口沿用现问题的时间范围，没有延长来增加人数。','',
        '|技术候选中的资格资料|结果|','|---|---:|']
    labels={'known_prior_GC_order':'医嘱前有其他GC医嘱记录（并非实际给药）','known_prior_GC_history':'医嘱前已录入较早GC药史',
        'culture_before':'医嘱前有培养采样记录','infection_assessment_before':'医嘱前有感染照护评估',
        'cancellation_final':'最终取消标志为Yes（取消时刻未知）','HC_start_before_order':'计划HC开始早于医嘱录入',
        '候选所在ICU有入院药史记录':'所在ICU有入院药史记录','候选住院院内结局可判定':'本次住院院内结局可判定'}
    for k,label in labels.items():lines.append('|'+label+'|'+metric(k)+'|')
    lines+=['',
        '在医院内时间可判定的医嘱及既有药史来源中，未检出更早GC记录且60分钟内有正速率NE记录者约'+m.count((~ev.known_prior_GC_order&~ev.known_prior_GC_history&ev.recent_positive_NE_60).sum())+'次住院。这是限定捕获下的技术子集；没有把“未检出”当成已证明激素初用。培养与感染评估捕获不足以直接确认原MIMIC感染资格，未把最终诊断倒填为基线。诊断及临床病情表尚未提取，和“表内无记录”分开。','',
        '**决定：**eICU具有研究HC医嘱启动的候选规模，NE剂量信息也有部分可修复空间；目前不能将它写成实际HC首剂的同协议外部验证。HC主要通过medication观察，而该表是医嘱；HC输注证据涉及的住院为小单元，不能用全部HC医嘱人数替代。无需为证明这一点继续下载完整数据库。','',
        '若继续外部研究，先在MIMIC现有完整订单链明确一个两边都能表达的医嘱启动补充问题，再决定是否定向补eICU的admissionDx/diagnosis、感染记录和治疗前病情；这些属于不同测量定义的外部分析，原实际给药主结果单独保留。是否进一步获取lab/生命体征大表，应以药物定义和基线感染资格有可行映射为前提。','',
        '本轮没有重建治疗风险集、拟合新权重、计算治疗组死亡差异或运行bootstrap。取消状态、计划开始时间与院外激素均不能被静默转成已确认的治疗事实。','',
        '|新增源表|读取体积|读取及初步解析|峰值进程工作集|','|---|---:|---:|---:|']
    for r in b['runs']:
        if r['table'] in ['medication','infusiondrug']:lines.append('|'+r['table']+'|'+f"{r['bytes_read']/1e6:.2f}MB|{r['seconds']:.2f}秒|{r['peak_MB']:.1f}MB|" )
    if 'code_resolution' in d:
        r=d['code_resolution']
        lines.append('|medication：空药名编码定向复查|'+f"{r['bytes_read']/1e6:.2f}MB|{r['seconds']:.2f}秒|未另测|" )
    lines+=['',
        '两表解析到EOF，主键重复、外键及时间字段质量保存在私有缓存；输入SHA256与文件签名随读取保存。编码复查读取的文件SHA256与初轮一致。10项剂量、浓度、途径合成校验、独立住院计数、药物事件去重及编码补回依据核对通过。所有病例级材料仅在授权本地缓存中，公开计数按10取整，1–10及可反推比例抑制。','',
        '入口：`steroid_eicu.py drug-project` → `drug-code-project` → `drug-review` → `concentration-review` → `drug-validate` → `drug-publish`。复用`hc_eicu_feasibility.pkl.gz`，不额外输出数据平台或审计目录。','',
        '官方语义：[medication](https://eicu.mit.edu/eicutables/medication/)、[infusionDrug](https://eicu.mit.edu/eicutables/infusiondrug/)。','<!-- HC_EICU_DRUG_END -->','']
    block='\n'.join(lines)
    if '<!-- HC_EICU_DRUG_START -->' in old:
        start=old.index('<!-- HC_EICU_DRUG_START -->');end=old.index('<!-- HC_EICU_DRUG_END -->',start)+len('<!-- HC_EICU_DRUG_END -->')
        while end<len(old) and old[end]=='\n':end+=1
    else:
        start=end=old.index('### 实际代码与证据位置')
    after=(old[:start]+block+'\n'+old[end:]).replace('\n',newline).encode('utf-8');assert path.read_bytes()==before
    if after!=before:
        hist=d.setdefault('report_history',[])
        if not any(r['sha256']==sha(before) for r in hist):hist.append(dict(sha256=sha(before),bytes=before))
        d['publication']=dict(report_sha256=sha(after),code_sha256=sha(Path(__file__).read_bytes()));save(b);path.write_bytes(after)
    say(dict(stage='eicu_drug_publish',report=str(path),effects=0))


def base():
    s=importlib.util.spec_from_file_location('eicu_local_helpers',ROOT/'steroid_tte.py')
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def say(v):print(json.dumps(v,ensure_ascii=False,default=str),flush=True)
def sha(v):return hashlib.sha256(v).hexdigest()


def locate(name):
    matches=[p for p in RAW.rglob('*') if p.is_file() and p.name.lower() in [name.lower()+'.csv',name.lower()+'.csv.gz']]
    if len(matches)>1:raise ValueError('MULTIPLE_COPIES_REQUIRE_SOURCE_RESOLUTION: '+name)
    return matches[0] if matches else None


def save(b):
    tmp=STATE.with_name(STATE.name+'.writing')
    pd.to_pickle(b,tmp,compression={'method':'gzip','compresslevel':1});os.replace(str(tmp),str(STATE))


def load():
    if STATE.exists():
        b=pd.read_pickle(STATE);assert b['config']==CONFIG;return b
    b=dict(config=CONFIG,created_utc=base().utc(),inputs={},tables={},runs=[],summary=None,
           effects_estimated=0,models_fitted=0,source_initial=Path(__file__).read_bytes())
    save(b);return b


def read_chunks(b,name,consume,filter_description='source coverage; no NE/HC cohort selected'):
    path=locate(name)
    if path is None:return False
    st=path.stat();signature=(str(path),st.st_size,st.st_mtime_ns)
    if name in b['inputs']:
        assert signature==tuple(b['inputs'][name]['signature']),'INPUT_CHANGED: '+name
        return False
    assert set(COLS[name])<=set(pd.read_csv(path,nrows=0).columns)
    m=base();start=time.perf_counter();rows=0;last=start
    say(dict(stage='eicu_read',table=name,bytes=st.st_size,columns=COLS[name],filter=filter_description,chunk_rows=100000))
    reader=m.HashReader(path);buffer=io.BufferedReader(reader,buffer_size=4*1024*1024)
    stream=gzip.GzipFile(fileobj=buffer) if path.name.endswith('.gz') else buffer
    try:
        for z in pd.read_csv(stream,usecols=COLS[name],dtype=str,chunksize=100000):
            rows+=len(z);consume(z)
            if time.perf_counter()-last>=30:
                say(dict(stage='eicu_read_progress',table=name,seconds=round(time.perf_counter()-start),rows=m.count(rows)));last=time.perf_counter()
        digest=reader.h.hexdigest();assert reader.n==st.st_size
    finally:
        stream.close();buffer.close()
    assert (st.st_size,st.st_mtime_ns)==(path.stat().st_size,path.stat().st_mtime_ns)
    b['inputs'][name]=dict(signature=signature,sha256=digest,parsed_to_EOF=True,rows=rows,columns=COLS[name])
    b['runs'].append(dict(table=name,seconds=round(time.perf_counter()-start,2),bytes_read=st.st_size,peak_MB=m.peak_mb()))
    return True


def lower(s):return s.fillna('').str.strip().str.lower()


def prepare_patient(b):
    for name in ['patient','hospital']:
        parts=[]
        changed=read_chunks(b,name,lambda z:parts.append(z))
        if changed:
            z=pd.concat(parts,ignore_index=True)
            key='patientunitstayid' if name=='patient' else 'hospitalid'
            z[key]=pd.to_numeric(z[key],errors='raise').astype('int64')
            assert z[key].notna().all() and not z[key].duplicated().any()
            b['tables'][name]=z;save(b)
    p=b['tables']['patient'].copy()
    for key in ['patienthealthsystemstayid','hospitalid','wardid']:
        p[key]=pd.to_numeric(p[key],errors='raise').astype('int64')
    for key in ['hospitaladmitoffset','hospitaldischargeoffset','unitdischargeoffset','unitvisitnumber','admissionweight']:
        p[key]=pd.to_numeric(p[key],errors='coerce')
    text=lower(p.age);p['age_numeric']=pd.to_numeric(text,errors='coerce')
    p['age_over89']=text.str.fullmatch(r'>\s*89',na=False)
    p['adult']=p.age_numeric.ge(18)|p.age_over89
    p['age_unknown']=p.age_numeric.isna()&~p.age_over89
    p['hospital_duration']=p.hospitaldischargeoffset-p.hospitaladmitoffset
    p['unit_start_from_hospital']=-p.hospitaladmitoffset
    p['unit_end_from_hospital']=p.unitdischargeoffset-p.hospitaladmitoffset
    p['status']=lower(p.hospitaldischargestatus)
    return p


def admission_history(b,p):
    if 'admissiondrug' in b['inputs']:return
    ids=set(p.patientunitstayid);adult=set(p.loc[p.adult,'patientunitstayid'])
    coverage=pd.DataFrame(index=p.patientunitstayid,columns=['row_first','named_first_entered','named_first_both'],dtype=float)
    pieces=[];idparts=[];stats=dict(rows=0,unlinked_rows=0,missing_name_rows=0,missing_time_rows=0,entered_before_drug_rows=0,duplicate_primary_keys=0)
    def consume(z):
        stats['rows']+=len(z);key=pd.to_numeric(z.patientunitstayid,errors='coerce');idparts.append(pd.to_numeric(z.admissiondrugid,errors='raise').to_numpy('int64'))
        stats['unlinked_rows']+=int((~key.isin(ids)).sum());z=z[key.isin(ids)].copy();z['patientunitstayid']=key.loc[z.index].astype('int64')
        name=lower(z.drugname);named=~name.isin(['','unknown','none','nan','no known medications'])
        z['drugoffset']=pd.to_numeric(z.drugoffset,errors='coerce');z['drugenteredoffset']=pd.to_numeric(z.drugenteredoffset,errors='coerce')
        stats['missing_name_rows']+=int((~named).sum());stats['missing_time_rows']+=int(z[['drugoffset','drugenteredoffset']].isna().any(1).sum())
        stats['entered_before_drug_rows']+=int((z.drugenteredoffset<z.drugoffset).sum())
        z['both']=z[['drugoffset','drugenteredoffset']].max(1,skipna=False)
        for column,field,mask in [('row_first','drugenteredoffset',np.ones(len(z),bool)),('named_first_entered','drugenteredoffset',named),('named_first_both','both',named)]:
            v=z.loc[mask].groupby('patientunitstayid')[field].min()
            coverage.loc[v.index,column]=np.fmin(coverage.loc[v.index,column].to_numpy(float),v.to_numpy(float))
        steroid=z.drugname.fillna('').str.contains(GC)
        q=z[steroid].copy()
        if len(q):
            q['HC_name']=q.drugname.fillna('').str.contains(HC)
            q['explicit_topical_or_local_name']=q.drugname.fillna('').str.contains(TOPICAL)
            q['record_class']=np.where(q.explicit_topical_or_local_name,'explicit_local_name','route_not_established')
            pieces.append(q)
    if read_chunks(b,'admissiondrug',consume):
        keys=np.concatenate(idparts);stats['duplicate_primary_keys']=int(len(keys)-len(np.unique(keys)))
        # Repeated history records are not interpreted as repeated doses.
        b['tables']['admissiondrug']=dict(coverage=coverage,steroid_records=pd.concat(pieces,ignore_index=True).drop_duplicates() if pieces else pd.DataFrame(),stats=stats)
        save(b)


def small_sources(b,p):
    for name,idcol,timecol in [('microlab','microlabid','culturetakenoffset'),('careplaninfectiousdisease','cplinfectid','cplinfectdiseaseoffset')]:
        parts=[]
        if read_chunks(b,name,lambda z:parts.append(z)):
            z=pd.concat(parts,ignore_index=True)
            z['patientunitstayid']=pd.to_numeric(z.patientunitstayid,errors='coerce')
            z[timecol]=pd.to_numeric(z[timecol],errors='coerce')
            stats=dict(rows=len(z),unlinked_rows=int((~z.patientunitstayid.isin(p.patientunitstayid)).sum()),
                duplicate_primary_keys=int(z[idcol].duplicated().sum()),missing_time_rows=int(z[timecol].isna().sum()))
            z=z[z.patientunitstayid.isin(p.patientunitstayid)].copy();z['patientunitstayid']=z.patientunitstayid.astype('int64')
            if name=='microlab':
                z=z.drop_duplicates(['patientunitstayid','culturetakenoffset','culturesite'])
                stats['unit_time_site_groups']=len(z)
            else:
                z['supported_infection_assessment']=lower(z.infectdiseaseassessment).isin(['definite infection','probable infection','possible infection'])
            b['tables'][name]=dict(records=z,stats=stats);save(b)


def aggregate(b,p):
    adult=p[p.adult].copy();ref=p.set_index('patientunitstayid');ad=adult.set_index('patientunitstayid')
    rows=[]
    def add(section,metric,n,d=None,value=None,note=''):
        rows.append(dict(section=section,metric=metric,n=int(n) if n is not None else None,d=int(d) if d is not None else None,value=value,note=note))
    add('population','全部ICU记录',len(p));add('population','全部住院',p.patienthealthsystemstayid.nunique());add('population','全部患者',p.uniquepid.nunique())
    add('population','成人ICU记录',len(adult),len(p));add('population','成人相关住院',adult.patienthealthsystemstayid.nunique());add('population','成人患者',adult.uniquepid.nunique())
    add('population','年龄无法判断的ICU记录',p.age_unknown.sum(),len(p));add('population','成人ICU所在医院',adult.hospitalid.nunique());add('population','成人ICU所在病区',adult.wardid.nunique())
    add('population','患者医院键在hospital表缺失',(~p.hospitalid.isin(pd.to_numeric(b['tables']['hospital'].hospitalid))).sum(),len(p))
    hosp=[]
    for h,g in p.groupby('patienthealthsystemstayid',sort=False):
        states=set(g.loc[g.status.isin(['alive','expired']),'status'])
        outcome=next(iter(states)) if len(states)==1 else 'CONFLICT' if len(states)>1 else 'UNKNOWN'
        vals=g.hospital_duration.dropna()
        hosp.append(dict(hospital_stay=h,adult=bool(g.adult.any()),unit_rows=len(g),
            patient_conflict=g.uniquepid.nunique(dropna=False)!=1,hospital_conflict=g.hospitalid.nunique()!=1,
            outcome=outcome,any_status_missing=bool((~g.status.isin(['alive','expired'])).any()),
            duration_range=float(vals.max()-vals.min()) if len(vals) else np.nan,
            duration_missing=bool(g.hospital_duration.isna().any()),duration_nonpositive=bool(g.hospital_duration.le(0).any())))
    h=pd.DataFrame(hosp);ha=h[h.adult]
    add('outcome','成人住院有一致可判定院内结局',ha.outcome.isin(['alive','expired']).sum(),len(ha))
    for value in ['alive','expired','UNKNOWN','CONFLICT']:add('outcome','成人住院结局：'+value,ha.outcome.eq(value).sum(),len(ha))
    add('outcome','成人住院至少一条ICU记录缺少医院状态',ha.any_status_missing.sum(),len(ha))
    add('outcome','成人住院含多个ICU记录',ha.unit_rows.gt(1).sum(),len(ha))
    add('outcome','成人住院跨ICU状态完整且一致',((~ha.any_status_missing)&ha.outcome.isin(['alive','expired'])).sum(),len(ha))
    add('time','住院内患者键冲突',ha.patient_conflict.sum(),len(ha));add('time','住院内医院键冲突',ha.hospital_conflict.sum(),len(ha))
    add('time','跨ICU推算住院总时长差异超过1分钟',ha.duration_range.gt(1).sum(),len(ha))
    add('time','住院时长缺失或非正数', (ha.duration_missing|ha.duration_nonpositive).sum(),len(ha))
    add('time','成人ICU退出晚于医院出院超过1分钟',adult.unitdischargeoffset.gt(adult.hospitaldischargeoffset+1).sum(),len(adult))
    add('time','成人记录医院入院偏移为正数',adult.hospitaladmitoffset.gt(0).sum(),len(adult))
    add('time','成人ICU退出偏移为负数',adult.unitdischargeoffset.lt(0).sum(),len(adult))
    add('time','成人记录住院状态未知',(~adult.status.isin(['alive','expired'])).sum(),len(adult))
    for year,g in adult.groupby('hospitaldischargeyear'):add('population','医院出院年份 '+str(year),len(g),len(adult))
    add('population','成人记录入ICU体重为可解析正数',adult.admissionweight.gt(0).sum(),len(adult))
    # Source denominators remain adult ICU stays; no future qualifications.
    cov=pd.DataFrame(index=ad.index)
    history=b['tables']['admissiondrug']['coverage'];gcs=b['tables']['admissiondrug']['steroid_records']
    cov['admission_history_any_row']=history.row_first.notna().reindex(ad.index,fill_value=False)
    cov['admission_history_named']=history.named_first_entered.notna().reindex(ad.index,fill_value=False)
    micro=b['tables']['microlab']['records'];care=b['tables']['careplaninfectiousdisease']['records']
    cov['micro_any']=cov.index.isin(micro.patientunitstayid)
    cov['careplan_infection_any']=cov.index.isin(care.patientunitstayid)
    cov['careplan_supported_assessment']=cov.index.isin(care.loc[care.supported_infection_assessment,'patientunitstayid'])
    cov['micro_or_careplan']=cov.micro_any|cov.careplan_infection_any
    for col in cov:add('coverage',col,cov[col].sum(),len(cov))
    for minute in CONFIG['fixed_coverage_minutes']:
        add('clock','有药名的药史已录入 <'+str(minute)+'min',history.named_first_entered.reindex(ad.index).lt(minute).sum(),len(ad))
        add('clock','药史两个时间均早于 '+str(minute)+'min',history.named_first_both.reindex(ad.index).lt(minute).sum(),len(ad))
        add('clock','培养采样记录 <'+str(minute)+'min',int(cov.index.isin(micro.loc[micro.culturetakenoffset.lt(minute),'patientunitstayid']).sum()),len(ad),note='collection only; result availability unknown')
        c=care[care.cplinfectdiseaseoffset.lt(minute)&care.supported_infection_assessment]
        add('clock','感染评估已录入 <'+str(minute)+'min',int(cov.index.isin(c.patientunitstayid).sum()),len(ad))
        if len(gcs):
            q=gcs[gcs.patientunitstayid.isin(ad.index)&gcs.drugoffset.lt(minute)&gcs.drugenteredoffset.lt(minute)]
            add('history','GC药名且双时钟早于 '+str(minute)+'min',q.patientunitstayid.nunique(),len(ad))
            add('history','HC药名且双时钟早于 '+str(minute)+'min',q.loc[q.HC_name,'patientunitstayid'].nunique(),len(ad))
    gc=gcs[gcs.patientunitstayid.isin(ad.index)] if len(gcs) else gcs
    add('history','任意时间存在GC药名药史',gc.patientunitstayid.nunique(),len(ad))
    add('history','任意时间存在HC药名药史',gc.loc[gc.HC_name,'patientunitstayid'].nunique(),len(ad))
    for flag,label in [('explicit_topical_or_local_name','药名明确为局部制剂的GC记录')]:add('history',label,int(gc[flag].sum()),len(gc))
    d=pd.to_numeric(gc.drugdosage,errors='coerce');add('history','GC药史剂量可解析且正数',d.gt(0).sum(),len(gc))
    for k,v in b['tables']['admissiondrug']['stats'].items():add('admissiondrug_quality',k,v)
    for name in ['microlab','careplaninfectiousdisease']:
        for k,v in b['tables'][name]['stats'].items():add(name+'_quality',k,v)
    site=cov.assign(hospitalid=ad.hospitalid).groupby('hospitalid').agg(['sum','size'])
    site_rows=[]
    for col in cov:
        n=site[(col,'sum')];den=site[(col,'size')];rate=n/den
        add('site',col+'：至少一条记录的医院',n.gt(0).sum(),len(n))
        add('site',col+'：覆盖达到一半成人ICU记录的医院',rate.ge(.5).sum(),len(n))
        for quantile in [.25,.5,.75,.9]:add('site',col+'：医院覆盖率P'+str(int(100*quantile)),None,None,float(rate.quantile(quantile)))
        positive=n[n.gt(0)]
        add('site',col+'：覆盖记录前5医院占比',None,None,float(positive.nlargest(5).sum()/positive.sum()) if len(positive) else np.nan)
        for site_id in n.index:site_rows.append(dict(hospitalid=site_id,source=col,n=int(n.loc[site_id]),denominator=int(den.loc[site_id]),rate=float(rate.loc[site_id])))
    for name in ['admissiondrug','microlab','careplaninfectiousdisease']:
        z=gcs if name=='admissiondrug' else b['tables'][name]['records']
        col='drugoffset' if name=='admissiondrug' else 'culturetakenoffset' if name=='microlab' else 'cplinfectdiseaseoffset'
        end=z.patientunitstayid.map(ref.unitdischargeoffset);begin=z.patientunitstayid.map(ref.hospitaladmitoffset)
        add('source_time',name+'：记录事件早于本次医院入院',z[col].lt(begin).sum(),len(z))
        add('source_time',name+'：记录事件晚于本ICU出科',z[col].gt(end).sum(),len(z))
    b['patient']=p;b['hospital_summary']=h;b['unit_source_coverage']=cov;b['site_coverage']=pd.DataFrame(site_rows)
    b['summary']=pd.DataFrame(rows);b['checks']=dict(no_effects=True,no_new_model=True,keys_distinct_from_MIMIC=True,
        input_parse_to_EOF=all(v['parsed_to_EOF'] for v in b['inputs'].values()),no_future_outcome_eligibility=True)
    b['source_current']=Path(__file__).read_bytes();save(b)


def tests():
    # Different ICU offsets must yield the same hospital endpoint.
    assert 10960-(-5040)==15280-(-720)==16000
    assert 4320-(-5040)==9360 and 2160-(-720)==2880
    # A late-entered admission-history event is unavailable earlier despite an early drug offset.
    drug=np.array([-30.,-20.,10.]);entry=np.array([60.,-10.,15.])
    assert (np.maximum(drug,entry)<0).tolist()==[False,True,False]
    assert (np.maximum(drug,entry)<60).tolist()==[False,True,True]
    assert bool(HC.search('Hydrocortisone')) and not bool(GC.search('fludrocortisone'))
    return 5


def foundation():
    assert tests()==5;b=load()
    if b['summary'] is not None:
        for name,info in b['inputs'].items():
            f=locate(name);assert f is not None,'INPUT_NO_LONGER_PRESENT: '+name
            assert tuple(info['signature'])==(str(f),f.stat().st_size,f.stat().st_mtime_ns),'INPUT_CHANGED: '+name
        available={n:str(locate(n)) if locate(n) else None for n in EXPECTED}
        if b.get('available_now')!=available:b['available_now']=available;save(b)
        say(dict(stage='eicu_foundation',reuse=True,missing=[n for n,v in available.items() if not v],
                 new_drug_tables=[n for n in ['medication','infusiondrug'] if available[n]],effects=0))
        return
    p=prepare_patient(b);admission_history(b,p);small_sources(b,p);aggregate(b,p)
    b['available_now']={n:str(locate(n)) if locate(n) else None for n in EXPECTED};save(b)
    m=base();s=b['summary']
    for r in s[s.section.isin(['population','outcome','coverage'])].itertuples():
        say(dict(section=r.section,metric=r.metric,n=m.count(r.n),denominator=m.count(r.d) if pd.notna(r.d) else '',percent=m.pct(r.n,r.d) if pd.notna(r.d) else ''))
    say(dict(stage='eicu_foundation_complete',missing=[n for n,v in b['available_now'].items() if not v],
        scan_seconds=round(sum(r['seconds'] for r in b['runs']),2),effects=0,models=0))


def publish():
    b=load();m=base();s=b['summary'];assert s is not None
    path=ROOT/'steroid_tte.md';old_bytes=path.read_bytes();old=old_bytes.decode('utf-8');newline='\r\n' if '\r\n' in old else '\n';old=old.replace('\r\n','\n')
    def value(r):
        if pd.notna(r.n):
            answer=m.count(r.n)
            if pd.notna(r.d):
                p=m.pct(r.n,r.d)
                formatted='<0.1' if p==0 and r.n>0 else '>99.9' if p==100 and r.n<r.d else str(p)
                answer+=' / '+m.count(r.d)+('（比例抑制）' if p=='' else '（'+formatted+'%）')
            return answer
        return f'{100*r.value:.1f}%' if pd.notna(r.value) else 'UNKNOWN'
    lines=['<!-- HC_EICU_SOURCE_START -->','### eICU现有源表实测与外部研究准备','',
        '初轮已完成patient、hospital、admissionDrug、microLab和carePlanInfectiousDisease的完整解析及关联检查。本节分母为全库成人ICU记录，用于评估来源；不是NE治疗人群或合格HC首剂队列。两张药物表到位后的实测另见后续药物核查小节。','',
        '|输入表|实际读取体积|读取及初步解析耗时|本轮用途|','|---|---:|---:|---|']
    uses={'patient':'患者/住院/ICU关联、相对时间与住院结局','hospital':'医院元数据关联','admissiondrug':'既往用药记录与录入时间，不作ICU新给药','microlab':'采样记录及医院捕获，不使用菌种或药敏做基线','careplaninfectiousdisease':'感染评估录入时间；排除出院活跃状态和治疗后反应字段'}
    for r in b['runs']:
        if r['table'] in uses:lines.append('|'+r['table']+'|'+f"{r['bytes_read']/1e6:.2f} MB|{r['seconds']:.2f}秒|"+uses[r['table']]+'|')
    lines+=['','所有输入均解析到EOF，读取前后大小/修改时间一致；SHA256、字段、绝对路径和运行成本保存在私有缓存中。这证明本次读取一致，不等于已与官方发布校验文件核对。目录采用“表名.csv/表名.csv”的解压结构，入口已兼容，不需要移动原文件。','']
    sections=[('population','数据规模与基础关联'),('outcome','整次住院随访'),('time','时间与重复ICU记录'),('coverage','来源覆盖：分母为全部成人ICU记录'),('clock','固定ICU时间点的记录覆盖'),('history','入ICU药史中的激素记录'),('site','医院之间的捕获差异')]
    for section,title in sections:
        lines+=['#### '+title,'','|检查|结果|','|---|---|']
        for r in s[s.section.eq(section)].itertuples():lines.append('|'+r.metric+'|'+value(r)+'|')
        lines.append('')
    lines+=['#### 事件表读取质量','','|来源|主键重复|未关联到patient的行|事件时间缺失|','|---|---:|---:|---:|']
    for name in ['admissiondrug','microlab','careplaninfectiousdisease']:
        stats=b['tables'][name]['stats']
        lines.append('|'+name+'|'+m.count(stats['duplicate_primary_keys'])+'|'+m.count(stats['unlinked_rows'])+'|'+m.count(stats['missing_time_rows'])+'|')
    lines.append('')
    lines+=['计数依项目规则按10取整，小单元及可反推小单元的比例抑制。百分比使用内部准确分母；同一患者可有多次住院，每次住院可有多个ICU记录，不将这些计数互换。0/6/24小时均相对各ICU入科，仅用于记录覆盖，未作为新资格或HC起点。','',
        '#### 目前能够支持什么','',
        '患者、医院和完整住院结局的连接已建立；后续NE/HC表可在同版键上直接关联。住院结局按patientHealthSystemStayID汇总一致的Alive/Expired，冲突与未知单列。跨ICU按“事件偏移−hospitalAdmitOffset”换算至医院入院时间，不能按ID大小排序。存活出院仅表示本次医院未记录死亡；转院后死亡与固定日死亡不由此得到。','',
        'admissionDrug提供记录的入院药史。药名、剂量和药史时间可以提示既往激素，但缺失不证明未用药，也不能用它代替新的ICU HC执行记录。药名明确局部制剂与途径未确认者分开；没有凭剂量单位推断静脉给药。','',
        'microLab只提供采样时间与微生物记录。本轮未读取organism、antibiotic或sensitivityLevel：其中antibiotic是药敏相关字段，不能当作实际抗菌治疗。相同ICU、采样时间和部位的行合并为记录组合，不能据此声称已识别独立标本。采样早于索引不证明结果当时已返回。carePlanInfectiousDisease的offset是感染评估录入时间，其出院活跃状态和后续治疗反应均未使用。','',
        '#### 初轮确定的药物核查任务','',
        '1. infusionDrug：核查NE名称、单位、体重、正速率及连续记录；报告首次观察NE是否可能左截断，并查HC是否有可用执行证据。缺少一个时间段的记录不自动填成停药。','2. medication：核查HC药名/途径、订单和开始/停止偏移、取消状态，并与NE轨迹及infusionDrug交叉检查。此表官方定位为药物医嘱，不能把drugStartOffset直接命名为实际首剂给药。','3. 仅在上述映射可解释后建立技术候选，按医院报告捕获、感染资格和已知既往激素，再判断同协议复现还是需明确修改测量定义的外部分析。当前不追加生命体征大表，也不将本次全库人数当作可用TTE样本量。','',
        '初轮状态：基础数据准备完成时，两张关键药物表尚未取得，HC/NE暴露与外部可比性为UNKNOWN。后续已到位表的实测结果见药物核查小节；不将初轮未知解释为整个eICU不能开展研究。MIMIC主结果与分型材料保持不变。','',
        '官方定义：[patient](https://eicu.mit.edu/eicutables/patient/)、[admissionDrug](https://eicu.mit.edu/eicutables/admissiondrug/)、[microLab](https://eicu.mit.edu/eicutables/microlab/)、[carePlanInfectiousDisease](https://eicu.mit.edu/eicutables/careplaninfectiousdisease/)、[medication](https://eicu.mit.edu/eicutables/medication/)。','',
        '复现入口：`steroid_eicu.py foundation`与`steroid_eicu.py publish`。已读取且签名一致的输入复用缓存；若源表被替换则停止并提示来源变化。私有缓存为`hc_eicu_feasibility.pkl.gz`，包含必要的源投影、聚合表、哈希和本节前版，不额外生成审计报告。','',
        '<!-- HC_EICU_SOURCE_END -->','']
    block='\n'.join(lines)
    if '<!-- HC_EICU_SOURCE_START -->' in old:
        start=old.index('<!-- HC_EICU_SOURCE_START -->');end=old.index('<!-- HC_EICU_SOURCE_END -->',start)+len('<!-- HC_EICU_SOURCE_END -->')
        while end<len(old) and old[end]=='\n':end+=1
    else:start=end=old.index('### 实际代码与证据位置')
    out=(old[:start]+block+'\n'+old[end:]).replace('\n',newline).encode('utf-8')
    if out==old_bytes:say(dict(stage='eicu_publish',reuse=True));return
    assert path.read_bytes()==old_bytes
    history=b.setdefault('report_history',[])
    if not any(v['sha256']==sha(old_bytes) for v in history):history.append(dict(sha256=sha(old_bytes),bytes=old_bytes))
    b['publication']=dict(before_hash=sha(old_bytes),report_hash=sha(out),source_hash=sha(Path(__file__).read_bytes()))
    b['source_current']=Path(__file__).read_bytes();save(b);path.write_bytes(out)
    say(dict(stage='eicu_publish',report=str(path),effects=0))


DX_CONFIG=dict(
    purpose='pretreatment infection documentation and order-based supplementary eligibility, no outcome contrasts',
    source_population='all adult NE-positive hospital stays and their ICU rows, not restricted to later HC recipients',
    time='diagnosis entered offset minus source-unit hospitaladmitoffset; strictly before each order or comparison index',
    validity='finite timestamp within source hospital and no later than source ICU discharge; negative ICU offset separately flagged',
    primary_candidate='explicit acute infection or sepsis label before index, NE positive recorded within prior60min, no earlier recorded systemic GC order/history',
    sensitivity='explicit sepsis/septic shock labels separately; no future diagnosis used to qualify',
    diagnosis_caveat='recorded clinical diagnosis, not disease-onset time, confirmed active infection or MIMIC antibiotic-culture equivalence',
    controls='same hospital site and same elapsed time since first observed NE as the order case; current ICU and prior NE record; later HC allowed',
    source_missing='missing or late diagnosis not classified as no infection; no record is not steroid-naive',
    exposure='HC medication order; final cancellation and future start/stop not eligibility filters',
    prohibited=['activeUponDischarge','admitDxText','death contrasts','weight fitting','new bootstrap'],
    existing_results_seen=True)


def diagnosis_project():
    b=load();m=base();assert 'technical_candidates' in b['drug']
    if 'diagnosis_extension' not in b:
        b['diagnosis_extension']=dict(config=DX_CONFIG,frozen_utc=m.utc(),source_at_freeze=Path(__file__).read_bytes(),tables={})
        save(b)
    x=b['diagnosis_extension'];assert x['config']==DX_CONFIG
    ref=b['patient'].set_index('patientunitstayid');ne=b['drug']['NE_records']
    hospitals=set(ne.loc[ne.positive_record,'patienthealthsystemstayid'])
    units=set(ref.index[ref.patienthealthsystemstayid.isin(hospitals)&ref.adult])
    for name,col in [('admissiondx','admitdxenteredoffset'),('diagnosis','diagnosisoffset')]:
        if name in x['tables']:
            f=locate(name);assert tuple(b['inputs'][name]['signature'])==(str(f),f.stat().st_size,f.stat().st_mtime_ns)
            continue
        parts=[];primary=[];stats=dict(rows=0,unlinked_rows=0,missing_time_rows=0)
        def consume(z):
            stats['rows']+=len(z);key=pd.to_numeric(z.patientunitstayid,errors='coerce')
            primary.append(pd.to_numeric(z[name+'id'],errors='raise').to_numpy('int64'))
            stats['unlinked_rows']+=int((~key.isin(ref.index)).sum())
            stats['missing_time_rows']+=int(pd.to_numeric(z[col],errors='coerce').isna().sum())
            q=z[key.isin(units)].copy()
            if not len(q):return
            q['patientunitstayid']=key.loc[q.index].astype('int64');q['entry_offset']=pd.to_numeric(q[col],errors='coerce')
            q['hospital']=q.patientunitstayid.map(ref.patienthealthsystemstayid)
            q['subject']=q.patientunitstayid.map(ref.uniquepid);q['site']=q.patientunitstayid.map(ref.hospitalid)
            q['entry']=q.entry_offset-q.patientunitstayid.map(ref.hospitaladmitoffset)
            q['valid_time']=q.entry.notna()&q.entry.ge(0)&q.entry_offset.le(q.patientunitstayid.map(ref.unitdischargeoffset))&q.entry_offset.le(q.patientunitstayid.map(ref.hospitaldischargeoffset))
            q['pre_ICU_entry']=q.entry_offset.lt(0)
            if name=='admissiondx':q['label']=lower(q.admitdxpath)+' | '+lower(q.admitdxname)
            else:q['label']=lower(q.diagnosisstring)
            parts.append(q)
        changed=read_chunks(b,name,consume,filter_description='all adult NE-positive hospital stays; structured diagnosis labels and entry offsets only')
        if not changed:raise ValueError('DIAGNOSIS_SOURCE_MISSING_OR_PARTIAL_CACHE: '+name)
        keys=np.concatenate(primary);stats['duplicate_primary_keys']=int(len(keys)-len(np.unique(keys)))
        records=pd.concat(parts,ignore_index=True);stats['retained_rows']=len(records)
        stats['invalid_time_rows']=int((~records.valid_time).sum());stats['pre_ICU_entry_rows']=int(records.pre_ICU_entry.sum())
        vocab=records.groupby('label').agg(rows=('patientunitstayid','size'),units=('patientunitstayid','nunique'),hospitals=('hospital','nunique')).reset_index()
        x['tables'][name]=dict(records=records,stats=stats,vocabulary=vocab)
        x['source_current']=Path(__file__).read_bytes();save(b)
        say(dict(stage='diagnosis_projection_done',table=name,retained_rows=m.count(len(records)),vocabulary_terms=len(vocab),effects=0))
    b['available_now']={n:str(locate(n)) if locate(n) else None for n in EXPECTED};save(b)


def diagnosis_flags(label,source):
    s=str(label).lower();parts=[p.strip() for p in s.split('|')]
    # Ignore directory names such as 'urinary tract / renal system infection'.
    # They also contain noninfectious descendants, e.g. nephrolithiasis.
    t=' | '.join(parts[2:]) if source=='diagnosis' else parts[-1]
    nonspecific='signs and symptoms of sepsis' in t or bool(re.search(r'\bsirs\b',t))
    neg=bool(re.search(r'non[- ]infectious|noninfectious|ruled out|rule out|\br/o\b|history of|hx of|eosinophilic pneumonia|carcinomatous meningitis|aspiration pneumonitis|lupus|abscess/cyst|complications of previous',t))
    # Aspiration alone can represent chemical injury; retain separately.
    aspiration='aspirat' in t
    uncertain=neg or aspiration or bool(re.search(r'likely abscess|etiology unknown|etiology unclear|cerebral mass',t))
    sepsis=bool(re.search(r'\bsepsis\b|\bsepticemia\b|\bseptic shock\b',t)) and not nonspecific and not neg
    septic_shock='septic shock' in t and not neg
    acute=bool(re.search(r'pneumonia|pyelonephritis|urinary tract infection|renal infection|cellulitis|bacteremia|fungemia|peritonitis|cholangitis|abscess|endocarditis|meningitis|necrotizing fasciitis|empyema|mediastinitis|wound infection|sternal infection|vascular catheter infection|vascular graft infection|bacterial infection|fungal infection|septic thrombophlebitis|arthritis, septic|septic arthritis|c\. difficile|clostridium difficile',t))
    explicit_sirs=nonspecific and 'due to infectious process' in t and 'non-infectious' not in t
    infection=sepsis or ((acute or explicit_sirs) and not uncertain)
    nonseptic_shock=bool(re.search(r'cardiogenic shock|shock, cardiogenic|hypovolemic shock|hemorrhagic shock|anaphylactic shock|neurogenic shock|obstructive shock|spinal shock',t)) and not neg
    adrenal=bool(re.search(r'adrenal insufficiency|adrenal crisis|adrenalectomy',t)) and not neg
    airway=bool(re.search(r'acute copd exacerbation|asthma|bronchospasm',t)) and not neg
    limitation=bool(re.search(r'comfort care|comfort measures|withdrawal of life|withdrawal of care|do not resuscitate|do not intubate|\bdnr\b|\bdni\b',t))
    return dict(sepsis=sepsis,septic_shock=septic_shock,infection=infection,
        infection_ambiguous=bool((acute or nonspecific or 'infect' in t) and not infection),
        nonseptic_shock=nonseptic_shock,adrenal=adrenal,airway=airway,limitation=limitation)


def diagnosis_tests():
    cases=[('cardiovascular|shock / hypotension|septic shock','diagnosis','sepsis',True),
        ('cardiovascular|shock / hypotension|signs and symptoms of sepsis (sirs)|due to non-infectious process with organ dysfunction','diagnosis','infection',False),
        ('cardiovascular|shock / hypotension|signs and symptoms of sepsis (sirs)|due to infectious process with organ dysfunction','diagnosis','infection',True),
        ('cardiovascular|shock / hypotension|signs and symptoms of sepsis (sirs)|due to infectious process with organ dysfunction','diagnosis','sepsis',False),
        ('renal|disorder of urinary tract / renal system infection|nephrolithiasis','diagnosis','infection',False),
        ('pulmonary|pulmonary infections|pneumonia|aspiration pneumonitis','diagnosis','infection',False),
        ('pulmonary|pulmonary infections|pneumonia|community-acquired','diagnosis','infection',True),
        ('cardiovascular|vascular disorders|swollen extremity, etiology unknown|r/o cellulitis','diagnosis','infection',False),
        ('admission diagnosis|all diagnosis|non-operative|diagnosis|cardiovascular|sepsis, pulmonary | sepsis, pulmonary','admissiondx','sepsis',True),
        ('admission diagnosis|all diagnosis|non-operative|diagnosis|gastrointestinal|gi abscess/cyst | gi abscess/cyst','admissiondx','infection',False),
        ('endocrine|adrenal disorders|hypotension / pressor dependent|r/o adrenal insufficiency','diagnosis','adrenal',False)]
    for label,source,key,want in cases:assert diagnosis_flags(label,source)[key]==want
    assert not 60.<60. and 59.<60.
    assert 120.-(-60.)==180.
    return len(cases)+2


def diagnosis_review():
    b=load();x=b['diagnosis_extension'];assert x['config']==DX_CONFIG;m=base()
    if 'summary' in x:say(dict(stage='diagnosis_review',reuse=True));return
    tests_passed=diagnosis_tests()
    # Dictionary is fixed before inspecting case/control or mortality summaries.
    mapping=[]
    for source,d in x['tables'].items():
        for label in d['vocabulary'].label:
            mapping.append(dict(source=source,label=label,**diagnosis_flags(label,source)))
    dictionary=pd.DataFrame(mapping);x['mapping']=dictionary;x['mapping_source']=Path(__file__).read_bytes()
    x['mapping_notes']='Structured diagnosis categories only; generic hierarchy, noninfectious SIRS, rule-out and ambiguous aspiration excluded from primary infection mapping. ICD fields retained for traceability, not used to override text.'
    save(b)
    parts=[];kinds=['sepsis','septic_shock','infection','infection_ambiguous','nonseptic_shock','adrenal','airway','limitation']
    for source,d in x['tables'].items():
        q=d['records'].merge(dictionary[dictionary.source.eq(source)],on='label',how='left',validate='many_to_one')
        parts.append(q[['patientunitstayid','hospital','subject','site','entry','valid_time','pre_ICU_entry','source']+kinds])
    records=pd.concat(parts,ignore_index=True)
    valid=records[records.valid_time].copy();ev=b['drug']['technical_candidates'].copy()
    times={}
    for kind in kinds:
        times[kind]=valid.loc[valid[kind]].groupby('hospital').entry.min()
        t=ev.hospital.map(times[kind]);ev['dx_'+kind+'_before']=t.lt(ev['index']);ev['dx_'+kind+'_tie']=t.eq(ev['index'])
        ev['dx_'+kind+'_late_only']=t.gt(ev['index']);ev['dx_'+kind+'_unobserved']=t.isna()
        ev['dx_'+kind+'_first_time']=t
    any_time=valid.groupby('hospital').entry.min();ev['any_dx_before']=ev.hospital.map(any_time).lt(ev['index'])
    for source in x['tables']:
        u=valid[valid.source.eq(source)];ev[source+'_before']=ev.hospital.map(u.groupby('hospital').entry.min()).lt(ev['index'])
        ev[source+'_infection_before']=ev.hospital.map(u[u.infection].groupby('hospital').entry.min()).lt(ev['index'])
        ev[source+'_sepsis_before']=ev.hospital.map(u[u.sepsis].groupby('hospital').entry.min()).lt(ev['index'])
    ev['no_earlier_recorded_GC']=~ev.known_prior_GC_order&~ev.known_prior_GC_history
    ev['primary_mapping']=ev.dx_infection_before&ev.recent_positive_NE_60&ev.no_earlier_recorded_GC
    ev['strict_sepsis_mapping']=ev.dx_sepsis_before&ev.recent_positive_NE_60&ev.no_earlier_recorded_GC
    rows=[]
    def add(scope,metric,n,den=None,value=None):rows.append(dict(scope=scope,metric=metric,n=n,den=den,value=value))
    add('cohort','existing_technical_candidates',len(ev));add('cohort','any_diagnosis_before',ev.any_dx_before.sum(),len(ev))
    for col in [c for c in ev if c.startswith('dx_') and not c.endswith('_first_time')]+['admissiondx_before','diagnosis_before','admissiondx_infection_before','diagnosis_infection_before','admissiondx_sepsis_before','diagnosis_sepsis_before']:
        add('cohort',col,ev[col].sum(),len(ev))
    for population,mask in [('infection_documented',ev.dx_infection_before),('primary_mapping',ev.primary_mapping),('strict_sepsis_mapping',ev.strict_sepsis_mapping)]:
        f=ev[mask];add(population,'orders',len(f));add(population,'patients',f.subject.nunique());add(population,'sites',f.site.nunique())
        for col in ['NE_dose_available','NE_dose_with_concentration_proxy','recent_positive_NE_30','recent_positive_NE_60','recent_positive_NE_120','known_prior_GC_order','known_prior_GC_history','HC_start_before_order','cancellation_final','dx_nonseptic_shock_before','dx_adrenal_before','dx_airway_before','dx_limitation_before']:
            add(population,col,f[col].sum(),len(f))
        add(population,'top5_sites_share',None,value=float(f.site.value_counts().nlargest(5).sum()/len(f)) if len(f) else np.nan)
        lag=f['index']-f.dx_infection_first_time
        add(population,'first_infection_entry_age_median_minutes',None,value=float(lag.median()))
    hosp=b['hospital_summary'].set_index('hospital_stay')
    add('primary_mapping','hospital_outcome_observed',ev.loc[ev.primary_mapping,'hospital'].map(hosp.outcome).isin(['alive','expired']).sum(),ev.primary_mapping.sum())
    ev['source_identity']=ev.medicationid.map(b['drug']['medication_resolved'].set_index('medicationid').identity_source)
    for identity,g in ev[ev.primary_mapping].groupby('source_identity'):add('identity',identity,len(g),ev.primary_mapping.sum())
    for name,d in x['tables'].items():
        for key in ['retained_rows','duplicate_primary_keys','unlinked_rows','missing_time_rows','invalid_time_rows','pre_ICU_entry_rows']:
            add('source_quality',name+'_'+key,d['stats'][key])
    # Identity and time comparisons are separately verified without future statuses.
    assert ev.loc[ev.dx_infection_before,'dx_infection_first_time'].lt(ev.loc[ev.dx_infection_before,'index']).all()
    assert (~ev.strict_sepsis_mapping|ev.primary_mapping).all()
    assert ev.hospital.is_unique
    x.update(summary=pd.DataFrame(rows),candidate_evidence=ev,diagnosis_time_features=times,
        all_diagnosis_first_time=any_time,source_valid_records=valid,
        checks=dict(synthetic_checks=tests_passed,strict_preindex_only=True,original_candidates_unchanged=True,
            primary_without_future_diagnosis=True,final_discharge_active_not_read=True,no_death_contrast=True,no_weight_model=True),source_current=Path(__file__).read_bytes())
    save(b);say(dict(stage='diagnosis_review_done',primary_mapping=m.count(ev.primary_mapping.sum()),strict_sepsis_mapping=m.count(ev.strict_sepsis_mapping.sum()),
        infection_before=m.count(ev.dx_infection_before.sum()),sepsis_before=m.count(ev.dx_sepsis_before.sum()),effects=0))


def dose_band(v):
    a=np.asarray(v,float);out=np.full(a.shape,'unknown',dtype=object)
    out[np.isfinite(a)&(a>=0)&(a<.1)]='<0.10'
    out[np.isfinite(a)&(a>=.1)&(a<.25)]='0.10-<0.25'
    out[np.isfinite(a)&(a>=.25)&(a<.5)]='0.25-<0.50'
    out[np.isfinite(a)&(a>=.5)]='>=0.50'
    return out


def diagnosis_support():
    b=load();x=b['diagnosis_extension'];assert x['config']==DX_CONFIG;m=base()
    if 'support' in x:say(dict(stage='diagnosis_support',reuse=True));return
    started=time.perf_counter();last=started;d=b['drug'];ref=b['patient'].set_index('patientunitstayid')
    cases=x['candidate_evidence'][x['candidate_evidence'].primary_mapping].copy().reset_index(drop=True)
    cases['case_id']=np.arange(len(cases));ne=d['NE_records'].copy()
    ne0=ne[ne.positive_record].groupby('patienthealthsystemstayid').infusionoffset_hospital.min()
    cases['NE_elapsed_min']=cases['index']-cases.hospital.map(ne0)
    med=d['medication_resolved'];gc=med[med.is_GC&med.order_in_hospital&~med.route_class.eq('other_nonIV_route')]
    first_gc_order=gc.groupby('patienthealthsystemstayid').drugorderoffset_hospital.min()
    history=b['tables']['admissiondrug']['steroid_records'].copy()
    history=history[~history.explicit_topical_or_local_name]
    history['hospital']=history.patientunitstayid.map(ref.patienthealthsystemstayid)
    history['available']=np.maximum(history.drugoffset,history.drugenteredoffset)-history.patientunitstayid.map(ref.hospitaladmitoffset)
    first_history=history.groupby('hospital').available.min()
    first_inf=x['diagnosis_time_features']['infection'];first_sepsis=x['diagnosis_time_features']['sepsis']
    ne['explicit_zero']=ne.rate_numeric.eq(0)|ne.infusion_numeric.eq(0)
    ne['dose_combined']=ne.dose_mcgkgmin.combine_first(ne.dose_concentration_proxy)
    cg=ne.groupby(['patientunitstayid','infusionoffset_hospital'],sort=False)
    collapsed=cg.agg(positive=('positive_record','all'),zero=('explicit_zero','any'),
        rows=('positive_record','size'),dose_n=('dose_combined','count'),dose_lo=('dose_combined','min'),dose_hi=('dose_combined','max')).reset_index()
    collapsed['usable_positive']=collapsed.positive&~collapsed.zero
    collapsed['dose']=collapsed.dose_lo.where(collapsed.dose_n.eq(collapsed.rows)&np.isclose(collapsed.dose_lo,collapsed.dose_hi))
    groups={u:g.sort_values('infusionoffset_hospital') for u,g in collapsed.groupby('patientunitstayid')}
    bysite={site:g for site,g in cases.groupby('site')};pieces=[];checked=0
    any_med_units=set(d['medication']['units_any'])
    # Same-site risksets, no fitted propensity or outcome used in their formation.
    for unit,g in groups.items():
        s=ref.loc[unit];case=bysite.get(s.hospitalid)
        if case is None:continue
        h=s.patienthealthsystemstayid
        if h not in ne0.index:continue
        t=float(ne0.loc[h])+case.NE_elapsed_min.to_numpy(float)
        keep=(t>=0)&(t>=float(s.unit_start_from_hospital))&(t<=float(s.unit_end_from_hospital))&(t<=float(s.hospital_duration))
        keep&=case.subject.to_numpy()!=s.uniquepid
        keep&=float(first_inf.get(h,np.inf))<t
        keep&=float(first_gc_order.get(h,np.inf))>t
        keep&=float(first_history.get(h,np.inf))>=t
        # Historical entry exactly at index is not prior information; current
        # GC orders are not classified as an untreated control.
        if not keep.any():continue
        pos=np.searchsorted(g.infusionoffset_hospital.to_numpy(float),t,side='left')-1
        ix=np.maximum(pos,0);age=t-g.infusionoffset_hospital.to_numpy(float)[ix]
        keep&=(pos>=0)&(age>0)&(age<=60)&g.usable_positive.to_numpy(bool)[ix]
        if keep.any():
            f=case.loc[keep,['case_id']].copy();f['hospital']=h;f['subject']=s.uniquepid;f['site']=s.hospitalid
            f['patientunitstayid']=unit;f['index']=t[keep];f['NE_age_min']=age[keep];f['NE_dose']=g.dose.to_numpy(float)[ix][keep]
            f['sepsis_before']=float(first_sepsis.get(h,np.inf))<t[keep]
            f['medication_source_present_in_unit']=unit in any_med_units
            f['age_numeric']=s.age_numeric;f['admission_weight']=s.admissionweight
            pieces.append(f)
        checked+=1
        if time.perf_counter()-last>=30:
            say(dict(stage='diagnosis_support_progress',seconds=round(time.perf_counter()-started),unit_groups=m.count(checked)));last=time.perf_counter()
    controls=pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame(columns=['case_id','hospital','subject','NE_dose','sepsis_before','site','medication_source_present_in_unit'])
    duplicated=controls.duplicated(['case_id','hospital'],keep=False)
    ambiguous=controls.loc[duplicated].copy();controls=controls.loc[~duplicated].copy()
    case_dose=[];case_age=[]
    for r in cases.itertuples():
        g=groups[r.patientunitstayid];pos=np.searchsorted(g.infusionoffset_hospital.to_numpy(float),r.index,side='left')-1
        assert pos>=0 and g.usable_positive.iloc[pos]
        age=r.index-g.infusionoffset_hospital.iloc[pos];assert 0<age<=60
        case_age.append(age);case_dose.append(g.dose.iloc[pos])
    cases['NE_dose']=case_dose;cases['NE_age_min_recomputed']=case_age;cases['NE_dose_band']=dose_band(cases.NE_dose)
    controls['NE_dose_band']=dose_band(controls.NE_dose)
    pools=controls.groupby('case_id').agg(control_hospitals=('hospital','nunique'),control_patients=('subject','nunique'),
        dose_observed=('NE_dose','count'),medication_source_present=('medication_source_present_in_unit','sum'))
    cases=cases.join(pools,on='case_id');cols=['control_hospitals','control_patients','dose_observed','medication_source_present']
    cases[cols]=cases[cols].fillna(0).astype('int64')
    bands=controls.groupby(['case_id','NE_dose_band']).size()
    cases['same_dose_band_controls']=[int(bands.get((r.case_id,r.NE_dose_band),0)) if r.NE_dose_band!='unknown' else np.nan for r in cases.itertuples()]
    strict_controls=controls[controls.sepsis_before&controls.case_id.isin(cases.loc[cases.strict_sepsis_mapping,'case_id'])]
    strict_pool=strict_controls.groupby('case_id').hospital.nunique()
    rows=[]
    def add(scope,metric,n=None,den=None,value=None):rows.append(dict(scope=scope,metric=metric,n=n,den=den,value=value))
    add('primary','cases',len(cases));add('primary','cases_with_any_same_site_control',cases.control_hospitals.gt(0).sum(),len(cases))
    add('primary','distinct_control_hospital_stays',controls.hospital.nunique());add('primary','distinct_control_patients',controls.subject.nunique())
    add('primary','control_memberships',len(controls));add('primary','sites',cases.site.nunique())
    for q in [.1,.25,.5,.75,.9]:add('primary','control_pool_P'+str(int(q*100)),value=float(cases.control_hospitals.quantile(q)))
    add('primary','treated_NE_dose_observed',cases.NE_dose.notna().sum(),len(cases))
    add('primary','control_memberships_NE_dose_observed',controls.NE_dose.notna().sum(),len(controls))
    add('primary','dose_observed_cases_with_same_dose_band_control',cases.same_dose_band_controls.gt(0).sum(),cases.NE_dose.notna().sum())
    add('primary','control_memberships_medication_source_in_unit',controls.medication_source_present_in_unit.sum(),len(controls))
    add('primary','ambiguous_unit_overlap_memberships',len(ambiguous))
    strict=cases[cases.strict_sepsis_mapping]
    add('strict','cases',len(strict));add('strict','cases_with_same_site_sepsis_control',strict.case_id.map(strict_pool).fillna(0).gt(0).sum(),len(strict))
    add('strict','distinct_control_patients',strict_controls.subject.nunique())
    # Planned start metadata is described, never used to choose qualifying cases.
    detail=cases.medicationid.map(med.set_index('medicationid').drugstartoffset_hospital)
    delta=cases['index']-detail
    add('clock','planned_start_precedes_order',delta.gt(0).sum(),len(cases))
    for minutes in [15,60,120]:add('clock','planned_start_precedes_order_more_than_'+str(minutes),delta.gt(minutes).sum(),len(cases))
    add('clock','earlier_planned_start_lag_median',value=float(delta[delta.gt(0)].median()))
    assert controls.index.is_unique and not controls.duplicated(['case_id','hospital']).any()
    own=cases.set_index('case_id');assert controls.subject.ne(controls.case_id.map(own.subject)).all()
    assert controls.site.eq(controls.case_id.map(own.site)).all()
    assert controls.NE_age_min.gt(0).all() and controls.NE_age_min.le(60).all()
    x['support']=dict(cases=cases,controls=controls,ambiguous=ambiguous,summary=pd.DataFrame(rows),
        contract='same hospital site; same NE elapsed minutes; strict prior infection, no earlier recorded GC; no propensity/ESS/effect estimated',
        checks=dict(same_site=True,same_patient_excluded=True,no_duplicate_hospital_memberships=True,strict_preindex_NE=True,
            no_future_diagnosis=True,original_first_HC_orders_unchanged=True,no_effects=True),seconds=time.perf_counter()-started)
    x['source_current']=Path(__file__).read_bytes();save(b)
    say(dict(stage='diagnosis_support_done',cases=m.count(len(cases)),cases_with_controls=m.count(cases.control_hospitals.gt(0).sum()),
        control_patients=m.count(controls.subject.nunique()),control_memberships=m.count(len(controls)),seconds=round(time.perf_counter()-started,1),effects=0))


def diagnosis_validate():
    """Independently rebuild bounded risksets from source projections, no outcomes."""
    b=load();x=b['diagnosis_extension'];s=x['support'];d=b['drug'];m=base()
    ref=b['patient'].set_index('patientunitstayid');cases=s['cases'];controls=s['controls'];ne=d['NE_records']
    assert set(x['candidate_evidence'].medicationid)==set(d['technical_candidates'].medicationid)
    assert cases.hospital.is_unique and not controls.duplicated(['case_id','hospital']).any()
    assert cases.dx_infection_first_time.lt(cases['index']).all()
    assert ref.loc[cases.patientunitstayid,'adult'].all() and ref.loc[controls.patientunitstayid,'adult'].all()
    first_ne=ne[ne.positive_record].groupby('patienthealthsystemstayid').infusionoffset_hospital.min()
    med=d['medication_resolved'];gc=med[med.is_GC&med.order_in_hospital&~med.route_class.eq('other_nonIV_route')]
    first_gc=gc.groupby('patienthealthsystemstayid').drugorderoffset_hospital.min()
    h=b['tables']['admissiondrug']['steroid_records'].copy();h=h[~h.explicit_topical_or_local_name]
    h['hospital']=h.patientunitstayid.map(ref.patienthealthsystemstayid)
    h['available']=np.maximum(h.drugoffset,h.drugenteredoffset)-h.patientunitstayid.map(ref.hospitaladmitoffset)
    assert h.available.notna().all(), 'UNKNOWN_HISTORY_TIME_REQUIRES_SEPARATE_HANDLING'
    first_history=h.groupby('hospital').available.min();first_dx=x['diagnosis_time_features']['infection']
    own=cases.set_index('case_id')
    assert controls.subject.ne(controls.case_id.map(own.subject)).all()
    assert controls.site.eq(controls.case_id.map(own.site)).all()
    assert np.allclose(controls['index']-controls.hospital.map(first_ne),controls.case_id.map(own.NE_elapsed_min))
    assert controls.hospital.map(first_dx).lt(controls['index']).all()
    assert controls.hospital.map(first_gc).fillna(np.inf).gt(controls['index']).all()
    assert controls.hospital.map(first_history).fillna(np.inf).ge(controls['index']).all()
    assert controls['index'].ge(controls.patientunitstayid.map(ref.unit_start_from_hospital)).all()
    assert controls['index'].le(controls.patientunitstayid.map(ref.unit_end_from_hospital)).all()
    assert controls['index'].le(controls.patientunitstayid.map(ref.hospital_duration)).all()
    # Fixed indices span cases; selection never uses outcome or pool size.
    selected=np.unique(np.linspace(0,len(cases)-1,min(20,len(cases)),dtype=int))
    for idx in selected:
        case=cases.iloc[idx]
        u=ref[ref.adult&ref.hospitalid.eq(case.site)&ref.uniquepid.ne(case.subject)].copy()
        u['index']=u.patienthealthsystemstayid.map(first_ne)+case.NE_elapsed_min
        t=u['index'];hospital=u.patienthealthsystemstayid
        keep=t.ge(0)&t.ge(u.unit_start_from_hospital)&t.le(u.unit_end_from_hospital)&t.le(u.hospital_duration)
        keep&=hospital.map(first_dx).lt(t)&hospital.map(first_gc).fillna(np.inf).gt(t)
        keep&=hospital.map(first_history).fillna(np.inf).ge(t)
        u=u[keep];q=ne[ne.patientunitstayid.isin(u.index)].copy()
        q['index']=q.patientunitstayid.map(u['index']);age=q['index']-q.infusionoffset_hospital
        q=q[age.gt(0)&age.le(60)]
        q=q[q.infusionoffset_hospital.eq(q.groupby('patientunitstayid').infusionoffset_hospital.transform('max'))]
        q['valid']=q.positive_record&~q.rate_numeric.eq(0)&~q.infusion_numeric.eq(0)
        valid=q.groupby('patientunitstayid').valid.all();u=u.loc[valid.index[valid]]
        u=u[~u.patienthealthsystemstayid.duplicated(keep=False)]
        observed=controls[controls.case_id.eq(case.case_id)]
        assert set(u.patienthealthsystemstayid)==set(observed.hospital)
        assert set(u.index)==set(observed.patientunitstayid)
    for name in ['admissiondx','diagnosis']:
        f=locate(name);info=b['inputs'][name]
        assert info['parsed_to_EOF'] and tuple(info['signature'])==(str(f),f.stat().st_size,f.stat().st_mtime_ns)
        assert 'activeupondischarge' not in info['columns'] and 'admitdxtext' not in info['columns']
    x['validation']=dict(synthetic_checks=diagnosis_tests(),independent_case_risksets=len(selected),
        all_memberships_temporal_checks=True,same_site_and_patient_exclusion=True,original_candidates_unchanged=True,
        source_signatures_unchanged=True,no_outcomes_used=True,source_sha256=sha(Path(__file__).read_bytes()))
    save(b);say(dict(stage='diagnosis_validation',checks=x['validation'],effects=0))


def diagnosis_publish():
    b=load();x=b['diagnosis_extension'];assert 'validation' in x;m=base()
    ev=x['candidate_evidence'];s=x['support'];cases=s['cases'];controls=s['controls'];summary=s['summary']
    path=ROOT/'steroid_tte.md';before=path.read_bytes();old=before.decode('utf-8')
    newline='\r\n' if '\r\n' in old else '\n';old=old.replace('\r\n','\n')
    def fmt(n,den=None):
        text=m.count(n)
        if den is not None:
            p=m.pct(n,den);text+=' / '+m.count(den)+('（比例抑制）' if p=='' else '（'+str(p)+'%）')
        return text
    def metric(scope,name):
        r=summary[summary.scope.eq(scope)&summary.metric.eq(name)].iloc[0]
        return fmt(r.n,r.den if pd.notna(r.den) else None) if pd.notna(r.n) else f'{r.value:.1f}'
    strict=cases[cases.strict_sepsis_mapping]
    lines=['<!-- HC_EICU_DIAGNOSIS_START -->','### eICU诊断到位：治疗前资格与同院候选对照','',
        '**决定：继续医嘱启动版的多中心补充分析。** 新取得的admissionDx和diagnosis已完整读取，前文“诊断尚未取得”的状态到此更新。治疗前感染记录与同院对照均已找到；下一步补足治疗前病情，再判断加权比较。当前没有计算新的死亡差异、权重或ESS，MIMIC实际首剂的主估计保持不变。','',
        '|新诊断资料带来的实际进展|住院/医嘱约数|','|---|---:|',
        '|原首次IV HC医嘱技术候选|'+fmt(len(ev))+'|',
        '|医嘱前已有明确感染诊断记录|'+fmt(ev.dx_infection_before.sum(),len(ev))+'|',
        '|原候选中仅在医嘱后出现感染诊断记录|'+fmt(ev.dx_infection_late_only.sum(),len(ev))+'|',
        '|感染记录在前、此前60分钟内NE正速率、未检出更早系统性GC记录|'+fmt(len(cases))+'|',
        '|上述候选中的独立患者 / 医院|'+fmt(cases.subject.nunique())+' / '+fmt(cases.site.nunique())+'|',
        '|进一步要求明确sepsis/septic shock标签的候选|'+fmt(len(strict))+'|','',
        '资格按诊断的录入偏移换算到同次医院时间，严格早于医嘱；“admission”名称不代表入院即已可知。只用结构化标签，不读admitDxText、出院活跃标志或诊断优先级。明确感染与明确sepsis分别保存，非感染性SIRS、待排/既往标签、吸入性肺炎样炎症和不确定脓肿不靠宽泛关键词纳入。该映射表示已记录的临床诊断，不是确诊感染性休克或疾病起始时刻，也不是MIMIC抗菌药—培养配对的同一操作定义。没有用后来诊断补基线。','',
        '|相同医院、相同首次观察NE经过时间的候选对照|结果|','|---|---:|',
        '|能够找到对照的HC医嘱|'+fmt(cases.control_hospitals.gt(0).sum(),len(cases))+'|',
        '|独立对照住院 / 对照患者|'+fmt(controls.hospital.nunique())+' / '+fmt(controls.subject.nunique())+'|',
        '|对照风险集成员记录（非独立患者）|'+fmt(len(controls))+'|',
        '|每个HC节点的对照数，中位数（P25–P75）|'+metric('primary','control_pool_P50')+'（'+metric('primary','control_pool_P25')+'–'+metric('primary','control_pool_P75')+'）|',
        '|明确sepsis子集中可找到同院sepsis对照的医嘱|'+metric('strict','cases_with_same_site_sepsis_control')+'|',
        '|HC候选NE剂量可解析：直接记录加唯一配方代理|'+metric('primary','treated_NE_dose_observed')+'|',
        '|对照成员记录中NE剂量可解析|'+metric('primary','control_memberships_NE_dose_observed')+'|',
        '|已知剂量HC节点中，同院同剂量档仍有对照|'+metric('primary','dose_observed_cases_with_same_dose_band_control')+'|','',
        '对照来自全部成人NE来源住院，不局限后来开HC者。每个节点须仍在ICU，已有感染记录，60分钟内最后NE记录为正，且未检出更早GC医嘱/已录药史；同时开GC者不当作未启动对照。允许以后开HC，同一患者的其他住院不与其本人配对；重复参与不同风险集单独计数。这里对齐的是NE经过时间，未声称同一日历时刻。没有缺记录即停药的插值，也未按未来执行、最终取消或结局筛人。','',
        '同剂量档采用已有临床区间<0.10、0.10–<0.25、0.25–<0.50及≥0.50 µg/kg/min，仅用于支持度描述，不是新的政策阈值或入组条件。剂量约六成可解释，其余仍未知；配方代理与直接剂量分别保留。这些人数说明可建立候选比较，尚不代表病情已经平衡、足够的加权ESS或临床可互换性。','',
        '|医嘱时间核查：不用于事后删选|结果|','|---|---:|',
        '|计划开始早于医嘱时间|'+metric('clock','planned_start_precedes_order')+'|',
        '|上述时间差中位数|'+metric('clock','earlier_planned_start_lag_median')+'分钟|',
        '|计划开始早于医嘱超过15分钟|'+metric('clock','planned_start_precedes_order_more_than_15')+'|','',
        '该差异不能全部归为补录，也不能证明计划开始等于实际给药。后续主时钟继续采用医嘱录入时间；如检查计划开始口径，资格和变量需要一起重锚，不能只移动暴露。另一休克病因、肾上腺疾病和急性气道疾病标签单列，不据院内死亡猜测HC用途。诊断表没有可靠的病情消退/排除时刻；治疗限制的专门记录仍待补充，未知不能写为full code。','',
        '#### 下一批需要的临床文件','',
        '现有9表已能构建上述候选。下一批先补以下同版文件，可保留.csv.gz压缩格式；不需要下载全库，也不需要重下已处理药物表。','',
        '|文件|要解决的具体问题|','|---|---|',
        '|lab.csv(.gz)|医嘱前乳酸、肾功能、血细胞、血气等；区分采样/修订录入时间|',
        '|vitalPeriodic.csv(.gz)、vitalAperiodic.csv(.gz)|医嘱前MAP、HR、RR、SpO₂及其变化；保留动脉与无创来源|',
        '|nurseCharting.csv(.gz)|治疗前GCS/RASS、体温与护理记录；区分观察和录入时间|',
        '|respiratoryCare.csv(.gz)、respiratoryCharting.csv(.gz)|治疗前气道/呼吸支持状态及实际设置；不把设备报警阈值当FiO₂或PEEP|',
        '|carePlanGeneral.csv(.gz)|已记录DNR/DNI/CMO或治疗限制及辅助支持状态；不使用出院活跃状态|','',
        '上述7文件是优先的病情比较资料。若同时下载方便，可补treatment.csv(.gz)作为CRRT等治疗记录的辅助来源；其记录也不等于执行金标准，不将它作为本轮候选人数的条件。现有infusionDrug后续只需针对VP/EPI等伴随支持补投影，无须再次下载。原MIMIC的变量集不能直接假定在eICU完整可得，后续调整集须依据这些源表的实际覆盖冻结。','',
        '**保留的备选：**若病情记录不能支持可解释的比较，继续以MIMIC实际首剂为主稿，eICU保留为医嘱启动路径的描述性补充。两者不合并患者键或效应，不将医嘱分析称为已完成的实际首剂外部验证。这个备选不因本轮候选增加而删除。','',
        '#### 已完成读取与验证','',
        '|源表|实际读取体积|读取与初步投影|峰值进程工作集|','|---|---:|---:|---:|']
    for r in b['runs']:
        if r['table'] in ['admissiondx','diagnosis']:
            lines.append('|'+r['table']+'|'+f"{r['bytes_read']/1e6:.2f}MB|{r['seconds']:.2f}秒|{r['peak_MB']:.1f}MB|" )
    lines+=['',
        '两表逐块解析至EOF，输入SHA256与签名保存在原私有缓存。主键重复、无法关联患者和缺时间行均为0；超出医院/来源ICU时间包络的条目单列，不用于基线。风险集构建复用缓存，耗时'+f"{s['seconds']:.1f}"+'秒，未再读原始大表。13项标签/时间边界检查通过；全量候选对照的既往诊断、药物、同院/异患者及ICU在院条件均通过验证；另对固定选取的20个节点用源投影独立重建对照集合，结果一致。所有患者级材料留在本地私有缓存，本节计数按10取整，1–10及其可反推比例抑制。','',
        '复现入口：`steroid_eicu.py diagnosis-project` → `diagnosis-review` → `diagnosis-support` → `diagnosis-validate` → `diagnosis-publish`。聚合表、逐节点证据、配置、检查及历史均复用`hc_eicu_feasibility.pkl.gz`；只更新本报告，没有新增报告副本。','',
        '语义依据：[admissionDx](https://eicu.mit.edu/eicutables/admissiondx/)、[diagnosis](https://eicu.mit.edu/eicutables/diagnosis/)、[medication](https://eicu.mit.edu/eicutables/medication/)。后续字段依据：[lab](https://eicu.mit.edu/eicutables/lab/)、[生命体征](https://eicu.mit.edu/eicutables/vitalperiodic/)、[无创血压](https://eicu.mit.edu/eicutables/vitalaperiodic/)、[护理记录](https://eicu.mit.edu/eicutables/nursecharting/)、[呼吸支持](https://eicu.mit.edu/eicutables/respiratorycare/)、[呼吸设置](https://eicu.mit.edu/eicutables/respiratorycharting/)、[治疗限制/照护计划](https://eicu.mit.edu/eicutables/careplangeneral/)。','<!-- HC_EICU_DIAGNOSIS_END -->','']
    block='\n'.join(lines)
    if '<!-- HC_EICU_DIAGNOSIS_START -->' in old:
        start=old.index('<!-- HC_EICU_DIAGNOSIS_START -->');end=old.index('<!-- HC_EICU_DIAGNOSIS_END -->',start)+len('<!-- HC_EICU_DIAGNOSIS_END -->')
        while end<len(old) and old[end]=='\n':end+=1
    else:start=end=old.index('### 实际代码与证据位置')
    after=(old[:start]+block+'\n'+old[end:]).replace('\n',newline).encode('utf-8')
    assert path.read_bytes()==before
    if after!=before:
        hist=x.setdefault('report_history',[])
        if not any(v['sha256']==sha(before) for v in hist):hist.append(dict(sha256=sha(before),bytes=before))
        x['publication']=dict(report_sha256=sha(after),code_sha256=sha(Path(__file__).read_bytes()))
        save(b);path.write_bytes(after)
    say(dict(stage='diagnosis_publish',report=str(path),effects=0))


def multicenter_support():
    """Describe site-dependent measurement support from cached, pre-index rows.

    No site exclusion, imputation, cohort revision, weights, or outcome analysis.
    Site rates describe these HC nodes, not a validated hospital interface status.
    """
    started=time.perf_counter();b=load();a=b['diagnosis_extension']['support'];m=base()
    case_cols=['case_id','patientunitstayid','site','index','NE_elapsed_min','NE_dose',
        'NE_age_min_recomputed','control_hospitals','dx_sepsis_before','source_identity']
    control_cols=['case_id','hospital','subject','site','NE_dose','medication_source_present_in_unit']
    c=a['cases'][case_cols].copy();q=a['controls'][control_cols].copy()
    assert not any('death' in v.lower() or 'outcome' in v.lower() for v in case_cols+control_cols)
    assert c.case_id.is_unique and not q.duplicated(['case_id','hospital']).any()
    c['dose_known']=c.NE_dose.notna();q['dose_known']=q.NE_dose.notna()
    sites=c.groupby('site').agg(nodes=('case_id','size'),dose_known=('dose_known','sum'))
    sites['dose_fraction']=sites.dose_known/sites.nodes
    pools=q.groupby('case_id').agg(memberships=('hospital','size'),dose_known=('dose_known','sum'))
    c['known_dose_controls']=c.case_id.map(pools.dose_known).fillna(0).astype('int64')
    c['site_no_known_case_dose']=c.site.isin(sites.index[sites.dose_known.eq(0)])
    # Attribute missing doses to the actual last pre-index NE source records.
    ne=b['drug']['NE_records'];groups={u:g for u,g in ne.groupby('patientunitstayid')}
    reasons=[]
    for r in c.itertuples():
        g=groups[r.patientunitstayid];g=g[g.infusionoffset_hospital.lt(r.index)]
        assert len(g)>0
        g=g[g.infusionoffset_hospital.eq(g.infusionoffset_hospital.max())]
        assert np.isclose(r.index-g.infusionoffset_hospital.iloc[0],r.NE_age_min_recomputed)
        if pd.notna(r.NE_dose):reason='dose_available'
        elif g.unit.eq('ml/hr').all():reason='volume_rate_without_usable_concentration_or_weight'
        elif g.unit.eq('unknown').any():reason='unresolved_rate_unit'
        else:reason='weight_or_duplicate_record_conflict'
        reasons.append(reason)
    c['dose_support_reason']=reasons
    rows=[]
    def add(metric,n=None,den=None,value=None):
        rows.append(dict(metric=metric,n=None if n is None else int(n),
                         denominator=None if den is None else int(den),value=value))
    add('HC_order_nodes',len(c));add('hospital_sites',len(sites))
    add('nodes_with_known_NE_dose',c.dose_known.sum(),len(c))
    add('nodes_at_sites_with_no_known_dose_at_any_HC_node',c.site_no_known_case_dose.sum(),len(c))
    add('missing_dose_nodes_from_zero_observed_case_dose_sites',c.site_no_known_case_dose.sum(),(~c.dose_known).sum())
    add('nodes_with_no_dose_known_same_site_controls',c.known_dose_controls.eq(0).sum(),len(c))
    add('nodes_with_known_dose_and_some_dose_known_control',(c.dose_known&c.known_dose_controls.gt(0)).sum(),len(c))
    for k in [1,5,10]:add('nodes_in_largest_'+str(k)+'_sites',sites.nodes.nlargest(k).sum(),len(c))
    for k,v in c.dose_support_reason.value_counts().items():add('dose_reason_'+k,v,len(c))
    profiles=[]
    for known,g in c.groupby('dose_known'):
        profiles.append(dict(dose_known=bool(known),nodes=len(g),
            NE_elapsed_median_min=float(g.NE_elapsed_min.median()),
            NE_record_age_median_min=float(g.NE_age_min_recomputed.median()),
            control_pool_median=float(g.control_hospitals.median()),
            sepsis_record_before=int(g.dx_sepsis_before.sum())))
    result=dict(config=dict(denominator='existing primary HC order nodes and their same-site same-NE-elapsed controls',
        source='existing eICU cache only; no raw read',outcomes_used=False,
        site_interpretation='source support at these nodes, not proven site-wide interface absence',
        action='descriptive only; no site dropped, no dose imputed, no eligibility changed'),
        summary=pd.DataFrame(rows),site_table=sites,node_evidence=c,profiles=profiles,
        checks=dict(case_count_unchanged=len(c)==len(a['cases']),control_count_unchanged=len(q)==len(a['controls']),
                    prior_source_rederived=True,unique_riskset_members=True,no_outcome_columns=True),
        seconds=time.perf_counter()-started,source_current=Path(__file__).read_bytes())
    prior=b.get('multicenter_support')
    if prior is not None:pd.testing.assert_frame_equal(prior['summary'],result['summary'])
    b['multicenter_support']=result;save(b)
    published=[]
    for r in rows:
        # Do not disclose partitions that reconstruct suppressed residual cells.
        if r['metric'].startswith('dose_reason_') or r['metric']=='nodes_with_known_dose_and_some_dose_known_control':continue
        published.append(dict(metric=r['metric'],n=m.count(r['n']) if r['n'] is not None else None,
            pct=m.pct(r['n'],r['denominator']) if r['denominator'] is not None else None))
    say(dict(stage='multicenter_support',summary=published,profiles=[dict(v,nodes=m.count(v['nodes']),
        sepsis_record_before=m.count(v['sepsis_record_before'])) for v in profiles],
        seconds=round(result['seconds'],2),effects=0))


CLINICAL_CONFIG = dict(
    version='HC_order_covariates_20261031',
    population='unchanged diagnosis_extension.support cases and complete same-site control memberships',
    purpose='pre-order covariate observability and local clinical support; no mortality comparison or new weights',
    primary_clock='observation strictly before index; lab also revised-entry strictly before index',
    strict_clock='all available observation/entry pairs strictly before index; no inferred first-visible time',
    windows_minutes=dict(vitals=60,temperature_GCS_respiratory=240,lactate_bloodgas=360,other_labs=1440,infusion=60,urine=360),
    limitation='all available same-hospital structured entries strictly before index; current status and historical positive evidence separate; unknown not full care',
    units='numeric lab units checked from retained vocabulary before feature derivation; NE conversion unchanged',
    absence='no-record is unknown, never normal/no therapy; treatment table is documentation evidence',
    local_support='descriptive same-site known NE dose band and MAP band; not propensity positivity or ESS',
    outcomes_used=False,prior_MIMIC_effects_seen=True)
CLINICAL_SPECS={
    'lab':(['labid','patientunitstayid','labresultoffset','labresultrevisedoffset','labtypeid','labname','labresult','labmeasurenamesystem','labmeasurenameinterface'],'labresultoffset',1440),
    'vitalPeriodic':(['vitalperiodicid','patientunitstayid','observationoffset','temperature','sao2','heartrate','respiration','systemicmean'],'observationoffset',240),
    'vitalAperiodic':(['vitalaperiodicid','patientunitstayid','observationoffset','noninvasivemean'],'observationoffset',60),
    'nurseCharting':(['nursingchartid','patientunitstayid','nursingchartoffset','nursingchartentryoffset','nursingchartcelltypecat','nursingchartcelltypevallabel','nursingchartcelltypevalname','nursingchartvalue'],'nursingchartoffset',240),
    'respiratoryCharting':(['respchartid','patientunitstayid','respchartoffset','respchartentryoffset','respcharttypecat','respchartvaluelabel','respchartvalue'],'respchartoffset',240),
    'respiratoryCare':(['respcareid','patientunitstayid','respcarestatusoffset','airwaytype','ventstartoffset','ventendoffset'],'respcarestatusoffset',240),
    'carePlanGeneral':(['cplgeneralid','patientunitstayid','cplitemoffset','cplgroup','cplitemvalue'],'cplitemoffset',None),
    'treatment':(['treatmentid','patientunitstayid','treatmentoffset','treatmentstring'],'treatmentoffset',240),
    'infusionDrug':(['infusiondrugid','patientunitstayid','infusionoffset','drugname','drugrate','infusionrate','drugamount','volumeoffluid','patientweight'],'infusionoffset',60),
    'intakeOutput':(['intakeoutputid','patientunitstayid','intakeoutputoffset','intakeoutputentryoffset','cellpath','celllabel','cellvaluenumeric'],'intakeoutputoffset',360)}


def clinical_path(name):
    # Inventory explicitly selects the verified uncompressed CSV. Gzip copies
    # remain in place; no ambiguous recursive first-match selection.
    p=RAW/(name+'.csv')
    for q in [p/(name+'.csv'),p,RAW/(name+'.csv.gz')]:
        if q.is_file():return q
    raise FileNotFoundError(str(p))


def clinical_requests(b):
    s=b['diagnosis_extension']['support'];cols=['case_id','patientunitstayid','hospital','subject','site','index','NE_dose']
    a=s['cases'][cols].copy();a['treated']=1
    c=s['controls'][cols].copy();c['treated']=0
    r=pd.concat([a,c],ignore_index=True);r['request_id']=np.arange(len(r))
    ref=b['patient'].set_index('patientunitstayid')
    r['unit_index']=r['index']+r.patientunitstayid.map(ref.hospitaladmitoffset)
    r['NE_elapsed_min']=r.case_id.map(s['cases'].set_index('case_id').NE_elapsed_min)
    assert r.unit_index.ge(0).all() and not r.duplicated(['case_id','hospital']).any()
    return r


def clinical_plan():
    b=load();r=clinical_requests(b)
    if 'clinical_extension' not in b:
        b['clinical_extension']=dict(config=CLINICAL_CONFIG,created_utc=base().utc(),raw={},runs={},
            requests=r,source_before=Path(__file__).read_bytes(),protected={})
        # Small public frozen artifacts are hashed; large private evidence uses
        # size/mtime invariants and is never rewritten by this extension.
        for f in [ROOT/'steroid_tte_aggregates.csv',ROOT/'TTE_Phenotype_Research_Decisions.html',ROOT/'icu_data_inventory.html']:
            b['clinical_extension']['protected'][str(f)]=sha(f.read_bytes())
        for f in [PRIVATE/'steroid_sequential.pkl.gz',PRIVATE/'phenotype_icu.pkl.gz']:
            b['clinical_extension']['protected'][str(f)]=(f.stat().st_size,f.stat().st_mtime_ns)
        save(b)
    x=b['clinical_extension'];assert x['config']==CLINICAL_CONFIG
    pd.testing.assert_frame_equal(x['requests'],r)
    files=[]
    for name,(cols,clock,window) in CLINICAL_SPECS.items():
        p=clinical_path(name);assert set(cols)<=set(pd.read_csv(p,nrows=0).columns)
        files.append(dict(table=name,path=str(p),bytes=p.stat().st_size,columns=cols,
            time_field=clock,lookback_minutes=window,reuse=name in x['raw']))
    say(dict(stage='clinical_plan',files=files,units=base().count(r.patientunitstayid.nunique()),
        parsed_bytes_if_not_cached=sum(v['bytes'] for v in files if not v['reuse']),
        chunksize=150000,estimated_minutes='5-15 based on prior local scans; not a guarantee',
        memory='chunked; projected data only, aim below 3 GiB; no whole-table load',effects=0))


def clinical_project():
    b=load();x=b['clinical_extension'];assert x['config']==CLINICAL_CONFIG
    r=x['requests'];ref=b['patient'].set_index('patientunitstayid');envelope=r.groupby('patientunitstayid').unit_index.agg(['min','max'])
    hmax=r.groupby('hospital')['index'].max();ids=set(envelope.index);m=base()
    for name,(cols,clock,window) in CLINICAL_SPECS.items():
        path=clinical_path(name);st=path.stat();signature=(str(path),st.st_size,st.st_mtime_ns)
        if name in x['raw']:
            assert tuple(x['runs'][name]['signature'])==signature
            say(dict(stage='clinical_project',table=name,reuse=True));continue
        started=time.perf_counter();last=started;parts=[];rows=0
        say(dict(stage='clinical_scan_start',table=name,bytes=st.st_size,columns=cols,
            filter='existing case/control ICU units, pre-index window union; CarePlan all linked hospital units before latest index',effects=0))
        reader=m.HashReader(path);buf=io.BufferedReader(reader,buffer_size=4*1024*1024)
        stream=gzip.GzipFile(fileobj=buf) if path.suffix=='.gz' else buf
        try:
            for z in pd.read_csv(stream,usecols=cols,dtype=str,chunksize=150000):
                rows+=len(z);u=pd.to_numeric(z.patientunitstayid,errors='coerce')
                if name=='carePlanGeneral':keep=u.map(ref.patienthealthsystemstayid).isin(hmax.index)
                else:keep=u.isin(ids)
                q=z[keep].copy()
                if len(q):
                    q['patientunitstayid']=u.loc[q.index].astype('int64');t=pd.to_numeric(q[clock],errors='coerce')
                    hospital_t=t-q.patientunitstayid.map(ref.hospitaladmitoffset)
                    if name=='carePlanGeneral':
                        hi=q.patientunitstayid.map(ref.patienthealthsystemstayid).map(hmax)
                        keep=hospital_t.ge(0)&hospital_t.lt(hi)
                    else:
                        lo=q.patientunitstayid.map(envelope['min'])-window;hi=q.patientunitstayid.map(envelope['max'])
                        keep=t.ge(lo)&t.lt(hi)&hospital_t.ge(0)
                        if name!='lab':keep &= t.ge(0)
                    q=q[keep].copy()
                    if name=='nurseCharting':
                        label=lower(q.nursingchartcelltypevallabel)+'|'+lower(q.nursingchartcelltypevalname)
                        q=q[label.str.contains('gcs|glasgow|rass|sedation|temperature|blood pressure|heart rate|respiratory rate|o2 saturation',regex=True)]
                    if name=='infusionDrug':
                        q=q[q.drugname.fillna('').str.contains(r'norepineph|noradren|levophed|vasopressin|(?<!nor)epinephrine|dopamine|dobutamine|phenylephrine|propofol|midazolam|dexmedetomidine|fentanyl',case=False,regex=True)]
                    if name=='intakeOutput':
                        label=lower(q.celllabel);q=q[label.str.contains('urine')&~label.str.contains('count|occurrence|incontinence')]
                    if len(q):parts.append(q)
                if time.perf_counter()-last>25:
                    say(dict(stage='clinical_scan_progress',table=name,seconds=round(time.perf_counter()-started),rows=m.count(rows)));last=time.perf_counter()
            assert reader.n==st.st_size;digest=reader.h.hexdigest()
        finally:stream.close();buf.close()
        q=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=cols)
        duplicates=int(q.duplicated(cols[0]).sum());assert duplicates==0,'duplicate source keys: '+name
        assert signature==(str(path),path.stat().st_size,path.stat().st_mtime_ns)
        # Identical sparse/row values may have different source keys; selection
        # handles same-time conflicts and never sums duplicate infusion rates.
        for col in q.select_dtypes(include='object'):
            if col not in [cols[0]] and q[col].nunique(dropna=False)<max(2000,len(q)*.1):q[col]=q[col].astype('category')
        x['raw'][name]=q;x['runs'][name]=dict(signature=signature,sha256=digest,rows=rows,retained=len(q),
            parsed_to_EOF=True,seconds=round(time.perf_counter()-started,2),peak_MB=m.peak_mb(),duplicate_keys=duplicates,columns=cols)
        x['source_current']=Path(__file__).read_bytes();save(b)
        say(dict(stage='clinical_scan_done',table=name,retained=m.count(len(q)),seconds=x['runs'][name]['seconds'],peak_MB=m.peak_mb()))


CLINICAL_LABS={
    'lactate':('lactate',360,(0.1,40),'mmol/l'),
    'creatinine':('creatinine',1440,(0.1,30),'mg/dl'),
    'platelets x 1000':('platelet',1440,(1,2000),'k/mcl'),
    'total bilirubin':('bilirubin',1440,(0,80),'mg/dl'),
    'pH':('pH',360,(6.5,8.0),''),
    'paO2':('pO2',360,(10,800),'mm hg'),
    'paCO2':('pCO2',360,(5,200),'mm hg'),
    'bicarbonate':('bicarbonate',1440,(2,60),'mmol/l'),
    'HCO3':('bloodgas_bicarbonate',360,(2,60),'mmol/l'),
    'WBC x 1000':('wbc',1440,(0.1,200),'k/mcl'),
    'Hgb':('hemoglobin',1440,(2,25),'g/dl'),
    'sodium':('sodium',1440,(100,180),'mmol/l'),
    'potassium':('potassium',1440,(1.5,10),'mmol/l'),
    'albumin':('albumin',1440,(0.3,7),'g/dl'),
    'BUN':('bun',1440,(1,300),'mg/dl'),
    'glucose':('glucose',1440,(10,1500),'mg/dl')}


def clinical_event(q,obs,entry,value):
    # One table at a time; retain invalid latest measurements as missing rather
    # than silently selecting an older normal reading.
    z=pd.DataFrame(dict(key=q.patientunitstayid.to_numpy('int64'),
        obs=pd.to_numeric(q[obs],errors='coerce').to_numpy(float),
        value=np.asarray(value)))
    z['entry']=pd.to_numeric(q[entry],errors='coerce').to_numpy(float) if entry else z.obs
    z['available']=np.maximum(z.obs,z.entry)
    return z[z.obs.notna()].reset_index(drop=True)


def clinical_lookup(events,requests,window,strict=True,key_col='patientunitstayid',time_col='unit_index'):
    """Select latest observation, then latest known revision; tied conflicts unknown."""
    n=len(requests);out=np.full(n,np.nan);age=np.full(n,np.nan);available_age=np.full(n,np.nan)
    seen=np.zeros(n,bool);conflict=np.zeros(n,bool)
    groups={k:g[['obs','available','value']].to_numpy(float) for k,g in events.groupby('key',sort=False)}
    for key,idx in requests.groupby(key_col,sort=False).groups.items():
        a=groups.get(key)
        if a is None:continue
        obs,avail,values=a.T
        for i in idx:
            t=float(requests.at[i,time_col]);keep=(obs<t)&(obs>=t-window)
            if strict:keep &= avail<t
            if not keep.any():continue
            last=np.max(obs[keep]);pick=keep&(obs==last)
            if np.isfinite(avail[pick]).any():pick &= avail==np.nanmax(avail[pick])
            vals=values[pick];seen[i]=True;age[i]=t-last
            available_age[i]=t-np.nanmax(avail[pick]) if np.isfinite(avail[pick]).any() else np.nan
            finite=vals[np.isfinite(vals)]
            if len(finite)==len(vals) and len(vals) and np.allclose(finite,finite[0],rtol=0,atol=1e-8):out[i]=finite[0]
            else:conflict[i]=len(vals)>1
    return dict(value=out,age=age,available_age=available_age,seen=seen,conflict=conflict)


def clinical_lookup_tests():
    ev=pd.DataFrame(dict(key=[1,1,1,1,1],obs=[1.,3.,4.,6.,6.],available=[5.,3.,7.,6.,6.],value=[1.,2.,4.,6.,7.]))
    r=pd.DataFrame(dict(patientunitstayid=[1,1,1,1],unit_index=[4.,5.,7.,8.]))
    s=clinical_lookup(ev,r,10,True);p=clinical_lookup(ev,r,10,False)
    assert s['value'][0]==2 and s['value'][1]==2 # late older observation never displaces a newer one
    assert s['value'][2]!=s['value'][2] and s['conflict'][2] # same-time disagreement
    assert p['value'][1]==4 and p['available_age'][1]<0 # measurement-only scenario is explicitly traceable
    assert clinical_lookup(ev,r.iloc[:1].copy(),1,True)['value'][0]==2 # inclusive lower boundary
    bad=pd.DataFrame(dict(key=[1,1],obs=[1.,3.],available=[1.,3.],value=[1.,np.nan]))
    assert np.isnan(clinical_lookup(bad,r.iloc[:1].copy(),10,True)['value'][0])
    return 6


def clinical_crrt_events(q):
    labels=q.treatmentstring.astype('object').fillna('').str.lower().str.replace(' ','',regex=False)
    # Source pick-list uses spaced abbreviations: C V V H and C V V H D.
    # Do not include dialysis catheters or intermittent haemodialysis.
    mask=labels.str.contains(r'continuousrenalreplacement|(?:^|\|)cvvh(?:d|df)?(?:\||$)',regex=True)
    z=q[mask].copy()
    return clinical_event(z,'treatmentoffset',None,np.ones(len(z)))


def clinical_features():
    started=time.perf_counter();b=load();x=b['clinical_extension'];assert x['config']==CLINICAL_CONFIG
    if 'features' in x:say(dict(stage='clinical_features',reuse=True));return
    assert all(n in x['raw'] for n in CLINICAL_SPECS);r=x['requests'];raw=x['raw'];m=base()
    tests_passed=clinical_lookup_tests();features={clock:r.copy() for clock in ['primary','strict']}
    contracts={};unit_checks=[]
    def add(var,ev,window,always_strict=False,scope='unit',meaning='numeric observed value'):
        key,timecol=('hospital','index') if scope=='hospital' else ('patientunitstayid','unit_index')
        for clock,f in features.items():
            z=clinical_lookup(ev,r,window,always_strict or clock=='strict',key,timecol)
            extra={var+suffix:z[values] for suffix,values in [('','value'),('_age_min','age'),('_record_seen','seen'),('_tie_conflict','conflict'),('_available_age_min','available_age')]}
            features[clock]=pd.concat([f,pd.DataFrame(extra,index=f.index)],axis=1)
        contracts[var]=dict(lookback_minutes=window,always_strict=always_strict,scope=scope,meaning=meaning)
    q=raw['lab']
    for label,(var,window,(lo,hi),unit) in CLINICAL_LABS.items():
        z=q[q.labname.eq(label)].copy();v=pd.to_numeric(z.labresult,errors='coerce')
        sysunit=z.labmeasurenamesystem.astype('object').fillna('').str.lower().str.strip()
        interface=z.labmeasurenameinterface.astype('object').fillna('').str.lower().str.replace(' ','',regex=False)
        good=sysunit.eq(unit)
        # A clearly conflicting per-mL interface label is not silently treated as per-L.
        unit_conflict=interface.str.contains(r'meq/ml|mmol/ml',regex=True)
        good &= ~unit_conflict
        unit_checks.append(dict(variable=var,system_units=sorted(set(sysunit)),rows=len(z),
            incompatible_unit_rows=int((~good).sum()),out_of_range_rows=int((~v.between(lo,hi)).sum())))
        v=v.where(good&v.between(lo,hi))
        add(var,clinical_event(z,'labresultoffset','labresultrevisedoffset',v),window,True,meaning='standardized lab value; revised-entry is a conservative availability proxy')
    q=raw['vitalPeriodic']
    for col,var,ran,window in [('heartrate','HR',(20,250),60),('respiration','RR',(2,80),60),('sao2','SpO2',(30,100),60),('systemicmean','MAP_arterial',(20,180),60),('temperature','temperature_monitor',(25,45),240)]:
        z=q[q[col].notna()].copy();v=pd.to_numeric(z[col],errors='coerce');v=v.where(v.between(*ran))
        add(var,clinical_event(z,'observationoffset',None,v),window,meaning='monitor five-minute median record; not clinician-validated')
    q=raw['vitalAperiodic'];z=q[q.noninvasivemean.notna()].copy();v=pd.to_numeric(z.noninvasivemean,errors='coerce')
    add('MAP_noninvasive',clinical_event(z,'observationoffset',None,v.where(v.between(20,180))),60)
    q=raw['nurseCharting'];name=q.nursingchartcelltypevalname.astype('object').fillna('').str.lower();label=q.nursingchartcelltypevallabel.astype('object').fillna('').str.lower()
    for var,mask,ran,convert in [('GCS',name.eq('gcs total'),(3,15),False),
        ('RASS',label.eq('rass'),(-5,4),False),('temperature_nurse_C',name.eq('temperature (c)'),(25,45),False),
        ('temperature_nurse_F',name.eq('temperature (f)'),(25,45),True)]:
        z=q[mask].copy();v=pd.to_numeric(z.nursingchartvalue,errors='coerce')
        if convert:v=(v-32)*5/9
        add(var,clinical_event(z,'nursingchartoffset','nursingchartentryoffset',v.where(v.between(*ran))),240)
    q=raw['respiratoryCharting'];label=q.respchartvaluelabel.astype('object').fillna('').str.lower().str.strip()
    for var,labels,ran in [('FiO2',['fio2','fio2 (%)','o2 percentage','set fraction of inspired oxygen (fio2)'],(21,100)),
        ('PEEP',['peep','peep/cpap'],(0,40))]:
        z=q[label.isin(labels)].copy();v=pd.to_numeric(z.respchartvalue.astype('object').str.replace('%','',regex=False).str.strip(),errors='coerce')
        if var=='FiO2':v=v.where(~v.between(.21,1),v*100)
        add(var,clinical_event(z,'respchartoffset','respchartentryoffset',v.where(v.between(*ran))),240,meaning='documented delivered setting; alarm and apnea backup fields excluded')
    # Exact structured airway states, rather than treating zero end offsets as stops.
    q=raw['respiratoryCare'];s=q.airwaytype.astype('object').fillna('').str.lower()
    value=s.map({'oral ett':1.,'nasal ett':1.,'tracheostomy':1.,'no artificial airway':0.})
    add('artificial_airway_resp',clinical_event(q,'respcarestatusoffset',None,value),240)
    q=raw['carePlanGeneral'];ref=b['patient'].set_index('patientunitstayid')
    group=q.cplgroup.astype('object').fillna('').str.lower();value=q.cplitemvalue.astype('object').fillna('').str.lower().str.strip()
    x['careplan_vocabulary']=q[['cplgroup','cplitemvalue']].drop_duplicates().copy()
    airway=q[group.eq('airway')].copy();val=value.loc[airway.index]
    v=pd.Series(np.nan,index=airway.index);v.loc[val.str.startswith('intubated/')]=1.;v.loc[val.str.startswith('not intubated/')]=0.
    add('artificial_airway_careplan',clinical_event(airway,'cplitemoffset',None,v),240)
    vent=q[group.eq('ventilation')].copy();val=value.loc[vent.index]
    v=pd.Series(np.nan,index=vent.index);v.loc[val.str.startswith('ventilated')]=2.;v.loc[val.eq('non-invasive ventilation')]=1.;v.loc[val.str.startswith('spontaneous')]=0.
    add('ventilation_careplan',clinical_event(vent,'cplitemoffset',None,v),240)
    lim=q[group.isin(['care limitation','code status'])].copy();lim['value_clean']=value.loc[lim.index]
    lim['hospital']=lim.patientunitstayid.map(ref.patienthealthsystemstayid)
    lim['time']=pd.to_numeric(lim.cplitemoffset,errors='coerce')-lim.patientunitstayid.map(ref.hospitaladmitoffset)
    lg={h:g for h,g in lim.groupby('hospital')};flags={k:np.zeros(len(r),bool) for k in ['limitation_record','DNR_record','DNI_record','CMO_record','no_augmentation_record','no_pressors_record','full_therapy_record','limitation_conflict','historical_CMO_record']}
    patterns={'DNR_record':{'do not resuscitate','no cpr'},'DNI_record':{'no intubation'},'CMO_record':{'comfort measures only'},
        'no_augmentation_record':{'no augmentation of care'},'no_pressors_record':{'no vasopressors/inotropes'},'full_therapy_record':{'full therapy'}}
    for row in r.itertuples():
        g=lg.get(row.hospital)
        if g is None:continue
        g=g[g.time<row.index]
        if not len(g):continue
        flags['historical_CMO_record'][row.Index]=g.value_clean.eq('comfort measures only').any()
        vals=set(g.loc[g.time.eq(g.time.max()),'value_clean']);flags['limitation_record'][row.Index]=True
        for k,choices in patterns.items():flags[k][row.Index]=bool(vals&choices)
        flags['limitation_conflict'][row.Index]=flags['full_therapy_record'][row.Index] and any(flags[k][row.Index] for k in ['DNR_record','DNI_record','CMO_record','no_augmentation_record','no_pressors_record'])
    for f in features.values():
        for k,v in flags.items():f[k]=v
    # Other infusion support and sedation are evidence of recent documentation,
    # not a validated full drug history or proof of absent treatment.
    q=raw['infusionDrug'];nm=q.drugname.astype('object').fillna('');rate=pd.to_numeric(q.drugrate,errors='coerce');inf=pd.to_numeric(q.infusionrate,errors='coerce')
    for var,pattern in [('VP_documented','vasopressin'),('EPI_documented',r'(?<!nor)epinephrine'),
        ('sedation_infusion_documented','propofol|midazolam|dexmedetomidine')]:
        z=q[nm.str.contains(pattern,case=False,regex=True)].copy()
        rr=rate.loc[z.index];ii=inf.loc[z.index];v=pd.Series(np.nan,index=z.index)
        v.loc[(rr>0)|(ii>0)]=1.;v.loc[((rr==0)|(ii==0))&~((rr>0)|(ii>0))]=0.
        add(var,clinical_event(z,'infusionoffset',None,v),60)
    add('CRRT_documented',clinical_crrt_events(raw['treatment']),240,meaning='recent treatment documentation; absent remains unknown')
    q=raw['intakeOutput'];v=pd.to_numeric(q.cellvaluenumeric,errors='coerce')
    # No summation: cell labels may duplicate/cumulate, and interval length is
    # unresolved. This assesses numeric source coverage only.
    add('urine_numeric_record',clinical_event(q,'intakeoutputoffset','intakeoutputentryoffset',v.where(v.between(0,10000))),360,True,meaning='latest candidate urine record; not six-hour volume or urine rate')
    ne=b['drug']['NE_records'].copy();ne['value']=ne.dose_mcgkgmin.combine_first(ne.dose_concentration_proxy)
    ne['time']=ne.infusionoffset_hospital
    ng={u:g for u,g in ne.groupby('patientunitstayid')};delta=np.full(len(r),np.nan);span=delta.copy();peak=delta.copy()
    for row in r.itertuples():
        g=ng.get(row.patientunitstayid)
        if g is None or not np.isfinite(row.NE_dose):continue
        q=g[g.time.lt(row.index)&g.time.ge(row.index-60)].copy()
        if not len(q):continue
        by=q.groupby('time').value.agg(['size','count','min','max'])
        # Missing or contradictory values are not interpolated across.
        good=by['size'].eq(by['count'])&np.isclose(by['min'],by['max'])
        if not good.all() or len(by)<2:continue
        dt=float(by.index.max()-by.index.min())
        if dt<15:continue
        delta[row.Index]=float(by['min'].iloc[-1]-by['min'].iloc[0]);span[row.Index]=dt;peak[row.Index]=float(by['max'].max())
    for f in features.values():
        f['NE_delta_1h']=delta;f['NE_observed_span_min']=span;f['NE_peak_1h']=peak
        f['MAP']=f.MAP_arterial.combine_first(f.MAP_noninvasive)
        f['MAP_age_min']=f.MAP_arterial_age_min.where(f.MAP_arterial.notna(),f.MAP_noninvasive_age_min)
        f['MAP_source']=np.where(f.MAP_arterial.notna(),'arterial',np.where(f.MAP_noninvasive.notna(),'noninvasive','unknown'))
        f['temperature']=f.temperature_monitor.combine_first(f.temperature_nurse_C).combine_first(f.temperature_nurse_F)
        f['respiratory_status_observed']=f.artificial_airway_resp.notna()|f.artificial_airway_careplan.notna()|f.ventilation_careplan.notna()
        f['NE_dose_known']=f.NE_dose.notna()
        f['core_circulation_observed']=f.NE_dose.notna()&f.MAP.notna()&f.lactate.notna()
        f['core_plus_renal_observed']=f.core_circulation_observed&f.creatinine.notna()
        f['MAP_band']=pd.cut(f.MAP,[-np.inf,65,75,np.inf],right=False,labels=['<65','65-<75','>=75']).astype('object').fillna('unknown')
        f['lactate_band']=pd.cut(f.lactate,[-np.inf,2,4,np.inf],right=False,labels=['<2','2-<4','>=4']).astype('object').fillna('unknown')
        f['NE_band']=dose_band(f.NE_dose)
    x.update(features=features,variable_contracts=contracts,unit_checks=unit_checks,
        feature_checks=dict(synthetic=tests_passed,strict_prior=True,no_outcomes=True,requests_unchanged=True),
        feature_seconds=round(time.perf_counter()-started,2),source_current=Path(__file__).read_bytes())
    save(b);say(dict(stage='clinical_features_done',seconds=x['feature_seconds'],numeric_variables=len(contracts),checks=x['feature_checks'],effects=0))


def clinical_fix_crrt():
    b=load();x=b['clinical_extension'];m=base()
    if 'crrt_picklist_fix' in x:say(dict(stage='clinical_fix_crrt',reuse=True));return
    old={clock:f.filter(regex=r'^CRRT_documented').copy() for clock,f in x['features'].items()}
    assert all(f.CRRT_documented.isna().all() for f in old.values())
    evidence=clinical_crrt_events(x['raw']['treatment']);assert len(evidence)>0
    # Verify exact positive and negative dictionary examples without outcomes.
    toy=pd.DataFrame(dict(patientunitstayid=[1,1,1,1],treatmentoffset=[1,2,3,4],treatmentstring=[
        'renal|dialysis|C V V H','renal|dialysis|C V V H D',
        'renal|dialysis|hemodialysis','renal|dialysis|insertion of venous catheter for hemodialysis']))
    assert set(clinical_crrt_events(toy).obs)=={1,2}
    reason='Source pick-list C V V H / C V V H D contains spaces missed by the initial abbreviation parser; no window, cohort or model change.'
    fix=dict(created_utc=m.utc(),reason=reason,old_features=old,source_before=x['source_current'],
        source_after=Path(__file__).read_bytes(),dictionary_checks=4,outcomes_used=False)
    for clock,f in x['features'].items():
        z=clinical_lookup(evidence,x['requests'],240,clock=='strict')
        for suffix,key in [('','value'),('_age_min','age'),('_record_seen','seen'),('_tie_conflict','conflict'),('_available_age_min','available_age')]:
            f['CRRT_documented'+suffix]=z[key]
    # Preserve the initial diagnostic fits, then rerun the same fixed model.
    if 'assignment_diagnostics' in x:
        x.setdefault('assignment_diagnostics_history',[]).append(dict(reason=reason,diagnostics=x.pop('assignment_diagnostics')))
    x['crrt_picklist_fix']=fix;x['source_current']=Path(__file__).read_bytes();save(b)
    f=x['features']['primary'];a=f[f.treated.eq(1)];c=f[f.treated.eq(0)]
    say(dict(stage='clinical_fix_crrt',reason=reason,affected_cases=m.count(a.CRRT_documented.notna().sum()),
        affected_control_memberships=m.count(c.CRRT_documented.notna().sum()),effects=0))


def clinical_summarize():
    b=load();x=b['clinical_extension'];r=x['requests'];m=base();rows=[];support=[];site_rows=[]
    numeric=['NE_dose','NE_delta_1h','MAP','HR','RR','SpO2','temperature','lactate','pH','pO2','creatinine',
        'platelet','bilirubin','bicarbonate','wbc','hemoglobin','FiO2','PEEP','GCS','RASS','urine_numeric_record','CRRT_documented']
    indicators=['respiratory_status_observed','limitation_record','DNR_record','DNI_record','CMO_record',
        'historical_CMO_record','no_augmentation_record','no_pressors_record','full_therapy_record','limitation_conflict',
        'core_circulation_observed','core_plus_renal_observed']
    for clock,f in x['features'].items():
        a=f[f.treated.eq(1)];c=f[f.treated.eq(0)];w=1/c.groupby('case_id').case_id.transform('size')
        for var in numeric+indicators:
            va=a[var].astype(bool) if var in indicators else a[var].notna()
            vc=c[var].astype(bool) if var in indicators else c[var].notna()
            rr=dict(clock=clock,variable=var,case_n=int(va.sum()),case_den=len(a),control_n=int(vc.sum()),control_den=len(c),
                control_equal_riskset_fraction=float(np.average(vc,weights=w)),
                observed_control_patients=int(c.loc[vc,'subject'].nunique()),
                case_median=float(a[var].median()) if var in numeric and va.sum()>10 else np.nan,
                case_age_median_min=float(a.loc[va,var+'_age_min'].median()) if var+'_age_min' in a and va.sum()>10 else np.nan)
            rows.append(rr)
        cs=a.set_index('case_id');matched=c.copy()
        for var in ['NE_band','MAP_band','lactate_band']:
            matched['same_'+var]=matched[var].ne('unknown')&matched[var].eq(matched.case_id.map(cs[var]))
        one=matched.same_NE_band&matched.same_MAP_band
        two=one&matched.same_lactate_band
        for label,cmask,amask in [('same_NE_MAP',one,a.NE_dose.notna()&a.MAP.notna()),
            ('same_NE_MAP_lactate',two,a.core_circulation_observed)]:
            pool=matched[cmask].groupby('case_id').subject.nunique();n=a.case_id.map(pool).fillna(0)
            support.append(dict(clock=clock,comparison=label,case_den=len(a),case_measurements_known=int(amask.sum()),
                cases_with_ge1=int(n.ge(1).sum()),cases_with_ge5=int(n.ge(5).sum()),
                controls_median_for_known=float(n[amask].median()) if amask.any() else np.nan,
                controls_P25_for_known=float(n[amask].quantile(.25)) if amask.any() else np.nan,
                controls_P75_for_known=float(n[amask].quantile(.75)) if amask.any() else np.nan))
        for site,g in a.groupby('site'):
            row=dict(clock=clock,site=site,n=len(g))
            for v in ['NE_dose','MAP','lactate','GCS']:row[v+'_n']=int(g[v].notna().sum())
            row['limitation_n']=int(g.limitation_record.sum());row['respiratory_n']=int(g.respiratory_status_observed.sum())
            site_rows.append(row)
    x['coverage']=pd.DataFrame(rows);x['local_support']=pd.DataFrame(support);x['site_coverage']=pd.DataFrame(site_rows)
    # Count membership/outcome completeness, never outcome rates by treatment.
    h=b['hospital_summary'].set_index('hospital_stay');case=r[r.treated.eq(1)];ctrl=r[r.treated.eq(0)]
    x['outcome_completeness']=dict(cases_known=int(case.hospital.map(h.outcome).isin(['alive','expired']).sum()),cases=len(case),
        control_memberships_known=int(ctrl.hospital.map(h.outcome).isin(['alive','expired']).sum()),control_memberships=len(ctrl))
    # Measurement clocks retain exactly the same cohort, action and index.
    pd.testing.assert_frame_equal(x['features']['primary'][list(r.columns)],x['features']['strict'][list(r.columns)])
    x['summary_checks']=dict(same_members_and_indices_across_clocks=True,summary_stage_no_weight_or_outcome_fit=True,
        source_rows_deduplicated=True,no_future_lab_revision=True,no_future_careplan_status=True)
    save(b)
    published=[]
    for v in rows:
        if v['variable'] not in ['NE_dose','NE_delta_1h','MAP','HR','lactate','creatinine','pH','GCS','FiO2','respiratory_status_observed','limitation_record','core_circulation_observed','core_plus_renal_observed']:continue
        published.append(dict(clock=v['clock'],variable=v['variable'],case_pct=m.pct(v['case_n'],v['case_den']),
            controls_equal_riskset_pct=round(100*v['control_equal_riskset_fraction'],1)))
    public_support=[dict(clock=v['clock'],comparison=v['comparison'],case_measured=m.count(v['case_measurements_known']),
        cases_with_ge1=m.count(v['cases_with_ge1']),cases_with_ge5=m.count(v['cases_with_ge5']),controls_median=v['controls_median_for_known']) for v in support]
    say(dict(stage='clinical_summary',coverage=published,local_support=public_support,effects=0))


def clinical_validate():
    b=load();x=b['clinical_extension'];r=x['requests'];checks={}
    pd.testing.assert_frame_equal(r,clinical_requests(b));checks['unchanged_cases_controls_indices']=True
    assert not any(re.search('mortality|death|outcome',v,re.I) for v in r.columns)
    for name,run in x['runs'].items():
        p=clinical_path(name);assert run['parsed_to_EOF']
        assert tuple(run['signature'])==(str(p),p.stat().st_size,p.stat().st_mtime_ns)
        assert 'activeupondischarge' not in run['columns'] and 'labresulttext' not in run['columns']
    checks['source_eof_signatures_and_no_final_status']=True
    for clock,f in x['features'].items():
        pd.testing.assert_frame_equal(r,f[list(r.columns)])
        for var,contract in x['variable_contracts'].items():
            seen=f[var+'_record_seen'];age=f.loc[seen,var+'_age_min']
            assert age.gt(0).all() and age.le(contract['lookback_minutes']).all(),(clock,var)
            if clock=='strict' or contract['always_strict']:
                assert f.loc[seen,var+'_available_age_min'].gt(0).all(),(clock,var)
        assert (f.GCS.dropna().between(3,15)).all()
        assert f.FiO2.dropna().between(21,100).all()
    checks['all_selected_observations_and_required_entries_preindex']=True
    independent_branch_changes=[]
    for path,sig in x['protected'].items():
        p=Path(path)
        same=(sha(p.read_bytes())==sig) if isinstance(sig,str) else (p.stat().st_size,p.stat().st_mtime_ns)==tuple(sig)
        if not same and p.name in ['icu_data_inventory.html','phenotype_icu.pkl.gz']:
            # Independent work may update its own products concurrently. This
            # extension neither reads those products into its cohort nor writes
            # them. Preserve the initial signature and report the change.
            independent_branch_changes.append(p.name)
        else:assert same,path
    checks['MIMIC_frozen_results_and_shared_HC_HTML_unchanged']=True
    checks['independent_branch_changes_observed_not_overwritten']=independent_branch_changes
    checks['synthetic_boundary_checks']=clinical_lookup_tests()
    # Independent direct filtering at fixed request rows for two key variables.
    testids=np.unique(np.linspace(0,len(r)-1,30,dtype=int));checks['direct_source_requests']=len(testids)
    q=x['raw']['lab'];z=q[q.labname.eq('lactate')].copy()
    for i in testids:
        row=r.loc[i];g=z[z.patientunitstayid.eq(row.patientunitstayid)].copy()
        obs=pd.to_numeric(g.labresultoffset,errors='coerce');entry=pd.to_numeric(g.labresultrevisedoffset,errors='coerce')
        g=g[(obs<row.unit_index)&(obs>=row.unit_index-360)&(entry<row.unit_index)]
        seen=bool(len(g));assert seen==bool(x['features']['strict'].at[i,'lactate_record_seen'])
        if seen:
            last=pd.to_numeric(g.labresultoffset,errors='raise').max()
            assert np.isclose(row.unit_index-last,x['features']['strict'].at[i,'lactate_age_min'])
    x['validation']=checks;x['source_current']=Path(__file__).read_bytes();save(b)
    say(dict(stage='clinical_validation',checks=checks,effects=0))


CLINICAL_SCORE_CONFIG=dict(
    purpose='fixed outcome-blind initiation score diagnostic; not a new death estimate',
    reason='new pre-order covariates now observed; evaluate whether conditional weighting can balance measured states',
    estimator='reuse existing HC conditional riskset ATT implementation, fixed sum-loss L2=1; no tuning or truncation',
    eligible_sets='existing risksets with at least one eligible control; unsupported sets counted separately, not disguised as comparable',
    clocks=['primary','strict'],prior_MIMIC_outcomes_seen=True,
    numeric=['age_numeric','admission_weight','ICU_hours','hospital_hours','log_NE','NE_delta_1h','MAP','HR','RR','SpO2',
        'temperature','log_lactate','pH','pO2','log_creatinine','platelet','log_bilirubin','bicarbonate',
        'log_wbc','hemoglobin','FiO2','PEEP','GCS'],
    categorical=['gender','unittype','unitadmitsource','hospitaldischargeyear',
        'limitation_record','DNR_record','DNI_record','CMO_record','no_augmentation_record','no_pressors_record',
        'full_therapy_record','limitation_conflict','artificial_airway_resp','artificial_airway_careplan','ventilation_careplan',
        'VP_documented','EPI_documented','sedation_infusion_documented','CRRT_documented',
        'dx_nonseptic_shock_before','dx_adrenal_before','dx_airway_before'],
    diagnostic_numeric=['NE_dose','lactate','creatinine','bilirubin','wbc'],
    missing='weighted medians plus explicit indicators in existing implementation; does not identify latent dose in source-poor hospitals',
    endpoints='patient ESS, concentration, observed/missing balance, per-riskset support; no death contrast')


def clinical_score():
    b=load();x=b['clinical_extension'];assert 'validation' in x
    if 'assignment_diagnostics' not in x:
        x['assignment_diagnostics']=dict(config=CLINICAL_SCORE_CONFIG,created_utc=base().utc(),results={},
            source_at_freeze=Path(__file__).read_bytes(),outcomes_used=False)
        save(b)
    d=x['assignment_diagnostics'];assert d['config']==CLINICAL_SCORE_CONFIG
    spec=importlib.util.spec_from_file_location('hc_existing_conditional_score',ROOT/'steroid_sequential.py')
    est=importlib.util.module_from_spec(spec);spec.loader.exec_module(est)
    ref=b['patient'].set_index('patientunitstayid');m=base()
    for clock in CLINICAL_SCORE_CONFIG['clocks']:
        if clock in d['results']:say(dict(stage='clinical_score',clock=clock,reuse=True));continue
        f=x['features'][clock].copy();eligible=set(f.loc[f.treated.eq(0),'case_id'])
        unsupported=f[f.treated.eq(1)&~f.case_id.isin(eligible)].copy();f=f[f.case_id.isin(eligible)].copy()
        f['age_numeric']=f.patientunitstayid.map(ref.age_numeric)
        # The >89 category is retained as 90, a top-coded age proxy, not exact age.
        f.loc[f.patientunitstayid.map(ref.age_over89),'age_numeric']=90.
        f['admission_weight']=f.patientunitstayid.map(ref.admissionweight).where(lambda z:z.between(20,300))
        f['ICU_hours']=f.unit_index/60.;f['hospital_hours']=f['index']/60.
        for col in ['gender','unittype','unitadmitsource','hospitaldischargeyear']:f[col]=f.patientunitstayid.map(ref[col])
        for var in ['NE_dose','lactate','creatinine','bilirubin','wbc']:
            name='log_NE' if var=='NE_dose' else 'log_'+var;f[name]=np.log1p(f[var].where(f[var]>=0))
        for var in ['nonseptic_shock','adrenal','airway']:
            f['dx_'+var+'_before']=f.hospital.map(b['diagnosis_extension']['diagnosis_time_features'][var]).lt(f['index'])
        cols=['case_id','subject','hospital','treated']+CLINICAL_SCORE_CONFIG['numeric']+CLINICAL_SCORE_CONFIG['categorical']+CLINICAL_SCORE_CONFIG['diagnostic_numeric']
        # Discharge year is a calendar label, not discharge time/status or outcome.
        frame=f[cols].copy();assert not any(re.search('death|mortality|status$',v) for v in cols)
        plan=est.make_plan(frame,CLINICAL_SCORE_CONFIG['numeric'],CLINICAL_SCORE_CONFIG['categorical'],
            patient_col='subject',admission_col='hospital',diagnostic_numeric=CLINICAL_SCORE_CONFIG['diagnostic_numeric'])
        say(dict(stage='clinical_score_start',clock=clock,cases=m.count(plan['n_groups']),
            numeric=len(CLINICAL_SCORE_CONFIG['numeric']),categorical=len(CLINICAL_SCORE_CONFIG['categorical']),outcomes_used=False))
        fit=est.fit_score(plan,l2=1.);weights=est.att_weights(plan,fit['linear_predictor'],fit['mult'])
        bal=est.balance_diagnostics(plan,weights);conc,per=est.concentration_diagnostics(plan,weights)
        local=est.local_similarity_diagnostics(plan,weights)
        d['results'][clock]=dict(fit=fit['fit'],beta=fit['beta'],linear_predictor=fit['linear_predictor'],
            plan_sha256=plan['plan_sha256'],balance=bal,concentration=conc,per_case=per,local=local,
            unsupported_sets=unsupported[['case_id','hospital','subject','site']],n_cases=plan['n_groups'],
            estimator_source_sha256=sha((ROOT/'steroid_sequential.py').read_bytes()))
        d['models_fitted']=len(d['results']);save(b)
        say(dict(stage='clinical_score_done',clock=clock,fit_seconds=round(fit['fit']['seconds'],1),
            converged=fit['fit']['success'],warnings=fit['fit']['warnings'],
            maximum_abs_SMD=float(bal.smd_after.abs().max()),control_patient_ESS=round(conc['control_patient']['ess'],1),
            largest_patient_share=round(conc['control_patient']['maximum_share'],4),
            per_set_control_ESS_median=round(per.control_patient_ess.median(),1),effects=0))
    b['models_fitted']=len(d['results'])+sum(len(z['diagnostics']['results']) for z in x.get('assignment_diagnostics_history',[]))
    b['effects_estimated']=x.get('order_estimation',{}).get('death_contrasts',0);x['source_current']=Path(__file__).read_bytes();save(b)


def clinical_publish():
    b=load();x=b['clinical_extension'];m=base();assert 'validation' in x
    fits=x['assignment_diagnostics']['results'];assert set(fits)=={'primary','strict'}
    cov=x['coverage'].set_index(['clock','variable']);r=x['requests'];a=r[r.treated.eq(1)];c=r[r.treated.eq(0)]
    def percentage(var,clock='primary'):
        z=cov.loc[(clock,var)];p=m.pct(z.case_n,z.case_den)
        return str(p)+'%' if p!='' else '比例抑制（小单元保护）'
    def controls(var,clock='primary'):
        z=cov.loc[(clock,var)]
        if m.pct(z.control_n,z.control_den)=='':return '比例抑制'
        return f'{100*z.control_equal_riskset_fraction:.1f}%'
    marker='<!-- HC_EICU_CLINICAL_START -->';endmarker='<!-- HC_EICU_CLINICAL_END -->'
    lines=[marker,'### 2026-10-31：新增临床表后的HC双库实测与决定','',
        '**决定：继续HC医嘱启动版的跨库补充，当前不需要追加核心下载。** 已将新增临床表用于现有HC候选及其同院对照，完成治疗前测量、时间核查和固定治疗分配加权。两种时钟下观测平衡与患者权重集中均有可解释的结果；NE剂量的医院性缺失和医嘱—实际给药差异仍需在补充分析中单独呈现。本轮没有计算新的死亡风险或bootstrap，原MIMIC实际首剂主结果不变。','',
        '本节更新前文“临床表尚未取得”的历史状态。这里的分母是HC研究的候选和风险集，不是独立世界模型分支的2,000人检查样本，也不使用其覆盖率代替本研究结果。F盘入口下的eICU链接仍指向E盘原文件，未搬移或删除。','',
        '|实际完成的内容|结果|','|---|---|',
        '|原HC医嘱候选 / 医院|约'+m.count(len(a))+' / 约'+m.count(a.site.nunique())+'|',
        '|至少有同院、同NE经过时间对照的节点|约'+m.count(fits['primary']['n_cases'])+'|',
        '|没有候选对照的节点|'+m.count(len(fits['primary']['unsupported_sets']))+'，保留在流程分母，不伪造可比性|',
        '|独立对照患者 / 对照风险集成员记录|约'+m.count(c.subject.nunique())+' / 约'+m.count(len(c))+'|',
        '|新解析的临床来源|lab、vitalPeriodic、vitalAperiodic、nurseCharting、respiratoryCharting、respiratoryCare、carePlanGeneral、treatment、infusionDrug、intakeOutput|',
        '|新增死亡效果 / bootstrap|0 / 0|','',
        '候选与对照仍使用之前保存的感染记录、NE正速率及既往激素规则；不按未来HC执行、最终取消、死亡或治疗成功筛人。既有小部分无同院对照节点不进入可计算的条件加权诊断，其目标范围因此限于有支持的医嘱启动节点。','',
        '#### 治疗前信息：按变量使用时间窗','',
        '|变量 / 记录|HC节点：测量时钟|HC节点：严格录入时钟|对照：测量时钟覆盖*|',
        '|---|---:|---:|---:|']
    names=[('NE_dose','当前NE可解释剂量，60分钟内'),('NE_delta_1h','NE过去1小时变化（至少两次、跨15分钟）'),
        ('MAP','MAP，1小时'),('HR','心率，1小时'),('SpO2','SpO₂，1小时'),
        ('lactate','乳酸，6小时'),('pH','pH，6小时'),('creatinine','肌酐，24小时'),
        ('platelet','血小板，24小时'),('bilirubin','胆红素，24小时'),
        ('FiO2','FiO₂，4小时'),('GCS','总GCS，4小时'),('respiratory_status_observed','气道/通气状态记录，4小时'),
        ('limitation_record','此前治疗限制相关条目'),('core_circulation_observed','NE剂量＋MAP＋乳酸同时可得'),
        ('core_plus_renal_observed','上述三项再加肌酐')]
    for v,label in names:lines.append('|'+label+'|'+percentage(v)+'|'+percentage(v,'strict')+'|'+controls(v)+'|')
    lines+=['',
        '*对照先在各风险集内均分权重，再对风险集等权汇总；没有把反复参与或大医院产生的大量行当作独立患者。此表为加权模型之前的覆盖率。所有窗口在查看本轮分组结局前固定；是本地操作化选择，不称为指南规定。','',
        '两时钟使用完全相同的患者、医嘱起点与风险集。检验均要求采样和修订录入严格早于索引；主口径床旁观测用记录测量时刻，严格口径同时要求录入在前。修订/录入时间不是已经证实的医生首次可见时刻。eICU监护表本身只有一个offset，不能据两口径相同就认定没有晚录；5分钟监护汇总也不是分钟级原始波形。GCS覆盖由'+percentage('GCS')+'降至'+percentage('GCS','strict')+'，反映时间口径敏感性，而非改变入组。','',
        '治疗限制采用同次住院、索引前的结构化条目；DNR、DNI、CMO、不得追加治疗与full therapy分别保留，未知不是full code。该表转科时可能沿用记录，90.7%的条目覆盖不代表90.7%经过即时临床核验。气道/通气记录只表示过去4小时有明确状态，缺记录不当作未通气；ventEndOffset=0不用于推导停机。','',
        '本次尿量只恢复最近的候选数字记录，未确定每条的收集时段及累计含义，不能直接生成6小时尿量率或SOFA分值。CRRT采用treatment的结构化治疗记录，仍是记录证据。字典复查发现初版漏匹配“C V V H / C V V H D”中的空格，已修复并重跑相同诊断；原映射和原诊断留在同一缓存。没有把未记录CRRT编码成未治疗。','',
        '来源：[lab](https://eicu.mit.edu/eicutables/lab/)、[vitalPeriodic](https://eicu.mit.edu/eicutables/vitalperiodic/)、[nurseCharting](https://eicu.mit.edu/eicutables/nursecharting/)、[respiratoryCharting](https://eicu.mit.edu/eicutables/respiratorycharting/)、[carePlanGeneral](https://eicu.mit.edu/eicutables/careplangeneral/)、[intakeOutput](https://eicu.mit.edu/eicutables/intakeoutput/)。','',
        '#### 固定条件加权：实际运行结果','',
        '复用MIMIC既有条件风险集ATT实现：启动者每个节点贡献1，对照按条件启动分数在同风险集内归一到1；不是ATO或简单全库PS匹配。固定23项连续概念及22项分类/记录状态，L2=1，不搜索模型、不截断选择最优结果。数值预处理含中位数及缺失指标，作用是保持观测样本和记录状态，不能恢复无来源医院的真实剂量。患者ESS将重复住院/风险集贡献先合并至患者后计算，不能拿成员行ESS当患者信息量。','',
        '|诊断|测量时钟|严格录入时钟|','|---|---:|---:|']
    def getfit(k,fun):return fun(fits[k])
    diag=[('最大绝对SMD（含原值、变换值与缺失状态）',lambda z:f"{z['balance'].smd_after.abs().max():.3f}"),
        ('启动患者ESS',lambda z:f"{z['concentration']['treated_patient']['ess']:.1f}"),
        ('对照患者ESS',lambda z:f"{z['concentration']['control_patient']['ess']:.1f}"),
        ('最大单个对照患者权重份额',lambda z:f"{100*z['concentration']['control_patient']['maximum_share']:.2f}%"),
        ('前10名对照患者累计份额',lambda z:f"{100*z['concentration']['control_patient']['top10_share']:.2f}%"),
        ('单风险集对照患者ESS中位数',lambda z:f"{z['per_case'].control_patient_ess.median():.1f}"),
        ('单风险集ESS不足5的节点（描述，非排除）',lambda z:'约'+m.count(z['per_case'].control_patient_ess.lt(5).sum())),
        ('优化收敛 / 警告',lambda z:str(z['fit']['success'])+' / '+str(len(z['fit']['warnings'])))]
    for label,fn in diag:lines.append('|'+label+'|'+getfit('primary',fn)+'|'+getfit('strict',fn)+'|')
    lines+=['',
        '主口径最大残余SMD来自已测NE剂量的log1p值，约由0.765降至0.084；本轮没有发现单个患者支配总体权重。此结果支持继续估计观测可比的医嘱启动关联，但不证明无未测混杂。两次优化均无警告；CRRT解析修正前后的诊断同时保留，未按结果方向挑选。','',
        '#### 整体平衡以外，仍有哪些真实限制','',
        '|固定支持描述|节点约数|','|---|---:|']
    ss=x['local_support'];ss=ss[ss.clock.eq('primary')].set_index('comparison')
    for key,label in [('same_NE_MAP','NE剂量与MAP均有记录'),('same_NE_MAP_lactate','再要求乳酸有记录')]:
        z=ss.loc[key];lines.append('|'+label+'：有测量 / 至少一个同粗分层对照|'+m.count(z.case_measurements_known)+' / '+m.count(z.cases_with_ge1)+'|')
    local=fits['primary']['local'];known=local.case_NE_observed_positive&local.case_MAP_observed
    mass=local.loc[known,'close_NE_MAP_control_mass']
    lines+=['',
        '粗分层仅用既有NE剂量段、MAP及乳酸区间描述，未作为删选条件。另一个既定邻近状态描述采用NE剂量相差不超过2倍、MAP相差不超过15mmHg；在启动者两值均已知的节点中，符合此描述的对照权重份额中位数为'+f'{100*mass.median():.1f}%'+'。它不是临床可互换性的金标准，但提示总体SMD较好不代表每个节点都以相似状态对照为主。约'+m.count(mass.eq(0).sum())+'个此类已知节点没有这样的对照。不能只展示总体SMD后宣布所有分支支持充分。','',
        '既有NE缺失结构仍成立：61.5%的HC节点剂量可解释；29.8%的全部节点来自本队列HC节点均缺剂量的医院，这些医院占缺剂量节点的77.5%。新补的lab/vitals不能补出未记录的浓度。NE变化在1小时窗内只有'+percentage('NE_delta_1h')+'可计算，与输注记录较稀疏有关，不能推断其他人NE未上调。没有采用标准药袋、固定80kg体重或重复盐基换算。','',
        '院内结局可以通过住院键关联；HC组仍有1–10例结局未知/不一致（小单元抑制），对照成员记录结局可确定比例约99.4%。本轮只核查完整性，不读取分组死亡率。后续不得把未知填为存活，也不得将ICU转出当医院随访终点。','',
        '#### 与MIMIC桥接及下一步的具体决定','',
        'MIMIC现有POE桥接已再次只读核查：约230个合格首医嘱，60分钟内有HC实际执行确认的比例为50.6%，6小时内79.0%。该样本局限既有风险集住院；不表示全库医嘱启动人群，也不证明eICU执行率相同。首医嘱选取没有要求未来执行，窗口未确认者没有全部归为未治疗。原主结果与主缓存签名仍一致。','',
        '当前为**Conditional go：进入有限的医嘱版补充比较，而非已通过同协议外部验证**。执行范围是保留MIMIC实际首剂主分析，另以医嘱起点独立建立MIMIC桥接及eICU补充估计；两库先分别报告，不拼接不同版本患者键，不直接合并实际给药与医嘱效应。','',
        '后续只需落实两件事：①在查看eICU死亡比较前，写定缺剂量医院及剂量已测子集的解释和并列敏感性，展示目标人群改变、局部支持及大医院影响，不能以中位数补值或漂亮SMD代替；②将明确限制治疗条目、未知住院结局和两种时间口径的处理与MIMIC医嘱映射一起固定，再估计风险/RD/RR及患者聚类区间。既有MIMIC死亡结果已经看过，新增分析不重新称为完全预设。','',
        '这轮需要用户新下载的文件：**无**。剩余问题主要是记录定义和可识别范围，不是缺少这10张表。无需因未到齐31/31张表而下载整个数据库；如果下一项具体判断确实需要未取得字段，再列精确用途。eICU只有医嘱而缺可靠执行链的问题，也不能通过随意增加另一张一般临床表宣称解决。','',
        '#### 成本、复现与保留','',
        '|步骤|实际成本|','|---|---:|',
        '|一次定向解析10表|'+f"{sum(v['signature'][1] for v in x['runs'].values())/2**30:.2f} GiB"+'|',
        '|读取与初步解析耗时（不含保存）|'+f"{sum(v['seconds'] for v in x['runs'].values())/60:.1f}分钟"+'|',
        '|扫描峰值进程内存|'+f"{max(v['peak_MB'] for v in x['runs'].values()):.1f} MB"+'|',
        '|36项时点特征生成|'+f"{x['feature_seconds']/60:.1f}分钟"+'|',
        '|两口径固定分配模型实际拟合|'+f"{sum(z['fit']['seconds'] for z in fits.values()):.1f}秒"+'|','',
        '来源均解析至EOF，记录字段、文件签名与流式SHA256。6项时间/修订边界测试、30个固定查询的独立源行复查及CRRT字典正反例检查通过；每个风险集权重总量与启动者质量一致。数据检查只保证所述实现，不代替临床确认。','',
        '入口：`steroid_eicu.py clinical-plan / clinical-project / clinical-features / clinical-summary / clinical-validate / clinical-score / clinical-publish`。上述顺序支持缓存复用；`clinical-fix-crrt`记录本轮一次解析修正的历史，不是新模型搜索。只更新本入口、原私有缓存和本报告。MIMIC冻结结果CSV、主缓存及共享HC会议HTML保持不变；独立世界模型分支的inventory/缓存在工作期间有外部更新，未覆盖或将其当作本研究输入。计数十位取整，小单元及可重建其数量的比例抑制。',
        '',endmarker,'']
    path=ROOT/'steroid_tte.md';before=path.read_bytes();old=before.decode('utf-8');nl='\r\n' if '\r\n' in old else '\n';old=old.replace('\r\n','\n')
    if marker in old:
        start=old.index(marker);end=old.index(endmarker,start)+len(endmarker)
        while end<len(old) and old[end]=='\n':end+=1
    else:start=end=old.index('### 实际代码与证据位置')
    out=old[:start]+'\n'.join(lines)+'\n'+old[end:]
    legacy='**仍需既定数据，不新增数据库：**'
    out=out.replace(legacy,'**2026-10-29时的待下载清单（已由下方2026-10-31实测更新）：**',1)
    after=out.replace('\n',nl).encode('utf-8')
    if after==before:say(dict(stage='clinical_publish',reuse=True));return
    assert path.read_bytes()==before
    history=b.setdefault('report_history',[])
    if not any(v['sha256']==sha(before) for v in history):history.append(dict(sha256=sha(before),bytes=before))
    x['publication']=dict(before_hash=sha(before),report_hash=sha(after),source_hash=sha(Path(__file__).read_bytes()),effects=0)
    x['source_current']=Path(__file__).read_bytes();save(b);path.write_bytes(after)
    say(dict(stage='clinical_publish',report=str(path),effects=0))


ORDER_ESTIMATION_CONFIG=dict(
    version='eICU_HC_order_initiation_20261031',
    question='recorded IV HC order initiation now versus not yet ordered at the same observed NE elapsed time, then usual care',
    target='ATT of supported recorded HC-order initiation states; exploratory weighted association, not proven administration effect',
    prior_MIMIC_results_seen=True,eICU_death_contrasts_seen_at_freeze=False,
    cohort='existing primary diagnosis-supported HC first-order cases and complete same-hospital-site risksets',
    limitation_rule='exclude a node with pre-index latest CMO, no vasopressors/inotropes, or no augmentation record, including conflicting full-therapy entry; retain DNR/DNI and unknown',
    no_future_selection='no later death, execution, cancellation, clinical response, or survival-through-window rule',
    time_boundary='index strictly before ICU and hospital exit; same-time exits are not treated as an observable ongoing ICU decision',
    followup='from order or aligned control index through this hospital discharge; alive discharge=0; ICU exit does not end followup',
    endpoint='consistent hospital discharge alive/expired; missing or conflict remains unknown',
    missing_endpoint='retain all weights and risksets; reference risk renormalized over observed endpoints separately by arm; report full-population worst-case RD/RR bounds',
    missing_endpoint_assumption='reference point requires weighted outcome-observed members to represent unknown endpoints; bounds do not make that assumption',
    covariates=CLINICAL_SCORE_CONFIG,
    scenarios=['primary','strict','known_NE'],
    known_NE='measurement clock; require interpretable positive NE dose in cases and controls, recompute supported risksets and all weights; distinct selected target population',
    primary_selection='retain primary regardless of direction, precision, or relative diagnostic appearance; all scenarios reported',
    hospital_sensitivity='leave out largest initiating-site by case count determined before outcomes; no retuning; point estimate only',
    bootstrap_repetitions=100,bootstrap_seed=103151,
    bootstrap_rule='patient resampling feasibility checked before outcome fitting; all admissions and riskset roles share patient weights; preprocessing and score refitted every draw',
    uncertainty='100 normalized Exp(1) positive patient-weight draws (Dirichlet Bayesian bootstrap); percentile uncertainty intervals conditional on observed patients/sites, not ordinary resampling CIs',
    precision='pilot intervals; no formal effect claim or publication promise from 100 resamples',
    raw_read=False)


def order_estimator():
    spec=importlib.util.spec_from_file_location('hc_order_conditional_score',ROOT/'steroid_sequential.py')
    est=importlib.util.module_from_spec(spec);spec.loader.exec_module(est);return est


def order_covariates(b,clock):
    f=b['clinical_extension']['features'][clock].copy();ref=b['patient'].set_index('patientunitstayid')
    f['age_numeric']=f.patientunitstayid.map(ref.age_numeric)
    f.loc[f.patientunitstayid.map(ref.age_over89),'age_numeric']=90.
    f['admission_weight']=f.patientunitstayid.map(ref.admissionweight).where(lambda z:z.between(20,300))
    f['ICU_hours']=f.unit_index/60.;f['hospital_hours']=f['index']/60.
    for col in ['gender','unittype','unitadmitsource','hospitaldischargeyear']:f[col]=f.patientunitstayid.map(ref[col])
    for var in ['NE_dose','lactate','creatinine','bilirubin','wbc']:
        name='log_NE' if var=='NE_dose' else 'log_'+var;f[name]=np.log1p(f[var].where(f[var]>=0))
    for var in ['nonseptic_shock','adrenal','airway']:
        f['dx_'+var+'_before']=f.hospital.map(b['diagnosis_extension']['diagnosis_time_features'][var]).lt(f['index'])
    return f


def order_plan(frame,est):
    cfg=CLINICAL_SCORE_CONFIG
    columns=['case_id','subject','hospital','treated']+cfg['numeric']+cfg['categorical']+cfg['diagnostic_numeric']
    assert not any(re.search('death|mortality|outcome',v,re.I) for v in columns)
    return est.make_plan(frame[columns],cfg['numeric'],cfg['categorical'],patient_col='subject',
        admission_col='hospital',diagnostic_numeric=cfg['diagnostic_numeric'])


def order_prepare():
    b=load();x=b['clinical_extension'];m=base();est=order_estimator()
    if 'order_estimation' not in x:
        x['order_estimation']=dict(config=ORDER_ESTIMATION_CONFIG,created_utc=m.utc(),source_at_freeze=Path(__file__).read_bytes(),
            scenarios={},raw_scans=0,death_contrasts=0,results={})
        save(b)
    d=x['order_estimation']
    if d['config']!=ORDER_ESTIMATION_CONFIG:
        assert not d.get('prepared') and d['death_contrasts']==0
        old=d['config'].copy();allowed={'time_boundary','uncertainty'}
        assert all(old.get(k)==v for k,v in ORDER_ESTIMATION_CONFIG.items() if k not in allowed)
        d.setdefault('amendments',[]).append(dict(created_utc=m.utc(),previous_config=old,
            reason='Before outcomes: source-index boundary check identified exit-tied controls; outcome-blind ordinary patient resampling produced 76/100 empty-pool failures. Define half-open eligibility and positive patient-weight uncertainty without dropping failed draws.',
            death_contrasts_seen=False))
        d['config']=ORDER_ESTIMATION_CONFIG;save(b)
    if d.get('prepared'):say(dict(stage='order_prepare',reuse=True));return
    ref=b['patient'].set_index('patientunitstayid')
    for scenario in ORDER_ESTIMATION_CONFIG['scenarios']:
        f=order_covariates(b,'strict' if scenario=='strict' else 'primary');original=f.copy()
        limit=f.CMO_record|f.no_pressors_record|f.no_augmentation_record
        f=f[~limit].copy()
        if scenario=='known_NE':f=f[f.NE_dose.notna()&f.NE_dose.gt(0)].copy()
        end=f.patientunitstayid.map(ref.hospital_duration);ue=f.patientunitstayid.map(ref.unitdischargeoffset)
        bad_time=~end.gt(f['index'])|~ue.gt(f.unit_index)
        time_excluded=f.loc[bad_time,['case_id','hospital','subject','treated']].copy();f=f[~bad_time].copy()
        validcase=set(f.loc[f.treated.eq(1),'case_id']);validcontrol=set(f.loc[f.treated.eq(0),'case_id'])
        support=validcase&validcontrol;f=f[f.case_id.isin(support)].reset_index(drop=True)
        # Observe source-unit/hospital bounds before attaching any endpoint label.
        end=f.patientunitstayid.map(ref.hospital_duration)
        assert (end>f['index']).all(),'HOSPITAL_FOLLOWUP_TIME_CONFLICT'
        assert (f.unit_index<f.patientunitstayid.map(ref.unitdischargeoffset)).all()
        g=f.groupby('case_id');assert g.site.nunique().eq(1).all() and g.NE_elapsed_min.nunique().eq(1).all()
        plan=order_plan(f,est);rng=np.random.RandomState(ORDER_ESTIMATION_CONFIG['bootstrap_seed'])
        failures={}
        for i in range(100):
            try:est.multiplicities(plan,est.draw_patient_counts(plan,rng))
            except est.EstimatorFailure as exc:failures[str(exc)]=failures.get(str(exc),0)+1
        aa=f[f.treated.eq(1)];cc=f[f.treated.eq(0)]
        d['scenarios'][scenario]=dict(frame=f,plan_sha256=plan['plan_sha256'],
            flow=dict(original_cases=int(original.treated.sum()),restriction_cases=int((limit&original.treated.eq(1)).sum()),
                original_control_rows=int(original.treated.eq(0).sum()),restriction_control_rows=int((limit&original.treated.eq(0)).sum()),
                case_nodes=len(aa),case_patients=aa.subject.nunique(),control_patients=cc.subject.nunique(),control_rows=len(cc),sites=aa.site.nunique(),
                eligible_cases_without_controls=len(validcase-validcontrol),time_boundary_cases=int(time_excluded.treated.sum()),
                time_boundary_control_rows=int(time_excluded.treated.eq(0).sum())),ordinary_resampling_failures=failures,
            largest_site=aa.groupby('site').size().sort_values(ascending=False,kind='stable').index[0])
        say(dict(stage='order_prepare',scenario=scenario,cases=m.count(len(aa)),controls=m.count(cc.subject.nunique()),
            clinical_restriction_cases=m.count((limit&original.treated.eq(1)).sum()),
            ordinary_patient_resample_failed=sum(failures.values()),attempted=100,reason=failures,outcomes_used=False))
    # Outcomes have not entered preparation, model choice or this resampling check.
    d['prepared']=True;d['source_current']=Path(__file__).read_bytes();save(b)


def order_positive_multiplicities(plan,patient_weight):
    v=np.asarray(patient_weight,dtype=float)
    assert v.shape==(len(plan['patient_ids']),) and np.isfinite(v).all() and (v>0).all()
    rows=v[plan['patient']];cases=rows[plan['case_rows']];controls=np.where(plan['treated'],0.,rows)
    return dict(patient=v,row=rows,case=cases,controls=controls,pool=np.add.reduceat(controls,plan['starts']),
        likelihood=np.where(plan['treated'],1.,rows))


def order_fit_positive(est,plan,patient_weight):
    from scipy.optimize import minimize
    start=time.perf_counter();mult=order_positive_multiplicities(plan,patient_weight)
    matrix,prep=est.preprocess(plan,mult);matrix=np.asarray(matrix,dtype=np.float64)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        opt=minimize(est.conditional_loss_gradient,np.zeros(matrix.shape[1]),args=(matrix,plan,mult,1.),
            jac=True,method='L-BFGS-B',options=dict(maxiter=600,ftol=1e-11,gtol=1e-6,maxls=40))
    if not opt.success:raise est.EstimatorFailure('POSITIVE_WEIGHT_SCORE_NONCONVERGENCE: '+str(opt.message))
    return dict(linear_predictor=est._matvec(matrix,opt.x),mult=mult,beta=opt.x,
        fit=dict(success=True,iterations=int(opt.nit),maximum_abs_gradient=float(np.abs(opt.jac).max()),
            warnings=[str(w.message) for w in ws],seconds=time.perf_counter()-start),preprocessing=prep)


def order_endpoint_summary(plan,weights,y_original):
    y=np.asarray(y_original,float)[plan['order']];known=np.isin(y,[0.,1.]);ysafe=np.where(known,y,0.)
    out={}
    for name,key in [('initiated','treated'),('not_yet','control')]:
        w=weights[key];total=w.sum();observed=w[known].sum()
        if total<=0 or observed<=0:raise ValueError('NO_OBSERVED_WEIGHTED_ENDPOINT')
        deaths=np.dot(w,ysafe);missing=total-observed
        out[name+'_risk']=float(deaths/observed)
        out[name+'_risk_lower']=float(deaths/total);out[name+'_risk_upper']=float((deaths+missing)/total)
        out[name+'_unknown_weight_fraction']=float(missing/total)
    out['risk_difference']=out['initiated_risk']-out['not_yet_risk']
    out['risk_ratio']=out['initiated_risk']/out['not_yet_risk'] if out['not_yet_risk']>0 else np.nan
    out['RD_missing_lower']=out['initiated_risk_lower']-out['not_yet_risk_upper']
    out['RD_missing_upper']=out['initiated_risk_upper']-out['not_yet_risk_lower']
    out['RR_missing_lower']=out['initiated_risk_lower']/out['not_yet_risk_upper'] if out['not_yet_risk_upper']>0 else np.nan
    out['RR_missing_upper']=out['initiated_risk_upper']/out['not_yet_risk_lower'] if out['not_yet_risk_lower']>0 else np.nan
    return out


def order_estimation_tests(est):
    toy=pd.DataFrame(dict(case_id=[1,1,1,2,2,2],subject=['a','b','c','b','a','d'],hospital=[1,2,3,2,1,4],
        treated=[1,0,0,1,0,0],severity=[1.,2.,np.nan,2.,1.,3.]))
    p=est.make_plan(toy,['severity'],patient_col='subject',admission_col='hospital')
    ordinary=est.fit_score(p);positive=order_fit_positive(est,p,np.ones(len(p['patient_ids'])))
    assert np.allclose(ordinary['beta'],positive['beta'],atol=1e-9)
    pw=np.arange(1,len(p['patient_ids'])+1,dtype=float);mult=order_positive_multiplicities(p,pw)
    for j in range(len(pw)):assert np.allclose(mult['row'][p['patient']==j],pw[j])
    ww=est.att_weights(p,positive['linear_predictor'],mult)
    assert np.allclose(np.add.reduceat(ww['control'],p['starts']),mult['case'])
    yt=np.array([1.,0.,np.nan,0.,1.,0.]);v=order_endpoint_summary(p,ww,yt)
    assert v['RD_missing_lower']<=v['risk_difference']<=v['RD_missing_upper']
    assert v['initiated_unknown_weight_fraction']==0 and v['not_yet_unknown_weight_fraction']>0
    known=order_endpoint_summary(p,ww,np.nan_to_num(yt))
    assert np.isclose(known['RD_missing_lower'],known['RD_missing_upper'])
    return dict(uniform_weights_match_existing_fit=True,shared_patient_roles=True,within_set_mass=True,
        missing_endpoint_bounds=True,complete_endpoint_bound_collapse=True)


def order_estimate():
    b=load();x=b['clinical_extension'];d=x['order_estimation'];m=base();est=order_estimator()
    assert d['prepared'] and d['config']==ORDER_ESTIMATION_CONFIG
    checks=order_estimation_tests(est);d['estimator_tests']=checks
    h=b['hospital_summary'].set_index('hospital_stay');save(b)
    for scenario in ORDER_ESTIMATION_CONFIG['scenarios']:
        data=d['scenarios'][scenario];f=data['frame'];plan=order_plan(f,est)
        assert plan['plan_sha256']==data['plan_sha256']
        y=f.hospital.map(h.outcome).map({'alive':0.,'expired':1.}).to_numpy(float)
        if scenario not in d['results']:
            say(dict(stage='order_estimate_start',scenario=scenario,case_nodes=m.count(plan['n_groups']),raw_bytes=0,
                question='fixed order-based weighted hospital-death association, with unknown-endpoint bounds'))
            fit=est.fit_score(plan,l2=1.);ww=est.att_weights(plan,fit['linear_predictor'],fit['mult'])
            point=order_endpoint_summary(plan,ww,y)
            unweighted=dict(ww,control=ww['control_reference']);crude=order_endpoint_summary(plan,unweighted,y)
            bal=est.balance_diagnostics(plan,ww);conc,per=est.concentration_diagnostics(plan,ww)
            state=dict(point=point,crude=crude,fit=fit['fit'],balance=bal,concentration=conc,per_case=per,
                local=est.local_similarity_diagnostics(plan,ww),bootstrap=[],source_at_estimate=Path(__file__).read_bytes())
            # Record aggregate complete/unknown endpoints without printing rare cells.
            state['endpoint_counts']={str(arm):dict(nodes=int(f.treated.eq(arm).sum()),
                known=int((f.treated.eq(arm)&np.isfinite(y)).sum()),deaths=int((f.treated.eq(arm)&(y==1)).sum())) for arm in [0,1]}
            d['results'][scenario]=state;d['death_contrasts']+=1;save(b)
            say(dict(stage='order_point',scenario=scenario,risk_initiated=round(point['initiated_risk'],4),
                risk_not_yet=round(point['not_yet_risk'],4),RD_pp=round(100*point['risk_difference'],3),RR=round(point['risk_ratio'],3),
                max_SMD=round(float(bal.smd_after.abs().max()),3)))
        state=d['results'][scenario];rng=np.random.RandomState(ORDER_ESTIMATION_CONFIG['bootstrap_seed'])
        if 'rng_state' in state:rng.set_state(state['rng_state'])
        started=time.perf_counter()
        for rep in range(len(state['bootstrap']),ORDER_ESTIMATION_CONFIG['bootstrap_repetitions']):
            row=dict(replicate=rep,success=False)
            try:
                v=rng.exponential(1.,len(plan['patient_ids']));v=v*len(v)/v.sum()
                ff=order_fit_positive(est,plan,v);weights=est.att_weights(plan,ff['linear_predictor'],ff['mult'])
                row.update(order_endpoint_summary(plan,weights,y));row.update(success=True,warnings=ff['fit']['warnings'],
                    iterations=ff['fit']['iterations'],fit_seconds=ff['fit']['seconds'])
            except (est.EstimatorFailure,FloatingPointError,RuntimeError,ValueError) as exc:
                row.update(error_type=type(exc).__name__,error=str(exc))
            state['bootstrap'].append(row);state['rng_state']=rng.get_state()
            if (rep+1)%10==0:
                say(dict(stage='order_bootstrap_progress',scenario=scenario,attempted=rep+1,
                    failures=sum(not z['success'] for z in state['bootstrap']),seconds=round(time.perf_counter()-started)))
            if (rep+1)%25==0:save(b)
        bt=pd.DataFrame(state['bootstrap']);ok=bt[bt.success]
        state['uncertainty']={key:dict(low=float(ok[key].quantile(.025)),high=float(ok[key].quantile(.975)))
            for key in ['initiated_risk','not_yet_risk','risk_difference','risk_ratio','RD_missing_lower','RD_missing_upper']}
        state['bootstrap_failures']=int((~bt.success).sum());state['bootstrap_warning_draws']=int(sum(bool(z.get('warnings')) for z in state['bootstrap']))
        # 100 draws are a pilot; do not label endpoint variability as converged.
        state['interval_label']='100-draw patient positive-weight bootstrap percentile interval; preliminary precision'
        save(b);say(dict(stage='order_estimate_done',scenario=scenario,point=state['point'],intervals=state['uncertainty'],
            failures=state['bootstrap_failures'],warnings=state['bootstrap_warning_draws']))
    if 'largest_site_exclusion' not in d:
        f=d['scenarios']['primary']['frame'];site=d['scenarios']['primary']['largest_site'];z=f[f.site.ne(site)].reset_index(drop=True)
        p=order_plan(z,est);fit=est.fit_score(p);w=est.att_weights(p,fit['linear_predictor'],fit['mult'])
        y=z.hospital.map(h.outcome).map({'alive':0.,'expired':1.}).to_numpy(float)
        d['largest_site_exclusion']=dict(removed_site=site,remaining_cases=p['n_groups'],point=order_endpoint_summary(p,w,y),
            balance=est.balance_diagnostics(p,w),concentration=est.concentration_diagnostics(p,w)[0],fit=fit['fit'],
            meaning='one prespecified site influence analysis, changes target population; not external validation')
        d['death_contrasts']+=1;save(b)
    b['effects_estimated']=d['death_contrasts'];d['source_current']=Path(__file__).read_bytes();save(b)


def order_validate():
    """Check fixed estimates against direct risk calculations; no new analysis choices."""
    b=load();x=b['clinical_extension'];d=x['order_estimation'];est=order_estimator()
    assert d['prepared'] and d['config']==ORDER_ESTIMATION_CONFIG
    assert set(d['results'])==set(ORDER_ESTIMATION_CONFIG['scenarios'])
    h=b['hospital_summary'].set_index('hospital_stay');ref=b['patient'].set_index('patientunitstayid')
    checks=dict(raw_scans=d['raw_scans'],scenarios={},protected_HC_unchanged=True)
    for name,data in d['scenarios'].items():
        f=data['frame'];p=order_plan(f,est);s=d['results'][name]
        assert p['plan_sha256']==data['plan_sha256']
        assert (f.unit_index<f.patientunitstayid.map(ref.unitdischargeoffset)).all()
        assert (f['index']<f.patientunitstayid.map(ref.hospital_duration)).all()
        assert not (f.CMO_record|f.no_pressors_record|f.no_augmentation_record).any()
        assert f.groupby('case_id').site.nunique().eq(1).all()
        assert f.groupby('case_id').NE_elapsed_min.nunique().eq(1).all()
        fit=est.fit_score(p,l2=1.);w=est.att_weights(p,fit['linear_predictor'],fit['mult'])
        y=f.hospital.map(h.outcome).map({'alive':0.,'expired':1.}).to_numpy(float)
        v=order_endpoint_summary(p,w,y)
        assert all(np.isclose(v[k],s['point'][k],atol=1.e-12) for k in v)
        # Independently normalize relative initiation scores within each riskset.
        cw=np.zeros(p['n']);tw=np.zeros(p['n']);tw[p['case_rows']]=1.
        for lo,hi in zip(p['starts'],p['ends']):
            idx=np.arange(lo,hi)[~p['treated'][lo:hi]];eta=fit['linear_predictor'][idx]
            q=np.exp(eta-eta.max());cw[idx]=q/q.sum()
        assert np.allclose(cw,w['control'],atol=1.e-12) and np.array_equal(tw,w['treated'])
        yy=y[p['order']];known=np.isfinite(yy)
        t=np.dot(tw[known],yy[known])/tw[known].sum();c=np.dot(cw[known],yy[known])/cw[known].sum()
        assert np.isclose(t-c,s['point']['risk_difference'],atol=1.e-12)
        bt=pd.DataFrame(s['bootstrap']);assert list(bt.replicate)==list(range(100))
        assert int((~bt.success).sum())==s['bootstrap_failures']
        # This is Monte Carlo description, not a new interval or stopping rule.
        halves=[]
        for lo,hi in [(0,50),(50,100)]:
            z=bt.iloc[lo:hi];z=z[z.success]
            halves.append([float(z.risk_difference.quantile(.025)),float(z.risk_difference.quantile(.975))])
        s['interval_MC_check']=dict(first50_RD=halves[0],last50_RD=halves[1],
            maximum_endpoint_difference_pp=float(100*np.max(np.abs(np.array(halves[0])-halves[1]))),
            meaning='100 draws do not establish endpoint stability; two half-batches are a diagnostic only')
        s['point_model']=dict(beta=fit['beta'],linear_predictor=fit['linear_predictor'],
            preprocessing=fit['preprocessing'],estimator_sha256=sha((ROOT/'steroid_sequential.py').read_bytes()))
        checks['scenarios'][name]=dict(plan_unchanged=True,direct_risks_match=True,all_patient_roles_shared=True,
            same_site_and_NE_time=True,no_outcome_used_in_assignment=True,
            attempted=len(bt),failures=int((~bt.success).sum()),warning_draws=s['bootstrap_warning_draws'])
    keys=['case_id','subject','hospital','patientunitstayid','treated','index','unit_index']
    pd.testing.assert_frame_equal(d['scenarios']['primary']['frame'][keys],d['scenarios']['strict']['frame'][keys])
    for path,sig in x['protected'].items():
        p=Path(path)
        if p.name in ['icu_data_inventory.html','phenotype_icu.pkl.gz']:continue
        same=(sha(p.read_bytes())==sig) if isinstance(sig,str) else (p.stat().st_size,p.stat().st_mtime_ns)==tuple(sig)
        assert same,path
    d['validation']=checks;d['source_current']=Path(__file__).read_bytes();save(b)
    say(dict(stage='order_validate',checks=checks))


def order_publish():
    """Update existing author report and venue profile; no parallel manuscript."""
    b=load();x=b['clinical_extension'];d=x['order_estimation'];m=base();assert 'validation' in d
    results=d['results'];names={'primary':'医嘱主口径','strict':'严格录入时钟','known_NE':'NE剂量可解释子集'}
    marker='<!-- HC_EICU_ORDER_ESTIMATION_START -->';endmarker='<!-- HC_EICU_ORDER_ESTIMATION_END -->'
    lines=[marker,'### 2026-10-31：eICU医嘱启动补充估计已完成','',
        '**决定：继续HC双库研究的同口径桥接，JIC保留为首选候选；目前不将两库称为已完成外部验证。** 本轮复用已下载并投影的临床表，完成三项固定口径的实际院内死亡关联估计、各100次患者正权重bootstrap及一个预先指定的大医院影响检查。没有新扫原始表、下载数据或改变MIMIC实际首剂主结果。下列是本节之前“尚未估计”的后续结果。','',
        '这次增加的是可审查的跨医院临床结果，不只是又一轮字段覆盖检查。eICU的关联方向与MIMIC实际首剂不同。差异可能涉及暴露来源、人群与病情选择，也可能涉及未测混杂；当前设计不能分解各因素的贡献。不能选择其中较显著的一项作为结论，也不能把两库直接汇总。','',
        '#### 已保存的比较定义','',
        '在已有感染记录、正速率NE支持及既往激素规则下，比较首次记录IV HC医嘱启动与同医院、相同NE经过时间的尚未启动者；后续接受通常照护。目标为有同院对照的实际医嘱启动状态所代表的ATT式比较，不是所有感染性休克患者的平均效果。医嘱并不等于药物已经实际执行。eICU的NE经过时间从首次可见的NE记录计算，不等于已核实真实开始输注时刻；这也是跨库映射需要处理的差别。','',
        '主要、严格时钟和NE剂量可解释子集在首个eICU死亡比较前保存。保留DNR、DNI与未知状态，排除索引前明确CMO、不得追加治疗或不得使用升压药的节点；没有要求后续HC执行、取消状态、治疗成功或存活至某窗口。少量退出边界对照在查看死亡差异前被移除：索引须严格早于ICU及医院退出。原配置、一次修订原因及代码保存在原缓存内。该过程发生在MIMIC主结果已知之后，不能称整项研究完全预设。','',
        '结局跟随本次医院住院至出院，存活出院计0，ICU转出不结束住院随访。未知/冲突结局不填为存活：参考点估计在各组结局已知的加权质量中计算，同时报告全部候选的最坏情景界限。参考点需要结局已知记录能代表同组加权未知记录的假设。','',
        '|范围|医嘱主口径 / 严格时钟|NE剂量可解释子集|','|---|---:|---:|']
    a=d['scenarios']['primary']['flow'];k=d['scenarios']['known_NE']['flow']
    for key,label in [('case_nodes','启动节点'),('case_patients','启动患者'),('control_patients','独立对照患者'),('control_rows','对照风险集成员记录'),('sites','贡献启动节点的医院')]:
        lines.append('|'+label+'|约'+m.count(a[key])+'|约'+m.count(k[key])+'|')
    lines+=['',
        '原候选约'+m.count(a['original_cases'])+'个。明确治疗限制排除的启动节点、仍合格但无同院对照的节点，以及退出边界的对照均分别为1–10，按既有小单元规则抑制。主/严格时钟患者与行动完全相同，只改变允许进入调整的录入信息。NE子集同时限制启动者与对照、重建支持及权重，改变目标人群，不能把它当作同人群的缺失修复。约数可能不满足算术相加。','',
        '#### 全部固定估计','',
        '以下风险差均为“当前医嘱启动−当前尚未启动”，单位为百分点；RR为相同方向。区间是100次患者正权重bootstrap的2.5%–97.5%分位范围，属于初步不确定性估计，并非MIMIC原500次普通患者重抽样区间。','',
        '|分析|启动风险%〔区间〕|尚未启动风险%〔区间〕|风险差，百分点〔区间〕|RR〔区间〕|','|---|---:|---:|---:|---:|']
    def display(z,key,scale=1.,digits=2):
        p=z['point'][key]*scale;q=z['uncertainty'][key]
        return f"{p:.{digits}f}〔{q['low']*scale:.{digits}f}, {q['high']*scale:.{digits}f}〕"
    for key in ORDER_ESTIMATION_CONFIG['scenarios']:
        s=results[key]
        lines.append('|'+names[key]+'|'+display(s,'initiated_risk',100,1)+'|'+display(s,'not_yet_risk',100,1)+'|'+display(s,'risk_difference',100,2)+'|'+display(s,'risk_ratio',1,3)+'|')
    lines+=['',
        '原MIMIC实际首剂结果保持：风险差−2.47个百分点，95%患者bootstrap区间−7.72至+2.87；RR 0.956（0.863–1.050）。这个对照用于说明当前证据状态，不是同协议异质性检验。两库暴露、资格映射、观测支持及不确定性计算均有差别。','',
        '|并列检查|主口径|严格时钟|NE剂量可解释子集|','|---|---:|---:|---:|']
    rows=[('风险集内均分对照的未调整风险差，百分点',lambda s:f"{100*s['crude']['risk_difference']:.2f}"),
        ('最大绝对SMD',lambda s:f"{s['balance'].smd_after.abs().max():.3f}"),
        ('启动患者ESS',lambda s:f"{s['concentration']['treated_patient']['ess']:.1f}"),
        ('对照患者ESS',lambda s:f"{s['concentration']['control_patient']['ess']:.1f}"),
        ('最大单个对照患者权重份额',lambda s:f"{100*s['concentration']['control_patient']['maximum_share']:.2f}%"),
        ('前10名对照患者累计份额',lambda s:f"{100*s['concentration']['control_patient']['top10_share']:.2f}%"),
        ('单风险集对照ESS中位数',lambda s:f"{s['per_case'].control_patient_ess.median():.1f}"),
        ('单风险集ESS不足5的节点（未据此删除）',lambda s:'约'+m.count(s['per_case'].control_patient_ess.lt(5).sum())),
        ('bootstrap尝试 / 失败 / 警告',lambda s:f"{len(s['bootstrap'])} / {s['bootstrap_failures']} / {s['bootstrap_warning_draws']}"),
        ('前后50次区间端点最大差，百分点',lambda s:f"{s['interval_MC_check']['maximum_endpoint_difference_pp']:.2f}"),
        ('未知结局取极值时RD点值范围，百分点*',lambda s:f"{100*s['point']['RD_missing_lower']:.2f}至{100*s['point']['RD_missing_upper']:.2f}")]
    for label,fun in rows:lines.append('|'+label+'|'+'|'.join(fun(results[key]) for key in ORDER_ESTIMATION_CONFIG['scenarios'])+'|')
    large=d['largest_site_exclusion'];point=large['point'];local=results['primary']['local']
    on=local.case_NE_observed_positive&local.case_MAP_observed
    lines+=['',
        '*未知结局界限处理的是失访/状态冲突的不确定性，不包含抽样误差，也不覆盖未测混杂。具体少数未知患者计数及可重建其数量的比例不公开。未调整比较也先在各风险集内均分对照，不能把重复成员行当独立人。','',
        '主/严格时钟最大残余SMD均来自已测log(1+NE剂量)，分别约0.086/0.086；NE子集最高为既往VP记录约0.101，NE剂量约0.098。没有因子集出现0.101而搜索新模型，也没有把整体接近0.10说成消除了适应证混杂。分配模型固定23项连续变量及22项分类/记录概念，L2=1；没有按结果调参、换阈值或挑截断点。','',
        '按启动数事先选出的最大贡献医院被排除后，剩余约'+m.count(large['remaining_cases'])+'个节点：风险为'+f"{100*point['initiated_risk']:.1f}%与{100*point['not_yet_risk']:.1f}%"+'，RD '+f"{100*point['risk_difference']:+.2f}"+'个百分点，RR '+f"{point['risk_ratio']:.3f}"+'。最大SMD '+f"{large['balance'].smd_after.abs().max():.3f}"+'，对照患者ESS '+f"{large['concentration']['control_patient']['ess']:.1f}"+'。这一预定影响分析只计算点估计，改变目标人群；未对每家医院轮流删除、未据此选择结果。','',
        '整体权重没有被极少数患者主导，但局部支持仍有限：在主口径NE与MAP均有记录的节点，同NE剂量2倍范围且MAP差≤15mmHg的对照权重份额中位数仅'+f"{100*local.loc[on,'close_NE_MAP_control_mass'].median():.1f}%"+'；约'+m.count(local.loc[on,'close_NE_MAP_control_mass'].eq(0).sum())+'个节点没有这种邻近状态对照。这个描述不是临床可互换性判据，却是不能仅据良好总体SMD作强因果解释的具体理由。','',
        '#### 不确定性计算为什么另行说明','',
        '在读取eICU分组死亡效果前，普通患者有放回抽样的100次可计算性检查出现空对照池：最终主/严格队列70次，NE子集71次（最初未修订边界的检查为76次）。这些是重抽样风险集失去对照，不是数值优化失败，也不是原数据70%的患者无对照。没有删除失败重复或仅用成功重复算区间。','',
        '预先改用归一化Exp(1)患者正权重，等价于Dirichlet(1,…,1) Bayesian bootstrap：同一患者的全部住院、启动和对照角色共享权重，每次重新估计中位数、标准化及启动分配模型。本轮三个情景各100次均完成，0失败、0警告。它给出条件于当前患者与医院支持的探索性区间；不恢复重抽样时消失的临床支持，也不解决未测混杂或跨新医院推广。100次尾部端点仍是初步精度，不能把区间排除零称为已确认治疗效果。','',
        '[Rubin 1981](https://doi.org/10.1214/aos/1176345338)提供Dirichlet随机权重的一般基础；归一化指数权重是等价实现。本研究共享患者权重、固定风险集及重复拟合是本地应用，原文并未专门验证本估计器的有限样本覆盖率。','',
        '#### 对论文的含义与下一项必要工作','',
        '本轮可以支持：不同记录体系下的HC启动问题已有实际队列、治疗前信息、加权风险及可核查的敏感性结果。尚不能支持：HC有害、eICU成功验证MIMIC疗效、或者两个库相加就提供更可信的单一效应。未调整RD约+19.8个百分点，调整后约+7.6，说明观测病情差异已明显改变比较，残余严重度与用药目的仍可能重要。','',
        '可保留且需明示的局限包括NE浓度/剂量在部分医院不可解释、乳酸及NE变化测量稀疏、治疗限制记录可能沿转科保留、医嘱执行不完整。缺失指示、剂量子集及严格时钟是并列检验，不能当作已补齐真实病情。另须区分对同一临床策略的可迁移性与不同暴露定义的差异。','',
        '**下一项优先工作只做MIMIC内部医嘱—实际给药桥接。** 现有约230个医嘱节点来自旧实际首剂风险集住院，60分钟执行确认50.6%、6小时79.0%，暂非全库独立医嘱人群。需要在现有原始医嘱/处方和已投影临床记录上，以医嘱自身资格建立风险集，不能继承“后来实际启动”的选人条件或沿用实际给药权重。覆盖不足先区分未提取和未记录，再决定是否确有必要补字段；当前核心文件齐备，无新增下载要求。完成后两库先并列同定义估计，只有目标与测量能够协调时才考虑汇总。','',
        'Rajendran等将MIMIC与eICU合并开发、CEDAR验证，MIMIC激素脚本使用处方。该先例支持明确研究处方/医嘱暴露，不证明本地医嘱就是实际给药，也不保证相同发表层级。[全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12075795/)；[固定代码](https://raw.githubusercontent.com/surajraj99/Corticosteroids-in-Patients-with-Critical-Illness/e96952407e38b52807ecd521372eeb005c993f1f/data/MIMIC/scipts/MIMIC%20Steroid%20Use%20-%20Prescription.ipynb)。','',
        '已完成独立风险重算、风险集质量守恒、相同患者多角色权重、退出边界、未知结局界限及原MIMIC文件不变检查。100次配置、代码、随机状态、失败/警告和全部聚合结果保存在原私有缓存，未导出患者标识。入口为 `steroid_eicu.py order-prepare / order-estimate / order-validate / order-publish`。','',endmarker]
    path=ROOT/'steroid_tte.md';before=path.read_bytes();nl='\r\n' if b'\r\n' in before else '\n';old=before.decode('utf-8').replace('\r\n','\n')
    section='\n'.join(lines)+'\n'
    if marker in old:
        lo=old.index(marker);hi=old.index(endmarker,lo)+len(endmarker);new=old[:lo]+section.rstrip('\n')+old[hi:]
    else:
        anchor='### 实际代码与证据位置';assert old.count(anchor)==1;new=old.replace(anchor,section+'\n'+anchor,1)
    after=new.replace('\n',nl).encode('utf-8')
    if after!=before:
        assert path.read_bytes()==before
        history=b.setdefault('report_history',[])
        if not any(v['sha256']==sha(before) for v in history):history.append(dict(sha256=sha(before),bytes=before))
        d['publication']=dict(before_hash=sha(before),report_hash=sha(after),created_utc=m.utc())
        save(b);path.write_bytes(after)
    say(dict(stage='order_publish',report=str(path),results=len(results),bootstrap_each=100,raw_bytes=0))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['foundation','publish','test','drug-project','drug-code-project','drug-review','concentration-review','drug-validate','drug-publish','diagnosis-project','diagnosis-review','diagnosis-support','diagnosis-validate','diagnosis-publish','multicenter-support','clinical-plan','clinical-project','clinical-features','clinical-fix-crrt','clinical-summary','clinical-validate','clinical-score','clinical-publish','order-prepare','order-estimate','order-validate','order-publish']);a=p.parse_args()
    if a.stage=='foundation':foundation()
    elif a.stage=='publish':publish()
    elif a.stage=='drug-project':drug_project()
    elif a.stage=='drug-code-project':drug_code_project()
    elif a.stage=='drug-review':drug_review()
    elif a.stage=='concentration-review':concentration_review()
    elif a.stage=='drug-validate':drug_validate()
    elif a.stage=='drug-publish':drug_publish()
    elif a.stage=='diagnosis-project':diagnosis_project()
    elif a.stage=='diagnosis-review':diagnosis_review()
    elif a.stage=='diagnosis-support':diagnosis_support()
    elif a.stage=='diagnosis-validate':diagnosis_validate()
    elif a.stage=='diagnosis-publish':diagnosis_publish()
    elif a.stage=='multicenter-support':multicenter_support()
    elif a.stage=='clinical-plan':clinical_plan()
    elif a.stage=='clinical-project':clinical_project()
    elif a.stage=='clinical-features':clinical_features()
    elif a.stage=='clinical-fix-crrt':clinical_fix_crrt()
    elif a.stage=='clinical-summary':clinical_summarize()
    elif a.stage=='clinical-validate':clinical_validate()
    elif a.stage=='clinical-score':clinical_score()
    elif a.stage=='clinical-publish':clinical_publish()
    elif a.stage=='order-prepare':order_prepare()
    elif a.stage=='order-estimate':order_estimate()
    elif a.stage=='order-validate':order_validate()
    elif a.stage=='order-publish':order_publish()
    else:say(dict(synthetic_tests_passed=tests()))


if __name__=='__main__':main()
