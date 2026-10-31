"""Independent MIMIC HC-order cohort, support and exploratory association.

Uses all admissions in the existing adult NE source, not admissions selected by
the later actual-dose risksets. Adds one branch to the existing private POE
checkpoint; never writes the frozen actual-dose checkpoint.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

# Match the canonical CPU estimator before NumPy initializes its runtime.
for _thread_key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[_thread_key]='1'
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / 'analysis'))
from hc_paths import configured_path
import steroid_sequential as seq
import steroid_event_review as review

# Immutable first-stage source protocol. Later clinical and estimation phases
# retain their own explicit protocols; this dictionary is not their scope.
CONFIG = dict(
    version='independent_MIMIC_HC_order_source_20261031',
    source='all adult observed NE admissions in existing MIMIC-IV3.1 source; no later HC/riskset selection',
    action='first native POE-linked IV HC order in the admission; not actual administration',
    clock='POE ordertime; prescribed start and eMAR administration kept distinct',
    horizon='within first24h of first observed ICU NE, still in an observed positive NE interval immediately before index',
    infection='same original MIMIC actual-antibacterial and diagnostic-culture preindex rule; eICU infection proxy differs',
    prior_GC='exclude prior recorded systemic GC order or actual GC; uncertain actual history remains indeterminate',
    unknown_order_history='missing/conflicted POE with planned start at/before index flags uncertain order history; no assumed administration',
    ties='same-time other systemic GC order or actual GC distinct from HC makes version ambiguous',
    routes='IV HC action; prior IV, IM or enteral GC orders count; unknown route before index remains uncertain',
    future='no selection on later execution, cancellation, response, death, or survival through a grace period',
    comparator='other patients not yet ordered systemic GC at their same observed NE elapsed time; later HC allowed',
    clinical_support='measure source/envelope completeness before new feature extraction; no complete-case eligibility',
    next_stage='pre-index treatment-limit exclusions and confounder adjustment still required; this is a technical source cohort',
    outcomes='no death contrasts, weight fitting or bootstrap in this module',
    prior_results_seen='MIMIC actual-dose and eICU order associations already seen; new analysis exploratory',
    raw_reads='one targeted prescriptions pass outside old projected admissions, one POE pass for missing native keys')
RX_COLS=['subject_id','hadm_id','pharmacy_id','poe_id','poe_seq','starttime','stoptime',
         'drug_type','drug','dose_val_rx','dose_unit_rx','doses_per_24_hrs','route']
BRANCH='independent_order_bridge'
PHARM_COLS=['subject_id','hadm_id','pharmacy_id','poe_id','starttime','stoptime',
            'medication','proc_type','entertime','verifiedtime','route','status']
CLINICAL_CONFIG=dict(
    version='independent_order_preindex_clinical_extension_1',
    rationale='source and clocks differ from actual-dose risksets; fill projection gaps before interpreting absent records',
    primary_action='first record-qualified native IV HC order explicitly marked New; unresolved Change excluded from primary, retained sensitivity',
    change_sensitivity='all original native first-order candidates at original index; do not infer earlier drug from a later Change',
    confounders='same fixed original numerical and categorical set; add explicit POE code-source discordance as a separate state',
    windows='vitals1h; neurologic/device/temperature4h; lactate/gases6h; other labs24h; code/intubation current hospitalization',
    clocks='primary bedside chart; labs both chart/store; strict both for bedside; all strictly before own index',
    code_orders='native POE entry strictly before index; contemporaneous D/C expires linked directive; final status ignored',
    limits='latest explicit structured directive from POE or chart; conflicting simultaneous directives unknown; DNR DNI CMO separate',
    CMO='exclude explicit current CMO in primary clock from both case and control; same membership for strict-clock analysis',
    unknown='unknown limitation retained, never full code; no record never absence or normal',
    subsequent='no future execution/survival/response/CMO for eligibility; hospital followup at own index',
    analysis='no outcome contrasts until clinical construction and measured support assessed',
    prior_seen='original actual-dose and eICU results seen; new order comparison exploratory')


def say(x):
    print(json.dumps(x,ensure_ascii=True,default=str),flush=True)


def state_path():
    return configured_path('private_dir','../restricted/hc-private')/'hc_poe_linkage.pkl.gz'


def load():
    frozen=configured_path('private_dir','../restricted/hc-private')/'steroid_sequential.pkl.gz'
    b=pd.read_pickle(frozen);side=pd.read_pickle(state_path())
    assert side['base_frozen_signature']==review.frozen_signature(b)
    if BRANCH not in side:
        side[BRANCH]=dict(config=CONFIG,created_utc=seq.base().utc(),runs=[],source_at_freeze=Path(__file__).read_bytes(),
            original_checkpoint_signature=(frozen.stat().st_size,frozen.stat().st_mtime_ns),models=0,effects=0)
        save(side[BRANCH])
    d=side[BRANCH];assert d['config']==CONFIG
    assert tuple(d['original_checkpoint_signature'])==(frozen.stat().st_size,frozen.stat().st_mtime_ns)
    return b,side['event_review'],d


def save(d):
    path=state_path();side=pd.read_pickle(path)
    side[BRANCH]=d
    temp=path.with_name(path.name+'.writing')
    pd.to_pickle(side,temp,compression={'method':'gzip','compresslevel':1})
    os.replace(str(temp),str(path))


def plan():
    b,e,d=load();raw=configured_path('mimic_dir','../restricted/mimic-iv-3.1')
    rx=raw/'hosp'/'prescriptions.csv';poe=configured_path('poe_file',str(raw/'hosp'/'poe.csv'))
    m=seq.base();outside=set(b['source'].hadm_id)-set(b['full_risksets'].hadm_id)
    entries=[]
    for path,cols,why in [(rx,RX_COLS,'GC prescriptions in NE-source admissions outside the old projected risksets'),
                          (poe,review.POE_COLUMNS,'missing native steroid-order POE keys and direct discontinuation links')]:
        assert path.exists() and set(cols)<=set(pd.read_csv(path,nrows=0).columns)
        entries.append(dict(path=str(path),bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns,columns=cols,filter=why))
    if 'read_plan' in d:assert d['read_plan']==entries
    else:d['read_plan']=entries;save(d)
    say(dict(stage='plan',source_admissions=m.count(len(b['source'])),outside_old_projection=m.count(len(outside)),
        files=[dict(table=Path(x['path']).name,GiB=round(x['bytes']/2**30,2),columns=x['columns'],filter=x['filter']) for x in entries],
        estimated_minutes='3-10',estimated_peak_GiB='1-2',models=0,effects=0))


def project():
    b,e,d=load();assert 'read_plan' in d;m=seq.base()
    ref=b['source'].set_index('hadm_id');all_adm=set(ref.index);old_adm=set(b['full_risksets'].hadm_id)
    outside=all_adm-old_adm
    if 'rx' not in d:
        pieces=[];errors=0
        def consume(z):
            nonlocal errors
            hh=pd.to_numeric(z.hadm_id,errors='coerce');z=z[hh.isin(outside)].copy()
            if z.empty:return
            z=z[z.drug.str.contains(review.RX_GC,na=False)].copy()
            if z.empty:return
            z['hadm_id']=pd.to_numeric(z.hadm_id).astype('int64');z['subject_id']=pd.to_numeric(z.subject_id).astype('int64')
            ok=z.subject_id.eq(z.hadm_id.map(ref.subject_id));errors+=int((~ok).sum());pieces.append(z[ok])
        r=d['read_plan'][0];say(dict(stage='rx_start',bytes=r['bytes'],filter=r['filter'],columns=RX_COLS))
        run=review.stream_projection(Path(r['path']),RX_COLS,consume,m,'independent_order_rx')
        old=e['prescriptions']['records'];old=old[old.is_GC][RX_COLS].copy()
        z=pd.concat([old]+pieces,ignore_index=True).drop_duplicates()
        z['start']=review.secs(z.starttime);z['stop_rx']=review.secs(z.stoptime)
        z['is_HC']=z.drug.str.contains(review.RX_HC,na=False);z['route_class']=z.route.map(review.rx_route)
        d['rx']=z;d['rx_identity_conflicts']=errors;d['runs'].append(run);save(d)
        say(dict(stage='rx_done',retained_rows=m.count(len(z)),admissions=m.count(z.hadm_id.nunique()),seconds=run['seconds'],peak_MB=run['peak_MB']))
    if 'poe' not in d:
        needed=set(d['rx'].poe_id.dropna());old=e['prescriptions']['poe'];old=old[old.poe_id.isin(needed)].copy()
        missing=needed-set(old.poe_id);pieces=[];links=[]
        def consume_poe(z):
            got=z[z.poe_id.isin(missing)]
            if len(got):pieces.append(got.copy())
            link=z[z.discontinue_of_poe_id.isin(needed)|z.discontinued_by_poe_id.isin(needed)]
            if len(link):links.append(link.copy())
        r=d['read_plan'][1];say(dict(stage='poe_start',bytes=r['bytes'],new_native_keys=m.count(len(missing)),columns=review.POE_COLUMNS))
        if missing:
            run=review.stream_projection(Path(r['path']),review.POE_COLUMNS,consume_poe,m,'independent_order_poe');d['runs'].append(run)
        z=pd.concat([old[review.POE_COLUMNS]]+pieces,ignore_index=True).drop_duplicates()
        z['order']=review.secs(z.ordertime);d['poe']=z
        d['direct_links']=pd.concat(links,ignore_index=True).drop_duplicates() if links else pd.DataFrame(columns=review.POE_COLUMNS)
        d['direct_links']['order']=review.secs(d['direct_links'].ordertime)
        save(d);say(dict(stage='poe_done',native_records=m.count(len(z)),missing_keys=m.count(len(needed-set(z.poe_id))),models=0,effects=0))


def build_orders(rx,poe):
    """Require native identity; no future order_status/cancellation filter."""
    clean=poe.drop_duplicates(subset=review.POE_COLUMNS).copy()
    duplicate=set(clean.loc[clean.poe_id.duplicated(False),'poe_id'])
    clean=clean[~clean.poe_id.isin(duplicate)].set_index('poe_id')
    rows=[];bad=[]
    for key,g in rx.dropna(subset=['poe_id']).groupby('poe_id',sort=False):
        reason=''
        if key not in clean.index:reason='MISSING_OR_CONFLICTED_POE'
        elif g.hadm_id.nunique()!=1 or g.subject_id.nunique()!=1:reason='RX_IDENTITY_CONFLICT'
        else:
            p=clean.loc[key]
            if str(p.subject_id)!=str(g.subject_id.iloc[0]) or str(p.hadm_id)!=str(g.hadm_id.iloc[0]):reason='NATIVE_IDENTITY_CONFLICT'
            elif not np.isfinite(p.order):reason='UNKNOWN_ORDER_TIME'
        if reason:
            for h,gg in g.groupby('hadm_id'):
                bad.append(dict(hadm_id=int(h),poe_id=key,reason=reason,planned=gg.start.min(),is_HC=bool(gg.is_HC.any())))
            continue
        p=clean.loc[key];routes=set(g.route_class);route=next(iter(routes)) if len(routes)==1 else 'route_conflict'
        rows.append(dict(hadm_id=int(g.hadm_id.iloc[0]),subject_id=int(g.subject_id.iloc[0]),poe_id=key,
            index=float(p.order),route=route,is_HC=bool(g.is_HC.all()),mixed_drug=bool(g.is_HC.any() and not g.is_HC.all()),
            transaction_type=p.transaction_type,order_type=p.order_type,planned=g.start.min()))
    for t in rx[rx.poe_id.isna()].itertuples():
        bad.append(dict(hadm_id=int(t.hadm_id),poe_id=None,reason='RX_WITHOUT_POE',planned=t.start,is_HC=bool(t.is_HC)))
    return pd.DataFrame(rows),pd.DataFrame(bad,columns=['hadm_id','poe_id','reason','planned','is_HC'])


def active_ne_record(g,t):
    z=g[(g.start<t)&(g.end>=t)]
    if 'icu_lo' in z:z=z[(z.icu_lo<=t)&(z.icu_hi>t)]
    if z.empty:return False,np.nan,'NO_ACTIVE_POSITIVE_NE'
    if z.stay_id.nunique()!=1:return False,np.nan,'MULTIPLE_ACTIVE_ICU_STAYS'
    return True,int(z.stay_id.iloc[0]),'OBSERVED_ACTIVE_NE'


def order_history(orders,unresolved,t,case_key=None):
    if unresolved is not None and len(unresolved) and unresolved.planned.le(t).any():
        return 'UNCERTAIN_ORDER_HISTORY'
    if orders is None or orders.empty:return 'NO_PRIOR_RECORDED_SYSTEMIC_GC_ORDER'
    prior=orders[orders['index']<t]
    if prior.route.isin(['IV_prescribed','IM_prescribed','enteral_prescribed']).any():return 'PRIOR_SYSTEMIC_GC_ORDER'
    if prior.route.isin(['route_unknown','route_conflict']).any():return 'UNCERTAIN_PRIOR_ORDER_ROUTE'
    at=orders[orders['index'].eq(t)]
    if case_key is not None:at=at[at.poe_id.ne(case_key)]
    if at.route.isin(['IV_prescribed','IM_prescribed','enteral_prescribed','route_unknown','route_conflict']).any():
        return 'SAME_TIME_OTHER_SYSTEMIC_ORDER'
    return 'NO_PRIOR_RECORDED_SYSTEMIC_GC_ORDER'


def envelope_complete(rows,b,var,kind,hours):
    env=b['clinical_stay_envelope'] if kind=='chart' else b['clinical_envelope']
    key=rows.stay_id if kind=='chart' else rows.hadm_id
    admission=rows.hadm_id.map(b['source'].set_index('hadm_id').admit)
    minimum=key.map(env['min']);maximum=key.map(env['max'])
    wanted=np.maximum(admission,rows['index']-hours*3600)
    available=np.maximum(admission,minimum-hours*3600)
    return minimum.notna()&maximum.notna()&(wanted>=available)&rows['index'].le(maximum)


def evaluate():
    b,e,d=load();assert 'poe' in d;m=seq.base()
    if 'cases' in d:say(dict(stage='evaluate',reuse=True));return
    orders,unresolved=build_orders(d['rx'],d['poe']);ref=b['source'].set_index('hadm_id');ctx=seq.context(b)
    ng={int(h):g for h,g in b['NE'].groupby('hadm_id',sort=False)}
    og={int(h):g for h,g in orders.groupby('hadm_id',sort=False)}
    ug={int(h):g for h,g in unresolved.groupby('hadm_id',sort=False)}
    first=orders[orders.is_HC&orders.route.eq('IV_prescribed')&~orders.mixed_drug].sort_values(['index','poe_id']).drop_duplicates('hadm_id')
    rows=[];old_admissions=set(b['full_risksets'].hadm_id);actual=b['resolved_medications'];other=actual[actual.glucocorticoid&actual.actual_systemic&~actual.hc_action]
    sameother=set(zip(other.hadm_id,other.chart))
    for r in first.itertuples():
        h=int(r.hadm_id);s=ref.loc[h];t=float(r.index);stay=np.nan;strict=False
        if t<s.ne0:state='BEFORE_FIRST_OBSERVED_NE'
        elif t>s.ne0+86400:state='AFTER_NE_DAY1'
        elif t<s.admit or t>=s.discharge or (np.isfinite(s.death_time) and t>=s.death_time):state='OUTSIDE_OBSERVED_ALIVE_ADMISSION'
        else:
            active,stay,state=active_ne_record(ng[h],t)
            if active:
                state,strict=seq.qualify(h,t,ctx,m,True)
                if state=='QUALIFIED_RECORD_PROXY':
                    check=order_history(og.get(h),ug.get(h),t,r.poe_id)
                    if check!='NO_PRIOR_RECORDED_SYSTEMIC_GC_ORDER':state=check
                    if (h,t) in sameother:state='SAME_TIME_OTHER_ACTUAL_GC'
        rows.append(dict(hadm_id=h,subject_id=int(s.subject_id),poe_id=r.poe_id,index=t,stay_id=stay,elapsed_h=(t-s.ne0)/3600,
            state=state,strict_infection=strict,transaction_type=r.transaction_type,old_riskset_admission=h in old_admissions))
    first=pd.DataFrame(rows);cases=first[first.state.eq('QUALIFIED_RECORD_PROXY')].copy().reset_index(drop=True)
    cases['case_id']=np.arange(len(cases));cases['treated']=1
    # Subsequent actual execution is descriptive only, evaluated after eligibility.
    hc=actual[actual.hc_action];ag={int(h):g.chart.to_numpy(float) for h,g in hc.groupby('hadm_id')}
    for minutes in [30,60,120,360]:
        states=[]
        for r in cases.itertuples():
            times=ag.get(r.hadm_id,np.array([]));later=times[times>=r.index]
            when=float(later.min()) if len(later) else np.nan
            end=min(ref.loc[r.hadm_id,'med_end'],ref.loc[r.hadm_id,'discharge'])
            states.append(review.order_proxy_window(r.index,when,end,minutes))
        cases['execution_'+str(minutes)]=states
    coverage=[]
    for var,kind,hours in [('MAP','chart',1),('HR','chart',1),('RR','chart',1),('GCS','chart',4),('lactate','labs',6),('creatinine','labs',24)]:
        full=envelope_complete(cases,b,var,kind,hours)
        coverage.append(dict(variable=var,den=len(cases),full_window_cached=int(full.sum()),not_fully_projected=int((~full).sum())))
    d.update(orders=orders,unresolved_orders=unresolved,first_orders=first,cases=cases,case_envelope_coverage=pd.DataFrame(coverage),
        source_admissions=len(ref),source_patients=ref.subject_id.nunique(),old_riskset_admissions=b['full_risksets'].hadm_id.nunique(),
        source_current=Path(__file__).read_bytes())
    save(d)
    say(dict(stage='independent_order_candidates',source_admissions=m.count(len(ref)),first_IV_orders=m.count(len(first)),
        qualified_record_proxy=m.count(len(cases)),outside_old_risksets=m.count((~cases.old_riskset_admission).sum()),
        qualification={k:m.count(v) for k,v in first.state.value_counts().items()},
        case_envelopes=[dict(variable=z['variable'],full=m.count(z['full_window_cached']),not_projected=m.count(z['not_fully_projected'])) for z in coverage],models=0,effects=0))


def support():
    b,e,d=load();assert 'cases' in d;m=seq.base()
    if 'risksets' in d:say(dict(stage='support',reuse=True));return
    ref=b['source'].set_index('hadm_id');ctx=seq.context(b);ng={int(h):g for h,g in b['NE'].groupby('hadm_id')}
    og={int(h):g for h,g in d['orders'].groupby('hadm_id')};ug={int(h):g for h,g in d['unresolved_orders'].groupby('hadm_id')}
    first_gc=ctx[0];uncertain_gc=ctx[1];first_order=d['orders'][d['orders'].route.isin(['IV_prescribed','IM_prescribed','enteral_prescribed'])].groupby('hadm_id')['index'].min()
    ne=b['NE'];ne0=ne.hadm_id.map(ref.ne0);start=ne.start-ne0;end=ne.end-ne0
    uncertain_order=d['orders'][d['orders'].route.isin(['route_unknown','route_conflict'])].groupby('hadm_id')['index'].min()
    unresolved_start=d['unresolved_orders'].groupby('hadm_id').planned.min()
    prior=pd.DataFrame(index=ref.index)
    prior['actual']=pd.Series(first_gc);prior['uncertain_actual']=pd.Series(uncertain_gc)
    prior['systemic_order']=first_order;prior['uncertain_order']=uncertain_order;prior['unresolved']=unresolved_start
    prior=prior.fillna(np.inf).min(axis=1)
    infection_possible=ref.index.isin(set(ctx[2])&set(ctx[3]))
    allrows=[];flow=[];began=time.perf_counter();last=began;equivalence_checked=False
    for case in d['cases'].itertuples():
        elapsed=case.elapsed_h*3600;tall=ref.ne0+elapsed;tn=ne0+elapsed
        active=(start<elapsed)&(end>=elapsed)&ne.start.lt(tn)&ne.end.ge(tn)
        if 'icu_lo' in ne:active&=ne.icu_lo.le(tn)&ne.icu_hi.gt(tn)
        locations=ne.loc[active,['hadm_id','stay_id']].drop_duplicates()
        locations=locations[~locations.hadm_id.duplicated(False)].set_index('hadm_id').stay_id
        eligible=infection_possible&ref.subject_id.ne(case.subject_id)&tall.lt(ref.discharge)&(ref.death_time.isna()|tall.lt(ref.death_time))&tall.lt(prior)
        ids=ref.index[eligible&ref.index.isin(locations.index)]
        candidates=ref.loc[ids,['subject_id','ne0']].copy();candidates['index']=tall.loc[ids];candidates['stay_id']=locations.reindex(ids)
        controls=[]
        for s in candidates.itertuples():
            h=int(s.Index);t=float(s.index)
            state,strict=seq.qualify(h,t,ctx,m,False)
            if state!='QUALIFIED_RECORD_PROXY':continue
            controls.append(dict(case_id=case.case_id,hadm_id=h,subject_id=int(s.subject_id),stay_id=int(s.stay_id),index=t,elapsed_h=case.elapsed_h,treated=0,strict_infection=strict))
        if not equivalence_checked:
            # Fixed first case, full original scalar reference; no outcomes.
            scalar=set();possible=set(ne.loc[(start<elapsed)&(end>=elapsed),'hadm_id'])
            for h in sorted(possible):
                s=ref.loc[h];t=float(s.ne0+elapsed)
                if s.subject_id==case.subject_id or t>=s.discharge or (np.isfinite(s.death_time) and t>=s.death_time):continue
                if first_gc.get(h,np.inf)<=t or uncertain_gc.get(h,np.inf)<=t or first_order.get(h,np.inf)<=t:continue
                on,stay,_=active_ne_record(ng[h],t)
                if not on or order_history(og.get(h),ug.get(h),t)!='NO_PRIOR_RECORDED_SYSTEMIC_GC_ORDER':continue
                state,strict=seq.qualify(h,t,ctx,m,False)
                if state=='QUALIFIED_RECORD_PROXY':scalar.add((int(h),int(stay),t,bool(strict)))
            vector={(r['hadm_id'],r['stay_id'],r['index'],bool(r['strict_infection'])) for r in controls}
            assert scalar==vector,'BATCH_SCALAR_RISKSET_MISMATCH'
            equivalence_checked=True;say(dict(stage='batch_scalar_equivalence',passed=True,controls=m.count(len(scalar))))
        flow.append(dict(case_id=case.case_id,controls=len(controls)))
        allrows.append(dict(case_id=case.case_id,hadm_id=case.hadm_id,subject_id=case.subject_id,stay_id=int(case.stay_id),index=case.index,elapsed_h=case.elapsed_h,treated=1,strict_infection=case.strict_infection));allrows.extend(controls)
        if time.perf_counter()-last>30:
            say(dict(stage='order_risksets',completed=m.count(len(flow)),total=m.count(len(d['cases'])),seconds=round(time.perf_counter()-began)));last=time.perf_counter()
    panel=pd.DataFrame(allrows);flow=pd.DataFrame(flow);cov=[]
    assert not any('death' in c or 'outcome' in c for c in panel.columns)
    for var,kind,hours in [('MAP','chart',1),('GCS','chart',4),('lactate','labs',6),('creatinine','labs',24)]:
        full=envelope_complete(panel,b,var,kind,hours)
        for arm in [1,0]:
            mask=panel.treated.eq(arm);cov.append(dict(variable=var,arm=arm,rows=int(mask.sum()),full_window_cached=int((full&mask).sum()),
                incomplete_patients=panel.loc[mask&~full,'subject_id'].nunique(),incomplete_admissions=panel.loc[mask&~full,'hadm_id'].nunique()))
    assert panel.groupby('case_id').elapsed_h.nunique().eq(1).all()
    case_subject=panel[panel.treated.eq(1)].set_index('case_id').subject_id
    assert panel.loc[panel.treated.eq(0),'subject_id'].ne(panel.loc[panel.treated.eq(0),'case_id'].map(case_subject)).all()
    d.update(risksets=panel,support=flow,panel_envelope_coverage=pd.DataFrame(cov),support_seconds=round(time.perf_counter()-began,1),
        support_implementation=dict(reason='Batch identical history and active-interval conditions after slow scalar run; no new model or clinical rule',scalar_first_case_equals_batch=equivalence_checked,initial_source_commit='be59458'),
        checks=dict(independent_source=True,native_identity_only=True,eligibility_before_execution=True,no_future_survival=True,
            distinct_control_patient=True,same_NE_elapsed=True,no_outcome_model=True,frozen_actual_cache_unchanged=True))
    save(d)
    say(dict(stage='order_support_done',cases=m.count(len(flow)),without_controls=m.count(flow.controls.eq(0).sum()),
        control_patients=m.count(panel.loc[panel.treated.eq(0),'subject_id'].nunique()),riskset_rows=m.count(len(panel)),
        control_pool_median=float(flow.controls.median()),seconds=d['support_seconds'],models=0,effects=0))


def table_file(module,name):
    raw=configured_path('mimic_dir','../restricted/mimic-iv-3.1')
    for path in [raw/module/(name+'.csv'),raw/module/(name+'.csv')/(name+'.csv'),raw/module/(name+'.csv.gz')]:
        if path.is_file():return path
    raise FileNotFoundError('SAME_VERSION_TABLE_NOT_LOCATED: '+module+'/'+name)


def lineage():
    """Resolve native predecessor identities, without using execution or outcome."""
    b,e,d=load();m=seq.base()
    if 'lineage_pharmacy' in d:say(dict(stage='lineage',reuse=True));return
    p=pd.concat([d['poe'],d['direct_links']],ignore_index=True).drop_duplicates(subset=review.POE_COLUMNS)
    native=set(p.poe_id.dropna())|set(p.discontinue_of_poe_id.dropna())|set(p.discontinued_by_poe_id.dropna())
    pharm=set(d['rx'].pharmacy_id.dropna().astype(str));admissions=set(b['source'].hadm_id)
    path=table_file('hosp','pharmacy');assert set(PHARM_COLS)<=set(pd.read_csv(path,nrows=0).columns)
    plan=dict(path=str(path),bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns,columns=PHARM_COLS,
        filter='native steroid POE keys, immediate predecessor/discontinuation keys or known pharmacy IDs; same source admissions',
        reason='resolve whether Change refers to an earlier GC decision; no final status/execution/outcome filtering',
        estimate_minutes='1-3',estimate_peak_MB=1800,utc=m.utc())
    d['lineage_plan']=plan;save(d);pieces=[]
    def consume(z):
        hit=z.poe_id.isin(native)|z.pharmacy_id.isin(pharm)
        z=z[hit].copy();hh=pd.to_numeric(z.hadm_id,errors='coerce');z=z[hh.isin(admissions)].copy()
        if len(z):pieces.append(z)
    run=review.stream_projection(path,PHARM_COLS,consume,m,'native_order_lineage_pharmacy')
    z=pd.concat(pieces,ignore_index=True).drop_duplicates() if pieces else pd.DataFrame(columns=PHARM_COLS)
    for col in ['subject_id','hadm_id']:z[col]=pd.to_numeric(z[col],errors='raise').astype('int64')
    z['entry']=review.secs(z.entertime);z['start']=review.secs(z.starttime)
    z['is_GC']=z.medication.str.contains(review.RX_GC,na=False);z['is_HC']=z.medication.str.contains(review.RX_HC,na=False)
    z['route_class']=z.route.map(review.rx_route)
    d['lineage_pharmacy']=z;d['runs'].append(run);d['source_current']=Path(__file__).read_bytes();save(d)
    say(dict(stage='lineage_pharmacy_done',retained=m.count(len(z)),seconds=run['seconds'],peak_MB=run['peak_MB'],effects=0))


def lineage_classify(d):
    p=pd.concat([d['poe'],d['direct_links']],ignore_index=True).drop_duplicates(subset=review.POE_COLUMNS)
    p=p.drop_duplicates('poe_id').set_index('poe_id');ph=d['lineage_pharmacy'];rows=[]
    for c in d['cases'].itertuples():
        current=p.loc[c.poe_id];key=current.discontinue_of_poe_id
        if c.transaction_type=='New' and pd.isna(key):state='EXPLICIT_NEW_NO_BACKWARD_LINK'
        elif pd.isna(key):state='CHANGE_OR_OTHER_NO_PREDECESSOR_KEY'
        elif key not in p.index:state='PREDECESSOR_POE_NOT_CAPTURED'
        else:
            prev=p.loc[key]
            valid=str(prev.subject_id)==str(c.subject_id) and str(prev.hadm_id)==str(c.hadm_id) and prev.order<c.index
            if not valid:state='PREDECESSOR_IDENTITY_OR_TIME_CONFLICT'
            else:
                zz=ph[ph.poe_id.eq(key)&ph.hadm_id.eq(c.hadm_id)&ph.subject_id.eq(c.subject_id)]
                if zz.empty:state='EARLIER_NATIVE_ORDER_DRUG_UNRESOLVED'
                elif zz.is_GC.any():state='EARLIER_NATIVE_GC_ORDER'
                elif zz.medication.notna().all():state='EARLIER_NATIVE_NONGC_ORDER'
                else:state='EARLIER_NATIVE_ORDER_DRUG_UNRESOLVED'
        rows.append(dict(case_id=c.case_id,state=state,primary_new=c.transaction_type=='New' and state=='EXPLICIT_NEW_NO_BACKWARD_LINK'))
    return pd.DataFrame(rows)


def clinical_bounds(b,p,name):
    """Requested and already fully scanned variable envelopes, half-open."""
    m=seq.base();q=b['source'].set_index('hadm_id');parts=[]
    variables=seq.HC_CF_LABS if name=='labs' else seq.HC_CF_CHART
    ages=m.LAB_HOURS if name=='labs' else m.CHART_HOURS
    for var in variables:
        hospital=name=='labs' or var in ['code_status','intubation_status'];key='hadm_id' if hospital else 'stay_id'
        z=p.groupby(key)['index'].agg(['min','max']);own=p.drop_duplicates(key).set_index(key)
        z['hadm_id']=z.index if hospital else own.hadm_id.reindex(z.index)
        admission=z.hadm_id.map(q.admit);old=b['clinical_envelope'] if hospital else b['clinical_stay_envelope']
        iscode=var in ['code_status','intubation_status'];age=float(ages[var])*3600
        z['lo']=admission if iscode else np.maximum(admission,z['min']-age);z['hi']=z['max']
        z['old_lo']=admission if iscode else np.maximum(admission,old['min'].reindex(z.index)-age)
        z['old_hi']=old['max'].reindex(z.index)
        z['variable']=var;z['key']=z.index.astype('int64');z['hospital_key']=hospital
        z['gap']=z.old_hi.isna()|z.old_lo.isna()|z.lo.lt(z.old_lo)|z.hi.gt(z.old_hi)
        parts.append(z.reset_index(drop=True))
    return pd.concat(parts,ignore_index=True)


def outside_cached_window(time,lo,hi,old_lo,old_hi):
    t=np.asarray(time,dtype=float);a=np.asarray(old_lo,dtype=float);e=np.asarray(old_hi,dtype=float)
    return (t>=np.asarray(lo))&(t<np.asarray(hi))&(~np.isfinite(a)|~np.isfinite(e)|(t<a)|(t>=e))


def clinical_plan():
    b,e,d=load();m=seq.base();assert 'lineage_pharmacy' in d
    if 'clinical_phase' in d:
        assert d['clinical_phase']['config']==CLINICAL_CONFIG;say(dict(stage='clinical_plan',reuse=True));return
    p=d['risksets'];bounds={k:clinical_bounds(b,p,k) for k in ['labs','chart']};plans={}
    for name in ['labs','chart']:
        path=m.PATHS[name];st=path.stat();plans[name]=dict(path=str(path),bytes=st.st_size,mtime_ns=st.st_mtime_ns,
            columns=m.COLS[name],mapping=m.LAB_MAP if name=='labs' else m.CHART_MAP,
            filter='only requested variable/hospital/ICU windows not covered by previous clinical projection',
            estimate_minutes='8-18' if name=='chart' else '4-10',estimate_peak_MB=2000)
    for name,cols in [('poe_detail',['poe_id','poe_seq','subject_id','field_name','field_value']),
                      ('poe',review.POE_COLUMNS)]:
        path=table_file('hosp',name) if name=='poe_detail' else configured_path('poe_file','../restricted/mimic-iv-3.1/hosp/poe.csv')
        plans[name]=dict(path=str(path),bytes=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns,columns=cols,
            filter='Code status field for current riskset patients; then native code POE IDs and linked D/C entries',
            estimate_minutes='1-3',estimate_peak_MB=2000)
    state=lineage_classify(d)
    d['clinical_phase']=dict(config=CLINICAL_CONFIG,utc=m.utc(),plans=plans,bounds=bounds,lineage=state,
        source_at_freeze=Path(__file__).read_bytes(),analysis_set=seq.ANALYSIS_CONFIG.copy(),no_new_outcome_contrasts=True)
    save(d)
    say(dict(stage='clinical_plan',read_GiB=round(sum(x['bytes'] for x in plans.values())/2**30,2),
        primary_new_nodes=m.count(state.primary_new.sum()),all_native_nodes=m.count(len(state)),
        lineage={k:m.count(v) for k,v in state.state.value_counts().items()},effects=0))


def clinical_project():
    b,e,d=load();m=seq.base();phase=d['clinical_phase'];assert phase['config']==CLINICAL_CONFIG
    q=b['source'].set_index('hadm_id');adm=b['admission_link'];ag={s:g for s,g in adm.groupby('subject_id',sort=False)}
    src=pd.read_pickle(m.SOURCE);native=src['icustays'].set_index('stay_id')[['hadm_id','subject_id']];del src
    for name in ['labs','chart']:
        target='supplement_'+name
        if target in phase:continue
        bounds=phase['bounds'][name];gaps=bounds[bounds.gap].copy();adms=set(gaps.hadm_id);subs=set(q.loc[list(adms),'subject_id'])
        plan=phase['plans'][name];path=Path(plan['path']);assert (path.stat().st_size,path.stat().st_mtime_ns)==(plan['bytes'],plan['mtime_ns'])
        mapping={int(i):v for v,ids in plan['mapping'].items() for i in ids if v in set(bounds.variable)}
        byvar={v:g.set_index('key') for v,g in gaps.groupby('variable')};pieces=[];counts=dict(native_identity_conflict=0,unresolved_hadm=0,stay_identity_conflict=0)
        candidates=q.loc[list(adms)].reset_index()
        def consume(raw):
            item=pd.to_numeric(raw.itemid,errors='coerce');su=pd.to_numeric(raw.subject_id,errors='coerce')
            z=raw[item.isin(mapping)&su.isin(subs)].copy()
            if z.empty:return
            z['itemid']=item.loc[z.index].astype('int64');z['subject_id']=su.loc[z.index].astype('int64')
            z['hadm_id']=pd.to_numeric(z.hadm_id,errors='coerce');z=z[z.hadm_id.isna()|z.hadm_id.isin(adms)].copy()
            if z.empty:return
            z['chart']=m.sec(z.charttime);z['store']=m.sec(z.storetime);z['join_method']='native_hadm'
            if name=='labs':
                z=seq.assign_missing(z,candidates,adm,'chart',groups=ag);counts['unresolved_hadm']+=int(z.hadm_id.isna().sum())
            z=z[z.hadm_id.isin(adms)].copy();valid=z.subject_id.eq(z.hadm_id.map(q.subject_id));counts['native_identity_conflict']+=int((~valid).sum());z=z[valid].copy()
            z=z[z.chart.ge(z.hadm_id.map(q.admit))&z.chart.lt(z.hadm_id.map(q.discharge))].copy()
            z['variable']=z.itemid.map(mapping)
            if name=='chart':
                z['stay_id']=pd.to_numeric(z.stay_id,errors='coerce');ok=z.stay_id.isna()|(z.stay_id.map(native.hadm_id).eq(z.hadm_id)&z.stay_id.map(native.subject_id).eq(z.subject_id))
                counts['stay_identity_conflict']+=int((~ok).sum());z=z[ok].copy()
            keep=np.zeros(len(z),dtype=bool)
            for var,positions in z.groupby('variable',sort=False).indices.items():
                g=byvar.get(var)
                if g is None:continue
                zz=z.iloc[positions];hospital=bool(g.hospital_key.iloc[0]);keys=zz.hadm_id if hospital else zz.stay_id
                ranges=g.reindex(keys)
                keep[positions]=outside_cached_window(zz.chart,ranges.lo,ranges.hi,ranges.old_lo,ranges.old_hi)
            z=z.iloc[np.flatnonzero(keep)].copy()
            if z.empty:return
            z['hadm_id']=z.hadm_id.astype('int64');z['valuenum']=pd.to_numeric(z.valuenum,errors='coerce')
            cols=list(b['clinical_'+name].columns)
            if name=='chart':
                z['stay_id']=z.stay_id.fillna(-1).astype('int64')
                z.loc[~z.variable.isin(list(seq.HC_CF_CATEGORICAL)+['GCS_verbal']),'value']=''
            z=z[cols]
            for col in ['variable','valueuom','join_method']+(['value'] if name=='chart' else []):z[col]=z[col].astype('category')
            pieces.append(z)
        say(dict(stage='clinical_gap_read_start',table=name,GiB=round(plan['bytes']/2**30,2),columns=plan['columns'],filter=plan['filter']))
        run=review.stream_projection(path,plan['columns'],consume,m,'order_gap_'+name)
        z=pd.concat(pieces,ignore_index=True).drop_duplicates().reset_index(drop=True) if pieces else b['clinical_'+name].iloc[:0].copy()
        assert counts['native_identity_conflict']==0 and counts['stay_identity_conflict']==0,counts
        phase[target]=z;phase.setdefault('projection_checks',{})[name]=counts;d['runs'].append(run);save(d)
        say(dict(stage='clinical_gap_read_done',table=name,rows=m.count(len(z)),seconds=run['seconds'],peak_MB=run['peak_MB'],effects=0))
    if 'poe_code_detail' not in phase:
        plan=phase['plans']['poe_detail'];path=Path(plan['path']);subs=set(d['risksets'].subject_id);pieces=[]
        def code_detail(z):
            su=pd.to_numeric(z.subject_id,errors='coerce');z=z[z.field_name.eq('Code status')&su.isin(subs)].copy()
            if len(z):pieces.append(z)
        run=review.stream_projection(path,plan['columns'],code_detail,m,'order_code_detail')
        phase['poe_code_detail']=pd.concat(pieces,ignore_index=True).drop_duplicates() if pieces else pd.DataFrame(columns=plan['columns'])
        d['runs'].append(run);save(d)
    if 'poe_code' not in phase:
        ids=set(phase['poe_code_detail'].poe_id);plan=phase['plans']['poe'];pieces=[]
        def code_poe(z):
            z=z[z.poe_id.isin(ids)|z.discontinue_of_poe_id.isin(ids)].copy()
            if len(z):pieces.append(z)
        run=review.stream_projection(Path(plan['path']),plan['columns'],code_poe,m,'order_code_poe')
        z=pd.concat(pieces,ignore_index=True).drop_duplicates() if pieces else pd.DataFrame(columns=plan['columns'])
        z['order']=review.secs(z.ordertime);phase['poe_code']=z;d['runs'].append(run);save(d)
    say(dict(stage='clinical_project_complete',new_effects=0,all_planned_projections=True))


def history_at(b,z,m,src,services):
    """Same covariate definitions as the actual-dose implementation, new indices."""
    q=b['source'].set_index('hadm_id');meta=src['adult_stays'].set_index('stay_id');x=pd.DataFrame(index=z.index)
    for col in ['age','gender','anchor_year_group','ne_start_left_censored']:x[col]=z.hadm_id.map(q[col])
    offset=pd.to_datetime(z['index'],unit='s').dt.year.to_numpy()-z.hadm_id.map(q.anchor_year).to_numpy()
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
    x['NE_state']='UNKNOWN';groups=list(z.groupby('stay_id',sort=False));ng={s:g for s,g in b['NE'].groupby('stay_id',sort=False)}
    started=time.perf_counter();last=started
    for j,(s,idx) in enumerate(groups):
        g=ng.get(s)
        if g is None:continue
        t=idx['index'].to_numpy();g=g[g.start.lt(t.max())&g.end.ge(t.min()-14400)].copy()
        dose,weight,flag=seq.ne_values_at(g,t)
        x.loc[idx.index,'NE_dose']=dose;x.loc[idx.index,'weight']=weight;x.loc[idx.index,'NE_state']=flag
        for hour in [1,4]:x.loc[idx.index,'NE_dose_'+str(hour)+'h']=seq.ne_values_at(g,t-hour*3600)[0]
        nodes=np.unique(np.r_[g.start.to_numpy(),g.end.to_numpy()]);mid=(nodes[:-1]+nodes[1:])/2
        vv,_,_=seq.ne_values_at(g,mid);valid=np.isfinite(vv);aa=nodes[:-1][valid];ee=nodes[1:][valid];dd=vv[valid]
        overlap=np.maximum(0,np.minimum(ee[:,None],t[None,:])-np.maximum(aa[:,None],t[None,:]-3600))
        x.loc[idx.index,'NE_prior1h_covered_min']=overlap.sum(axis=0)/60
        peak=np.max(np.where(overlap>0,dd[:,None],-np.inf),axis=0) if len(dd) else np.full(len(t),np.nan)
        peak[~np.isfinite(peak)]=np.nan;x.loc[idx.index,'NE_prior1h_peak']=peak
        if time.perf_counter()-last>30:say(dict(stage='order_history',percent=round(100*(j+1)/len(groups),1)));last=time.perf_counter()
    x['NE_change_1h']=x.NE_dose-x.NE_dose_1h;x['NE_change_4h']=x.NE_dose-x.NE_dose_4h
    gate=m.import_file('hc_order_history_normalizer',m.OLD/'clinical_tte_gate.py');norm=gate.normalize(src['events'],src['icustays'])
    for label,ids in {'VP':[222315],'EPI':[221289,229617],'PHENYLEPHRINE':[221749,229630,229631,229632],'DOPAMINE':[221662]}.items():
        x[label+'_record']='NO_PRIOR_RECORD';gg={s:g for s,g in norm[norm.itemid.isin(ids)].groupby('stay_id',sort=False)}
        for s,idx in groups:
            g=gg.get(s)
            if g is None:continue
            tt=idx['index'].to_numpy();aa=g.start.to_numpy();ee=g.end.to_numpy();rr=g.rate.to_numpy()
            active=(aa[:,None]<tt[None,:])&(ee[:,None]>=tt[None,:]);prior=(aa[:,None]<tt[None,:]).any(axis=0)
            val=np.full(len(tt),'NO_PRIOR_RECORD',dtype=object);val[prior]='PRIOR_RECORD_CURRENT_UNKNOWN'
            val[(active&np.isfinite(rr[:,None])&(rr[:,None]>0)).any(axis=0)]='ACTIVE_RECORDED'
            val[(active&~np.isfinite(rr[:,None])).any(axis=0)]='ACTIVE_DOSE_UNKNOWN'
            x.loc[idx.index,label+'_record']=val
    x['service']='UNKNOWN';sg={h:g.sort_values('t') for h,g in services.groupby('hadm_id',sort=False)}
    for h,idx in z.groupby('hadm_id',sort=False):
        g=sg.get(h)
        if g is None:continue
        loc=np.searchsorted(g.t.to_numpy(),idx['index'].to_numpy(),side='left')-1;good=loc>=0
        x.loc[idx.index[good],'service']=g.curr_service.to_numpy()[loc[good]]
    for col in x.select_dtypes(include=['float']).columns:x[col]=x[col].astype('float32')
    for col in x.select_dtypes(include=['object']).columns:x[col]=x[col].astype('category')
    return x


def limitation_bits(value):
    """Structured code values only; unknown dimensions are never filled with zero."""
    s=str(value).strip().lower()
    dnr=bool(re.search(r'\bdnr\b|\bdnar\b|do not resusc|not attempt resusc',s))
    dni=bool(re.search(r'\bdni\b|do not intubat|no intubation',s))
    cmo=bool(re.search(r'\bcmo\b|comfort',s));full=bool(re.search(r'full\s*code',s))
    if full and (dnr or dni or cmo):return np.nan
    bits=int(full)+2*int(dnr)+4*int(dni)+8*int(cmo)
    return float(bits) if bits else np.nan


def code_order_events(phase,source):
    detail=phase['poe_code_detail'];poe=phase['poe_code'];q=source.set_index('hadm_id');rows=[];checks=dict(identity_conflicts=0,unlinked_detail=0,ambiguous_native_keys=0)
    pg={k:g for k,g in poe.groupby('poe_id',sort=False)}
    stops={k:g for k,g in poe[poe.transaction_type.eq('D/C')].groupby('discontinue_of_poe_id',sort=False)}
    for key,g in detail.groupby('poe_id',sort=False):
        z=pg.get(key)
        if z is None:checks['unlinked_detail']+=1;continue
        if len(z)!=1:checks['ambiguous_native_keys']+=1;continue
        r=z.iloc[0];hh=pd.to_numeric(r.hadm_id,errors='coerce');ss=pd.to_numeric(r.subject_id,errors='coerce')
        if not np.isfinite(hh) or int(hh) not in q.index:continue
        h=int(hh)
        if ss!=q.loc[h,'subject_id'] or not pd.to_numeric(g.subject_id,errors='coerce').eq(ss).all():checks['identity_conflicts']+=1;continue
        t=float(r.order)
        if not q.loc[h,'admit']<=t<q.loc[h,'discharge']:continue
        parsed=g.field_value.map(limitation_bits);unique=parsed.dropna().unique()
        bits=float(unique[0]) if len(unique)==1 and parsed.notna().all() and r.transaction_type in ['New','Change'] else np.nan
        stop=stops.get(key,poe.iloc[:0])
        stop=stop[pd.to_numeric(stop.hadm_id,errors='coerce').eq(h)&pd.to_numeric(stop.subject_id,errors='coerce').eq(ss)&stop.order.ge(t)]
        end=float(stop.order.min()) if len(stop) else np.inf
        rows.append(dict(hadm_id=h,time=t,bits=bits,end=end,poe_id=key))
    out=pd.DataFrame(rows,columns=['hadm_id','time','bits','end','poe_id'])
    assert checks['identity_conflicts']==0,checks
    return out,checks


def order_code_at(events,queries,admit):
    n=len(queries);vals=np.full(n,np.nan);times=np.full(n,np.nan);flags=np.zeros(n,dtype='uint8')
    if events.empty:return vals,times,flags
    g=events.sort_values(['time','poe_id']);duplicate=g.time.duplicated(False);values=g.bits.to_numpy(float)
    valid=np.isfinite(values)&~duplicate.to_numpy();tt=g.time.to_numpy(float)
    got=seq.hc_cf_asof(tt,tt,values,valid,queries,admit,True)
    vals=got[0].astype(float);times=got[3];flags=got[2]
    ends=g.drop_duplicates('time').set_index('time').end
    selected_end=pd.Series(times).map(ends).to_numpy()
    expired=np.isfinite(times)&(selected_end<queries)
    vals[expired]=np.nan;flags[expired]=4;times[expired]=selected_end[expired]
    return vals,times,flags


def combine_directives(chart_value,chart_time,order_value,order_time):
    cv=np.asarray(chart_value,float);ct=np.asarray(chart_time,float);ov=np.asarray(order_value,float);ot=np.asarray(order_time,float)
    use_order=np.isfinite(ot)&(~np.isfinite(ct)|(ot>ct));value=np.where(use_order,ov,cv)
    tie=np.isfinite(ot)&np.isfinite(ct)&(ot==ct)
    agree=np.isfinite(cv)&np.isfinite(ov)&(cv==ov)
    value[tie&~agree]=np.nan
    disagreement=np.where(np.isfinite(cv)&np.isfinite(ov),cv!=ov,np.nan)
    return value,disagreement


def add_code_features(phase,b,p,features,chart):
    q=b['source'].set_index('hadm_id');times=p['index'].to_numpy();admit=p.hadm_id.map(q.admit).to_numpy()
    events,checks=code_order_events(phase,b['source']);eg={k:g for k,g in events.groupby('hadm_id',sort=False)}
    cg={k:g for k,g in chart[chart.variable.eq('code_status')].groupby('hadm_id',sort=False)}
    poev=np.full(len(p),np.nan);poet=np.full(len(p),np.nan);poef=np.zeros(len(p),dtype='uint8')
    chartvals={k:np.full(len(p),np.nan) for k in ['primary','strict']};charttimes={k:np.full(len(p),np.nan) for k in chartvals}
    for h,ix in p.groupby('hadm_id',sort=False).indices.items():
        g=eg.get(h)
        if g is not None:poev[ix],poet[ix],poef[ix]=order_code_at(g,times[ix],admit[ix])
        g=cg.get(h)
        if g is None:continue
        val=g.value.astype(object).map(limitation_bits).to_numpy(float);good=np.isfinite(val)
        for clock in chartvals:
            got=seq.hc_cf_asof(g.chart.to_numpy(float),g.store.to_numpy(float),val,good,times[ix],admit[ix],clock=='strict')
            chartvals[clock][ix]=got[0];charttimes[clock][ix]=got[3]
    for clock in ['primary','strict']:
        f=features[clock];value,discord=combine_directives(chartvals[clock],charttimes[clock],poev,poet)
        f['POE_code_known']=np.isfinite(poev).astype('uint8');f['code_source_discordance']=discord.astype('float32')
        bits=np.nan_to_num(value,nan=0).astype('uint8');full=(bits&1)>0
        for name,bit in [('recorded_DNR',2),('recorded_DNI',4),('recorded_CMO',8)]:
            f['chart_'+name]=f[name].copy();out=np.full(len(f),np.nan,dtype='float32')
            out[full]=0.;out[(bits&bit)>0]=1.;f[name]=out
        f['chart_code_status_unknown']=f.code_status_unknown.copy()
        f['recorded_full_code']=np.where(np.isfinite(value),full,np.nan).astype('float32')
        f['code_status_unknown']=(~np.isfinite(value)).astype('uint8')
    phase['code_events']=events;phase['code_checks']=checks
    phase['code_selected']=dict(poe_value=poev,poe_time=poet,poe_flag=poef,chart_value=chartvals,chart_time=charttimes)


def clinical_features():
    b,e,d=load();m=seq.base();phase=d['clinical_phase'];assert phase['config']==CLINICAL_CONFIG
    assert all(k in phase for k in ['supplement_labs','supplement_chart','poe_code'])
    if 'features' in phase:say(dict(stage='clinical_features',reuse=True));return
    start=time.perf_counter();p=d['risksets'].reset_index(drop=True);src=pd.read_pickle(m.SOURCE)
    native=src['icustays'].set_index('stay_id')
    assert native.index.is_unique
    assert p.stay_id.map(native.hadm_id).eq(p.hadm_id).all()
    assert p.stay_id.map(native.subject_id).eq(p.subject_id).all()
    phase['source_reference']=dict(path=str(m.SOURCE),bytes=m.SOURCE.stat().st_size,mtime_ns=m.SOURCE.stat().st_mtime_ns)
    service=table_file('hosp','services');ss=pd.read_csv(service,usecols=['hadm_id','transfertime','curr_service']);ss['t']=m.sec(ss.transfertime)
    phase['services_source']=dict(path=str(service),bytes=service.stat().st_size,sha256=seq.sha(service))
    # Fixed old cases validate reuse, without fitting or consulting outcomes.
    old=b['full_risksets'][b['full_risksets'].treated.eq(1)].iloc[::11].copy();ref=b['history_features'].loc[old.index].reset_index(drop=True)
    old=old.reset_index(drop=True);check=history_at(b,old,m,src,ss)
    for col in ref:
        if pd.api.types.is_numeric_dtype(ref[col]):assert np.allclose(ref[col],check[col],equal_nan=True,rtol=1e-5,atol=1e-5),col
        else:assert ref[col].astype(object).fillna('').astype(str).equals(check[col].astype(object).fillna('').astype(str)),col
    hist=history_at(b,p,m,src,ss);del src
    temp=dict(source=b['source'],full_risksets=p)
    for name in ['labs','chart']:
        temp['clinical_'+name]=pd.concat([b['clinical_'+name],phase['supplement_'+name]],ignore_index=True).drop_duplicates().reset_index(drop=True)
    feat=seq.hc_cf_features(temp,m);add_code_features(phase,b,p,feat,temp['clinical_chart'])
    primary_ids=set(phase['lineage'].loc[phase['lineage'].primary_new,'case_id'])
    cmo=feat['primary'].recorded_CMO.eq(1);case_cmo=set(p.loc[p.treated.eq(1)&cmo,'case_id'])
    scenario_ids=dict(primary_new=primary_ids,all_native=set(d['cases'].case_id))
    masks={label:p.case_id.isin(ids-case_cmo)&~cmo for label,ids in scenario_ids.items()}
    support=[]
    for label,mask in masks.items():
        keep=p[mask];cnt=keep.groupby('case_id').treated.agg(['sum','size']);assert cnt['sum'].eq(1).all()
        empty=set(cnt.index[cnt['size'].le(1)])
        masks[label]=mask&~p.case_id.isin(empty);retained=p[masks[label]]
        support.append(dict(scenario=label,cases=int(retained.treated.sum()),empty_risksets=len(empty),
            control_patients=retained.loc[retained.treated.eq(0),'subject_id'].nunique(),
            CMO_cases=len(case_cmo&scenario_ids[label]),
            CMO_control_rows=int((p.case_id.isin(scenario_ids[label]-case_cmo)&cmo&p.treated.eq(0)).sum())))
    checks=seq.hc_cf_tests();checks.update(history_reproduces_frozen_sample=True,source_keys_match_same_version_reference=True,
        all_new_projection_complete=True,own_index_cutoffs=True,strict_and_primary_membership_same=True,
        unknown_limitations_not_full_code=True,no_new_outcome_contrasts=True)
    q=b['source'].set_index('hadm_id')
    endpoint_checks={}
    for label,mask in masks.items():
        z=p.loc[mask];a=q.reindex(z.hadm_id);tt=z['index'].to_numpy()
        endpoint_checks[label]=dict(
            admissions=int(z.hadm_id.nunique()),
            missing_or_invalid_expire_flag=int(a.loc[~a.hospital_expire_flag.isin([0,1])].index.nunique()),
            invalid_admission_index_rows=int((~((a.admit.to_numpy()<=tt)&(tt<a.discharge.to_numpy()))).sum()),
            death_at_or_before_index_rows=int((a.death_time.to_numpy()<=tt).sum()),
            alive_flag_with_inhospital_death=int(a.loc[a.hospital_expire_flag.eq(0)&a.death_time.ge(a.admit)&a.death_time.le(a.discharge)].index.nunique()),
            ICU_transfer_not_censoring=True,alive_discharge_is_terminal_zero=True)
    phase.update(history=hist,features=feat,analysis_masks=masks,final_support=pd.DataFrame(support),tests=checks,
        endpoint_checks=endpoint_checks,
        feature_seconds=round(time.perf_counter()-start,2),feature_peak_MB=m.peak_mb(),source_current=Path(__file__).read_bytes())
    save(d)
    say(dict(stage='clinical_features_done',seconds=phase['feature_seconds'],peak_MB=phase['feature_peak_MB'],
        support=[dict(scenario=r['scenario'],cases=m.count(r['cases']),empty_risksets=m.count(r['empty_risksets']),control_patients=m.count(r['control_patients']),CMO_cases=m.count(r['CMO_cases'])) for r in support],effects=0))


def analysis_frame(d,scenario):
    phase=d['clinical_phase'];clock='strict' if scenario=='strict_new' else 'primary'
    membership='all_native' if scenario=='primary_all_native' else 'primary_new'
    bundle=dict(full_risksets=d['risksets'],history_features=phase['history'],new_clinical_features=phase['features'])
    x=seq.analysis_frame(bundle,clock)
    return x.loc[phase['analysis_masks'][membership]].reset_index(drop=True)


def analysis_freeze():
    b,e,d=load();phase=d['clinical_phase'];m=seq.base();assert 'features' in phase
    spec=dict(version='independent_HC_order_ATT_1',numerical=seq.ANALYSIS_CONFIG['numerical'],
        categorical=seq.ANALYSIS_CONFIG['categorical']+['POE_code_known','code_source_discordance'],
        rationale='reuse fixed illness/care-setting/NE-trajectory set at order index; added code-source presence/difference because newly available directive source can alter recorded treatment limits',
        scenarios=['primary_new','strict_new','primary_all_native'],
        interpretation='ATT for recorded first HC ordering decisions during NE; subsequent usual care; exploratory, clinical indication not confirmed',
        score='fixed conditional logistic sum-loss L2=1; no tuning, clipping or selection by outcome',
        weights='case mass1; full same-elapsed control pool normalized to1 for each case',
        outcome='hospital_expire_flag in complete index hospitalization; alive discharge0, no ICU-transfer censoring; missing never alive',
        uncertainty='ordinary source-patient bootstrap, all admissions and riskset roles together, preprocessing and score refit;100 initial then500 after validity review',
        bootstrap_seed=523,bootstrap_repetitions_initial=100,
        reporting='all specified scenarios; intervals describe uncertainty rather than equivalence or treatment recommendations',
        prior_outcomes_seen='actual-dose MIMIC and eICU order results seen; current order-group mortality contrasts not seen')
    if 'analysis_phase' in d:assert d['analysis_phase']['config']==spec;say(dict(stage='analysis_freeze',reuse=True));return
    d['analysis_phase']=dict(config=spec,utc=m.utc(),signature=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest(),
        source_at_freeze=Path(__file__).read_bytes(),models=0,effects=0)
    save(d);say(dict(stage='analysis_freeze',scenarios=spec['scenarios'],numerical=len(spec['numerical']),categorical=len(spec['categorical']),effects=0))


def clinical_score():
    b,e,d=load();m=seq.base();phase=d['clinical_phase'];ap=d['analysis_phase'];spec=ap['config']
    assert ap['signature']==hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    phase.setdefault('assignment',{})
    for scenario in spec['scenarios']:
        if scenario in phase['assignment']:continue
        frame=analysis_frame(d,scenario);assert not any('death' in c or 'outcome' in c for c in frame)
        plan=seq.make_plan(frame,spec['numerical'],spec['categorical'],diagnostic_numeric=['NE_dose'])
        say(dict(stage='order_score_start',scenario=scenario,rows=m.count(len(frame)),cases=m.count(frame.treated.sum()),question='fixed pretreatment support, no new mortality contrast'))
        rng=np.random.RandomState(spec['bootstrap_seed']);resampling_failures={}
        for _ in range(spec['bootstrap_repetitions_initial']):
            try:seq.multiplicities(plan,seq.draw_patient_counts(plan,rng,b['source'].subject_id.unique()))
            except seq.EstimatorFailure as exc:resampling_failures[str(exc)]=resampling_failures.get(str(exc),0)+1
        fit=seq.fit_score(plan,l2=1.);ww=seq.att_weights(plan,fit['linear_predictor'],fit['mult'])
        bal=seq.balance_diagnostics(plan,ww);con,per=seq.concentration_diagnostics(plan,ww)
        within=seq.within_set_balance(plan,ww,['NE_dose','MAP','lactate','NE_change_1h']);local=seq.local_similarity_diagnostics(plan,ww)
        phase['assignment'][scenario]=dict(plan_sha256=plan['plan_sha256'],beta=fit['beta'],linear_predictor=fit['linear_predictor'],fit=fit['fit'],
            balance=bal,concentration=con,per_case=per,within=within,local=local,cases=int(frame.treated.sum()),rows=len(frame),
            ordinary_resampling=dict(attempts=spec['bootstrap_repetitions_initial'],failures=resampling_failures,outcomes_used=False))
        ap['models']+=1;save(d)
        say(dict(stage='order_score_done',scenario=scenario,seconds=round(fit['fit']['seconds'],1),
            maximum_abs_SMD=round(bal.smd_after.abs().max(),4),control_patient_ESS=round(con['control_patient']['ess'],1),
            max_patient_weight_percent=round(100*con['control_patient']['maximum_share'],2),warnings=len(fit['fit']['warnings']),
            resampling_structure_failures=sum(resampling_failures.values()),effects=0))
        del frame,plan,fit,ww


def estimation_review():
    """Record the outcome-blind review; no automatic SMD-only causal gate."""
    b,e,d=load();phase=d['clinical_phase'];ap=d['analysis_phase']
    if 'support_decision' in ap:say(dict(stage='estimation_review',reuse=True));return
    assert set(phase['assignment'])==set(ap['config']['scenarios'])
    assert ap['effects']==0
    for checks in phase['endpoint_checks'].values():
        for key in ['missing_or_invalid_expire_flag','invalid_admission_index_rows',
                    'death_at_or_before_index_rows','alive_flag_with_inhospital_death']:
            assert checks[key]==0,(key,checks[key])
    assert phase['code_checks']['identity_conflicts']==0
    assert phase['code_checks']['unlinked_detail']==phase['code_checks']['ambiguous_native_keys']==0
    # This rationale follows explicit inspection of all three fixed diagnostics,
    # measurement coverage and outcome linkage, before any new death contrast.
    ap['support_decision']=dict(
        decision='PROCEED_EXPLORATORY_ORDER_ASSOCIATION',utc=seq.base().utc(),
        rationale='可以进入已固定的探索性医嘱关联估计。独立起点、院内结局链和记录资格可解释；MAP、乳酸、NE历史及其他病情记录有调整基础，三套模型未见明显残余整体失衡或少数患者主导。局部支持仍不均匀，完整GCS覆盖较低，部分治疗限制与HC实际用途未知，因此不把本轮结果解释为已验证的HC治疗效果。上述未知不要求清零，也不以SMD代替识别假设。',
        before_new_mortality_contrasts=True,
        ordinary_structure_checks={name:phase['assignment'][name]['ordinary_resampling'] for name in ap['config']['scenarios']},
        plan_hashes={name:phase['assignment'][name]['plan_sha256'] for name in ap['config']['scenarios']},
        workers=1,execution_reason='serial CPU estimation preserves Python3.8 per-fit warning capture and bounds memory; no statistical settings changed',
        uncertainty='all100 planned source-patient draws retained; no replacement of failures; percentile intervals among valid estimates with denominators reported; incomplete intervals require review')
    ap['source_at_estimation_review']=Path(__file__).read_bytes()
    save(d);say(dict(stage='estimation_review',decision=ap['support_decision']['decision'],effects=0))


def bootstrap_summary(table):
    result=dict(attempts=int(len(table)),successful=int(table.success.sum()),
                failures=int((~table.success).sum()),intervals={})
    result['failure_reasons']=table.loc[~table.success,'error'].value_counts().to_dict() if 'error' in table else {}
    result['warning_replicates']=int(sum(bool(x) for x in table.loc[table.success,'warnings'])) if 'warnings' in table else 0
    for field in ['risk_treated','risk_control','risk_difference','risk_ratio']:
        values=pd.to_numeric(table.loc[table.success,field],errors='coerce') if field in table else pd.Series(dtype=float)
        finite=values[np.isfinite(values)]
        limits=np.quantile(finite.to_numpy(),[.025,.975]) if len(finite) else [None,None]
        result['intervals'][field]=dict(finite_n=int(len(finite)),undefined_n=int(len(values)-len(finite)),
            lower=float(limits[0]) if len(finite) else None,
            upper=float(limits[1]) if len(finite) else None)
    result['preliminary_100']=True
    if len(table)>=20:
        half=len(table)//2;limits=[]
        for part in [table.iloc[:half],table.iloc[half:]]:
            values=pd.to_numeric(part.loc[part.success,'risk_difference'],errors='coerce')
            values=values[np.isfinite(values)]
            limits.append(np.quantile(values,[.025,.975]) if len(values) else np.full(2,np.nan))
        result['RD_halves_endpoint_max_difference_pp']=float(100*np.max(np.abs(limits[0]-limits[1])))
        result['MC_interpretation']='first/second half comparison is a rough endpoint-sensitivity check, not proof of interval convergence'
    return result


def order_estimate():
    import threading
    b,e,d=load();phase=d['clinical_phase'];ap=d['analysis_phase'];spec=ap['config'];decision=ap['support_decision']
    assert decision['decision']=='PROCEED_EXPLORATORY_ORDER_ASSOCIATION'
    assert ap['signature']==hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    assert decision['plan_hashes']=={s:phase['assignment'][s]['plan_sha256'] for s in spec['scenarios']}
    ap.setdefault('estimates',{});lock=threading.Lock();m=seq.base();source=b['source'].set_index('hadm_id')
    def run_one(scenario):
        if ap['estimates'].get(scenario,{}).get('complete'):
            say(dict(stage='order_estimate',scenario=scenario,reuse=True));return
        frame=analysis_frame(d,scenario)
        plan=seq.make_plan(frame,spec['numerical'],spec['categorical'],diagnostic_numeric=['NE_dose'])
        assert plan['plan_sha256']==decision['plan_hashes'][scenario]
        y=frame.hadm_id.map(source.hospital_expire_flag).to_numpy(float)
        assignment=phase['assignment'][scenario];weights=seq.att_weights(plan,assignment['linear_predictor'])
        point=seq.binary_risks(plan,weights,y)
        crude=seq.binary_risks(plan,dict(treated=weights['treated'],control=weights['control_reference']),y)
        with lock:
            state=ap['estimates'].setdefault(scenario,{})
            if 'point' in state:assert state['point']==point and state['crude']==crude
            else:
                state.update(point=point,crude=crude,plan_sha256=plan['plan_sha256'],first_estimated_utc=m.utc())
                ap['effects']+=1
            save(d)
        say(dict(stage='order_estimate_started',scenario=scenario,repetitions=spec['bootstrap_repetitions_initial']))
        def progress(rec):
            if (rec['replicate']+1)%10==0 or not rec['success']:
                say(dict(stage='order_patient_bootstrap',scenario=scenario,completed=rec['replicate']+1,
                    success=rec['success'],replicate_seconds=round(rec['seconds'],1)))
        def checkpoint(payload):
            with lock:state['bootstrap_state']=payload;save(d)
        table=seq.bootstrap_att(plan,y,repetitions=spec['bootstrap_repetitions_initial'],seed=spec['bootstrap_seed'],
            source_patient_ids=b['source'].subject_id.unique(),l2=1.,progress=progress,
            checkpoint_callback=checkpoint,checkpoint_every=10,resume_state=state.get('bootstrap_state'),
            analysis_signature=ap['signature'])
        summary=bootstrap_summary(table)
        with lock:
            state.update(bootstrap=table,summary=summary,complete=len(table)==spec['bootstrap_repetitions_initial'],
                completed_utc=m.utc(),peak_process_MB=m.peak_mb())
            save(d)
        say(dict(stage='order_estimate_complete',scenario=scenario,successful=summary['successful'],
            failures=summary['failures'],warning_replicates=summary['warning_replicates'],
            RD_percentage_points=round(100*point['risk_difference'],3),
            RD_interval={k:(round(100*v,3) if k in ['lower','upper'] and v is not None else v)
                for k,v in summary['intervals']['risk_difference'].items()}))
    for scenario in spec['scenarios']:run_one(scenario)
    ap['result_status']='THREE_FIXED_ORDER_ASSOCIATIONS_100_SOURCE_PATIENT_BOOTSTRAPS'
    ap['source_current']=Path(__file__).read_bytes();save(d)
    say(dict(stage='order_estimate_all_complete',effects=ap['effects'],bootstrap_per_scenario=spec['bootstrap_repetitions_initial'],
        frozen_actual_dose_unchanged=True,peak_process_MB=m.peak_mb()))


def order_validate():
    b,e,d=load();ap=d['analysis_phase'];spec=ap['config'];phase=d['clinical_phase']
    assert ap['result_status']=='THREE_FIXED_ORDER_ASSOCIATIONS_100_SOURCE_PATIENT_BOOTSTRAPS'
    q=b['source'].set_index('hadm_id');checks={}
    for name in spec['scenarios']:
        r=ap['estimates'][name];f=analysis_frame(d,name)
        plan=seq.make_plan(f,spec['numerical'],spec['categorical'],diagnostic_numeric=['NE_dose'])
        assert r['plan_sha256']==plan['plan_sha256']==phase['assignment'][name]['plan_sha256']
        weights=seq.att_weights(plan,phase['assignment'][name]['linear_predictor'])
        assert np.allclose(np.add.reduceat(weights['control'],plan['starts']),1.)
        assert np.isclose(weights['control'].sum(),weights['treated'].sum())
        y=f.hadm_id.map(q.hospital_expire_flag).to_numpy(float)[plan['order']]
        risks={}
        for arm in ['treated','control']:
            den=np.bincount(plan['patient'],weights=weights[arm],minlength=len(plan['patient_ids']))
            num=np.bincount(plan['patient'],weights=weights[arm]*y,minlength=len(plan['patient_ids']))
            risks[arm]=num.sum()/den.sum()
            assert np.isclose(risks[arm],r['point']['risk_'+arm],rtol=0,atol=1e-12)
        assert np.isclose(risks['treated']-risks['control'],r['point']['risk_difference'],rtol=0,atol=1e-12)
        table=r['bootstrap'];state=r['bootstrap_state'];n=spec['bootstrap_repetitions_initial']
        assert table.replicate.tolist()==list(range(n)) and len(state['results'])==n
        assert state['specification']['analysis_signature']==ap['signature']
        assert state['specification']['plan_sha256']==plan['plan_sha256']
        assert state['specification']['seed']==spec['bootstrap_seed']
        prior=r['summary'];fresh=bootstrap_summary(table)
        assert fresh['intervals']==prior['intervals']
        r.setdefault('initial_summary',prior);r['summary']=fresh
        checks[name]=dict(riskset_mass=True,patient_aggregate_risks_agree=True,
            complete_planned_replicate_sequence=True,no_silent_replacement=True,
            source_patient_roles_shared=True,preprocessing_and_score_refit_each_draw=True,
            diagnostics_match_estimation_frame=True,intervals_unchanged=True)
    a=analysis_frame(d,'primary_new');s=analysis_frame(d,'strict_new')
    assert a[['case_id','subject_id','hadm_id','treated']].equals(s[['case_id','subject_id','hadm_id','treated']])
    ap['validation']=dict(checks=checks,strict_membership_same=True,original_actual_dose_cache_unchanged=True,
        no_new_fits=True,utc=seq.base().utc(),source=Path(__file__).read_bytes())
    save(d)
    say(dict(stage='order_validate',passed=True,new_fits=0,MC_endpoint_half_difference_pp={
        name:round(ap['estimates'][name]['summary']['RD_halves_endpoint_max_difference_pp'],2) for name in spec['scenarios']}))


PRECISION_CONFIG=dict(
    version='HC_order_precision_1',initial_target=500,maximum_target=1000,
    scenarios=['mimic:primary_new','mimic:strict_new','mimic:primary_all_native',
               'eicu:primary','eicu:strict','eicu:known_NE'],
    fixed='same populations, plan hashes, preprocessing, L2, outcomes, seeds and estimands; resume original100 draws',
    mimic_inference='ordinary source-patient bootstrap; all admissions and riskset roles together',
    eicu_inference='normalized Exp(1) patient weights, conditional on observed patients/sites; not an ordinary bootstrap CI or new-hospital inference',
    MC_rule='RD percentile endpoint binomial-order-statistic95% Monte Carlo bands; extend500 to1000 if either band width exceeds2 percentage points; retain unresolved numerical imprecision at1000',
    MC_max_band_width_pp=2.,workers=2,checkpoint_every=25,
    failures='retain every attempt and warning, no replacement or seed search; any failure requires interpretation before interval use',
    bias='ATT additive scenario: bias=gamma0*(p_U_treated-p_U_weighted_control); constant untreated outcome contrast gamma0 and prevalence difference across measured covariate strata',
    bias_gamma=[.10,.20,.30],bias_delta=[-.30,-.20,-.10,0.,.10,.20,.30],
    bias_interpretation='researcher-selected scenarios, not estimated or clinically validated; shift RD and conditional sampling interval by the same fixed bias, not a new identified effect',
    prior_results_seen=True,raw_scans=0)


def mc_endpoint_bands(values,alpha=.05):
    """Order-statistic bands for finite simulation error, not clinical uncertainty."""
    from scipy.stats import binom
    values=np.sort(np.asarray(values,float));values=values[np.isfinite(values)];n=len(values)
    out=[]
    for p in [.025,.975]:
        # With K~Binomial(n,p), X_(lo) <= q_p <= X_(hi) iff lo<=K<hi.
        lo=int(binom.ppf(alpha/2,n,p));hi=int(binom.ppf(1-alpha/2,n,p))+1
        lower=float(values[lo-1]) if lo>=1 else -np.inf
        upper=float(values[hi-1]) if hi<=n else np.inf
        coverage=float(binom.cdf(hi-1,n,p)-binom.cdf(lo-1,n,p))
        out.append(dict(probability=p,lower=lower,upper=upper,width_pp=100*(upper-lower),
            lower_rank=lo,upper_rank=hi,binomial_coverage=coverage))
    return out


def standardized_bootstrap(table,kind):
    t=pd.DataFrame(table).copy()
    if kind=='eicu':t=t.rename(columns={'initiated_risk':'risk_treated','not_yet_risk':'risk_control'})
    return t


def precision_summary(records,kind):
    t=standardized_bootstrap(records,kind);s=bootstrap_summary(t)
    s['preliminary_100']=len(t)<500
    v=t.loc[t.success,'risk_difference'].to_numpy(float)
    s['MC_bands']=mc_endpoint_bands(v)
    s['MC_numerical_target_met']=all(z['width_pp']<=PRECISION_CONFIG['MC_max_band_width_pp'] for z in s['MC_bands'])
    s['cumulative_RD_intervals']={}
    for n in [100,250,500,1000]:
        if len(t)>=n:
            z=t.iloc[:n];v=z.loc[z.success,'risk_difference'].to_numpy(float)
            s['cumulative_RD_intervals'][n]=np.quantile(v,[.025,.975]).tolist()
    return s


def precision_plan():
    import copy
    import steroid_eicu as eicu
    b,e,d=load();ap=d['analysis_phase'];assert 'validation' in ap
    if 'precision_phase' in d:
        assert d['precision_phase']['config']==PRECISION_CONFIG
        say(dict(stage='precision_plan',reuse=True));return
    eb=eicu.load();ed=eb['clinical_extension']['order_estimation']
    assert ed['config']==eicu.ORDER_ESTIMATION_CONFIG and 'validation' in ed
    signature=(eicu.STATE.stat().st_size,eicu.STATE.stat().st_mtime_ns)
    px=dict(config=PRECISION_CONFIG,utc=seq.base().utc(),jobs={},
        original_mimic_analysis_signature=ap['signature'],eicu_source_signature=signature,
        eicu_source_sha256=seq.sha(eicu.STATE),source_at_freeze=Path(__file__).read_bytes(),
        existing_eicu_site_sensitivity=copy.deepcopy(ed['largest_site_exclusion']))
    for key in PRECISION_CONFIG['scenarios']:
        kind,name=key.split(':')
        if kind=='mimic':
            old=ap['estimates'][name];state=copy.deepcopy(old['bootstrap_state']);point=old['point']
        else:
            old=ed['results'][name];p=eicu.order_plan(ed['scenarios'][name]['frame'],seq)
            assert p['plan_sha256']==ed['scenarios'][name]['plan_sha256']
            # Confirm the saved random stream really follows the original100 draws.
            rng=np.random.RandomState(ed['config']['bootstrap_seed'])
            for _ in range(100):rng.exponential(1.,len(p['patient_ids']))
            assert rng_equal(rng.get_state(),old['rng_state'])
            state=dict(results=copy.deepcopy(old['bootstrap']),next_replicate=100,rng_state=old['rng_state'],
                specification=dict(plan_sha256=p['plan_sha256'],seed=ed['config']['bootstrap_seed'],
                    config_sha256=hashlib.sha256(json.dumps(ed['config'],sort_keys=True).encode()).hexdigest()))
            point=dict(old['point'],risk_treated=old['point']['initiated_risk'],risk_control=old['point']['not_yet_risk'])
        assert state['next_replicate']==100
        px['jobs'][key]=dict(kind=kind,scenario=name,state=state,original100=copy.deepcopy(state['results']),
            point=copy.deepcopy(point),target=500,complete=False,prior_intervals=copy.deepcopy(old.get('summary',old.get('uncertainty'))))
    d['precision_phase']=px;save(d)
    say(dict(stage='precision_plan',scenarios=6,initial_repetitions=500,maximum_repetitions=1000,
        workers=2,raw_scans=0,expected_peak_total_GiB='6-9',expected_minutes='60-90',prior100_preserved=True))


def rng_equal(a,b):
    return a[0]==b[0] and np.array_equal(a[1],b[1]) and a[2:]==b[2:]


def _precision_init(channel):
    global _PRECISION_CHANNEL
    _PRECISION_CHANNEL=channel


def _precision_worker(job):
    """Pure worker: private arrays in memory; one parent owns all checkpoint I/O."""
    key=job['key'];kind=job['kind'];p=job['plan'];y=job['outcome'];start=time.perf_counter()
    def checkpoint(payload):_PRECISION_CHANNEL.put((key,'checkpoint',payload))
    if kind=='mimic':
        table=seq.bootstrap_att(p,y,repetitions=job['target'],seed=job['seed'],source_patient_ids=job['population'],
            l2=1.,checkpoint_callback=checkpoint,checkpoint_every=PRECISION_CONFIG['checkpoint_every'],
            resume_state=job['state'],analysis_signature=job['analysis_signature'])
        state=table.attrs['checkpoint_state']
    else:
        import steroid_eicu as eicu
        state=job['state'];assert state['specification']['plan_sha256']==p['plan_sha256']
        rng=np.random.RandomState(state['specification']['seed']);rng.set_state(state['rng_state'])
        records=[dict(z) for z in state['results']]
        for rep in range(len(records),job['target']):
            row=dict(replicate=rep,success=False)
            try:
                v=rng.exponential(1.,len(p['patient_ids']));v=v*len(v)/v.sum()
                fit=eicu.order_fit_positive(seq,p,v);w=seq.att_weights(p,fit['linear_predictor'],fit['mult'])
                row.update(eicu.order_endpoint_summary(p,w,y));row.update(success=True,warnings=fit['fit']['warnings'],
                    iterations=fit['fit']['iterations'],fit_seconds=fit['fit']['seconds'])
            except (seq.EstimatorFailure,FloatingPointError,RuntimeError,ValueError) as exc:
                row.update(error_type=type(exc).__name__,error=str(exc))
            records.append(row)
            if len(records)%PRECISION_CONFIG['checkpoint_every']==0:
                checkpoint(dict(state,results=records,next_replicate=len(records),rng_state=rng.get_state()))
        state=dict(state,results=records,next_replicate=len(records),rng_state=rng.get_state())
        checkpoint(state)
    return dict(state=state,summary=precision_summary(state['results'],kind),seconds=time.perf_counter()-start,
        peak_worker_MB=seq.base().peak_mb())


def precision_run():
    import concurrent.futures as cf
    import multiprocessing as mp
    import queue
    import gc
    import steroid_eicu as eicu
    b,e,d=load();px=d['precision_phase'];assert px['config']==PRECISION_CONFIG
    assert px['original_mimic_analysis_signature']==d['analysis_phase']['signature']
    assert tuple(px['eicu_source_signature'])==(eicu.STATE.stat().st_size,eicu.STATE.stat().st_mtime_ns)
    eb=eicu.load();ed=eb['clinical_extension']['order_estimation'];plans={}
    for key in PRECISION_CONFIG['scenarios']:
        state=px['jobs'][key]
        if state['complete']:continue
        kind,name=key.split(':')
        if kind=='mimic':
            f=analysis_frame(d,name);s=d['analysis_phase']['config']
            p=seq.make_plan(f,s['numerical'],s['categorical'],diagnostic_numeric=['NE_dose'])
            y=f.hadm_id.map(b['source'].set_index('hadm_id').hospital_expire_flag).to_numpy(float)
            extra=dict(population=b['source'].subject_id.unique(),seed=s['bootstrap_seed'],analysis_signature=d['analysis_phase']['signature'])
        else:
            f=ed['scenarios'][name]['frame'];p=eicu.order_plan(f,seq)
            y=f.hospital.map(eb['hospital_summary'].set_index('hospital_stay').outcome).map({'alive':0.,'expired':1.}).to_numpy(float)
            extra={}
        assert p['plan_sha256']==state['state']['specification']['plan_sha256']
        plans[key]=dict(key=key,kind=kind,plan=p,outcome=y,**extra)
    del b,e,eb,ed;gc.collect()
    if not plans:say(dict(stage='precision_run',reuse=True));return
    ctx=mp.get_context('spawn')
    with ctx.Manager() as manager:
        channel=manager.Queue()
        with cf.ProcessPoolExecutor(max_workers=PRECISION_CONFIG['workers'],mp_context=ctx,
                                    initializer=_precision_init,initargs=(channel,)) as pool:
            pending={};waiting=list(plans)
            while waiting or pending:
                while waiting and len(pending)<PRECISION_CONFIG['workers']:
                    key=waiting.pop(0);state=px['jobs'][key]
                    payload=dict(plans[key],state=state['state'],target=state['target'])
                    pending[pool.submit(_precision_worker,payload)]=key
                    say(dict(stage='precision_start',scenario=key,from_replicate=state['state']['next_replicate'],target=state['target'],raw_scans=0))
                try:
                    key,kind,payload=channel.get(timeout=.5)
                    if payload['next_replicate']>px['jobs'][key]['state']['next_replicate']:
                        px['jobs'][key]['state']=payload;save(d)
                        say(dict(stage='precision_checkpoint',scenario=key,attempts=payload['next_replicate'],
                            failures=sum(not z['success'] for z in payload['results'])))
                except queue.Empty:pass
                for future in list(pending):
                    if not future.done():continue
                    key=pending.pop(future);result=future.result();state=px['jobs'][key]
                    state.update(result);summary=result['summary'];state.setdefault('round_summaries',{})[state['target']]=summary
                    # Numerical criterion only. No effect, sign, P value or desired interval enters.
                    extend=state['target']==500 and not summary['MC_numerical_target_met'] and summary['failures']==0
                    if extend:state['target']=1000;waiting.append(key)
                    else:state['complete']=True
                    save(d)
                    say(dict(stage='precision_round_done',scenario=key,attempts=summary['attempts'],failures=summary['failures'],
                        warnings=summary['warning_replicates'],MC_max_band_width_pp=round(max(z['width_pp'] for z in summary['MC_bands']),2),extend_to1000=extend))
    px['complete']=all(z['complete'] for z in px['jobs'].values());px['source_current']=Path(__file__).read_bytes();save(d)
    say(dict(stage='precision_run_complete',complete=px['complete'],raw_scans=0))


def additive_bias_scenarios(point,interval,gammas,deltas):
    """ATT scenario under a constant untreated-risk contrast and imbalance."""
    rows=[]
    for gamma in gammas:
        for delta in deltas:
            bias=gamma*delta
            control_cf=point['risk_control']+bias
            rows.append(dict(gamma0=gamma,prevalence_difference=delta,bias=bias,
                RD=point['risk_difference']-bias,lower=interval['lower']-bias,upper=interval['upper']-bias,
                counterfactual_control_risk=control_cf,aggregate_risk_feasible=0<=control_cf<=1,
                assumptions_not_estimated=True))
    return pd.DataFrame(rows)


def precision_validate():
    import steroid_eicu as eicu
    b,e,d=load();px=d['precision_phase'];assert px['complete'] and px['config']==PRECISION_CONFIG
    eb=eicu.load();ed=eb['clinical_extension']['order_estimation'];checks={}
    assert seq.sha(eicu.STATE)==px['eicu_source_sha256']
    assert d['analysis_phase']['signature']==px['original_mimic_analysis_signature']
    for key,j in px['jobs'].items():
        kind,name=key.split(':');state=j['state'];rows=state['results'];n=len(rows)
        assert n in [500,1000] and state['next_replicate']==n and [r['replicate'] for r in rows]==list(range(n))
        saved=pd.DataFrame(j['original100']);first=pd.DataFrame(rows[:100])
        pd.testing.assert_frame_equal(first[saved.columns],saved,check_dtype=False,check_exact=True)
        if kind=='mimic':
            cfg=d['analysis_phase']['config'];f=analysis_frame(d,name)
            p=seq.make_plan(f,cfg['numerical'],cfg['categorical'],diagnostic_numeric=['NE_dose'])
            y=f.hadm_id.map(b['source'].set_index('hadm_id').hospital_expire_flag).to_numpy(float)
            lp=d['clinical_phase']['assignment'][name]['linear_predictor'];w=seq.att_weights(p,lp)
            original=d['analysis_phase']['estimates'][name]
            assert original['point']==j['point']
            direct=seq.binary_risks(p,w,y)
            pd.testing.assert_frame_equal(saved[original['bootstrap'].columns],original['bootstrap'],check_dtype=False,check_exact=True)
            rng=np.random.RandomState(cfg['bootstrap_seed'])
            for _ in range(n):seq.draw_patient_counts(p,rng,b['source'].subject_id.unique())
        else:
            f=ed['scenarios'][name]['frame'];p=eicu.order_plan(f,seq)
            y=f.hospital.map(eb['hospital_summary'].set_index('hospital_stay').outcome).map({'alive':0.,'expired':1.}).to_numpy(float)
            original=ed['results'][name];w=seq.att_weights(p,original['point_model']['linear_predictor'])
            direct=eicu.order_endpoint_summary(p,w,y)
            pd.testing.assert_frame_equal(saved,pd.DataFrame(original['bootstrap']),check_dtype=False,check_exact=True)
            rng=np.random.RandomState(ed['config']['bootstrap_seed'])
            for _ in range(n):rng.exponential(1.,len(p['patient_ids']))
        assert p['plan_sha256']==state['specification']['plan_sha256']
        assert rng_equal(rng.get_state(),state['rng_state'])
        assert all(np.isclose(direct[k],j['point'][k],atol=1e-12,equal_nan=True) for k in direct)
        assert np.allclose(np.add.reduceat(w['control'],p['starts']),1.)
        fresh=precision_summary(rows,kind);assert fresh==j['summary']
        assert not fresh['failures'],'Failed draws require interpretation before interval publication'
        j['bias_scenarios']=additive_bias_scenarios(j['point'],fresh['intervals']['risk_difference'],
            px['config']['bias_gamma'],px['config']['bias_delta'])
        checks[key]=dict(first100_exact=True,all_draw_indices_present=True,random_stream_continuous=True,
            point_unchanged=True,original_data_signature=True,plan_unchanged=True,all_riskset_mass=True,
            no_new_model_selection=True,MC_target_met=fresh['MC_numerical_target_met'])
    ref=b['precision_extension']['summary']
    r=ref[ref.clock.eq('primary')&ref.attempts.eq(500)].set_index('metric')
    pt={k:float(r.loc[k,'point']) for k in ['risk_treated','risk_control','risk_difference','risk_ratio']}
    bounds={k:float(r.loc['risk_difference',k]) for k in ['lower','upper']}
    px['actual_dose_bias_scenarios']=additive_bias_scenarios(pt,bounds,px['config']['bias_gamma'],px['config']['bias_delta'])
    px['validation']=dict(checks=checks,original_actual_dose_unchanged=True,eicu_original100_unchanged=True,
        uncertainty_scopes='MIMIC source-patient bootstrap; eICU patient positive-weight bootstrap conditional on observed sites; no new-hospital uncertainty estimated',
        bias_formula='ATT_obs - ATT_true = E_X|A=1[gamma0(X)*(p1(X)-p0(X))]; constant gamma0 and delta assumed for scenario grid',
        bias_source='https://doi.org/10.1097/EDE.0b013e3181f74493',
        MC_source='https://www-users.york.ac.uk/~mb55/intro/cicent4.htm',
        bootstrap_MC_source='https://doi.org/10.1080/00031305.2015.1089789',
        utc=seq.base().utc(),new_fits=0,source=Path(__file__).read_bytes())
    save(d);say(dict(stage='precision_validate',passed=True,scenarios=len(checks),new_fits=0,
        unresolved_MC=[k for k,v in checks.items() if not v['MC_target_met']]))


def precision_progress(d,m):
    px=d.get('precision_phase',{})
    if 'validation' not in px:return []
    labels={'mimic:primary_new':'MIMIC新医嘱','mimic:strict_new':'MIMIC严格时钟',
            'mimic:primary_all_native':'MIMIC包含Change','eicu:primary':'eICU医嘱',
            'eicu:strict':'eICU严格时钟','eicu:known_NE':'eICU可解释NE剂量子集'}
    def cell(j,key,scale):
        z=j['summary']['intervals'][key];digits=3 if scale==1 else 2
        return f'{scale*j["point"][key]:.{digits}f}〔{scale*z["lower"]:.{digits}f}, {scale*z["upper"]:.{digits}f}〕'
    lines=['','### M3：区间精度与固定敏感性结果','',
        '**以下为续跑后的当前区间，前文100次结果保留为历史首版。** 六套队列、起点、协变量、评分模型和点估计均不变；从原随机状态续跑，并逐条核对前100次完全一致。原MIMIC实际首剂500次结果保留。','',
        '|固定口径|重抽样次数|启动风险%，95%区间|对照风险%，95%区间|RD，百分点，95%区间|RR，95%区间|',
        '|---|---:|---:|---:|---:|---:|']
    for key in px['config']['scenarios']:
        j=px['jobs'][key];s=j['summary']
        lines.append('|'+labels[key]+'|'+str(s['attempts'])+'|'+'|'.join(cell(j,k,v) for k,v in
            [('risk_treated',100),('risk_control',100),('risk_difference',100),('risk_ratio',1)])+'|')
    lines+=['','MIMIC为普通源患者重抽样百分位区间；eICU为患者Exp(1)正权重百分位不确定性区间，条件于已观察患者与医院支持。二者不是同一种抽样推断，均未估计推广至新医院的误差。患者的所有住院和病例/对照角色共享一次权重，每次重新拟合预处理和原评分模型。','',
        '|固定口径|失败 / 警告 / RR未定义|RD区间半宽（百分点）|100次→当前端点最大变化|MC端点带最大宽度（百分点）|数值目标|',
        '|---|---:|---:|---:|---:|---|']
    for key in px['config']['scenarios']:
        j=px['jobs'][key];s=j['summary'];v=s['intervals']['risk_difference']
        drift=100*np.max(np.abs(np.asarray(s['cumulative_RD_intervals'][100])-[v['lower'],v['upper']]))
        width=max(q['width_pp'] for q in s['MC_bands'])
        lines.append(f'|{labels[key]}|{s["failures"]} / {s["warning_replicates"]} / {s["intervals"]["risk_ratio"]["undefined_n"]}|{50*(v["upper"]-v["lower"]):.2f}|{drift:.2f}|{width:.2f}|'+('达到' if s['MC_numerical_target_met'] else '尚未达到')+'|')
    lines+=['','续跑前固定数值规则：先至500次；用二项分布秩界给出每个RD区间端点的95% Monte Carlo带，任一带宽超过2个百分点则续至1000次。2个百分点是本轮计算精度标准，不是临床重要性或发表门槛，也不是治疗效果区间的宽度。到1000次仍不满足时照实报告，不按方向、显著性或预期结论继续择优。端点带为边际带，不主张两个端点同时95%覆盖。',
        '±5和±10个百分点继续作为区间半宽的规划参照，并非临床认可标准；上表直接给出实际半宽。更多重抽样不会提高独立患者信息量，也不修复有限支持。参考：[分位数二项秩界](https://www-users.york.ac.uk/~mb55/intro/cicent4.htm)、[bootstrap计算误差](https://doi.org/10.1080/00031305.2015.1089789)。','',
        '#### 缺失结局与预定医院影响','',
        '|eICU固定口径|未知结局极值界限，保守向外取整（百分点）|','|---|---:|']
    for name in ['primary','strict','known_NE']:
        p=px['jobs']['eicu:'+name]['point']
        lines.append(f'|{labels["eicu:"+name]}|{int(np.floor(100*p["RD_missing_lower"]))}至{int(np.ceil(100*p["RD_missing_upper"]))}|')
    site=px['existing_eicu_site_sensitivity'];p=site['point']
    lines+=['','这些是固定权重下未知结局的极值界限，不包含抽样误差或未测混杂；未知仍保留，未填为存活。为避免由风险与界限联立反推少数未知结局的数量，公开表向外取整至整数百分点；精确值保存在原私有缓存中。',
        f'既定最大贡献医院排除分析沿用原结果：剩余约{m.count(site["remaining_cases"])}个启动节点，RD {100*p["risk_difference"]:+.2f}个百分点，RR {p["risk_ratio"]:.3f}；最大绝对SMD {site["balance"].smd_after.abs().max():.3f}，对照患者ESS {site["concentration"]["control_patient"]["ess"]:.1f}。仅有点估计，不添加未经计算的区间；它改变目标人群，不能代替医院整群不确定性或外部验证。','',
        '#### 未测混杂：有限的定量情景','',
        '令U表示索引前尚未记录、可能影响开药选择的临床状态，且在两组均有定义；δ为启动者与已加权对照中U的比例差，γ₀为在暂不启动HC条件下U关联的死亡风险差。在已测协变量层内γ₀与δ均恒定的简化ATT模型中，偏倚为γ₀×δ；情景RD等于观察RD减去该偏倚。[VanderWeele与Arah的偏倚公式](https://doi.org/10.1097/EDE.0b013e3181f74493)为此提供一般框架。',
        '本轮在续跑前固定γ₀=10、20、30个百分点，δ=−30、−20、−10、0、10、20、30个百分点，全部组合保留。它们是研究者情景，不是从HC用途或限制治疗资料估计的比例；正负失衡均评估。对固定偏倚平移抽样区间，没有为这些未知参数捏造概率分布。','',
        '|γ₀ \u00d7 δ|δ=10个百分点|δ=20个百分点|δ=30个百分点|','|---|---:|---:|---:|',
        '|γ₀=10个百分点|1个百分点|2个百分点|3个百分点|',
        '|γ₀=20个百分点|2个百分点|4个百分点|6个百分点|',
        '|γ₀=30个百分点|3个百分点|6个百分点|9个百分点|','',
        'δ为负时，上表偏倚取相反符号。下面并列同一固定示例γ₀=20个百分点、δ=±20个百分点；完整21格情景由代码保存在现有私有缓存中。','',
        '|主视角|δ=+20个百分点：情景RD〔条件区间〕|δ=−20个百分点：情景RD〔条件区间〕|','|---|---:|---:|']
    for label,table in [('MIMIC实际首剂',px['actual_dose_bias_scenarios']),
                        ('MIMIC新医嘱',px['jobs']['mimic:primary_new']['bias_scenarios']),
                        ('eICU医嘱',px['jobs']['eicu:primary']['bias_scenarios'])]:
        cells=[]
        for delta in [.20,-.20]:
            r=table[np.isclose(table.gamma0,.20)&np.isclose(table.prevalence_difference,delta)].iloc[0]
            cells.append(f'{100*r.RD:+.2f}〔{100*r.lower:+.2f}, {100*r.upper:+.2f}〕')
        lines.append('|'+label+'|'+'|'.join(cells)+'|')
    lines+=['','该情景计算没有证明U存在或达到某个强度，也不能修复医嘱、实际执行和临床用途的错分。其用途是说明剩余混杂在什么量级上会改变解释，不从情景中选择最有利的结果。','',
        '**阶段决定：**保留原MIMIC实际首剂为主分析，MIMIC医嘱与eICU医嘱分别作为相关决策的补充证据。进入稿件整合，不合并为共同治疗效果，不宣称已经验证HC伤害或获益；HC用途、真实决定时间与局部支持仍决定临床解释范围。当前无需追加原始表。']
    return lines


def partition_percentages(counts,total):
    """Suppress rare cells plus a complement so row totals cannot reveal them."""
    counts=[int(n) for n in counts]
    assert total>0 and sum(counts)==total
    hidden={i for i,n in enumerate(counts) if 0<n<=10 or 0<total-n<=10}
    if len(hidden)==1:
        candidates=[i for i,n in enumerate(counts) if i not in hidden and n>0]
        if candidates:hidden.add(min(candidates,key=lambda i:counts[i]))
    return ['小样本或互补抑制' if i in hidden else f'{100*n/total:.1f}%'
            for i,n in enumerate(counts)]


def clinical_progress(d,m):
    phase=d.get('clinical_phase',{})
    if 'features' not in phase:return []
    def percentage(n,total):
        value=m.pct(int(n),int(total))
        return '小样本抑制' if value=='' else f'{value:.1f}%'
    p=d['risksets'];mask=phase['analysis_masks']['primary_new'];z=p.loc[mask]
    case=z.treated.eq(1);control=~case;f=phase['features']
    lines=['','**新医嘱起点的临床资料已重建。** 原窗口缺口已经补取；没有把尚未提取当作临床无记录，也没有要求完整病例。','',
        '|最终记录资格风险集|启动节点约数|独立对照患者约数|无对照风险集约数|',
        '|---|---:|---:|---:|']
    labels={'primary_new':'主分析：明确New医嘱','all_native':'敏感性：包含版本不确定医嘱'}
    for r in phase['final_support'].itertuples():
        lines.append('|'+labels[r.scenario]+'|'+m.count(r.cases)+'|'+m.count(r.control_patients)+'|'+m.count(r.empty_risksets)+'|')
    selected=set(z.loc[case,'case_id']);actions=d['cases'][d['cases'].case_id.isin(selected)]
    lines+=['','|医嘱之后的记录窗|观察到HC实际给药记录|投影覆盖窗末但无该记录|投影提前结束|',
        '|---|---:|---:|---:|']
    for minutes in [30,60,120,360]:
        states=actions['execution_'+str(minutes)]
        counts=[states.eq(value).sum() for value in
            ['CONFIRMED_ADMIN_IN_WINDOW','NO_CONFIRMED_ADMIN_IN_COMPLETE_WINDOW','CAPTURE_ENDS_BEFORE_WINDOW']]
        cells=partition_percentages(counts,len(states))
        lines.append('|'+str(minutes)+'分钟|'+'|'.join(cells)+'|')
    lines+=['','上述为同次住院内医嘱后的HC实际记录路径，不等于每条给药均已通过原生键确认执行了该条医嘱；投影覆盖也不保证临床记录没有漏记。没有记录不解释为未给药或撤回。后续路径仅作描述，不用于入组或分组。涉及小单元时同时隐藏另一个单元，避免由行合计反推。']
    lines+=['','主分析和严格记录时钟使用相同风险集；严格时钟改变可进入调整的信息，不重新选择患者。显式当前CMO按主时钟排除；未知治疗限制保留。资格和协变量不使用后来执行、死亡、治疗反应或后来CMO。','',
        '|主分析风险集内的信息|启动者：主时钟 / 严格时钟|对照索引行：主时钟 / 严格时钟|',
        '|---|---:|---:|']
    for var,label in [('MAP','MAP（1h）'),('HR','HR（1h）'),('lactate','乳酸（6h，采样/录入均在前）'),
                      ('creatinine','肌酐（24h，采样/录入均在前）'),('GCS_simultaneous','同刻GCS组分（4h）'),
                      ('NE_dose','NE剂量（记录左极限）')]:
        cols=[]
        for arm in [case,control]:
            vals=[]
            for clock in ['primary','strict']:
                values=phase['history'].loc[mask,var] if var=='NE_dose' else f[clock].loc[mask,var]
                vals.append(percentage(values[arm].notna().sum(),arm.sum()))
            cols.append(' / '.join(vals))
        lines.append('|'+label+'|'+'|'.join(cols)+'|')
    lines+=['','对照覆盖以可重复的索引行为分母；这不是独立患者覆盖率。MAP等床旁主时钟使用测量记录时间，严格时钟额外要求入库；实验室两者均须在索引前。storetime不等同医生首次可见时间。NE仍是记录中的输注左极限，不声称实时可见，也不额外盐基换算。','',
        '|治疗限制来源|启动者|对照索引行|','|---|---:|---:|']
    for variable,label,kind in [('chart_code_status_unknown','仅Chart：限制状态未知','positive'),
        ('code_status_unknown','合并POE后：限制状态未知','positive'),
        ('POE_code_known','有可解释POE限制指令','positive'),
        ('code_source_discordance','两来源末次值不同','positive')]:
        cols=[]
        for arm in [case,control]:
            val=f['primary'].loc[mask,variable];cols.append(percentage((val.eq(1)&arm).sum(),arm.sum()))
        lines.append('|'+label+'|'+'|'.join(cols)+'|')
    lines+=['','DNR、DNI、CMO分别编码；未明确的维度不填为否。两来源末次值不同可能反映不同时间的指令变化，不自动视为错误。新的撤销记录使被撤销指令失效，不用最终医嘱状态反向改写基线。参考：[POE定义](https://mimic.mit.edu/docs/iv/modules/hosp/poe.html)、[结构化POE明细](https://mimic.mit.edu/docs/iv/modules/hosp/poe_detail.html)。']
    assignments=phase.get('assignment',{})
    if assignments:
        labels2={'primary_new':'明确New / 主时钟','strict_new':'明确New / 严格时钟','primary_all_native':'含Change / 主时钟'}
        lines+=['','|固定分配模型|最大绝对SMD|对照患者ESS|最大单人权重份额|前10人合计份额|100次普通患者抽样的结构失败|',
                '|---|---:|---:|---:|---:|---:|']
        for name in d['analysis_phase']['config']['scenarios']:
            if name not in assignments:continue
            a=assignments[name];con=a['concentration']['control_patient'];fails=sum(a['ordinary_resampling']['failures'].values())
            lines.append('|'+labels2[name]+'|'+f'{a["balance"].smd_after.abs().max():.3f}|{con["ess"]:.1f}|{100*con["maximum_share"]:.2f}%|{100*con["top10_share"]:.2f}%|{fails}/100|')
        a=assignments['primary_new'];bal=a['balance'].copy();bal['absolute']=bal.smd_after.abs()
        lines+=['','|主时钟下较大的剩余差异|类型|加权前SMD|加权后SMD|','|---|---|---:|---:|']
        display={'oxygen_device_group=7.0':'其他已记录氧疗设备','oxygen_device_group=3.0':'CPAP/BiPAP面罩记录',
                 'NE_dose':'当前NE记录剂量','calendar_era=THROUGH_2013':'日历区间：截至2013年','recorded_DNI=1.0':'明确DNI记录'}
        for r in bal.sort_values('absolute',ascending=False).head(5).itertuples():
            lines.append(f'|{display.get(r.variable,r.variable)}|{r.kind}|{r.smd_before:.3f}|{r.smd_after:.3f}|')
        loc=a['local'];v=loc.close_NE_MAP_control_mass.dropna()
        lines+=['',f'局部支持：约{m.count(len(v))}个启动节点同时具备可解释NE剂量和MAP；其对照权重中，处于NE剂量两倍范围且MAP相差≤15mmHg的比例，中位数为{100*v.median():.1f}%。该尺度是固定描述规则，不是入组卡尺或临床可互换性标准。',
            'ESS按患者累计其全部住院和风险集角色后计算，不能以约数十万索引行充当独立样本。整体平衡与局部相似性并列解释；均不能消除未记录的HC用途及病情选择。',
            '上表的100次结构检查只验证患者重抽后风险集能否保留对照，本身不拟合模型或产生区间。正式患者bootstrap作为独立步骤执行，每次重新拟合预处理和分配模型。']
    checks=phase['endpoint_checks'];bad={k:v for k,v in checks['primary_new'].items() if k not in ['admissions','ICU_transfer_not_censoring','alive_discharge_is_terminal_zero']}
    lines+=['', '完整住院结局链检查：'+ '；'.join(k+'='+m.count(v) for k,v in bad.items())+'。存活出院记为本次院内死亡未发生，转出ICU不终止随访。',
        f'临床变量重建耗时{phase["feature_seconds"]/60:.1f}分钟，峰值进程内存{phase["feature_peak_MB"]:.1f}MB。固定旧样本的历史变量与原分析一致，变量截止的200组合成核对通过；原实际首剂缓存未写入。']
    decision=d.get('analysis_phase',{}).get('support_decision')
    if decision:lines+=['','**本阶段决定：**'+decision['rationale']]
    return lines


def estimation_progress(d):
    ap=d.get('analysis_phase',{});estimates=ap.get('estimates',{})
    if not estimates or not all(estimates.get(s,{}).get('complete') for s in ap['config']['scenarios']):return []
    labels={'primary_new':'明确New / 主时钟','strict_new':'明确New / 严格时钟','primary_all_native':'含Change / 主时钟'}
    def interval_text(rec,key,scale=1.):
        bounds=rec['summary']['intervals'][key]
        point=scale*rec['point'][key]
        if bounds['lower'] is None or bounds['upper'] is None:return f'{point:.2f}〔区间不可计算〕'
        return f'{point:.2f}〔{scale*bounds["lower"]:.2f}, {scale*bounds["upper"]:.2f}〕'
    lines=['','### 已完成的探索性医嘱关联比较','',
        '结局为本次完整住院的院内死亡；比较当前记录启动HC医嘱与当前尚未启动，之后通常照护。该目标不是实际执行、完整HC疗程、永不治疗或固定推迟时长。下列95%百分位区间各基于100次普通患者bootstrap，属于初步区间精度。','',
        '|口径|当前医嘱组风险%，95%区间|尚未医嘱组加权风险%，95%区间|RD，百分点，95%区间|RR，95%区间|',
        '|---|---:|---:|---:|---:|']
    for name in ap['config']['scenarios']:
        r=estimates[name]
        lines.append('|'+labels[name]+'|'+'|'.join(interval_text(r,k,scale) for k,scale in
            [('risk_treated',100),('risk_control',100),('risk_difference',100),('risk_ratio',1)])+'|')
    lines+=['','|口径|未调整风险集比较RD（百分点）|计划/成功重抽样|失败|拟合警告重复数|RR不可计算重复数|前后半数区间端点最大差（百分点）|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name in ap['config']['scenarios']:
        r=estimates[name];s=r['summary']
        mc=s.get('RD_halves_endpoint_max_difference_pp',np.nan)
        lines.append(f'|{labels[name]}|{100*r["crude"]["risk_difference"]:.2f}|{s["attempts"]}/{s["successful"]}|{s["failures"]}|{s["warning_replicates"]}|{s["intervals"]["risk_ratio"]["undefined_n"]}|{mc:.2f}|')
    r=estimates['primary_new'];bounds=r['summary']['intervals']['risk_difference']
    if bounds['lower'] is not None and bounds['lower']<=0<=bounds['upper']:
        finding='主口径区间跨过零，仍容纳两个方向的院内死亡风险差；这不是等效性检验，也不支持按当前结果推荐启动或等待。'
    else:
        finding='主口径显示有方向性的加权关联，但区间不能覆盖未测量HC用途和病情选择造成的系统偏倚；不能直接据此推荐临床用药。'
    lines+=['',finding,
        '每次从原始NE来源患者总体重抽，患者的全部住院、启动者与对照角色共享抽样次数，并重新拟合预处理和固定分配模型。未替换失败抽样、未搜索种子、模型或截断值。未调整对照为每个风险集等总质量的均匀对照，不是将所有重复索引行直接混成一组。',
        '另按患者累计加权质量独立重算两组风险，核对与原风险集估计一致；三套估计的风险集与已报告诊断相同。前后各50次区间端点差仅检查Monte Carlo敏感性，不代表尾部已经收敛。',
        '三套固定口径均报告。严格时钟保持主分析人群，改变可用协变量；纳入Change的敏感性改变人群。原MIMIC实际首剂结果、eICU医嘱结果及各自区间均保留，当前三者不汇总为一个治疗效果。',
        '当前MIMIC医嘱区间与原实际首剂500次区间、eICU100次患者正权重区间的抽样支持不同。下一步先完善不确定性解释及三种记录决策的差异，再扩至既定精度；增加重复次数不增加独立患者。']
    return lines


def cross_database_progress(d,m,actual):
    """Describe existing independent estimates; no pooled fit or new selection."""
    if 'assignment' not in d.get('clinical_phase',{}):return []
    import steroid_eicu as eicu
    signature=(eicu.STATE.stat().st_size,eicu.STATE.stat().st_mtime_ns)
    cached=d.get('cross_database_context')
    if cached is None or tuple(cached['source_signature'])!=signature or 'eicu_point' not in cached:
        b=eicu.load();old=b['clinical_extension']['order_estimation']
        assert old['config']==eicu.ORDER_ESTIMATION_CONFIG
        count=int(old['scenarios']['primary']['frame'].treated.sum())
        d['cross_database_context']=dict(source_signature=signature,source_sha256=seq.sha(eicu.STATE),
            eicu_case_nodes=count,eicu_balance=old['results']['primary']['balance'].copy(),
            eicu_point=old['results']['primary']['point'],eicu_intervals=old['results']['primary']['uncertainty'],
            existing_eicu_definition=old['config'],eicu_state_read_only=True)
        del b
    cached=d['cross_database_context'];own=d['clinical_phase']['assignment']['primary_new']
    def measured_cell(balance,key,n):
        row=balance[balance.variable.eq(key)&balance.kind.eq('observed_value')].iloc[0]
        observed=int(round(row.treated_coverage*n));cover=m.pct(observed,n)
        mean=f'{row.treated_mean:.2f}' if observed>10 else '小样本抑制'
        return mean+'；'+('比例抑制' if cover=='' else f'{cover:.1f}%')
    lines=['','### 两个医嘱分析的共同点与尚未消除的差别','',
        '|构件|MIMIC独立医嘱|eICU已有医嘱|','|---|---|---|',
        '|记录行动|原生POE关联IV HC处方，主口径明确New|medication中的IV HC医嘱；不代表已实际执行|',
        '|索引字段|POE.ordertime，处方计划starttime另存|medication.drugorderoffset转为同住院时间；计划drugstartoffset另存|',
        '|感染资格|索引前实际抗菌药与诊断培养配对|索引前明确急性感染/脓毒症诊断记录|',
        '|NE状态与共同时间轴|首次观察ICU NE后首24h；当前正速率区间左极限|首次NE记录后的相同经过时间；此前60分钟有正速率记录|',
        '|对照来源|单中心全部合格NE来源，在各自相同NE经过时间|同一家医院内、各自相同NE经过时间的合格患者|',
        '|治疗限制|主时钟当前明确CMO排除，其他状态及未知分别保留|明确CMO、不得追加治疗、不得用升压药排除，DNR/DNI及未知保留|',
        '|主要目标量|本库有明确新医嘱、符合记录资格启动状态的ATT|本库有同院对照、符合记录资格启动状态的ATT|',
        '|医院结局|当前分析住院标志完整，存活出院0|少量未知/冲突；参考风险与未知结局界限并报|',
        '|区间|普通源患者重抽，每次重拟合|患者正权重重抽，每次重拟合；条件于现有患者/医院支持|',
        '', '下表描述两库主口径**启动节点**的已测均数与记录覆盖率，未作库间显著性检验。药物剂量沿用各库已核查的单位映射，不新增盐基换算；覆盖率分母都是本库启动节点。','',
        '|变量|MIMIC：已测均数；覆盖率|eICU：已测均数；覆盖率|','|---|---:|---:|']
    for variable,label in [('NE_dose','NE，记录换算μg/kg/min'),('MAP','MAP，mmHg'),
                           ('HR','心率，次/min'),('lactate','乳酸，mmol/L'),('creatinine','肌酐，mg/dL')]:
        lines.append('|'+label+'|'+measured_cell(own['balance'],variable,own['cases'])+'|'+
            measured_cell(cached['eicu_balance'],variable,cached['eicu_case_nodes'])+'|')
    lines+=['','“比例抑制”遵守小单元规则，不表示整项测量未知。已测均数来自选择性测量者，不能直接当作两库全部患者的严重度比较。GCS来源与完整性规则不同，因此没有把两库GCS覆盖率包装成同一测量的优劣比较。',
        '**综合判断：**两库能够研究相邻的HC启动决策，但当前资格、NE时间代理、限制治疗记录和不确定性支持仍不完全相同。先分别报告三个估计及这些差别，暂不合并患者或汇总一个共同治疗效果。方向相同或不同都不能单独证明哪种差别造成了结果。',
        '下一轮优先完善这三种记录决策的解释和已固定区间精度；核心原始表已经具备，当前无新增下载要求。']
    current=d.get('analysis_phase',{}).get('estimates',{}).get('primary_new',{})
    if current.get('complete'):
        lines+=['','|三个主视角（分别估计）|启动风险%，95%区间|同期对照风险%，95%区间|RD，百分点，95%区间|RR，95%区间|',
            '|---|---:|---:|---:|---:|']
        reference=actual['precision_extension']['summary']
        rows=reference[reference.clock.eq('primary')&reference.attempts.eq(500)].set_index('metric')
        def value_text(point,lo,hi,scale):
            digits=3 if scale==1 else 2
            return f'{scale*point:.{digits}f}〔{scale*lo:.{digits}f}, {scale*hi:.{digits}f}〕'
        for label,kind in [('MIMIC实际首剂；原500次','actual'),('MIMIC新医嘱；100次','order'),('eICU医嘱；100次正权重','eicu')]:
            cells=[]
            for key,scale in [('risk_treated',100),('risk_control',100),('risk_difference',100),('risk_ratio',1)]:
                if kind=='actual':
                    row=rows.loc[key];v,lo,hi=row.point,row.lower,row.upper
                elif kind=='order':
                    bounds=current['summary']['intervals'][key];v,lo,hi=current['point'][key],bounds['lower'],bounds['upper']
                else:
                    ek={'risk_treated':'initiated_risk','risk_control':'not_yet_risk'}.get(key,key)
                    bounds=cached['eicu_intervals'][ek];v,lo,hi=cached['eicu_point'][ek],bounds['low'],bounds['high']
                cells.append(value_text(v,lo,hi,scale))
            lines.append('|'+label+'|'+'|'.join(cells)+'|')
        lines+=['','这不是三种估计的直接假设检验或共同效应meta分析。MIMIC两版来自同一数据资源但启动节点、人群和部分调整资料不同；eICU另有医院及记录机制差别。区间重叠不证明结果一致，点估计变号也不证明治疗效果反转。']
    return lines


def publish():
    b,e,d=load();assert 'risksets' in d;m=seq.base();p=d['risksets'];c=d['cases'];flow=d['support']
    co=p[p.treated.eq(0)];cov=d['panel_envelope_coverage'];first=d['first_orders']
    raw_bytes=sum(r['bytes_read'] for r in d['runs']);seconds=sum(r['seconds'] for r in d['runs'])
    peak=max(r['peak_MB'] for r in d['runs'])
    counts=[('全部成人NE来源住院',len(b['source'])),('首次原生关联IV HC医嘱',len(first)),
        ('符合当前记录资格的技术启动节点',len(c)),('来自旧风险集范围之外的合格节点',int((~c.old_riskset_admission).sum())),
        ('没有同经过时间对照的节点',int(flow.controls.eq(0).sum())),('独立对照患者',co.subject_id.nunique()),
        ('全部风险集成员行（不是独立患者）',len(p))]
    complete='features' in d.get('clinical_phase',{})
    effects_complete=d.get('analysis_phase',{}).get('result_status')=='THREE_FIXED_ORDER_ASSOCIATIONS_100_SOURCE_PATIENT_BOOTSTRAPS'
    precision_complete='validation' in d.get('precision_phase',{})
    if effects_complete:assert 'validation' in d['analysis_phase']
    heading='**已完成独立医嘱队列、临床重建、固定支持诊断与三套探索性关联估计。**' if effects_complete else ('**已完成独立来源、新起点临床重建及固定支持诊断。**' if complete else '**已完成独立来源及技术风险集；尚未完成新起点的全部临床调整。**')
    inference_status=(' 原100次保留，当前500或1000次区间见M3；原实际首剂主结果保留。' if precision_complete
                      else ' 每套完成100次患者bootstrap，原实际首剂主结果保留。')
    lines=[heading+(inference_status if effects_complete else ' 没有计算新的MIMIC医嘱死亡差异。'),'',
        '|本轮实际结果|数量约数|','|---|---:|']
    for label,n in counts:lines.append('|'+label+'|'+m.count(n)+'|')
    lines+=['','全部候选从完整成人NE来源独立选取，结果恰好没有增加旧风险集范围外的合格启动者。旧约230个桥接节点与本轮约210个节点的差异涉及新增的此前系统激素医嘱、同时间其他激素及未知途径规则；这是分析对象定义的变化，不是按效果或样本量选择。新结果不能称为扩样成功。','',
        '首条关联POE的transaction_type：'+ '；'.join(str(k)+'约'+m.count(v) for k,v in c.transaction_type.value_counts(dropna=False).items())+'。原生前驱链和pharmacy表核查后，Change前驱的药名仍无法确认。因此以明确New且无前驱链接为主，包含Change作为固定敏感性；不按后来执行、存活或死亡选择，也不将后来的Change反推为已知既往HC。','',
        '|补取前的变量回看窗|启动节点：窗口完整缓存 / 尚未完整投影|对照成员：窗口完整缓存 / 总成员|','|---|---:|---:|']
    for var in ['MAP','GCS','lactate','creatinine']:
        t=cov[cov.variable.eq(var)&cov.arm.eq(1)].iloc[0];a=cov[cov.variable.eq(var)&cov.arm.eq(0)].iloc[0]
        lines.append('|'+var+'|'+m.count(t.full_window_cached)+' / '+m.count(t.rows-t.full_window_cached)+'|'+m.count(a.full_window_cached)+' / '+m.count(a.rows)+'|')
    lines+=['','窗口完整缓存表示原提取范围覆盖所需观察区间，不表示存在该项测量，更不表示医生当时已知。尚未投影不能记成临床缺失；不按完整病例缩减此队列。']
    lines+=clinical_progress(d,m)
    lines+=estimation_progress(d)
    lines+=cross_database_progress(d,m,b)
    lines+=precision_progress(d,m)
    lines+=['',f'本阶段累计原始读取{raw_bytes/2**30:.2f} GiB，解析{seconds/60:.1f}分钟，扫描进程峰值内存{peak:.1f} MB；风险集构建约{d["support_seconds"]:.1f}秒。处方、版本链、临床窗口与结构化限制均按所需字段/范围投影并核对EOF、输入签名和流式哈希；不是将这些大表整体载入内存，也不是新增下载体积。',
        '批量风险集版本在固定首节点与完整原规则核对一致；资格与时钟未修改。已完成投影自动复用，无需再次下载或重扫。',
        '与原实际首剂研究相比，本医嘱分析同时重建了起点、既往医嘱资格和治疗限制资料，目标人群也有变化。因此两版估计的差异不能全部归因于医嘱与给药的时间差；它们是同一临床主题下不同记录决策的比较。','',
        '**下一依赖：**完成三个记录决策的差异解释与不确定性完善，再更新稿件核心结果；不将增加bootstrap次数等同扩样。两库结果不会因方向不同而更换临床问题、阈值或模型。']
    body='\n'.join(lines)
    if precision_complete:
        body=body.replace('**下一依赖：**完成三个记录决策的差异解释与不确定性完善，再更新稿件核心结果；不将增加bootstrap次数等同扩样。两库结果不会因方向不同而更换临床问题、阈值或模型。',
            '**下一依赖：**将已完成的主分析、医嘱补充及固定敏感性整合进现有稿件；不增加平行分析或新的报告。两库结果分别呈现，不因方向不同而更换临床问题、阈值或模型。')
        body=body.replace('下一轮优先完善这三种记录决策的解释和已固定区间精度；核心原始表已经具备，当前无新增下载要求。',
            '三种记录决策的差异与续跑后的区间在下方M3一并报告；核心原始表已经具备，当前无新增下载要求。')
        body=body.replace('当前MIMIC医嘱区间与原实际首剂500次区间、eICU100次患者正权重区间的抽样支持不同。下一步先完善不确定性解释及三种记录决策的差异，再扩至既定精度；增加重复次数不增加独立患者。',
            '以上为首版100次历史区间。M3已续跑至500或1000次，普通患者重抽与患者正权重重抽的支持差异仍保留；增加重复次数不增加独立患者。')
    readme=REPO/'README.md';old=readme.read_text(encoding='utf-8');start='<!-- ORDER_BRIDGE_PROGRESS_START -->';end='<!-- ORDER_BRIDGE_PROGRESS_END -->'
    lo=old.index(start)+len(start);hi=old.index(end,lo)
    new=old[:lo]+'\n'+body+'\n'+old[hi:]
    for text in ['在原缓存范围外补取系统激素处方与必要POE键，记录字段、过滤、字节数、耗时和完整读取检查。',
                 '在医嘱时点重新判断NE、此前感染证据、实际及医嘱激素历史，区分冲突、未提取与未记录。',
                 '由全来源建立同NE经过时间的未启动对照，保留后续允许启动的通常照护比较。']:
        new=new.replace('- [ ] '+text,'- [x] '+text)
    if complete:
        new=new.replace('### M1 · MIMIC医嘱来源和独立风险集（当前优先，1–2个有效工作日）',
            '### M1 · MIMIC医嘱来源和独立风险集（已完成首版）')
        for task in ['核清首条POE为Change的原生版本链；区分真正首次决定、既有医嘱变更与无法判断，不按后来执行选择。',
            '按变量回看窗核对既有病情缓存是否覆盖新起点；只有能改变判断的缺口才定向补取。',
            '纳入索引前治疗限制与关键病情资料后，确定最终分析风险集；不把技术候选称为临床核验人群。']:
            new=new.replace('- [ ] '+task,'- [x] '+task)
    if 'analysis_phase' in d:
        task='固定MIMIC医嘱分析的调整集、风险计算和不确定性方法，再读取新分组死亡差异；已看过其他结果的事实保留。'
        new=new.replace('- [ ] '+task,'- [x] '+task)
    if len(d.get('clinical_phase',{}).get('assignment',{}))==3:
        task='独立拟合医嘱风险集权重，报告病例、对照、缺失状态、局部支持、ESS及权重集中；不沿用实际首剂权重。'
        new=new.replace('- [ ] '+task,'- [x] '+task)
        new=new.replace('当前唯一下一工作包：**M1独立MIMIC医嘱队列与起点覆盖**。','当前唯一下一工作包：**M2固定医嘱比较的估计与两库差异解释**。')
    if effects_complete:
        new=new.replace('当前阶段：**已有MIMIC主分析和eICU补充估计，正在解决两者能否形成同一临床问题下可解释的证据。**',
            '当前阶段：**MIMIC实际首剂、独立MIMIC医嘱和eICU医嘱三种比较均有实际结果；保持分别报告，进入不确定性完善。**')
        new=new.replace('### M2 · 同口径比较与两库差异解释（M1通过后，2–3个有效工作日）',
            '### M2 · 同口径比较与两库差异解释（首版已完成，当前不汇总）')
        task='并列MIMIC实际首剂、MIMIC医嘱和eICU医嘱的风险、RD、RR与区间，保留所有已指定分析。'
        new=new.replace('- [ ] '+task,'- [x] '+task)
        for task in ['比较治疗前严重程度、感染代理、NE时间来源、医嘱执行路径及医院缺失结构，区分“可能解释”与已有证据支持的解释。',
            '只有临床策略、资格和目标人群能够协调时，才讨论汇总；分别报告是允许的最终方案。']:
            new=new.replace('- [ ] '+task,'- [x] '+task)
        estimates=d['analysis_phase']['estimates'];r=estimates['primary_new'];bounds=r['summary']['intervals']['risk_difference']
        table_row=f'|MIMIC独立医嘱补充|明确New且符合记录资格的HC医嘱；约200个节点、3,900位独立对照|{100*r["point"]["risk_difference"]:+.2f}〔{100*bounds["lower"]:+.2f}, {100*bounds["upper"]:+.2f}〕|100次普通患者bootstrap；完整固定敏感性见下方|'
        anchor='|eICU-CRD2.0补充|'
        if '|MIMIC独立医嘱补充|' in new:
            new=re.sub(r'^\|MIMIC独立医嘱补充\|.*$',table_row,new,flags=re.M)
        else:new=new.replace(anchor,table_row+'\n'+anchor,1)
        for old_next in ['M2固定医嘱比较的估计与两库差异解释','M2两库差异解释与M3不确定性完善']:
            new=new.replace('当前唯一下一工作包：**'+old_next+'**。','当前唯一下一工作包：**M3不确定性完善与已指定敏感性**。')
        if 'validation' in d['analysis_phase']:
            task='复核小风险集下的不确定性方法，明确患者所有住院及多种风险集角色共享权重、每次重拟合预处理与评分。'
            new=new.replace('- [ ] '+task,'- [x] '+task)
    if 'validation' in d.get('precision_phase',{}):
        px=d['precision_phase']
        new=new.replace('### 已完成的探索性医嘱关联比较','### 首版探索性医嘱关联比较（100次历史区间）')
        new=new.replace('|三个主视角（分别估计）|','|三个主视角（首版历史区间；当前区间见M3）|')
        new=new.replace('### M3 · 完善推断与有限敏感性（约3–5个有效工作日）',
            '### M3 · 完善推断与有限敏感性（本轮完成；精度见实测端点带）')
        for task in ['区分患者抽样、固定医院支持与推广至新医院三种不确定性；不以医院数多就自动声称可推广。',
                     '方法确定后，将必要区间扩至至少500次并检查端点Monte Carlo稳定性；记录全部失败和警告。',
                     '固定报告严格时钟、NE剂量可解释子集、未知结局界限及已指定的大医院影响，不搜索最有利版本。',
                     '针对HC用途和未测病情制定一个可解释的定量偏倚情景；不把它当作消除混杂的证明。']:
            new=new.replace('- [ ] '+task,'- [x] '+task)
        new=new.replace('当前唯一下一工作包：**M3不确定性完善与已指定敏感性**。',
            '当前唯一下一工作包：**M4整合主分析、医嘱补充及解释边界，形成稿件**。')
        for label,key,desc in [('MIMIC独立医嘱补充','mimic:primary_new','明确New且符合记录资格的HC医嘱；约200个节点、3,900位独立对照'),
            ('eICU-CRD2.0补充','eicu:primary','HC医嘱节点；约480次启动、3,840位独立对照、约50家贡献启动节点的医院'),
            ('eICU严格录入时钟','eicu:strict','相同队列与行动，改变可用调整信息'),
            ('eICU NE剂量可解释子集','eicu:known_NE','同时限制启动者及对照，目标人群改变')]:
            j=px['jobs'][key];s=j['summary'];v=s['intervals']['risk_difference']
            kind='普通患者bootstrap' if key.startswith('mimic') else '患者正权重区间；条件于现有支持'
            row=f'|{label}|{desc}|{100*j["point"]["risk_difference"]:+.2f}〔{100*v["lower"]:+.2f}, {100*v["upper"]:+.2f}〕|{s["attempts"]}次{kind}|'
            new=re.sub(r'^\|'+re.escape(label)+r'\|.*$',row,new,flags=re.M)
        new=new.replace('三个口径各100次、无优化失败或警告。','首版三个口径各100次、无优化失败或警告；续跑精度与全部运行状态见M3结果。')
        new=new.replace('保持分别报告，进入不确定性完善。','保持分别报告，完成本轮不确定性与敏感性分析，进入稿件整合。')
        new=new.replace('python -X utf8 hc_order_bridge.py order_validate\npython -X utf8 hc_order_bridge.py publish',
            'python -X utf8 hc_order_bridge.py order_validate\npython -X utf8 hc_order_bridge.py precision_plan\npython -X utf8 hc_order_bridge.py precision_run\npython -X utf8 hc_order_bridge.py precision_validate\npython -X utf8 hc_order_bridge.py publish')
        note='精度续跑由`precision_plan/run/validate`执行，保留原100次及两库原缓存；两个独立进程计算、单个父进程保存现有检查点，避免线程共享警告或并发覆盖。数值规则见M3；合成数据已核对串行、并行和中断续跑结果相同。'
        if note not in new:new=new.replace('**当前复现边界：**',note+'\n\n**当前复现边界：**')
        # Keep prior-generation narratives identifiable without altering estimates.
        body=body.replace('### 已完成的探索性医嘱关联比较','### 首版探索性医嘱关联比较（100次历史区间）')
        body=body.replace('|三个主视角（分别估计）|','|三个主视角（首版历史区间；当前区间见M3）|')
    readme.write_bytes(new.encode('utf-8'))
    report=configured_path('canonical_source_dir','../HTE_sepsis')/'steroid_tte.md'
    before=report.read_bytes();nl='\r\n' if b'\r\n' in before else '\n';old=before.decode('utf-8').replace('\r\n','\n')
    marker='<!-- HC_MIMIC_INDEPENDENT_ORDER_START -->';ending='<!-- HC_MIMIC_INDEPENDENT_ORDER_END -->'
    section=marker+'\n### 独立MIMIC医嘱比较：来源、临床重建与支持\n\n'+body+'\n\n研究执行清单维护在[独立HC仓库README](../HC_TTE/README.md)，[私有GitHub仓库](https://github.com/Sheldon0106/hc-initiation-tte)同步代码与计划；原HC源码、实际首剂结果、Word稿及会议HTML保留。\n'+ending
    if marker in old:
        lo=old.index(marker);hi=old.index(ending,lo)+len(ending);out=old[:lo]+section+old[hi:]
    else:
        anchor='### 实际代码与证据位置';assert old.count(anchor)==1;out=old.replace(anchor,section+'\n\n'+anchor,1)
    after=out.replace('\n',nl).encode('utf-8')
    if after!=before:
        history=d.setdefault('report_history',[]);digest=hashlib.sha256(before).hexdigest()
        if not any(z['sha256']==digest for z in history):history.append(dict(sha256=digest,bytes=before))
        assert report.read_bytes()==before
        report.write_bytes(after)
    d['publication']=dict(report_hash=hashlib.sha256(after).hexdigest(),README_hash=hashlib.sha256(new.encode('utf-8')).hexdigest())
    d['source_current']=Path(__file__).read_bytes();save(d)
    say(dict(stage='publish',canonical_reports_updated=2,new_report_files=0,models=d.get('analysis_phase',{}).get('models',0),effects=d.get('analysis_phase',{}).get('effects',0)))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['plan','project','evaluate','support','lineage','clinical_plan','clinical_project','clinical_features','analysis_freeze','clinical_score','estimation_review','order_estimate','order_validate','precision_plan','precision_run','precision_validate','publish']);a=p.parse_args()
    globals()[a.stage]()


if __name__=='__main__':main()
