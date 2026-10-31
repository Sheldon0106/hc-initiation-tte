"""Independent MIMIC HC-order source cohort; no mortality-effect estimation.

Uses all admissions in the existing adult NE source, not admissions selected by
the later actual-dose risksets. Adds one branch to the existing private POE
checkpoint; never writes the frozen actual-dose checkpoint.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / 'analysis'))
from hc_paths import configured_path
import steroid_sequential as seq
import steroid_event_review as review

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
    ne=b['NE'];start=ne.start-ne.hadm_id.map(ref.ne0);end=ne.end-ne.hadm_id.map(ref.ne0)
    allrows=[];flow=[];began=time.perf_counter();last=began
    for case in d['cases'].itertuples():
        elapsed=case.elapsed_h*3600;possible=set(ne.loc[(start<elapsed)&(end>=elapsed),'hadm_id'])
        controls=[]
        for h in sorted(possible):
            s=ref.loc[h];t=float(s.ne0+elapsed)
            if s.subject_id==case.subject_id or t>=s.discharge or (np.isfinite(s.death_time) and t>=s.death_time):continue
            if first_gc.get(h,np.inf)<=t or uncertain_gc.get(h,np.inf)<=t or first_order.get(h,np.inf)<=t:continue
            active,stay,_=active_ne_record(ng[h],t)
            if not active:continue
            if order_history(og.get(h),ug.get(h),t)!='NO_PRIOR_RECORDED_SYSTEMIC_GC_ORDER':continue
            state,strict=seq.qualify(h,t,ctx,m,False)
            if state!='QUALIFIED_RECORD_PROXY':continue
            controls.append(dict(case_id=case.case_id,hadm_id=int(h),subject_id=int(s.subject_id),stay_id=int(stay),index=t,elapsed_h=case.elapsed_h,treated=0,strict_infection=strict))
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
        checks=dict(independent_source=True,native_identity_only=True,eligibility_before_execution=True,no_future_survival=True,
            distinct_control_patient=True,same_NE_elapsed=True,no_outcome_model=True,frozen_actual_cache_unchanged=True))
    save(d)
    say(dict(stage='order_support_done',cases=m.count(len(flow)),without_controls=m.count(flow.controls.eq(0).sum()),
        control_patients=m.count(panel.loc[panel.treated.eq(0),'subject_id'].nunique()),riskset_rows=m.count(len(panel)),
        control_pool_median=float(flow.controls.median()),seconds=d['support_seconds'],models=0,effects=0))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['plan','project','evaluate','support']);a=p.parse_args()
    globals()[a.stage]()


if __name__=='__main__':main()
