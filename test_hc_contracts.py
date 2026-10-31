"""Synthetic contracts only: no local data, patient IDs or network access."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent/'analysis'))
import steroid_eicu as eicu
import steroid_sequential as seq
import steroid_event_review as review
import hc_order_bridge as bridge
import sync_sources


class TimeAndSelection(unittest.TestCase):
    def test_execution_is_description_not_survival_requirement(self):
        self.assertEqual(review.order_proxy_window(0,60,60,1),'CONFIRMED_ADMIN_IN_WINDOW')
        self.assertEqual(review.order_proxy_window(0,np.nan,30,1),'CAPTURE_ENDS_BEFORE_WINDOW')
        self.assertEqual(review.order_proxy_window(0,np.nan,60,1),'NO_CONFIRMED_ADMIN_IN_COMPLETE_WINDOW')
        self.assertEqual(review.order_proxy_window(0,-1,60,1),'ADMIN_BEFORE_ORDER')

    def test_ne_left_limit_and_observation_end(self):
        g=pd.DataFrame(dict(start=[0.],end=[60.],stay_id=[1],icu_lo=[0.],icu_hi=[120.]))
        self.assertFalse(bridge.active_ne_record(g,0)[0])
        self.assertTrue(bridge.active_ne_record(g,60)[0])
        self.assertFalse(bridge.active_ne_record(g,61)[0])
        g.loc[0,'icu_hi']=60
        self.assertFalse(bridge.active_ne_record(g,60)[0])

    def test_future_order_not_prior_exposure(self):
        orders=pd.DataFrame(dict(index=[100.],route=['IV_prescribed'],poe_id=['synthetic']))
        self.assertEqual(bridge.order_history(orders,None,99),'NO_PRIOR_RECORDED_SYSTEMIC_GC_ORDER')
        self.assertEqual(bridge.order_history(orders,None,100),'SAME_TIME_OTHER_SYSTEMIC_ORDER')
        self.assertEqual(bridge.order_history(orders,None,100,'synthetic'),'NO_PRIOR_RECORDED_SYSTEMIC_GC_ORDER')
        self.assertEqual(bridge.order_history(orders,None,101),'PRIOR_SYSTEMIC_GC_ORDER')

    def test_unknown_order_not_assumed_untreated(self):
        orders=pd.DataFrame(dict(index=[1.],route=['route_unknown'],poe_id=['synthetic']))
        self.assertEqual(bridge.order_history(orders,None,2),'UNCERTAIN_PRIOR_ORDER_ROUTE')
        bad=pd.DataFrame(dict(planned=[1.]))
        self.assertEqual(bridge.order_history(None,bad,2),'UNCERTAIN_ORDER_HISTORY')

    def test_native_linkage_does_not_filter_on_final_status(self):
        rx=pd.DataFrame(dict(poe_id=['synthetic'],hadm_id=[1],subject_id=[2],is_HC=[True],route_class=['IV_prescribed'],start=[10.]))
        p=pd.DataFrame({c:[''] for c in review.POE_COLUMNS})
        for c,v in dict(poe_id='synthetic',hadm_id='1',subject_id='2',order_type='Medication',transaction_type='New',order_status='Inactive').items():p[c]=v
        p['order']=5.
        a,bad=bridge.build_orders(rx,p)
        self.assertEqual(len(a),1);self.assertTrue(bad.empty)
        p['order_status']='Cancelled';c,_=bridge.build_orders(rx,p)
        pd.testing.assert_frame_equal(a,c)
        p['hadm_id']='3';a,bad=bridge.build_orders(rx,p)
        self.assertTrue(a.empty);self.assertEqual(bad.reason.iloc[0],'NATIVE_IDENTITY_CONFLICT')


class EstimationAndExport(unittest.TestCase):
    def test_patient_weights_and_missing_endpoints(self):
        checks=eicu.order_estimation_tests(seq)
        self.assertTrue(all(checks.values()))

    def test_export_changes_paths_only(self):
        source='from pathlib import Path\nROOT=Path("old")\ndef risk(x):\n    return x.mean()\nREPORT="C:/Users/synthetic/Documents/ChatGPT/TTE/HTE_sepsis/private/example.csv"\n'
        out=sync_sources.export(source,{'ROOT':'ROOT = Path(__file__).resolve().parent'})
        self.assertEqual(sync_sources.normalized(source,{'ROOT'}),sync_sources.normalized(out,{'ROOT'}))
        self.assertNotIn('C:/Users/synthetic/',out)
        self.assertIn('../restricted/hc-private/example.csv',out)

    def test_bootstrap_failures_and_undefined_ratios_remain_visible(self):
        table=pd.DataFrame([
            dict(success=True,risk_treated=.2,risk_control=.1,risk_difference=.1,risk_ratio=2.,warnings=[]),
            dict(success=True,risk_treated=.3,risk_control=0.,risk_difference=.3,risk_ratio=np.nan,warnings=['synthetic warning']),
            dict(success=False,error='synthetic empty control pool')])
        result=bridge.bootstrap_summary(table)
        self.assertEqual((result['attempts'],result['successful'],result['failures']),(3,2,1))
        self.assertEqual(result['warning_replicates'],1)
        self.assertEqual(result['intervals']['risk_ratio']['undefined_n'],1)
        self.assertEqual(result['intervals']['risk_ratio']['finite_n'],1)
        self.assertAlmostEqual(result['intervals']['risk_difference']['lower'],.105)
        self.assertEqual(result['failure_reasons'],{'synthetic empty control pool':1})


class ClinicalExtension(unittest.TestCase):
    def test_partition_small_cells_have_complementary_suppression(self):
        values=bridge.partition_percentages([130,68,2],200)
        self.assertEqual(values,['65.0%','小样本或互补抑制','小样本或互补抑制'])
        self.assertEqual(bridge.partition_percentages([80,120,0],200),['40.0%','60.0%','0.0%'])
        with self.assertRaises(AssertionError):bridge.partition_percentages([80,119,0],200)

    def test_projection_half_open_gaps(self):
        t=np.array([-1.,0.,4.,5.,9.,10.])
        got=bridge.outside_cached_window(t,np.zeros(6),np.full(6,10),np.full(6,4),np.full(6,9))
        np.testing.assert_array_equal(got,[False,True,False,False,True,False])
        got=bridge.outside_cached_window(t,np.zeros(6),np.full(6,10),np.full(6,np.nan),np.full(6,np.nan))
        np.testing.assert_array_equal(got,[False,True,True,True,True,False])

    def test_separate_limitations(self):
        self.assertEqual(bridge.limitation_bits('Resuscitate (Full code)'),1)
        self.assertEqual(bridge.limitation_bits('DNR'),2)
        self.assertEqual(bridge.limitation_bits('Do Not Resuscitate / Do Not Intubate'),6)
        self.assertEqual(bridge.limitation_bits('Comfort measures only'),8)
        self.assertTrue(np.isnan(bridge.limitation_bits('Unknown')))
        self.assertTrue(np.isnan(bridge.limitation_bits('Full code; DNR')))

    def test_directive_expiry_and_preindex_boundary(self):
        ev=pd.DataFrame(dict(time=[0.,20.],bits=[1.,2.],end=[10.,np.inf],poe_id=['synthetic-a','synthetic-b']))
        queries=np.array([0.,5.,10.,11.,20.,21.]);v,t,f=bridge.order_code_at(ev,queries,np.full(6,-1.))
        np.testing.assert_allclose(v,[np.nan,1.,1.,np.nan,np.nan,2.],equal_nan=True)
        self.assertEqual(f[3],4);self.assertEqual(t[3],10.)
        # Cancelling an older directive cannot invalidate a newer active one.
        ev.loc[0,'end']=25.
        v,_,_=bridge.order_code_at(ev,np.array([26.]),np.array([-1.]))
        self.assertEqual(v[0],2.)

    def test_conflicting_or_newer_unknown_directive(self):
        v,d=bridge.combine_directives([1.,1.,2.],[5.,5.,5.],[2.,np.nan,1.],[5.,6.,4.])
        np.testing.assert_allclose(v,[np.nan,np.nan,2.],equal_nan=True)
        self.assertEqual(d[0],1.)

    def test_existing_two_clock_and_ne_contracts(self):
        checks=seq.hc_cf_tests()
        self.assertTrue(checks['two_clock_synthetic'])
        self.assertEqual(checks['random_bruteforce_groups'],200)


if __name__=='__main__':unittest.main()
