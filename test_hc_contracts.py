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
        source='from pathlib import Path\nROOT=Path("old")\ndef risk(x):\n    return x.mean()\n'
        out=sync_sources.export(source,{'ROOT':'ROOT = Path(__file__).resolve().parent'})
        self.assertEqual(sync_sources.normalized(source,{'ROOT'}),sync_sources.normalized(out,{'ROOT'}))


if __name__=='__main__':unittest.main()
