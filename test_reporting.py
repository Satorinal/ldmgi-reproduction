import tempfile,unittest,csv,json
from pathlib import Path
import numpy as np
from reporting import cluster_summary,write_diagnostics
class ReportingTests(unittest.TestCase):
    def test_matching_and_empty_cluster(self):
        y=np.array([0,0,0,1,1,2]);p=np.array([1,1,0,0,0,1])
        d=cluster_summary(y,p,3)
        self.assertEqual(sum(r['size'] for r in d['cluster_rows']),6)
        self.assertEqual(sum(r['matched_correct'] for r in d['cluster_rows']),4)
        self.assertAlmostEqual(d['acc'],4/6)
        self.assertEqual(d['cluster_rows'][2]['size'],0)
        self.assertIsNone(d['cluster_rows'][2]['purity'])
    def test_fewer_and_more_clusters(self):
        for p in ([0,0,0,0],[0,1,2,3]):
            d=cluster_summary([0,0,1,1],p)
            self.assertAlmostEqual(d['acc'],.5)
    def test_csv_and_sample_consistency(self):
        with tempfile.TemporaryDirectory() as tmp:
            y=np.array([0,0,1,1]);p=np.array([1,1,0,0])
            write_diagnostics(tmp,y,p,np.zeros((4,4)),(2,2),['a','b','c','d'],2)
            with open(Path(tmp)/'samples.csv') as f: rows=list(csv.DictReader(f))
            self.assertEqual(sum(int(r['correct']) for r in rows),4)
            self.assertTrue((Path(tmp)/'cluster_examples.png').exists())
if __name__=='__main__': unittest.main()
