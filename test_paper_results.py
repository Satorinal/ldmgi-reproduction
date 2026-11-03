import unittest
from make_paper_results import select_result
class PaperTableTests(unittest.TestCase):
    def test_representative_not_best_individual_seed(self):
        rows=[{'reg':1,'selected_by_objective':{'acc':.6,'nmi':.8},'mean_acc':.7,'mean_nmi':.8,
               'trials':[{'acc':.99},{'acc':.6}]},
              {'reg':10,'selected_by_objective':{'acc':.8,'nmi':.7},'mean_acc':.65,'mean_nmi':.75}]
        self.assertEqual(select_result(rows,'acc')['reg'],10)
        self.assertEqual(select_result(rows,'acc',True)['reg'],1)
        self.assertEqual(select_result(rows,'nmi')['reg'],1)
    def test_parameter_tie_and_invalid_representative(self):
        rows=[{'reg':10,'selected_by_objective':{'acc':.8}},
              {'reg':1,'selected_by_objective':{'acc':.8}},
              {'reg':.1,'selected_by_objective':None}]
        self.assertEqual(select_result(rows,'acc')['reg'],1)
if __name__=='__main__':unittest.main()
