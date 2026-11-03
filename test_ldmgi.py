import unittest
import numpy as np
from ldmgi import local_matrix,build_laplacian,embed,discretize
from run_evaluation import acc
from scipy.linalg import solve
class Tests(unittest.TestCase):
    def test_identity(self):
        rng=np.random.default_rng(0)
        for k,d,reg in [(3,8,.2),(7,3,2.)]:
            X=rng.normal(size=(k,d)); A=(X-X.mean(0)).T; H=np.eye(k)-np.ones((k,k))/k
            original=H-A.T@solve(A@A.T+reg*np.eye(d),A)
            np.testing.assert_allclose(original,reg*local_matrix(X,reg),atol=1e-10)
    def test_assembly_spectrum(self):
        X=np.random.default_rng(3).normal(size=(30,4)); L,_=build_laplacian(X)
        np.testing.assert_allclose(L.toarray(),L.toarray().T,atol=1e-12)
        np.testing.assert_allclose(L@np.ones(30),0,atol=1e-10)
        self.assertGreater(np.linalg.eigvalsh(L.toarray()).min(),-1e-10)
        F,v,r=embed(L,3); self.assertLess(r,1e-8)
        np.testing.assert_allclose(v,np.linalg.eigvalsh(L.toarray())[1:4],atol=1e-9)
        result=discretize(F); self.assertTrue(np.all(np.diff(result['history'])<1e-9))
    def test_disconnected(self):
        X=np.array([[0.],[1.],[2.],[100.],[101.],[102.]])
        L,cc=build_laplacian(X,k=3); self.assertEqual(cc,2)
        F,v,r=embed(L,2); self.assertLess(abs(v[0]),1e-9); self.assertLess(r,1e-8)
    def test_acc(self):
        self.assertEqual(acc([0,0,1,1],[7,7,3,3]),1.)
if __name__=='__main__': unittest.main()
