import unittest
import numpy as np
from baselines import discriminant_svd,discriminant_features,ncut_matrix,ncut_objective,llc_linear_matrix,neighbors_excluding_self,spectral_rows
class BaselineTests(unittest.TestCase):
    def test_discriminative_kernel(self):
        X=np.random.default_rng(0).normal(size=(14,5));A=X-X.mean(0);U,s=discriminant_svd(X)
        for reg in (.01,1,100):
            Z=discriminant_features(U,s,reg)
            expected=A@np.linalg.solve(A.T@A+reg*np.eye(5),A.T)
            np.testing.assert_allclose(Z@Z.T,expected,atol=1e-11)
    def test_llc_prediction_energy(self):
        X=np.random.default_rng(2).normal(size=(18,4));ids=neighbors_excluding_self(X,5);reg=.7
        L,Q=llc_linear_matrix(X,ids,reg);A=np.zeros((18,18))
        for i,j in enumerate(ids):A[i,j]=np.linalg.solve(X[j]@X[j].T+reg*np.eye(5),X[j]@X[i])
        np.testing.assert_allclose(L.toarray(),(np.eye(18)-A).T@(np.eye(18)-A),atol=1e-11)
        g=np.random.default_rng(1).normal(size=18)
        self.assertAlmostEqual(float(g@(L@g)),float(np.sum((g-A@g)**2)),places=10)
        _,d=spectral_rows(L,3,Q,reg);self.assertLess(d['eigen_residual'],1e-8)
    def test_ncut_energy(self):
        X=np.array([[0.],[.1],[1.],[1.1]]);ids=neighbors_excluding_self(X,2)
        L,degree,_=ncut_matrix(X,ids,1);pred=np.array([0,0,1,1])
        degree_array=np.exp(degree['log_degree']).astype(float)
        A=np.diag(np.sqrt(degree_array))@(np.eye(4)-L.toarray())@np.diag(np.sqrt(degree_array))
        expected=sum(A[np.ix_(pred==j,pred!=j)].sum()/degree_array[pred==j].sum() for j in range(2))
        self.assertAlmostEqual(ncut_objective(L,degree,pred,2),expected)
        tiny,logdeg,_=ncut_matrix(X,ids,1e-8)
        self.assertTrue(np.isfinite(tiny.data).all())
        self.assertTrue(np.isfinite(ncut_objective(tiny,logdeg,pred,2)))
        self.assertAlmostEqual(ncut_objective(tiny,logdeg,pred,2),0.,places=8)
if __name__=='__main__':unittest.main()
