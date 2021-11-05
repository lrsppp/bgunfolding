import numpy as np

def setup_0():
    from bgunfolding.ibu import IBU
    
    ibu = IBU(n_iterations = 1,
              x0 = np.array([3, 3]),
              epsilon = 1e-18)
    
    ibu.fit(f = np.array([2, 2]),
            g = np.array([3, 4]),
            b = np.array([1, 2]),
            A = np.array([[0.8, 0.2],
                          [0.2, 0.8]]))
    
    return ibu

def test_ibu_fit():
    
    ibu = setup_0()
    assert (ibu.g == ibu.A @ ibu.f + ibu.b).all() == True
    assert (ibu.A.sum(axis = 1) == np.ones(ibu.A.shape[0])).all() == True

def test_ibu_predict():
    
    ibu = setup_0()
    
    assert np.isclose(ibu.predict()[0], np.array([2.28, 2.37])).all() == True