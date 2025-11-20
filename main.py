import numpy as np
import scipy as sp
from scipy.sparse import issparse

def is_diagonally_dominant(A: np.ndarray | sp.sparse.csc_array) -> bool | None:

    if not isinstance(A, (np.ndarray, sp.sparse.csc_array)):
        return None
    
    if A.ndim !=2:
        return None
    
    if A.shape[0] != A.shape[1]:
        return None

    if issparse(A):
        A = A.toarray()

    diag_elements = np.abs(np.diagonal(A))

    row_sums = np.sum(np.abs(A), axis=1) - diag_elements

    return np.all(diag_elements > row_sums)

def residual_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray):
    if not isinstance(A, np.ndarray) or not isinstance(x, np.ndarray) or not isinstance(b, np.ndarray):
        return None
    
    if A.shape[1] != x.shape[0]:
        return None  
    if A.shape[0] != b.shape[0]:
        return None  

    if x.ndim != 1 or b.ndim != 1:
        return None
    
    residual = b - A.dot(x)
    norm = np.linalg.norm(residual)
    
    return norm
