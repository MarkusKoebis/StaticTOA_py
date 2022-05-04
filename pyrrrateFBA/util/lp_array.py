# -*- coding: utf-8 -*-

"""
LP array:
     Create a "container" such that numpy arrays and scipy.sparse arrays can be dealt with in a
     unified manner

     TODO: dtypes in the underlying (?)
"""

import numpy as np
import scipy.sparse as sp

DEFAULT_FORMAT = 'csr'

class LpArray:
    """
    This is the "container".
    """
    def __init__(self, orig_array):
        """
        Save array
        """
        self.data = orig_array
        if isinstance(orig_array, np.ndarray):
            array_type = 'np'
        elif isinstance(orig_array, sp.csr_matrix):
            array_type = 'csr'
        else:
            raise TypeError('So far, only scipy.sparse.csr and numpy.ndarray are supported')
        self.array_type = array_type

    # pass through properties
    shape = property(lambda self: self.data.shape)


    def __str__(self):
        """
        just re-route the original __str__
        """
        return 'LPARRAY: '+ self.data.__str__()

    def __getitem__(self, key):
        return LpArray(self.data.__getitem__(key))

    def __setitem__(self, key, val):
        self.data.__setitem__(key, _ensure_format(val, out_format=self.array_type))

    def toarray(self):
        """
        return an np.ndarray with the same content
        """
        return _ensure_np(self.data)


def bmat(blocks, out_format=None):
    """
    Build array from blocks (Basically copied from scipy.sparse)
    """
    blocks = np.asarray(blocks, dtype='object')

    if blocks.ndim != 2:
        raise ValueError('blocks must be 2-D')

    n_brows, n_bcols = blocks.shape
    if out_format is None:
        if min(n_brows, n_bcols) == 0:
            out_format = DEFAULT_FORMAT
        else:
            out_format = blocks[0, 0].array_type
    #
    if out_format == 'np':
        #use_lib = np
        new_array = np.bmat([[_ensure_np(i) for i in b] for b in blocks])
    elif out_format ==  'csr':
        #use_lib = sp
        new_array = sp.bmat([[_ensure_csr(i) for i in b] for b in blocks], format='csr')
    #
    #new_array = use_lib.bmat([[_ensure_format(b, out_format=out_format)])
    #
    return LpArray(new_array)



def hstack(blocks, out_format=None):
    """
    This is basically copied from the scipy.sparse equivalent.
    """
    return bmat([blocks], out_format=out_format)


def vstack(blocks, out_format=None):
    """
    This is basically copied from the scipy.sparse equivalent.
    """
    return bmat([[b] for b in blocks], out_format=out_format)



def _ensure_csr(mat_in):
    """
    If the matrix is not csr: Make it so
    TODO: To this point: copied from linalg
    """
    if isinstance(mat_in, LpArray):
        return _ensure_csr(mat_in.data)
    if isinstance(mat_in, sp.csr_matrix):
        return mat_in
    if isinstance(mat_in, np.ndarray):
        return sp.csr_matrix(mat_in)
    return mat_in.tocsr()


def _ensure_np(mat_in):
    """
    If the matrix is sparse: todense() it
    TODO: To this point: copied from linalg
    """
    if isinstance(mat_in, LpArray):
        return _ensure_np(mat_in.data)
    if isinstance(mat_in, sp.base.spmatrix):
        return mat_in.todense()
    return mat_in


def _ensure_format(mat_in, out_format):
    """
    make sure the array is in (np.ndarray / scipy.csr / ... ) format
    MAYBE: Add a copy=True/False option(?)
    """
    if out_format=='np':
        return _ensure_np(mat_in)
    if out_format=='csr':
        return _ensure_csr(mat_in)
    raise ValueError('Only np.ndarray and scipy.sparse.csr_matrix are supported.')
