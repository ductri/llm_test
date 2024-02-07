import torch
import numpy as np



def X_t_Db_t_Y(X, Db, Y):
    """
    X:  n1 x n2
    Db: n2 x B
    Y:  n2 x n3
    -------------
    Return Q: B x n1 x n2, where Q[b, :, :] = X @ diag(D[:, b] @ Y)
    """
    assert X.ndim == 2
    assert Db.ndim == 2
    assert Y.ndim == 2
    tmp = X_t_Db(X, Db)
    tmp = tmp.permute(2, 0, 1)
    tmp = Xb_t_Y(tmp, Y)
    return tmp


def X_t_diag_d(X, d):
    return X*d


def X_t_Db(X, D):
    """
    X: n1 x n2
    D: n2 x B
    --------------
    Return Q: n1 x n2 x B, where Q[:, :, b] = X @ diag(D[:, b])
    """
    return X[..., None] * D[None, ...]

def Xb_t_Y(X, Y):
    """
    Xb: B x n1 x n2
    Y: n2 x n3
    -------------------
    Return Q: B x n1 x n3, where Q[b, :, :] = Xb[b, :, :] @ Y
    """
    return X @ Y

def count_parameters(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return pytorch_total_params

