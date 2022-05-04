#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 13:11:32 2022

@author: markukob
"""


import numpy as np
import matplotlib.pyplot as plt

def protein_allocation(y_data, y_vec, constants, prot_list):
    n_t, n_y = y_data.shape
    macro_inds = [y_vec.index(m_name) for m_name in prot_list]
    rel_out = np.zeros((n_t, len(macro_inds)))
    for k in range(n_t):
        rel_out[k, :] = y_data[k, macro_inds]/sum(y_data[k, macro_inds])
    return rel_out


def biomass(y, wvec):
    #return np.inner(y.flatten(), wvec.T.flatten())
    return np.inner(y, wvec.T.flatten())


# FIXME: The wvec option appears to be broken (here and in plot_between)
def plot_ty(t, y, y_names, color_dict, wvec=None, line_style='-'):
    for i, y_name in enumerate(y_names):
        if wvec is None:
            plt.plot(t, y[:, i], color=color_dict[y_name], linestyle=line_style)    
            plt.ylabel('y(t)')
        else:
            tmp = y[:, i]
            for j, tval in enumerate(t):
                tmp[j] /= biomass(y[j, :], wvec)
            plt.plot(t, tmp, color=color_dict[y_name], linestyle=line_style)
            plt.ylabel('c(t)')


def plot_ty_between(t, y_1, y_2, y_names, color_dict, wvec=None, line_style='-'):
    for i, y_name in enumerate(y_names):
        if wvec is None:
            plt.fill_between(t, y_1[:, i], y_2[:, i], alpha=0.2, color=color_dict[y_name])
            plt.plot(t, y_1[:, i], color=color_dict[y_name], linestyle=line_style)    
            plt.plot(t, y_2[:, i], color=color_dict[y_name], linestyle=line_style)    
            plt.ylabel('y(t)')
        else:
            tmp_1 = y_1[:, i]
            tmp_2 = y_2[:, i]
            for j, tval in enumerate(t):
                tmp_1[j] /= biomass(y_1[j, :], wvec)
                tmp_2[j] /= biomass(y_2[j, :], wvec)
            plt.fill_between(t, tmp_1, tmp_2, alpha=0.3, color=color_dict[y_name])
            plt.plot(t, tmp_1, color=color_dict[y_name], linestyle=line_style)
            plt.plot(t, tmp_2, color=color_dict[y_name], linestyle=line_style)
            plt.ylabel('c(t)')

def plot_flux(out, u_names, col_dict):
    for i, u in enumerate(u_names):
        plt.plot(out['tgrid_u'], out['u_data'][:, i], color=col_dict[u])
        plt.ylabel('u(t)')


def MichMen(m):
    """
    Michaelis-Menten (with K=1)
    """
    return m/(1.0+m)

def InvMichMen(x):
    """
    inverse Michaelis-Menten (with K=1)
    """
    return x/(1.0-x)