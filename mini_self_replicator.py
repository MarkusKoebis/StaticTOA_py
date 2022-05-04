#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 14:02:48 2021

@author: markukob
"""

# By-hand creation of a deFBA/cFBA/RBA model
import numpy as np
import scipy.sparse as sp


def create_name_vectors():
    #        0     1    2
    y_vec = ['M', 'T', 'R']
    m_vec = []
    #        0     1     2     3      4
    u_vec = ['vN', 'vT', 'vR', 'vdT', 'vdR']
    return y_vec, m_vec, u_vec


def create_color_palette():
    cols = {'M': 'red',
            'T': 'blue',
            'R': 'gold',
            #
            'vN': 'black', 
            'vT': 'lightblue', 
            'vR': 'gold',
            'vdT': 'cyan',
            'vdR': 'orange'}
    return cols


def create_objective_vectors(flag_objective, constants):
    y_vec, _,_ = create_name_vectors()
    if (flag_objective==0) or (flag_objective is None): # optimize R production + end time R
        phi1 = np.zeros((len(y_vec), 1))
        phi1[y_vec.index('R'), 0] = -1.0
        phi3 = phi1
    elif flag_objective==1: # biomass (without Q) integral
        #
        phi1 = np.zeros((len(y_vec), 1))
        #
        phi1[y_vec.index('R'), 0] = -1.0*constants['npR']
        phi1[y_vec.index('T'), 0] = -1.0*constants['npT']
        #
        phi3 = 0.0*phi1
    else:
        raise ValueError(f'Unknown objective flag: {flag_objective}')
    return phi1, phi3


def create_constants(flag, alpha=None):
    #npT = constants['npT']
    #npR = constants['npR']
    #kdR    = constants['kdR']
    #kdT    = constants['kdT']
    #kdM    = constants['kdM']
    #kcatT  = constants['kcatT']
    #kcatR  = constants['kcatR']
    #npTred = constants['npTred']       
    #npRred = constants['npRred']        
    #ubvdT  = constants['ubvdT']
    #ubvdR  = constants['ubvdR']
    #
    c = {'npT': 10.,  # protein size transporter
         'npR': 15.,  # protein size ribosome
        #
         'kcatT':  10.0, # catalytic constants
         'kcatR':  10.0}
    #
    if flag==0:
        if alpha is None:
            alpha = 0.0
        c.update({#
             'kdR': 0., # degradation constants
             'kdT': 0.,
             'kdM': 0.,
             #
             'ubvdT': 0.0, # maximum protein degradation rates
             'ubvdR': 0.0
             })
    elif flag==1:
        if alpha is None: 
            alpha = 0.0
        c.update({#
             'kdR': 0., # degradation constants
             'kdT': 0.,
             'kdM': 0.,
             #
             'ubvdT': 1e40, # maximum protein degradation rates
             'ubvdR': 1e40
             })
    elif flag==2:
        if not alpha == None:
            raise ValueError('You cannot set alpha for flag == 2')
        alpha = 0.2
        c.update({#
             'kdR':  0., # degradation constants
             'kdT':  0.,
             'kdM':  0.,
             #
             'ubvdT': 1e40, # maximum protein degradation rates
             'ubvdR': 1e40
             })
    elif flag==3:
        if not alpha == None:
            raise ValueError('You cannot set alpha for flag == 3')
        alpha = 0.2
        c.update({'kdR':  0.02, # degradation constants
             'kdT':  0.01,
             'kdM':  0.005,
             #
             'ubvdT': 1e40, # maximum protein degradation rates
             'ubvdR': 1e40
             })
    else:
        raise ValueError('Unknown flag')
    #
    c['npTred'] = alpha*c['npT'] # active protein degradation: reduced effective protein number
    c['npRred'] = alpha*c['npR']
    #
    return c


def create_initial_values(flag, constants):
    y_vec, _, _ = create_name_vectors()
    if flag == 0:
        R0 = 0.01
        T0 = 0.1
        M0 = 1.0
    else:
        raise ValueError(f'Unknown flag for initial value creation: {flag}')
    #
    y0 = np.zeros((len(y_vec), 1))
    y0[[y_vec.index('R')], 0] = R0
    y0[[y_vec.index('T')], 0]  = T0   
    y0[[y_vec.index('M')], 0]  = M0
    return y0


def create_weight_vector(y_vec, constants):
    n_y = len(y_vec)
    wvec = np.zeros((n_y, 1))
    wvec[y_vec.index('M')] = 1.0
    wvec[y_vec.index('T')] = constants['npT']
    wvec[y_vec.index('R')] = constants['npR']
    #wvec[y_vec.index('T')] = 1.0
    #wvec[y_vec.index('R')] = 1.0
    return wvec


def create_reaction_matrices(y_vec, m_vec, u_vec, constants):
    n_y = len(y_vec)
    n_m = len(m_vec)
    n_u = len(u_vec)    
    npT = constants['npT']
    npR = constants['npR']
    npTred = constants['npTred']       
    npRred = constants['npRred']        
    smat2 = np.zeros((n_y, n_u))
    smat1 = np.zeros((n_m, n_u))
    # 0:  vN: N -> M
    smat2[y_vec.index('M'), u_vec.index('vN')] = 1.0
    # 1:  vT: M -> (npT - npTred)* T
    smat2[[y_vec.index('M'), y_vec.index('T')], u_vec.index('vT')] = [-(npT - npTred), 1.0]
    # 2:  vR: M -> (npR - npRred)*R
    smat2[[y_vec.index('M'), y_vec.index('R')], u_vec.index('vR')] = [-(npR - npRred), 1.0]
    # 3:  vdT npT*T -> M
    smat2[[y_vec.index('M'), y_vec.index('T')], u_vec.index('vdT')] = [npT, -1.0]
    # 4:  vdR npR*R -> M
    smat2[[y_vec.index('M'), y_vec.index('R')], u_vec.index('vdR')] = [npR, -1.0]    
    
    return smat1, smat2


def create_degredation(y_vec, constants):
    n_y = len(y_vec)
    smat4 = np.zeros((n_y, n_y))
    kdR    = constants['kdR']
    kdT    = constants['kdT']
    kdM    = constants['kdM']
    #
    smat4[y_vec.index('R'), y_vec.index('R')] = -kdR
    smat4[y_vec.index('T'), y_vec.index('T')] = -kdT
    smat4[y_vec.index('M'), y_vec.index('M')] = -kdM
    return smat4


def create_mixed_constraints(y_vec, u_vec, ext, constants):
    n_y = len(y_vec)
    n_u = len(u_vec)
    #
    npT = constants['npT']
    npR = constants['npR']
    kcatT  = constants['kcatT']
    kcatR  = constants['kcatR']
    #
    n_con = 2
    matrix_u = np.zeros((n_con, n_u))
    vec_h = np.zeros((n_con, 1))
    #
    def matrix_y(t):
        m = np.zeros((n_con, n_y))
        # 0: vN <= ex/(1+ex)*kcatT*T
        m[0, y_vec.index('T')] = -ext(t)/(1.0+ext(t))*kcatT
        # 1: npT/kcatR*vT + npR/kcatR*vR <= R
        m[1, y_vec.index('R')] = -1.0
        return sp.csr_matrix(m)
    # 0: vN <= ex/(1+ex)*kcatT*T
    matrix_u[0, u_vec.index('vN')] = 1.0
    # 1: npT/kcatR*vT + npR/kcatR*vR <= R
    matrix_u[1, [u_vec.index('vT'), u_vec.index('vR')]] = [npT/kcatR, npR/kcatR]
    return matrix_y, matrix_u, vec_h


def create_flux_bounds(u_vec, constants):
    n_u = len(u_vec)
    lbvec = np.zeros((n_u, 1))
    ubvec = 1e40*np.ones((n_u, 1))
    #        
    ubvdT  = constants['ubvdT']
    ubvdR  = constants['ubvdR']
    #
    ubvec[u_vec.index('vdT')]  = ubvdT
    ubvec[u_vec.index('vdR')]  = ubvdR
    #
    return lbvec, ubvec

