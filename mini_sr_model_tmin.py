#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 15:15:49 2021

@author: markukob
"""

import time
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import gridspec
import lowlevel_cFBAstuff as cFBA
import mini_self_replicator as mini
from itertools import chain, combinations
from scipy.optimize import bisect


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

"""
def plot_y(out, y_names, color_dict, wvec=None):
    for i, y in enumerate(y_names):
        if wvec is None:
            plt.plot(out['tgrid'], out['y_data'][:, i], color=color_dict[y])    
            plt.ylabel('y(t)')
        else:
            tmp = out['y_data'][:, i]
            for j, t in enumerate(out['tgrid']):
                tmp[j] /= biomass(out['y_data'][j, :], wvec)
            plt.plot(out['tgrid'], tmp, color=color_dict[y])
            plt.ylabel('c(t)')

"""
def plot_flux(out, u_names, color_dict):
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


    
    
y_vec, m_vec, u_vec = mini.create_name_vectors()
n_y, n_u = len(y_vec), len(u_vec)
col_dict = mini.create_color_palette()
prot_list = ['M', 'T', 'R']
#tspanRBA = np.linspace(0.0, 0.01, 2)
del_t_RBA = 0.01
# np_red propto protein oder nur fürs produzieren reduziert
# Überall (einfaches) deFVA dazu oder mit deterministischer degradation eindeutig machen
# Perfekt adaptiert: RBA ist super
# Shift von extern Zustand A nach extern Zustand B
# Nicht perfekt adaptiert: interessante Dynamik: Constraint fulfilment plot
# Nicht perfekt adaptiert: Speicher intern: Optimist (besser?)/Pessimist
# Inwieweit kann man das auch schon mit cFBA sehen?


'''
constants = mini.create_constants(flag=2)
y_vec, m_vec, u_vec = mini.create_name_vectors()
yinit = np.array([[2.4, 3.2, 3.4]]).T
ygoal = np.array([[5.3, 2.5, 3.6]]).T
wvec = mini.create_weight_vector(y_vec, constants)
ext_cond = lambda t: 0.2
tast, out, _ = cFBA.opt_jump_from_to(mini, yinit, ygoal, ext_cond, constants, # J0=None, Jend=None, wvec=None, t0=0.0,
                               #est_min=1e-3, est_max=1e3, t_estimates=None,
                               n_steps=100, verbosity_level=0)
#plt.plot(out['tgrid'], out['y_data'])
deFVAmin, deFVAmax, info_deFVA = cFBA.relative_tmin_deFVA(mini, yinit, ygoal, ext_cond, tast,
                                                          constants, wvec, verbosity_level=10,
                                                          t_extend_trials=(1.0e-7, 1.0e-5))
plt.plot(out['tgrid'], deFVAmin, marker='*')
plt.plot(out['tgrid'], deFVAmax, marker='*')
'''

'''
# Number crunshing -------------------------------------------------------------------------------
# tmin opt: Just "normal" and direct cell doubling
#cFBA.RBA_like(m_dict, t0, del_t, verbosity_level=0, mumin=1.0, mumax=2.0, y_start=None, wvec=None)
#cFBA.cFBA(m_dict, tspan, verbosity_level=0, mumin=1.0, mumax=5.0, y_start=None, wvec=None)
val_for_M = 0.0
v_level = 0
phi_fva = 0.1
ext0_vals = {10.0: 'optimistic',
             1.0: 'perfect', 
             0.5: 'pessimistic'
             }
j = 1
save_figures_to_files = True
#save_figures_to_files = False
for ext0_val, strategy_string in ext0_vals.items():
    print(f'j = {j}\n')
    for use_model_flag in [0, 1, 2, 3]:
        constants = mini.create_constants(use_model_flag)
        wvec = mini.create_weight_vector(y_vec, constants)
        # baseline solution RBA ------------------------------------------------------------------
        ext0 = lambda t: ext0_val
        m_dict = cFBA.create_m_dict(mini, ext0, constants)
        outRBA, mu0 = cFBA.RBA_like(m_dict, 0.0, del_t_RBA, verbosity_level=0, mumin=1.0, mumax=2.0, y_start=None, wvec=wvec)
        y0 = outRBA['y']
        y0[y_vec.index('M')] = val_for_M
        # new external conditions ----------------------------------------------------------------
        ext = lambda t: 1.0
        m_dict = cFBA.create_m_dict(mini, ext, constants)
        rba_new, munew = cFBA.RBA_like(m_dict, 0.0, del_t_RBA, verbosity_level=0, mumin=1.0, mumax=2.0, y_start=None, wvec=wvec)
        # for comparison: RBA with fixed (wrong) initial values ----------------------------------
        rba_it, muit = cFBA.RBA_like(m_dict, 0.0, del_t_RBA, verbosity_level=0, mumin=1.0, mumax=2.0, y_start=y0, wvec=None)
        #
        y1 = 2.0*y0
        # t-min-opt ------------------------------------------------------------------------------
        eye_n, zer_n = np.eye(n_y), np.zeros((n_y, n_y))
        m_dict['matrix_start'] = np.vstack([eye_n, zer_n])
        m_dict['matrix_end'] = np.vstack([zer_n, eye_n])
        m_dict['vec_bndry'] = np.vstack([y0, y1])
        n_steps = 101
        #n_steps = 11 # DEBUG
        tast, outtmin, _ = cFBA.t_min_opt(m_dict, est_min=0.05, est_max=20.0, n_steps=n_steps, verbosity_level=v_level)
        # deFVA ----------------------------------------------------------------------------------
        #y_min_all, y_max_all = cFBA.low_level_deFVA(outtmin, y_vec, verbosity_level=v_level, fva_level=3)# DEBUG
        y_min_all, y_max_all = cFBA.low_level_deFVA(outtmin, y_vec, verbosity_level=v_level, varphi=phi_fva, fva_level=2)
        #
        _, _, feas_result = cFBA.t_min_opt(m_dict, t_estimates=np.linspace(0.0, 2*tast, 51), n_steps=n_steps, verbosity_level=v_level)
        # Postprocessing -------------------------------------------------------------------------
        # growth constants
        sigma0 = np.log(mu0)/del_t_RBA
        sigma1 = np.log(munew)/del_t_RBA
        sigmait = np.log(muit)/del_t_RBA
        # metabolite amounts
        #print(
        mean_M = np.mean(outtmin['y_data'][:, y_vec.index('M')])
        mean_M_min = np.mean(y_min_all[:, y_vec.index('M')])
        max_M_min = np.max(y_min_all[:, y_vec.index('M')])
        #)
        #
        info_str = f'{strategy_string}:\n'                                   + \
                   f'Biomass at t = 0: {biomass(y0, wvec)}\n'                + \
                   f'Biomass at end point: {biomass(y1, wvec)}, \n'          + \
                   f'relation: {biomass(y1, wvec)/biomass(y0, wvec)}\n'      + \
                   f'Biomass prediction by RBA1: {np.exp(sigma0*tast)}\n'    + \
                   f'Biomass prediction by RBA2: {np.exp(sigma1*tast)}\n'    + \
                   f'Biomass prediction by it.RBA: {np.exp(sigmait*tast)}\n' + \
                   f'M values:\n    {mean_M}\n    {mean_M_min}\n    {max_M_min}'
        # plots setup
        fig = plt.figure(constrained_layout=True)
        fig.set_size_inches(3*4, 3*3)
        #gs = fig.add_gridspec(4, 2)
        #fig_ax_y, fig_ax_u, fig_ax_bisect = fig.add_subplot(gs[0:2, 0]), fig.add_subplot(gs[0:2, 1]), fig.add_subplot(gs[-1:, :])
        # solution plots
        gs = fig.add_gridspec(3, 3)
        fig_ax_y, fig_ax_u, fig_ax_cons, fig_ax_text = fig.add_subplot(gs[0:2, 0:2]), fig.add_subplot(gs[0:2, -1]), fig.add_subplot(gs[2, 0:2]), fig.add_subplot(gs[-1:, -1])
        #fig_ax_y, fig_ax_u, fig_ax_bisect = fig.add_subplot(gs[0:2, 0]), fig.add_subplot(gs[0:2, 1]), fig.add_subplot(gs[-1:, :])
        # solution plots
        plt.subplot(fig_ax_y)
        plot_ty_between(outtmin['tgrid'], y_min_all, y_max_all, y_vec, col_dict, wvec=None, line_style=':')
        #plot_y(outtmin, y_vec, col_dict)
        plot_ty(outtmin['tgrid'], outtmin['y_data'], y_vec, col_dict)
        #plt.subplot(fig_ax_y), plot_y(outtmin, y_vec, col_dict, wvec=wvec)
        plt.title(f'model_flag = {use_model_flag}, t_min = {tast}')
        plt.subplot(fig_ax_u), plot_flux(outtmin, u_vec, col_dict)
        fig_ax_text.text(0*tast/2.0, 0*2.0, info_str)
        #plt.subplot(fig_ax_bisect), plt.plot(feas_result[:, 0], feas_result[:, 1], marker='x')
        plt.subplot(fig_ax_cons)
        #cons_plot_kinds = ['flux_bounds', 'mixed', 'positivity', 'dyn', 'qssa']
        out_cons_fl, labels_fl = cFBA.constraint_fulfillment(outtmin, m_dict, plot_kind = 'flux_bounds')
        out_cons_mix, labels_mix = cFBA.constraint_fulfillment(outtmin, m_dict, plot_kind = 'mixed')
        out_cons_plot = np.hstack([out_cons_fl, out_cons_mix])
        plt.plot(outtmin['tgrid'], out_cons_plot)
        plt.legend(labels_fl + labels_mix, loc = 'upper left', ncol=6, mode='expand')
        plt.ylim(-0.01, 0.11)
        #plt.plot(outtmin['tgrid'], out_cons_mix)
        #plt.legend(labels_mix, loc = 'upper left', ncol=2)
        plt.subplot(fig_ax_y)
        # plot RBA solution (with falsely adapted values)
        plot_ty(outtmin['tgrid'], (np.exp(sigmait*outtmin['tgrid'])*y0).T, y_vec, color_dict=col_dict, line_style='dashdot')
        if save_figures_to_files:
            plt.savefig(f'./svg/fig_{j}.svg')
            plt.savefig(f'./png/fig_{j}.png')
            time.sleep(0.2)
            j += 1
        plt.show()
'''

'''
# (C) jump from one RBA solution to another
set_M_val = 0.0
v_level = 2 # verbosity level
ext1_vals = {20.0: 'bad to good'}#, 0.2: 'good to bad'}
ext1_vals = {0.5: 'good to bad'}#
#ext1_vals = {10.0: 'bad to good', 0.2: 'good to bad'}
ext1_vals = {2.0: 'bad to good', 0.5: 'good to bad'}
for ext1_val, situation_string in ext1_vals.items():
    for use_model_flag in [2]: #[0,1,2,3]:
        constants = mini.create_constants(use_model_flag)
        wvec = mini.create_weight_vector(y_vec, constants)
        # create RBA solution 0
        ext0 = lambda t: 1.0
        m_dict0 = cFBA.create_m_dict(mini, ext0, constants)
        outRBA0, mu0 = cFBA.RBA_like(m_dict0, 0.0, del_t_RBA, verbosity_level=0, 
                                     mumin=1.0, mumax=2.0, y_start=None, wvec=wvec)
        #outRBA0, mu0 = cFBA.RBA_like(mini, ext0, tspanRBA, constants)
        #y0 = np.array([outRBA0['ydata'][0, :]]).T
        y0 = outRBA0['y']
        y0[y_vec.index('M')] = set_M_val
        # create RBA solution 1
        ext1 = lambda t: ext1_val
        m_dict1 = cFBA.create_m_dict(mini, ext1, constants)
        outRBA1, mu1 = cFBA.RBA_like(m_dict1, 0.0, del_t_RBA, verbosity_level=0,
                                     mumin=1.0, mumax=2.0, y_start=None, wvec=wvec)
        #outRBA1, mu1 = cFBA.RBA_like(mini, ext1, tspanRBA, constants)
        #y1 = np.array([outRBA1['y_data'][0, :]]).T
        y1 = outRBA1['y']
        y1[y_vec.index('M')] = set_M_val
        #
        #print(y0, y1)
        #
        ext = ext1
        # t-min-opt in absolute amounts
        fig = plt.figure()
        #fig.title(f'flag = {use_model_flag}')
        fig.set_size_inches(3*4, 3*3)
        n_steps = 101
        #try:
        #tast, out, _ = cFBA.opt_jump_from_to(mini, y0, y1, ext, constants,
        #                                         J0=None, Jend=None,
        #                     wvec = None,
        #                     n_steps=n_steps, verbosity_level=v_level,
        #                     est_min=0.01, est_max=50.5)
         #   #plt.subplot(2,2,1); plot_y(out, y_vec, col_dict); plt.title('absolute formulation')
        #plt.subplot(2,2,1)
        #plot_ty(out['tgrid'], out['y_data'], y_vec, col_dict)#, wvec=None, line_style='-')
        #plt.title('absolute formulation')
        #plt.subplot(2,2,2); plot_flux(out, u_vec, col_dict); plt.title(situation_string)
        #except:
        #    pass
        #    plt.subplot(2,2,1); plt.title('No solution, ' + situation_string)
        #
        # t-min-opt in concentrations
        #try:
        tast, out, _ = cFBA.opt_jump_from_to(mini, y0, y1, ext, constants,
                                                 J0=None, Jend=None, est_min=0.01, est_max=2.5,
                                                 wvec = wvec,
                                                 n_steps= n_steps, verbosity_level=v_level)
        #plt.subplot(2,2,3);
        #plot_y(out, y_vec, col_dict)
        plot_ty(out['tgrid'], out['y_data'], y_vec, col_dict)
        plt.title('formulation in c')
        plt.show()
        #plt.subplot(2,2,4); plot_flux(out, u_vec, col_dict); plt.title(situation_string)
        #except:
        #    pass
        #    plt.subplot(2,2,3); plt.title('No solution, '+ situation_string)
'''    


'''
# alpha vs. M
use_model_flag = 1
alpha_vals = np.linspace(0.0, 0.9, n_alpha:=25)
#alpha_vals = np.linspace(0, 1, n_alpha:=5)# DEBUG
M_vals = np.zeros((n_alpha, 3))
v_level = 0
val_for_M = 0.0
phi_fva = 0.1
ext0_val = 10.0# optmistic behavior
save_figures_to_files = True
#save_figures_to_files = False
for k, alpha in enumerate(alpha_vals):
    print(k)
    constants = mini.create_constants(use_model_flag, alpha)
    wvec = mini.create_weight_vector(y_vec, constants)
    # baseline solution RBA ------------------------------------------------------------------
    ext0 = lambda t: ext0_val
    m_dict = cFBA.create_m_dict(mini, ext0, constants)
    outRBA, mu0 = cFBA.RBA_like(m_dict, 0.0, del_t_RBA, verbosity_level=0, mumin=1.0, mumax=2.0, y_start=None, wvec=wvec)
    y0 = outRBA['y']
    y0[y_vec.index('M')] = val_for_M
    # new external conditions ----------------------------------------------------------------
    ext = lambda t: 1.0
    m_dict = cFBA.create_m_dict(mini, ext, constants)
    #rba_new, munew = cFBA.RBA_like(m_dict, 0.0, del_t_RBA, verbosity_level=0, mumin=1.0, mumax=2.0, y_start=None, wvec=wvec)
    #
    y1 = 2.0*y0
    # t-min-opt ------------------------------------------------------------------------------
    eye_n, zer_n = np.eye(n_y), np.zeros((n_y, n_y))
    m_dict['matrix_start'] = np.vstack([eye_n, zer_n])
    m_dict['matrix_end'] = np.vstack([zer_n, eye_n])
    m_dict['vec_bndry'] = np.vstack([y0, y1])
    n_steps = 101
    #n_steps = 11 # DEBUG
    tast, outtmin, _ = cFBA.t_min_opt(m_dict, est_min=0.05, est_max=20.0, n_steps=n_steps, verbosity_level=v_level)
    # deFVA ----------------------------------------------------------------------------------
    #y_min_all, y_max_all = cFBA.low_level_deFVA(outtmin, y_vec, verbosity_level=v_level, fva_level=3)# DEBUG
    #y_min_all, y_max_all = cFBA.low_level_deFVA(outtmin, y_vec, verbosity_level=v_level, varphi=phi_fva, fva_level=2,
    #                                            y_indices=[y_vec.index('M')],
    #                                            minmax=('min'))
    mean_M = np.mean(outtmin['y_data'][:, y_vec.index('M')])
    #mean_M_min = np.mean(y_min_all[:, y_vec.index('M')])
    #max_M_min = np.max(y_min_all[:, y_vec.index('M')])
    M_vals[k, 0] = mean_M
    #M_vals[k, 1] = mean_M_min
    #M_vals[k, 2] = max_M_min
    #plot_ty(outtmin['tgrid'], outtmin['y_data'], y_vec, col_dict)
    #plt.show()

#plt.show()
plt.plot(alpha_vals, M_vals[:, 0], marker='.')
plt.xlabel('alpha')
plt.ylabel('avrg(M)')
if save_figures_to_files:
    plt.savefig('./svg/alpha_vs_M.svg')
    plt.savefig('./png/alpha_vs_M.png')
'''


'''
# optimism vs. M
use_model_flag = 2
#ext_vals = np.linspace(0.2, 2.0, n_ext:=82)# adapted to these values
#ext_vals = np.linspace(0.48, 0.52, n_ext:=12)# DEBUG
ext_vals = np.linspace(0.2, 2.0, n_ext:=15)# DEBUG
#ext_vals = np.linspace(2.0, 20.0, n_ext:=15)# DEBUG
relevant_indices = range(n_ext)
M_vals = np.zeros((n_ext, 9))
tast_vals = np.zeros((n_ext, 1))
mu_vals = np.zeros((n_ext, 4))
v_level = 0
val_for_M = 0.0
phi_fva = 0.1
Tminmax=3.5
save_figures_to_files = False
save_data_to_files = False
constants = mini.create_constants(use_model_flag)
#constants['npT'] = 15.0; Tminmax=4.4 # CAREFUL: Here, we play around with the model in an insecure way...
wvec = mini.create_weight_vector(y_vec, constants)
constants['kcatT'] = 20/11*constants['kcatT']# DEBUG, CAREFUL
#
# given external conditions --------------------------------------------------------------------
ext = lambda t: 1.0
#ext = lambda t: 10.0# DEBUG
m_dict = cFBA.create_m_dict(mini, ext, constants)
rba_env, mu_env = cFBA.RBA_like(m_dict, 0.0, del_t_RBA, verbosity_level=0, mumin=1.0, mumax=2.0,
                                y_start=None, wvec=wvec)
#
for k, ext_0_val in enumerate(ext_vals):
    print(k)
    # baseline solution RBA --------------------------------------------------------------------
    ext0 = lambda t: ext_0_val
    m_dict0 = cFBA.create_m_dict(mini, ext0, constants)
    outRBA, mu0 = cFBA.RBA_like(m_dict0, 0.0, del_t_RBA, verbosity_level=0, mumin=1.0, mumax=2.0,
                                y_start=None, wvec=wvec)
    y0 = outRBA['y']
    print(y0, mu0)
    y0[y_vec.index('M')] = val_for_M
    # given external conditions ----------------------------------------------------------------
    m_dict = cFBA.create_m_dict(mini, ext, constants)
    #rba_new, munew = cFBA.RBA_like(m_dict, 0.0, del_t_RBA, verbosity_level=0, mumin=1.0, mumax=2.0, y_start=None, wvec=wvec)
    # iterative RBA ----------------------------------------------------------------------------
    _, mu_it = cFBA.RBA_like(m_dict, 0.0, del_t_RBA, verbosity_level=0, mumin=1.0, mumax=2.0,
                             y_start=y0, wvec=None)
    y1 = 2.0*y0
    # t-min-opt --------------------------------------------------------------------------------
    eye_n, zer_n = np.eye(n_y), np.zeros((n_y, n_y))
    m_dict['matrix_start'] = np.vstack([eye_n, zer_n])
    m_dict['matrix_end'] = np.vstack([zer_n, eye_n])
    m_dict['vec_bndry'] = np.vstack([y0, y1])
    n_steps = 100
    tast, outtmin, _ = cFBA.t_min_opt(m_dict, est_min=0.05, est_max=20.0, n_steps=n_steps,
                                      verbosity_level=v_level)
    # deFVA -----------------------------------------------------------------------------------
    #y_min_all, y_max_all = cFBA.low_level_deFVA(outtmin, y_vec, verbosity_level=v_level, fva_level=3)# DEBUG
    y_min_all, y_max_all = cFBA.low_level_deFVA(outtmin, y_vec, verbosity_level=v_level, varphi=phi_fva, fva_level=1,
                                                #y_indices=[y_vec.index('M')],
                                                #minmax=('min')
                                                )
    min_M =   np.min( outtmin['y_data'][:, y_vec.index('M')] )
    mean_M = np.mean( outtmin['y_data'][:, y_vec.index('M')] )
    max_M =   np.max( outtmin['y_data'][:, y_vec.index('M')] )
    min_M_min =   np.min( y_min_all[:, y_vec.index('M')] )
    mean_M_min = np.mean( y_min_all[:, y_vec.index('M')] )
    max_M_min =   np.max( y_min_all[:, y_vec.index('M')] )
    min_M_max =   np.min( y_max_all[:, y_vec.index('M')] )
    mean_M_max = np.mean( y_max_all[:, y_vec.index('M')] )
    max_M_max =   np.max( y_max_all[:, y_vec.index('M')] )
    M_vals[k, 0] = min_M
    M_vals[k, 1] = mean_M
    M_vals[k, 2] = max_M
    M_vals[k, 3] = min_M_min
    M_vals[k, 4] = mean_M_min
    M_vals[k, 5] = max_M_min
    M_vals[k, 6] = min_M_max
    M_vals[k, 7] = mean_M_max
    M_vals[k, 8] = max_M_max
    tast_vals[k, 0] = tast
    mu_vals[k, 0] = mu_env
    mu_vals[k, 1] = mu0
    mu_vals[k, 2] = mu_it
    mu_tmin = np.exp(np.log(2.0)*del_t_RBA/tast)
    mu_vals[k, 3] = mu_tmin
    if k in relevant_indices:
        fig = plt.figure()
        fig.set_size_inches(3*4, 3*3)
        plot_ty(outtmin['tgrid'], outtmin['y_data'], y_vec, col_dict)
        plot_ty_between(outtmin['tgrid'], y_min_all, y_max_all, y_vec, col_dict, wvec=None, line_style=':')
        plt.title('adapted to N = %1.5f' % ext_0_val)
        plt.ylim(top=0.25)
        plt.xlim(right=Tminmax)
        if save_figures_to_files:
            plt.savefig(f'./png/optpess_M_{k}.png')
            plt.savefig(f'./svg/optpess_M_{k}.svg')
        plt.show()
    #plt.show()

#plt.show()
fig = plt.figure()
fig.set_size_inches(3*4, 3*3)
#plt.plot(ext_vals, M_vals[:, 0], marker='.')
plt.subplot(1, 3, 1)
plt.plot(ext_vals, M_vals[:, 0:3], marker='.')
plt.xlabel('ext'); plt.ylabel('M'); plt.legend(['min(M)', 'mean(M)', 'max(M)'])
plt.title('t-min-opt')
plt.subplot(1, 3, 2)
plt.plot(ext_vals, M_vals[:, 3:6], marker='.')
plt.xlabel('ext'); plt.ylabel('M'); plt.legend(['min(M)', 'mean(M)', 'max(M)'])
plt.title('min (deFVA)')
plt.subplot(1, 3, 3)
plt.plot(ext_vals, M_vals[:, 6:9], marker='.')
plt.xlabel('ext'); plt.ylabel('M'); plt.legend(['min(M)', 'mean(M)', 'max(M)'])
plt.title('max (deFVA)')
if save_figures_to_files:
    plt.savefig('./svg/ext_vs_M.svg')
    plt.savefig('./png/ext_vs_M.png')
#
fig = plt.figure()
fig.set_size_inches(3*4, 3*3)
plt.plot(ext_vals, tast_vals, marker='.')
plt.xlabel('ext')
plt.ylabel('T')
if save_figures_to_files:
    plt.savefig('./svg/ext_vs_tmin.svg')
    plt.savefig('./png/ext_vs_tmin.png')
#plt.show()
fig = plt.figure()
fig.set_size_inches(3*4, 3*3)
plt.plot(ext_vals, mu_vals, marker='.')
plt.xlabel('ext')
plt.ylabel('mu (delta)')
plt.legend(['RBA(env.)', 'RBA(adapt.)', 'it. RBA', 't-min-opt'])
if save_figures_to_files:
    plt.savefig('./svg/ext_vs_mu.svg')
    plt.savefig('./png/ext_vs_mu.png')
'''


'''
# remove requirements and check the tmin values ---------------------
use_model_flag = 2
ext_vals = np.linspace(0.2, 2.0, n_ext:=25)# adapted to these values
t_opt_vals = []
v_level = 0
val_for_M = 0
save_figures_to_files = True
constants = mini.create_constants(use_model_flag)
wvec = mini.create_weight_vector(y_vec, constants)
# adapted value
ext0 = lambda t: 10.0
m_dict = cFBA.create_m_dict(mini, ext0, constants)
outRBA, _ = cFBA.RBA_like(m_dict, 0.0, del_t_RBA, verbosity_level=0, mumin=1.0, mumax=2.0, y_start=None, wvec=wvec)
y0 = outRBA['y']
y0[y_vec.index('M')] = val_for_M
y1 = 2.0*y0
# new external conditions ----------------------------------------------------------------
ext = lambda t: 1.0
m_dict = cFBA.create_m_dict(mini, ext, constants)
eye_n, zer_n = np.eye(n_y), np.zeros((n_y, n_y))
m_dict['matrix_start'] = np.vstack([eye_n, zer_n])
m_dict['matrix_end'] = np.vstack([zer_n, eye_n])
m_dict['vec_bndry'] = np.vstack([y0, y1])
n_steps = 101
#n_steps = 11 # DEBUG
#
all_indices = [1, 2, 3, 4, 5, 6]
def _check_index_lists(tup1, tup2):
    # remove if all on one side
    if not tup1 or not tup2:
        return False
    # remove if none on both sides
    if sorted(tup1+tup2) in [[1, 2, 3], [1,2], [1,3], [2,3]]:
        return False
    else:
        return True

for sth in chain(combinations(all_indices, 6),
                 combinations(all_indices, 5),
                 combinations(all_indices, 4),
                 combinations(all_indices, 3),
                 combinations(all_indices, 2)):
    ind_start = [i for i in sth if i <4]
    ind_end = [i-3 for i in sth if i > 3]
    if _check_index_lists(ind_start, ind_end):
        #
        #tast, _, _ = cFBA.t_min_opt(m_dict, est_min=0.05, est_max=20.0, n_steps=n_steps, verbosity_level=v_level)
        try:
            tast, _, _ = cFBA.opt_jump_from_to(mini, y0, y1, ext, constants, J0=ind_start, Jend=ind_end,
                                               wvec=None, t0=0.0, est_min=1e-3, est_max=1e3, t_estimates=None,
                                               n_steps=n_steps, verbosity_level=0)
        except:
            tast = 0.0
    #
        t_opt_vals.append([ind_start, ind_end, tast])
        print(ind_start, ind_end, tast)
#print(t_opt_vals)
t_opt_vals.sort(key = lambda x: 3000*x[2]+len(x[0]+x[1]))
print(45*'*')
for l in t_opt_vals:
    print(l)
'''



'''
# PARETO optimality: t-min vs mu-max for a cell multiplication experiment
# Idea: Fix a t-value and max mu via cFBA and the other way around
eye_n, zer_n = np.eye(n_y), np.zeros((n_y, n_y))
use_model_flag = 2
n_givenvals = 15
v_level = 0
val_for_M = 0.0
n_steps_tmin = 100
n_steps_cfba = 100
#phi_fva = 0.1
save_figures_to_files = False
constants = mini.create_constants(use_model_flag)
wvec = mini.create_weight_vector(y_vec, constants)
#
# adapted to these conditions --------------------------------------------------------------
ext_adapt = lambda t: 1.0
m_dict_adapt = cFBA.create_m_dict(mini, ext_adapt, constants)
rba_adapt, mu_adapt = cFBA.RBA_like(m_dict_adapt, 0.0, del_t_RBA, verbosity_level=0,
                                    mumin=1.0, mumax=2.0, y_start=None, wvec=wvec)
y0 = rba_adapt['y']
y0[y_vec.index('M')] = val_for_M
# given external conditions --------------------------------------------------------------------
ext_vals = np.array([0.5, 1.0, 2.0])
col_string_vals = ['green','blue','black']
#
all_mu_vals = []
all_tast_vals = []
all_title_strings = []
#
for n, e in enumerate(ext_vals):
    print(n)
    ext = lambda t: e
    m_dict = cFBA.create_m_dict(mini, ext, constants)
    m_dict['matrix_start'] = np.vstack([eye_n, zer_n])
    m_dict['matrix_end'] = np.vstack([zer_n, eye_n])
    #
    tast_vals = np.zeros((n_givenvals, 2))
    mu_vals = np.zeros((n_givenvals, 2))
    mu_vals[:, 1] = np.linspace(1.1, 6.0, n_givenvals)
    mu_vals[:, 1] = np.linspace(1.05, 1.65, n_givenvals)
    for k, mu in enumerate(mu_vals[:, 1]):
        print('\t', k)
        # fix the end configuration
        y1 = mu*y0
        m_dict['vec_bndry'] = np.vstack([y0, y1])
        # run tminopt
        tast, outtmin, _ = cFBA.t_min_opt(m_dict, est_min=0.05, est_max=20.0, n_steps=n_steps_tmin,
                                          verbosity_level=v_level)
        tast_vals[k, 1] = tast
    tast_vals[:, 0] = np.linspace(np.min(tast_vals[:, 1]), np.max(tast_vals[:, 1]), n_givenvals)
    #
    for k, T in enumerate(tast_vals[:, 0]):
        print('\t', k)
        # fix the time horizon
        tspan = np.linspace(0.0, T, n_steps_cfba)
        _, muend = cFBA.cFBA(m_dict, tspan, verbosity_level=v_level,
                             mumin=max( mu_vals[k, 1]-1.0, 1.0),
                             mumax=mu_vals[k, 1]+1.0, y_start=y0, wvec=None)
        mu_vals[k, 0] = muend
    title_string = f'adapt. {ext_adapt(0)}, env. {ext(0.0)}'
    all_mu_vals.append(mu_vals)
    all_tast_vals.append(tast_vals)
    all_title_strings.append(title_string)
fig = plt.figure()
fig.set_size_inches(3*4, 3*3)
norm_plot = plt.subplot(1, 2, 1)
plt.xlabel('T')
plt.ylabel('mu')
log_plot = plt.subplot(1, 2, 2)
plt.xlabel('T')
plt.ylabel('mu')
for n, col_string in enumerate(col_string_vals):
    tast_vals = all_tast_vals[n]
    mu_vals = all_mu_vals[n]
    plt.subplot(norm_plot)    
    plt.plot(tast_vals[:, 1], mu_vals[:, 1], marker='.', color=col_string)
    plt.plot(tast_vals[:, 0], mu_vals[:, 0], marker='*', color=col_string)
    norm_plot.invert_yaxis()
    #
    plt.subplot(log_plot)
    plt.semilogy(tast_vals[:, 1], mu_vals[:, 1], marker='.', color=col_string)
    plt.semilogy(tast_vals[:, 0], mu_vals[:, 0], marker='*', color=col_string)
    log_plot.invert_yaxis()
#plt.title(title_string)
'''

'''
# PARETO optimality: t-min vs mu-max for an adaptation experiment
# Idea: Fix a t-value and max mu via cFBA and the other way around
#
# TODO: Additional straight lines, some time series plots to see the "exponential phase"
#
eye_n, zer_n = np.eye(n_y), np.zeros((n_y, n_y))
use_model_flag = 2
n_givenvals = 15
v_level = 0
val_for_M = 0.0
n_steps_tmin = 100
n_steps_cfba = 100
#phi_fva = 0.1
#save_figures_to_files = False
constants = mini.create_constants(use_model_flag)
wvec = mini.create_weight_vector(y_vec, constants)
#
# adapted at start to these conditions -----------------------------------------------------------
ext_adapt = lambda t: 1.0
m_dict_adapt = cFBA.create_m_dict(mini, ext_adapt, constants)
rba_adapt, mu_adapt = cFBA.RBA_like(m_dict_adapt, 0.0, del_t_RBA, verbosity_level=0,
                                    mumin=1.0, mumax=2.0, y_start=None, wvec=wvec)
y0 = rba_adapt['y']
y0[y_vec.index('M')] = val_for_M
# given external conditions --------------------------------------------------------------------
ext_vals = np.array([0.5, 1.0, 2.0])
col_string_vals = ['green', 'blue', 'black']
#
all_mu_vals = []
all_tast_vals = []
all_title_strings = []
#
for n, e in enumerate(ext_vals):
    print(n)
    ext = lambda t: e
    m_dict = cFBA.create_m_dict(mini, ext, constants)
    rba_sol, mu_sol = cFBA.RBA_like(m_dict, 0.0, del_t_RBA, verbosity_level=0,
                                    mumin=1.0, mumax=2.0, y_start=None, wvec=wvec)
    y1 = rba_sol['y']
    y1[y_vec.index('M')] = val_for_M
    #
    tast_vals = np.zeros((n_givenvals, 2))
    mu_vals = np.zeros((n_givenvals, 2))
    #mu_vals[:, 1] = np.linspace(1.1, 6.0, n_givenvals)
    mu_vals[:, 1] = np.linspace(1.05, 1.65, n_givenvals)
    for k, mu in enumerate(mu_vals[:, 1]):
        print('\t', k)
        # fix the end configuration
        y1 = mu*y0
        # run tminopt
        tast, out, _ = cFBA.opt_jump_from_to(mini, y0, y1, ext, constants,#J0=None, Jend=None, wvec=None, t0=0.0, est_min=1e-3, est_max=1e3, t_estimates=None, n_steps=50, verbosity_level=0)
                                        n_steps=n_steps_tmin, verbosity_level=v_level)
        tast_vals[k, 1] = tast
    tast_vals[:, 0] = np.linspace(np.min(tast_vals[:, 1]), np.max(tast_vals[:, 1]), n_givenvals)
    #
    m_dict['matrix_start'] = np.vstack([eye_n, zer_n])
    m_dict['matrix_end'] = np.vstack([zer_n, eye_n])
    m_dict['vec_bndry'] = np.vstack([y0, y1])
    for k, T in enumerate(tast_vals[:, 0]):
        print('\t', k)
        # fix the time horizon
        tspan = np.linspace(0.0, T, n_steps_cfba+1)
        _, muend = cFBA.cFBA(m_dict, tspan, verbosity_level=v_level,
                             mumin=max( mu_vals[k, 1]-1.0, 1.0),
                             mumax=mu_vals[k, 1]+1.0, y_start=y0, wvec=None)
        mu_vals[k, 0] = muend
    title_string = f'adapt. {ext_adapt(0)}, env. {ext(0.0)}'
    all_mu_vals.append(mu_vals)
    all_tast_vals.append(tast_vals)
    all_title_strings.append(title_string)
fig = plt.figure()
fig.set_size_inches(3*4, 3*3)
norm_plot = plt.subplot(1, 2, 1)
plt.xlabel('T')
plt.ylabel('mu')
log_plot = plt.subplot(1, 2, 2)
plt.xlabel('T')
plt.ylabel('mu')
for n, col_string in enumerate(col_string_vals):
    tast_vals = all_tast_vals[n]
    mu_vals = all_mu_vals[n]
    plt.subplot(norm_plot)    
    plt.plot(tast_vals[:, 1], mu_vals[:, 1], marker='.', color=col_string)
    plt.plot(tast_vals[:, 0], mu_vals[:, 0], marker='*', color=col_string)
    norm_plot.invert_yaxis()
    #
    plt.subplot(log_plot)
    plt.semilogy(tast_vals[:, 1], mu_vals[:, 1], marker='.', color=col_string)
    plt.semilogy(tast_vals[:, 0], mu_vals[:, 0], marker='*', color=col_string)
    log_plot.invert_yaxis()
#plt.title(title_string)
'''




# OLD/LITTLE STUFF
'''
# deFVA --------------------------------------------------
use_model_flag = 3
constants = mini.create_constants(use_model_flag)
wvec = mini.create_weight_vector(y_vec, constants)
tspan = np.linspace(0.0, 3.0, 73) 
ext = lambda t: 1.0
y0 = np.array([[0.3], [0.05], [0.01]])
m_dict = cFBA.create_m_dict(mini, y0, ext, constants) #, mu=None, flag_objective=0)
m_dict['phi1'] = -wvec #np.array([[0.0],[-1.0],[-5.0]])
m_dict['phi3'] *= 0.0
out = cFBA.deFBA(m_dict, tspan, varphi=0.1)
#print(m_dict['phi1'])
ty_axis = plt.subplot(1,2,1); plot_y(out, y_vec, col_dict)
plt.subplot(1,2,2); plot_flux(out, u_vec, col_dict)
#
y_min_all, y_max_all = cFBA.low_level_deFVA(out, y_vec, verbosity_level=1, fva_level=2, varphi=3.0)
#print(y_new)
#plt.figure()
plt.subplot(ty_axis)
#plt.figure()
plot_ty_between(out['tgrid'], y_min_all, y_max_all, y_vec, col_dict, wvec=None, line_style=':')
#plot_ty(out['tgrid'], y_min_all, y_vec, col_dict, line_style=':')
#plot_ty(out['tgrid'], y_max_all, y_vec, col_dict, line_style=':')
plt.show()
'''


'''
# RBA/cFBA ----------------------------------
use_model_flag = 2
constants = mini.create_constants(use_model_flag)
wvec = mini.create_weight_vector(y_vec, constants)
tspan = np.linspace(0.0, 2.0, 73) 
ext = lambda t: 1.0
m_dict = cFBA.create_m_dict(mini, ext, constants)
y0 = np.array([[0.1], [0.5], [0.05]])
#out, mu = cFBA.cFBA(m_dict, tspan, verbosity_level=0, mumin=1.0, mumax=5.0, y_start=None, wvec=wvec)
#out, mu = cFBA.cFBA(m_dict, tspan, verbosity_level=0, mumin=1.0, mumax=5.0, y_start=None, wvec=None)
#out, mu = cFBA.cFBA(m_dict, tspan, verbosity_level=2, mumin=1.0, mumax=5.0, y_start=y0, wvec=None)
#plot_ty(out['tgrid'], out['y_data'], y_vec, col_dict)
#plt.show()
t0, del_t = 0.0, 0.01
rba_sol, mu = cFBA.RBA_like(m_dict, t0, del_t, verbosity_level=0, mumin=1.0, mumax=2.0, y_start=None, wvec=None)
print(rba_sol, mu)
rba_sol, mu = cFBA.RBA_like(m_dict, t0, del_t, verbosity_level=0, mumin=1.0, mumax=2.0, y_start=y0, wvec=None)
print(rba_sol, mu)
rba_sol, mu = cFBA.RBA_like(m_dict, t0, del_t, verbosity_level=0, mumin=1.0, mumax=2.0, y_start=None, wvec=wvec)
print(rba_sol, mu)
'''


'''
# TEST: Is it the same if we change kcatT and/or nutrient supply?
# (a)
use_model_flag = 2
constantsa = mini.create_constants(use_model_flag)
wveca = mini.create_weight_vector(y_vec, constantsa)
comp_t_min_opt = True
# nutrient values old
exta_val = 2.0 # adapted to this
ext_conda_val = 10.0 
# nutrient values new
extb_val = None
ext_condb_val = 1.0
#
exta = lambda t: exta_val
m_dicta = cFBA.create_m_dict(mini, exta, constantsa)
outRBAa, _ = cFBA.RBA_like(m_dicta, 0.0, del_t_RBA, verbosity_level=0, mumin=1.0, mumax=2.0, y_start=None, wvec=wveca)
y0a = outRBAa['y']
#y0[y_vec.index('M')] = val_for_M
print(y0a)
yenda = 2.0*y0a
ext_conda = lambda t: ext_conda_val
print(MichMen(ext_conda(0.0))*constantsa['kcatT'])
if comp_t_min_opt:
    tasta, outa, _ = cFBA.opt_jump_from_to(mini, y0a, yenda, ext_conda, constantsa, # J0=None, Jend=None, wvec=None, t0=0.0,
                       #est_min=1e-3, est_max=1e3, t_estimates=None,
                       n_steps=100, verbosity_level=0)
    plt.plot(outa['tgrid'], outa['y_data'])

# (b) change kcatT such that exta/(1+exta)*kcatT_old == extb/(1+extb)*kcatT_new
kcatT_old = constantsa['kcatT']
kcatT_new = MichMen(ext_conda_val)*kcatT_old/MichMen(ext_condb_val)
print(kcatT_old, kcatT_new)
#
extb_val = InvMichMen( kcatT_old/kcatT_new*MichMen(exta_val) )
print(extb_val) 
constantsb = mini.create_constants(use_model_flag)
constantsb['kcatT'] = kcatT_new
wvecb = mini.create_weight_vector(y_vec, constantsb)
extb = lambda t: extb_val
m_dictb = cFBA.create_m_dict(mini, extb, constantsb)
outRBAb, _ = cFBA.RBA_like(m_dictb, 0.0, del_t_RBA, verbosity_level=0, mumin=1.0, mumax=2.0, y_start=None, wvec=wvecb)
y0b = outRBAb['y']
print(y0b)
yendb = 2.0*y0b
ext_condb = lambda t: ext_condb_val
print(MichMen(ext_condb(0.0))*constantsb['kcatT'])

if comp_t_min_opt:
    tastb, outb, _ = cFBA.opt_jump_from_to(mini, y0b, yendb, ext_condb, constantsb, # J0=None, Jend=None, wvec=None, t0=0.0,
                       #est_min=1e-3, est_max=1e3, t_estimates=None,
                       n_steps=100, verbosity_level=0)
    plt.plot(outb['tgrid'], outb['y_data'])

print(m_dicta['matrix_y'](0.0).todense())
print(m_dictb['matrix_y'](0.0).todense())
'''








'''
# actual data creation for the paper:
# (A) :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# the example network: growth curves, "RBA curves"
constants = mini.create_constants(use_model_flag:= 2)
wvec = mini.create_weight_vector(y_vec, constants)
extvals = np.linspace(0.3, 15.0, n_ext:=35)
plot_in_py = True
muvals = []
growthvals = []
allRBA_yvals = []
for i, ext_val in enumerate(extvals):
    ext_cond = lambda t: ext_val
    m_dictRBA = cFBA.create_m_dict(mini, ext_cond, constants)
    outRBA, muRBA = cFBA.RBA_like(m_dictRBA, 0.0, del_t_RBA, verbosity_level=0, mumin=1.0, mumax=2.0, y_start=None, wvec=wvec)
    muvals.append(muRBA)
    growthvals.append(np.log(muRBA)/del_t_RBA)
    allRBA_yvals.append(outRBA['y'].T.tolist()[0])
if plot_in_py:
    plt.plot(extvals, muvals)
    plt.show()
    plt.plot(extvals, np.array(allRBA_yvals))
filename_to_save = 'simple_model_RBA.dat'
data_to_save = np.hstack([np.array([extvals]).T, np.array([muvals]).T, np.array([growthvals]).T , np.array(allRBA_yvals)])
header_to_save = 'n\tmu\tgrowth\tM\tTr\tR'
#np.savetxt(filename_to_save, data_to_save, delimiter='\t', newline='\n', header=header_to_save, comments='')
'''

'''
# (Aii) ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# parameters for the growth curve
constants = mini.create_constants(use_model_flag:= 2)
wvec = mini.create_weight_vector(y_vec, constants)
#del_t_RBA /= 100.0
extvals = np.linspace(0.0, 150.0, n_ext:=2)
def calc_growth(nut_avail):
    ext_cond = lambda t: nut_avail
    m_dictRBA = cFBA.create_m_dict(mini, ext_cond, constants)
    outRBA, muRBA = cFBA.RBA_like(m_dictRBA, 0.0, del_t_RBA, verbosity_level=0, mumin=1.0, mumax=2.0, y_start=None, wvec=wvec)
    print(ext_cond(0), muRBA)
    print(np.log(muRBA)/del_t_RBA)
    return np.log(muRBA)/del_t_RBA

for i, ext_val in enumerate(extvals):
    growth_factor = calc_growth(ext_val)
    #print(growth_factor)
half_growth = 0.5*growth_factor
bisect(lambda n: calc_growth(n)-half_growth, 0.1, 4.0)
'''


'''
# (B0) ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# cell doubling for a not perfectly adapted cell
n_steps = 200
constants = mini.create_constants(use_model_flag:= 2)
#constants['npT'] = 10.0
wvec = mini.create_weight_vector(y_vec, constants)
plot_in_py = True
ext = lambda t: 1.0
ext_adapt = lambda t: 2.0
#
m_dictRBA = cFBA.create_m_dict(mini, ext_adapt, constants)
outRBA, muRBA = cFBA.RBA_like(m_dictRBA, 0.0, del_t_RBA, verbosity_level=0, mumin=1.0, mumax=2.0, y_start=None, wvec=wvec)
yinit = np.fmax(outRBA['y'], 0.0*outRBA['y'])# make sure, none are negative
ygoal = 2.0*yinit
tast, out, _ = cFBA.opt_jump_from_to(mini, yinit, ygoal, ext, constants, # J0=None, Jend=None, wvec=None, t0=0.0,
                               #est_min=1e-3, est_max=1e3, t_estimates=None,
                               n_steps=n_steps, verbosity_level=0)
#
out_biomass = biomass(out['y_data'], wvec)
#constants['npT']*out['y_data'][:, y_vec.index('T')]  + \
#          constants['npT']*out['y_data'][:, y_vec.index('T')]
      
# deFVA (not used here)
#min_y, max_y = cFBA.low_level_deFVA(out, y_vec, verbosity_level=0, varphi=0.0, fva_level=1)
# it. RBA/it. FBA
m_dict = cFBA.create_m_dict(mini, ext, constants)
out_it_RBA, mu_it_RBA = cFBA.RBA_like(m_dict, 0.0, del_t_RBA, verbosity_level=0, mumin=1.0, mumax=2.0, y_start=yinit, wvec=None)
_, mu_RBA_opt = cFBA.RBA_like(m_dict, 0.0, del_t_RBA, verbosity_level=0, mumin=1.0, mumax=2.0, y_start=None, wvec=wvec)
lambda_it = np.log(mu_it_RBA)/del_t_RBA
tgrid_itRBA = np.linspace(0.0, np.log(2)/lambda_it, n_steps+1)
print(np.log(2)/np.log(mu_RBA_opt)*del_t_RBA, tast, np.log(2)/lambda_it)
it_RBA_y = np.outer(yinit, np.exp(lambda_it*tgrid_itRBA)).T
uinit = np.fmax(out_it_RBA['u'], 0.0*out_it_RBA['u'])# make sure, none are negative
it_RBA_u = np.outer(uinit, np.exp(lambda_it*tgrid_itRBA)).T
out_biomass_RBA = biomass(it_RBA_y, wvec)
# Find the boundaries of the different areas in the time course
tmp = np.diff(out['y_data'],axis=0,n=2)
for k, t in enumerate( out['tgrid'][:-2] ):
    if np.abs(tmp[k, 2]) > 0.0001:
        print(out['tgrid'][k+1])
#
#outmixed, mixed_names = cFBA.constraint_fulfillment(out, m_dict, verbosity_level=0, plot_kind = 'mixed')
#outpos, pos_names = cFBA.constraint_fulfillment(out, m_dict, verbosity_level=0, plot_kind = 'positivity')
outfl, fl_names = cFBA.constraint_fulfillment(out, m_dict, verbosity_level=0, plot_kind = 'flux_bounds')
if plot_in_py:
    ax1 = plt.subplot(2,1,1)
    ax1.plot(out['tgrid'], out['y_data'])
    ax2 = ax1.twinx()
    ax2.semilogy(out['tgrid'], out_biomass)
    ax2.semilogy(tgrid_itRBA, out_biomass_RBA)
    ax1.plot(tgrid_itRBA, it_RBA_y)
    plt.subplot(2,1,2)
    #plt.plot(out['tgrid'], outmixed)
    #plt.plot(out['tgrid'], outpos)
    #plt.plot(out['tgrid'], outfl[:,5:])
    #plt.legend(fl_names[5:])
    #plt.ylim(-0.01, 0.1)
    plt.plot(out['tgrid_u'], out['u_data'])
    plt.legend(u_vec)
    plt.plot(tgrid_itRBA, it_RBA_u)
filename_y = 'cell_double_single.dat'
header_to_save_y = '\t'.join(['t', 'tRBA', 'M', 'Tr', 'R', 'MRBA', 'TrRBA', 'RRBA', 'biom', 'biomRBA'])
data_to_save_y = np.hstack([np.array([out['tgrid']]).T,
                          np.array([tgrid_itRBA]).T,
                          out['y_data'],
                          it_RBA_y,
                          np.array([out_biomass]).T,
                          np.array([out_biomass_RBA]).T]
                          )
filename_u = 'cell_double_single_flux.dat'
header_to_save_u = '\t'.join(['t', 'tRBA','vN', 'vT', 'vR', 'vdT', 'vdR', 'vNRBA', 'vTRBA',
                              'vRRBA', 'vdTRBA', 'vdRRBA'])
#data_to_save_u = np.hstack([np.array([np.hstack([0.0, out['tgrid_u']])]).T,
#                            np.array([tgrid_itRBA]).T,
#                            np.vstack([out['u_data'][0,:], out['u_data']]),
#                            it_RBA_u])

tgrid_tminoptu = out['tgrid']
tmin_opt_out_u = np.array([np.interp(tgrid_tminoptu, out['tgrid_u'], out['u_data'][:,i]) for i in range(n_u)]).T
data_to_save_u = np.hstack([np.array([tgrid_tminoptu]).T,
                            np.array([tgrid_itRBA]).T,
                            tmin_opt_out_u,
                            it_RBA_u])
#
np.savetxt(filename_y, data_to_save_y, delimiter='\t', newline='\n', header=header_to_save_y, comments='')
np.savetxt(filename_u, data_to_save_u, delimiter='\t', newline='\n', header=header_to_save_u, comments='')
'''



'''
# (B) :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# time course for different nutrient supply, cell doubling
constants = mini.create_constants(use_model_flag:= 2)
wvec = mini.create_weight_vector(y_vec, constants)
plot_in_py = True
save_results_to_file = False
#                 t1  t2  t3  M1  Tr1  R1  minM1  maxM1  minTr1  maxTr1  minR1  maxR1
#headertmin_to_save = 't1\tt2\tt3\tM1\tTr1\tR1\tminM1\tmaxM1\tminTr1\tmaxTr1\tminR1\tmaxR1\t' \
#                +            'M2\tTr2\tR2\tminM2\tmaxM2\tminTr2\tmaxTr2\tminR2\tmaxR2\t' \
#                +            'M3\tTr3\tR3\tminM3\tmaxM3\tminTr3\tmaxTr3\tminR3\tmaxR3'
header_to_save = 't\tM\tTr\tR\tMmin\tTrmin\tRmin\tMmax\tTrmax\tRmax\tMRBA\tTrRBA\tRRBA' # tmin, deFVA, itRBA
#
all_extvals = []
all_adapt_ext_vals = []
# (1, low nutrient supply)
all_extvals.append(0.5)
all_adapt_ext_vals.append([0.25, 0.5, 0.75])
#all_adapt_ext_vals = [[0.5]]# DEBUG
# (2, medium nutrient supply)
all_extvals.append(1.0)
#all_adapt_ext_vals.append([5.0])# DEBUG
all_adapt_ext_vals.append([0.5, 1.0, 5.0])
# (3, higher nutrient supply)
all_extvals.append(5.0)
all_adapt_ext_vals.append([1.0, 5.0, 10.0])
# (a) RBA
# DEBUG
#all_extvals.append(5.0), all_adapt_ext_vals.append([1.0])
for k, extval in enumerate(all_extvals):
    ext_cond = lambda t: extval
    m_dictRBA = cFBA.create_m_dict(mini, ext_cond, constants)
    outRBA, muRBA = cFBA.RBA_like(m_dictRBA, 0.0, del_t_RBA, verbosity_level=0, mumin=1.0, mumax=2.0, y_start=None, wvec=wvec)
    # (b) t-min-opt, optimally adapted
    adapt_ext_vals = all_adapt_ext_vals[k]
    for i, adapt_ext_val in enumerate(adapt_ext_vals):
        print(i)
        filename_to_save = f'cell_doubling_{k}_{i}.dat'
        ext_cond_adapt = lambda t: adapt_ext_val
        m_dict_adapt = cFBA.create_m_dict(mini, ext_cond_adapt, constants)
        out_adapt, mu_adapt = cFBA.RBA_like(m_dict_adapt, 0.0, del_t_RBA, verbosity_level=0, mumin=1.0, mumax=2.0, y_start=None, wvec=wvec)    
        yinit = np.fmax(out_adapt['y'], 0.0*out_adapt['y'])# make sure, none are negative
        ygoal = 2.0*yinit
        tast, out, _ = cFBA.opt_jump_from_to(mini, yinit, ygoal, ext_cond, constants, # J0=None, Jend=None, wvec=None, t0=0.0,
                               #est_min=1e-3, est_max=1e3, t_estimates=None,
                               n_steps=100, verbosity_level=0)
        # deFVA
        min_y, max_y = cFBA.low_level_deFVA(out, y_vec, verbosity_level=10, varphi=0.0, fva_level=1)
        #
        if plot_in_py:
            plt.plot(out['tgrid'], out['y_data'])
            plt.plot(out['tgrid'], min_y)
            plt.plot(out['tgrid'], max_y)
            #plt.plot(out['tgrid'], (max_y-min_y)[:,1:])
        # (c) "iterative" RBA
        out_it_RBA, mu_it_RBA = cFBA.RBA_like(m_dictRBA, 0.0, del_t_RBA, verbosity_level=0, mumin=1.0, mumax=2.0, y_start=yinit, wvec=None)
        #print(out_it_RBA['y'] - yinit)
        lambda_it_RBA = np.log(mu_it_RBA)/del_t_RBA
        it_RBA_y = np.outer(yinit, np.exp(lambda_it_RBA*out['tgrid'])).T
        if plot_in_py:
            #pass
            plt.plot(out['tgrid'], it_RBA_y)
        data_to_save = np.hstack([np.array([out['tgrid']]).T, out['y_data'], min_y, max_y, it_RBA_y])
        #print(data_to_save)
        if save_results_to_file:
            np.savetxt(filename_to_save, data_to_save, delimiter='\t', newline='\n', header=header_to_save, comments='')
    #
    if plot_in_py:
        plt.title(f"N = {extval}")
        plt.show()
'''

'''
# (C) :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# cell doubling: the role of optimism/pessimism for storage
# the cell lives in a surrounding of ext_cond but is adapted to ext_cond_adapt_val
constants = mini.create_constants(use_model_flag:= 2)
wvec = mini.create_weight_vector(y_vec, constants)
plot_in_py = True
ext_cond_adapt_vals = np.linspace(0.2, 2.0, n_ext:=45)
#ext_cond_adapt_vals = np.linspace(0.2, 2.0, n_ext:=85)# DEBUG for lambda values
#ext_cond_adapt_vals = np.linspace(0.2, 2.0, n_ext:=5)# DEBUG
#
export_indices = [0, n_ext//2, n_ext-1]
#
M_vals = np.zeros((n_ext, 9))
tast_vals = np.zeros((n_ext, 1))
mu_vals = np.zeros((n_ext, 4))
lambda_vals = np.zeros((n_ext, 4))
ext_cond = lambda t: 1.0
# baseline RBA for the given conditions
m_dictRBA_env = cFBA.create_m_dict(mini, ext_cond, constants)
_, muRBA_env = cFBA.RBA_like(m_dictRBA_env, 0.0, del_t_RBA, verbosity_level=0,
                             mumin=1.0, mumax=2.0, y_start=None, wvec=wvec)
#
for k, ext_cond_adapt_val in enumerate(ext_cond_adapt_vals):
    print(f'k = {k+1} of {n_ext}')
    ext_cond_adapt = lambda t: ext_cond_adapt_val
    # (i) compute RBA solution for adaptation value (utopia since the actual surrounding migt not
    #     be good enough)
    m_dictRBA_adapt = cFBA.create_m_dict(mini, ext_cond_adapt, constants)
    outRBA_adapt, muRBA_adapt = cFBA.RBA_like(m_dictRBA_adapt, 0.0, del_t_RBA, verbosity_level=0,
                                              mumin=1.0, mumax=2.0, y_start=None, wvec=wvec)
    # (ii) compute tminopt
    yinit = np.fmax(outRBA_adapt['y'], 0.0*outRBA_adapt['y'])# make sure, none are negative
    ygoal = 2.0*yinit
    tast, out, _ = cFBA.opt_jump_from_to(mini, yinit, ygoal, ext_cond, constants, # J0=None, Jend=None, wvec=None, t0=0.0,
                               #est_min=1e-3, est_max=1e3, t_estimates=None,
                               n_steps=100, verbosity_level=0)
    # (iii) compute deFVA
    #y_min, y_max = cFBA.low_level_deFVA(out, y_vec, verbosity_level=0, varphi=0.0, fva_level=1)
    #
    #deFVAmin, deFVAmax, info_deFVA
    y_min, y_max, egal = cFBA.relative_tmin_deFVA(mini, yinit, ygoal, ext_cond, tast,
                                                          constants, wvec, verbosity_level=0,
                                                          t_extend_trials=(1.0e-7, 1.0e-5))
    #plt.plot(out['tgrid'], deFVAmin, marker='*')   
    #plt.plot(out['tgrid'], deFVAmax, marker='*')
    #y_min_rel, y_max_rel = cFBA.relative_tmin_deFVA(mini, yinit, ygoal, ext_cond, tast, constants,
    #                                                wvec, n_steps=100, verbosity_level=1)
    # (iv) compute iterative RBA
    _, muRBA_it = cFBA.RBA_like(m_dictRBA_env, 0.0, del_t_RBA, verbosity_level=0,
                                mumin=1.0, mumax=2.0, y_start=yinit, wvec=None)
    # (v) compute averages (plain so far)
    #
    if k in export_indices:
        filename = f'opt_pess_adapt_{k}.dat'
        #                     t  M  Tr  R  Mmin  Trmin  Rmin  Mmax  Trmax Rmax
        headertmin_to_save = 't\tM\tTr\tR\tMmin\tTrmin\tRmin\tMmax\tTrmax\tRmax'
        data_to_save = np.hstack([np.array([out['tgrid']]).T, out['y_data'], y_min, y_max])
        np.savetxt(filename, data_to_save, delimiter='\t', newline='\n', header=headertmin_to_save, comments='')
    #
    min_M =   np.min( out['y_data'][:, y_vec.index('M')] )
    mean_M = np.mean( out['y_data'][:, y_vec.index('M')] )
    max_M =   np.max( out['y_data'][:, y_vec.index('M')] )
    min_M_min =   np.min( y_min[:, y_vec.index('M')] )
    mean_M_min = np.mean( y_min[:, y_vec.index('M')] )
    max_M_min =   np.max( y_min[:, y_vec.index('M')] )
    min_M_max =   np.min( y_max[:, y_vec.index('M')] )
    mean_M_max = np.mean( y_max[:, y_vec.index('M')] )
    max_M_max =   np.max( y_max[:, y_vec.index('M')] )
    #
    M_vals[k, 0] = min_M
    M_vals[k, 1] = mean_M
    M_vals[k, 2] = max_M
    M_vals[k, 3] = min_M_min
    M_vals[k, 4] = mean_M_min
    M_vals[k, 5] = max_M_min
    M_vals[k, 6] = min_M_max
    M_vals[k, 7] = mean_M_max
    M_vals[k, 8] = max_M_max
    #
    tast_vals[k, 0] = tast
    mu_vals[k, 0] = muRBA_env
    mu_vals[k, 1] = muRBA_adapt
    mu_vals[k, 2] = muRBA_it
    mu_tmin = np.exp(np.log(2.0)*del_t_RBA/tast)
    mu_vals[k, 3] = mu_tmin
    #
    lambda_vals[k, 0] = np.log(muRBA_env)/del_t_RBA
    lambda_vals[k, 1] = np.log(muRBA_adapt)/del_t_RBA
    lambda_vals[k, 2] = np.log(muRBA_it)/del_t_RBA
    lambda_vals[k, 3] = np.log(mu_tmin)/del_t_RBA
    #
    # (v) plot/export
    if plot_in_py:
        plot_ty_between(out['tgrid'], y_min, y_max, y_vec, col_dict)#, wvec=None, line_style='-'):
        plot_ty(out['tgrid'], out['y_data'], y_vec, col_dict)
        plt.title(f'N_adapt = {ext_cond_adapt_val}')
        #plt.plot(out['tgrid'], out['y_data'])
        #plt.plot(out['tgrid'], min_y)
        #plt.plot(out['tgrid'], max_y)
        plt.show()
if plot_in_py:
    plt.subplot(3, 1, 1)
    plt.plot(ext_cond_adapt_vals, M_vals[:,[4,7]])
    plt.subplot(3, 1, 2)
    plt.plot(ext_cond_adapt_vals, tast_vals)
    plt.subplot(3, 1, 3)
    plt.plot(ext_cond_adapt_vals, mu_vals)
#
filename = 'opt_pess_adapt.dat'   
header_to_save = '\t'.join(['N','Mmin','Mmean','Mmax','minMmin','minMmean','minMmax', 'maxMmin', 'maxMmean', 'maxMmax',
                            'muenv', 'muadapt', 'muit', 'mutmin', 'tast'])
data_to_save = np.hstack([np.array([ext_cond_adapt_vals]).T, M_vals, mu_vals, tast_vals])
np.savetxt(filename, data_to_save, delimiter='\t', newline='\n', header=header_to_save, comments='')

h_to_save='\t'.join(['N','lenv','ladapt','lit','ltmin'])
f_name = 'opt_pess_adapt_lam.dat'
d_to_save=np.hstack([np.array([ext_cond_adapt_vals]).T, lambda_vals])
np.savetxt(f_name, d_to_save, delimiter='\t', newline='\n', header=h_to_save, comments='')
'''


'''
# (D) :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# RBA jumps: We jump from one (constant) nutrient level to the next
constants = mini.create_constants(use_model_flag:= 2)
wvec = mini.create_weight_vector(y_vec, constants)
plot_in_py = True
init_Ns = [0.5, 1.0, 1.0, 5.0]
goal_Ns = [1.0, 0.5, 5.0, 1.0]
for k, (init_N, goal_N) in enumerate(zip(init_Ns, goal_Ns)):
    # RBA
    ext_init = lambda t: init_N
    m_dictRBA_init = cFBA.create_m_dict(mini, ext_init, constants)
    outRBA_init, _ = cFBA.RBA_like(m_dictRBA_init, 0.0, del_t_RBA, verbosity_level=0,
                                  mumin=1.0, mumax=2.0, y_start=None, wvec=wvec)
    ext_goal = lambda t: goal_N
    m_dictRBA_goal = cFBA.create_m_dict(mini, ext_goal, constants)
    outRBA_goal, _ = cFBA.RBA_like(m_dictRBA_goal, 0.0, del_t_RBA, verbosity_level=0,
                                   mumin=1.0, mumax=2.0, y_start=None, wvec=wvec)
    # t-min-opt
    yinit = np.fmax(outRBA_init['y'], 0.0*outRBA_init['y'])# make sure, none are negative
    ygoal = np.fmax(outRBA_goal['y'], 0.0*outRBA_goal['y'])# make sure, none are negative
    ext_cond = lambda t: goal_N
    tast, outtmin, _ = cFBA.opt_jump_from_to(mini, yinit, ygoal, ext_cond, constants, J0=None, Jend=None, wvec=wvec, t0=0.0,
                         est_min=1e-3, est_max=1e3, t_estimates=None, n_steps=100, verbosity_level=0)
    print(f'{init_N}\t{goal_N}\t{tast}')
    # deFVA
    y_min, y_max = cFBA.low_level_deFVA(outtmin, y_vec, verbosity_level=0, varphi=0.0, fva_level=1)
    #
    if plot_in_py:
        #print(outtmin['tgrid'])
        #plt.subplot(1,3,1)
        #plt.plot(outtmin['tgrid'], y_min)
        #plt.subplot(1,3,2)
        #plt.plot(outtmin['tgrid'], y_max)
        #plt.subplot(1,3,3)
        #plt.plot(outtmin['tgrid'], outtmin['y_data'])
        plot_ty_between(outtmin['tgrid'], y_min, y_max, y_vec, col_dict, line_style=':')#, wvec=None, line_style='-'):
        plot_ty(outtmin['tgrid'], outtmin['y_data'], y_vec, col_dict)
        plt.title(f'N from {init_N} to {goal_N}')
        plt.show()
    filename = f'rba_jumps_{k}.dat'
    header_to_save = '\t'.join(['t','M','Tr','R','Mmin','Trmin','Rmin', 'Mmax', 'Trmax', 'Rmax'])
    data_to_save = np.hstack([np.array([outtmin['tgrid']]).T, outtmin['y_data'], y_min, y_max])
    np.savetxt(filename, data_to_save, delimiter='\t', newline='\n', header=header_to_save, comments='')
# 
'''

'''
# (E) :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# PARETO optimality: t-min (?vs. mu-max?) 
eye_n, zer_n = np.eye(n_y), np.zeros((n_y, n_y))
constants = mini.create_constants(use_model_flag:=2)
wvec = mini.create_weight_vector(y_vec, constants)
n_givenvals = 3
plot_in_py = True
n_steps_tmin = 100
n_steps_cfba = 100
# adapted at start to these conditions -----------------------------------------------------------
e0 = 1.0
ext_adapt = lambda t: e0
m_dict_adapt = cFBA.create_m_dict(mini, ext_adapt, constants)
rba_adapt, mu_adapt = cFBA.RBA_like(m_dict_adapt, 0.0, del_t_RBA, verbosity_level=0,
                                    mumin=1.0, mumax=2.0, y_start=None, wvec=wvec)
yinit = np.fmax(rba_adapt['y'], 0.0*rba_adapt['y'])
# given external conditions --------------------------------------------------------------------
ext_vals, col_string_vals = np.array([0.5, 1.0, 2.0]), ['green', 'blue', 'black']
#ext_vals, col_string_vals = np.array([0.5, 5.0]), ['green', 'black']
#ext_vals, col_string_vals = np.array([0.5]), ['green']
#
all_mu_vals = []
all_tast_vals = []
all_title_strings = []
#
header_to_save = '\t'.join(['t', 'M', 'Tr', 'R', 'Mmin', 'Trmin', 'Rmin', 'Mmax', 'Trmax', 'Rmax'])
#
for n, e in enumerate(ext_vals):
    print(n)
    ext = lambda t: e
    m_dict = cFBA.create_m_dict(mini, ext, constants)
    rba_sol, mu_sol = cFBA.RBA_like(m_dict, 0.0, del_t_RBA, verbosity_level=0,
                                    mumin=1.0, mumax=2.0, y_start=None, wvec=wvec)
    ygoal_normed = np.fmax(rba_sol['y'], 0.0*rba_sol['y'])
    #
    tast_vals = np.nan*np.ones((n_givenvals, 2))
    mu_vals = np.nan*np.ones((n_givenvals, 2))
    #
    if e != e0:
        tast_min, out, _ = cFBA.opt_jump_from_to(mini, yinit, ygoal_normed, ext, constants,#J0=None, Jend=None,
                                             wvec=wvec,# t0=0.0, est_min=1e-3, est_max=1e3, t_estimates=None, n_steps=50, verbosity_level=0)
                                             n_steps=n_steps_tmin, verbosity_level=0)
        y_min, y_max = cFBA.low_level_deFVA(out, y_vec, verbosity_level=0, varphi=0.0, fva_level=1)
        mu_min = np.inner(wvec.flatten(), out['y_data'][-1,:].flatten())
    else:
        tast_min = 0.0
        mu_min = 1.0
    print(tast_min, mu_min)
    filename = f'Pareto_timeopt_{n}.dat'
    # MAYBE: deFVA also?
    data_to_save = np.hstack([np.array([out['tgrid']]).T, out['y_data'], y_min, y_max])
    np.savetxt(filename, data_to_save, delimiter='\t', newline='\n', header=header_to_save, comments='')
    #
    mu_vals[:, 1] = np.linspace(mu_min, 1.65, n_givenvals)
    tast_vals[0, 1] = tast_min
    #
    for k, mu in enumerate(mu_vals[1:, 1]):
        print('\t', k)
        # fix the end configuration
        ygoal = mu*ygoal_normed
        #print(np.inner(wvec.flatten(), ygoal.flatten()))
        # run tminopt
        tast, out, _ = cFBA.opt_jump_from_to(mini, yinit, ygoal, ext, constants,#J0=None, Jend=None, wvec=None, t0=0.0, 
                                             #est_min=1e-3, est_max=1e3, # t_estimates=None, n_steps=50, verbosity_level=0)
                                        n_steps=n_steps_tmin, verbosity_level=0)
        tast_vals[k+1, 1] = tast
        #print(tast, mu)
        if (k ==n_givenvals-2) and (e != e0):
            # export time course
            y_min, y_max = cFBA.low_level_deFVA(out, y_vec, verbosity_level=0, varphi=0.0, fva_level=1)
            print('exporting')
            filename = f'Pareto_time_{n}.dat'
            # MAYBE: deFVA also?
            data_to_save = np.hstack([np.array([out['tgrid']]).T, out['y_data'], y_min, y_max])
            np.savetxt(filename, data_to_save, delimiter='\t', newline='\n', header=header_to_save, comments='')
    #
    # check via cFBA (doesn't work often :-( )
    tast_vals[:, 0] = np.linspace(tast_min, np.max(tast_vals[:, 1]), n_givenvals)
    tast_vals[:, 0] = tast_vals[:, 1]
    """
    #mu_vals
    for k, T in enumerate( tast_vals[:, 0] ):
    #for k, T in enumerate( tast_vals[1:, 0] ):
        print('\t', k, T)
        #print(m_dict)
        # fix end time point
        tspan = np.linspace(0.0, T, n_steps_cfba+1)
        print(np.inner(wvec.flatten(), ygoal_normed.flatten()),
              np.inner(wvec.flatten(), yinit.flatten()))
        try:
            outcFBA, muend = cFBA.cFBA(m_dict, tspan, verbosity_level=10,
                             mumin=1.000, #mu_vals[k, 1]/1.00001,
                             mumax=2.00, #mu_vals[k, 1]*1.05, 
                             y_start=yinit, y_end_normed=ygoal_normed, wvec=None)
        except:
            print('no sol')
            muend = np.nan
            fuenf = vier + 1
        mu_vals[k, 0] = muend
        print(30*'_')
        #print(T, muend)
    """
    all_mu_vals.append(mu_vals)
    all_tast_vals.append(tast_vals)

filename = 'Pareto.dat'
header_to_save = '\t'.join(['mu1', 'mu2', 'mu3', 'T1', 'T2', 'T3'])
data_to_save = np.hstack([np.array([all_mu_vals[0][:,1]]).T,
                          np.array([all_mu_vals[1][:,1]]).T,
                          np.array([all_mu_vals[2][:,1]]).T,
                          np.array([all_tast_vals[0][:,1]]).T, 
                          np.array([all_tast_vals[1][:,1]]).T, 
                          np.array([all_tast_vals[2][:,1]]).T  ])
np.savetxt(filename, data_to_save, delimiter='\t', newline='\n', header=header_to_save, comments='')

print(40*'+')
for mu_vals in all_mu_vals:
    print(mu_vals)
print(40*'+')
for tast_vals in all_tast_vals:
    print(tast_vals)

if plot_in_py:
    for n, col_string in enumerate(col_string_vals):
        tast_vals = all_tast_vals[n]
        mu_vals = all_mu_vals[n]
        #
        plt.semilogy(tast_vals[:, 1], mu_vals[:, 1], marker='.', color=col_string)
        plt.semilogy(tast_vals[:, 0], mu_vals[:, 0], marker='*', color=col_string)
'''        
        
        
'''
# (Eii) :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# PARETO optimality: t-min (?vs. mu-max?) this time: all adapt to the same environment
eye_n, zer_n = np.eye(n_y), np.zeros((n_y, n_y))
constants = mini.create_constants(use_model_flag:=2)
wvec = mini.create_weight_vector(y_vec, constants)
n_givenvals = 26 # 26
plot_in_py = True
export_data = True
n_steps_tmin = 100
n_steps_cfba = 100
# adapted at start to these conditions -----------------------------------------------------------
e0 = 1.0
ext_adapt = lambda t: e0
m_dict_adapt = cFBA.create_m_dict(mini, ext_adapt, constants)
rba_adapt, mu_adapt = cFBA.RBA_like(m_dict_adapt, 0.0, del_t_RBA, verbosity_level=0,
                                    mumin=1.0, mumax=2.0, y_start=None, wvec=wvec)
ygoal_normed = np.fmax(rba_adapt['y'], 0.0*rba_adapt['y'])
# given external conditions --------------------------------------------------------------------
ext_vals, col_string_vals = np.array([0.2, 1.0, 10.0]), ['green', 'blue', 'black']
#ext_vals, col_string_vals = np.array([0.5, 5.0]), ['green', 'black']
#ext_vals, col_string_vals = np.array([0.5]), ['green']
#
all_mu_vals = []
all_tast_vals = []
all_title_strings = []
#
header_to_save = '\t'.join(['t', 'M', 'Tr', 'R', 'Mmin', 'Trmin', 'Rmin', 'Mmax', 'Trmax', 'Rmax'])
#
for n, e in enumerate(ext_vals):
    print(n)
    ext = lambda t: e
    m_dict = cFBA.create_m_dict(mini, ext, constants)
    rba_sol, mu_sol = cFBA.RBA_like(m_dict, 0.0, del_t_RBA, verbosity_level=0,
                                    mumin=1.0, mumax=2.0, y_start=None, wvec=wvec)
    yinit = np.fmax(rba_sol['y'], 0.0*rba_sol['y'])
    #
    tast_vals = np.nan*np.ones((n_givenvals+1, 2))
    mu_vals = np.nan*np.ones((n_givenvals+1, 2))
    #
    if e != e0:
        tast_min, out, _ = cFBA.opt_jump_from_to(mini, yinit, ygoal_normed, ext_adapt, constants,#J0=None, Jend=None,
                                             wvec=wvec,# t0=0.0, est_min=1e-3, est_max=1e3, t_estimates=None, n_steps=50, verbosity_level=0)
                                             n_steps=n_steps_tmin, verbosity_level=0)
        y_min, y_max = cFBA.low_level_deFVA(out, y_vec, verbosity_level=0, varphi=0.0, fva_level=1)
        mu_min = np.inner(wvec.flatten(), out['y_data'][-1,:].flatten())
    else:
        tast_min = 0.0
        mu_min = 1.0
    print(tast_min, mu_min)
    filename = f'Pareto_timeopt_to_one_{n}.dat'
    # MAYBE: deFVA also?
    data_to_save = np.hstack([np.array([out['tgrid']]).T, out['y_data'], y_min, y_max])
    if export_data:
        np.savetxt(filename, data_to_save, delimiter='\t', newline='\n', header=header_to_save, comments='')
    #
    #mu_vals[:, 1] = np.linspace(mu_min, 6.50, n_givenvals)
    #mu_vals[:, 1] = [i**2 for i in np.linspace(np.sqrt(mu_min), np.sqrt(6.50), n_givenvals)]
    mu_vals[:, 1] = np.hstack([[np.exp(i**2) for i in np.linspace(np.sqrt(np.log(mu_min)), np.sqrt(np.log(4.50)), n_givenvals)],[6.0]])
    tast_vals[0, 1] = tast_min
    #
    for k, mu in enumerate(mu_vals[1:, 1]):
        print('\t', k, '\tmu = ', mu)
        # fix the end configuration
        ygoal = mu*ygoal_normed
        #print(np.inner(wvec.flatten(), ygoal.flatten()))
        # run tminopt
        tast, out, _ = cFBA.opt_jump_from_to(mini, yinit, ygoal, ext_adapt, constants,#J0=None, Jend=None, wvec=None, t0=0.0, 
                                             #est_min=1e-3, est_max=1e3, # t_estimates=None, n_steps=50, verbosity_level=0)
                                        n_steps=n_steps_tmin, verbosity_level=0)
        tast_vals[k+1, 1] = tast
        #print(tast, mu)
        if (k ==n_givenvals-2) and (e != e0):
            # export time course
            y_min, y_max = cFBA.low_level_deFVA(out, y_vec, verbosity_level=0, varphi=0.0, fva_level=1)
            filename = f'Pareto_time_to_one_{n}.dat'
            # MAYBE: deFVA also?
            data_to_save = np.hstack([np.array([out['tgrid']]).T, out['y_data'], y_min, y_max])
            if export_data:
                print('exporting')
                np.savetxt(filename, data_to_save, delimiter='\t', newline='\n', header=header_to_save, comments='')
    #
    # check via cFBA (doesn't work often :-( )
    tast_vals[:, 0] = np.linspace(tast_min, np.max(tast_vals[:, 1]), n_givenvals+1)
    tast_vals[:, 0] = tast_vals[:, 1]
    #
    all_mu_vals.append(mu_vals)
    all_tast_vals.append(tast_vals)

filename = 'Pareto_to_one.dat'
header_to_save = '\t'.join(['mu1', 'mu2', 'mu3', 'T1', 'T2', 'T3'])
data_to_save = np.hstack([np.array([all_mu_vals[0][:,1]]).T,
                          np.array([all_mu_vals[1][:,1]]).T,
                          np.array([all_mu_vals[2][:,1]]).T,
                          np.array([all_tast_vals[0][:,1]]).T, 
                          np.array([all_tast_vals[1][:,1]]).T, 
                          np.array([all_tast_vals[2][:,1]]).T  ])
if export_data:
    np.savetxt(filename, data_to_save, delimiter='\t', newline='\n', header=header_to_save, comments='')

print(40*'+')
for mu_vals in all_mu_vals:
    print(mu_vals)
print(40*'+')
for tast_vals in all_tast_vals:
    print(tast_vals)

if plot_in_py:
    for n, col_string in enumerate(col_string_vals):
        tast_vals = all_tast_vals[n]
        mu_vals = all_mu_vals[n]
        #
        plt.semilogy(tast_vals[:, 1], mu_vals[:, 1], marker='.', color=col_string)
        plt.semilogy(tast_vals[:, 0], mu_vals[:, 0], marker='*', color=col_string)      
'''    
        
        
        
        
