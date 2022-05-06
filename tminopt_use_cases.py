#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 10:47:37 2022

@author: markukob
"""
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib import gridspec
import lowlevel_cFBAstuff as cFBA
import mini_self_replicator as mini
import toa_display_helper as toa_disp
#from itertools import chain, combinations
#from scipy.optimize import bisect


y_vec, m_vec, u_vec = mini.create_name_vectors()
n_y, n_u = len(y_vec), len(u_vec)
col_dict = mini.create_color_palette()
prot_list = ['M', 'T', 'R']
del_t_RBA = 0.01


def use_case_RBA_curves():
    '''
    the example network: growth curves, "RBA curves"
    '''
    constants = mini.create_constants(flag=2)
    wvec = mini.create_weight_vector(y_vec, constants)
    extvals = np.linspace(0.3, 15.0, 35)
    muvals = []
    growthvals = []
    allRBA_yvals = []
    for i, ext_val in enumerate(extvals):
        #print(i)
        ext_cond = lambda t: ext_val
        m_dictRBA = cFBA.create_m_dict(mini, ext_cond, constants)
        outRBA, muRBA = cFBA.RBA_like(m_dictRBA, 0.0, del_t_RBA, verbosity_level=0, mumin=1.0, mumax=2.0, y_start=None, wvec=wvec)
        muvals.append(muRBA)
        growthvals.append(np.log(muRBA)/del_t_RBA)
        allRBA_yvals.append(outRBA['y'].T.tolist()[0])
    plt.plot(extvals, muvals)
    plt.show()
    plt.plot(extvals, np.array(allRBA_yvals))
    

def use_case_cell_doubling():
    '''
    cell doubling for a not perfectly adapted cell
    '''
    n_steps = 200
    constants = mini.create_constants(flag=2)
    wvec = mini.create_weight_vector(y_vec, constants)
    ext = lambda t: 1.0
    ext_adapt = lambda t: 2.0
    #
    m_dictRBA = cFBA.create_m_dict(mini, ext_adapt, constants)
    outRBA, muRBA = cFBA.RBA_like(m_dictRBA, 0.0, del_t_RBA, verbosity_level=0, mumin=1.0, mumax=2.0, y_start=None, wvec=wvec)
    yinit = np.fmax(outRBA['y'], 0.0*outRBA['y'])# make sure, none are negative
    ygoal = 2.0*yinit
    tast, out, _ = cFBA.opt_jump_from_to(mini, yinit, ygoal, ext, constants, 
                               n_steps=n_steps, verbosity_level=0)
    #
    out_biomass = toa_disp.biomass(out['y_data'], wvec)
    #      
    m_dict = cFBA.create_m_dict(mini, ext, constants)
    out_it_RBA, mu_it_RBA = cFBA.RBA_like(m_dict, 0.0, del_t_RBA, verbosity_level=0, mumin=1.0, mumax=2.0, y_start=yinit, wvec=None)
    _, mu_RBA_opt = cFBA.RBA_like(m_dict, 0.0, del_t_RBA, verbosity_level=0, mumin=1.0, mumax=2.0, y_start=None, wvec=wvec)
    lambda_it = np.log(mu_it_RBA)/del_t_RBA
    tgrid_itRBA = np.linspace(0.0, np.log(2)/lambda_it, n_steps+1)
    print(np.log(2)/np.log(mu_RBA_opt)*del_t_RBA, tast, np.log(2)/lambda_it)
    it_RBA_y = np.outer(yinit, np.exp(lambda_it*tgrid_itRBA)).T
    uinit = np.fmax(out_it_RBA['u'], 0.0*out_it_RBA['u'])# make sure, none are negative
    it_RBA_u = np.outer(uinit, np.exp(lambda_it*tgrid_itRBA)).T
    out_biomass_RBA = toa_disp.biomass(it_RBA_y, wvec)
    # Find the boundaries of the different areas in the time course
    tmp = np.diff(out['y_data'],axis=0,n=2)
    for k, t in enumerate( out['tgrid'][:-2] ):
        if np.abs(tmp[k, 2]) > 0.0001:
            print(out['tgrid'][k+1])
    outfl, fl_names = cFBA.constraint_fulfillment(out, m_dict, verbosity_level=0, plot_kind = 'flux_bounds')
    # 
    ax1 = plt.subplot(2,1,1)
    ax1.plot(out['tgrid'], out['y_data'])
    ax2 = ax1.twinx()
    ax2.semilogy(out['tgrid'], out_biomass)
    ax2.semilogy(tgrid_itRBA, out_biomass_RBA)
    ax1.plot(tgrid_itRBA, it_RBA_y)
    plt.subplot(2,1,2)
    plt.plot(out['tgrid_u'], out['u_data'])
    plt.legend(u_vec)
    plt.plot(tgrid_itRBA, it_RBA_u)


def use_case_different_cell_doublings():
    '''
    # time course for different nutrient supply, cell doubling
    '''
    constants = mini.create_constants(flag=2)
    wvec = mini.create_weight_vector(y_vec, constants)
    all_extvals = []
    all_adapt_ext_vals = []
    # (1, low nutrient supply)
    all_extvals.append(0.5)
    all_adapt_ext_vals.append([0.25, 0.5, 0.75])
    # (2, medium nutrient supply)
    all_extvals.append(1.0)
    all_adapt_ext_vals.append([0.5, 1.0, 5.0])
    # (3, higher nutrient supply)
    all_extvals.append(5.0)
    all_adapt_ext_vals.append([1.0, 5.0, 10.0])
    # (a) RBA
    for k, extval in enumerate(all_extvals):
        ext_cond = lambda t: extval
        m_dictRBA = cFBA.create_m_dict(mini, ext_cond, constants)
        outRBA, muRBA = cFBA.RBA_like(m_dictRBA, 0.0, del_t_RBA, verbosity_level=0, mumin=1.0, mumax=2.0, y_start=None, wvec=wvec)
        # (b) t-min-opt, optimally adapted
        adapt_ext_vals = all_adapt_ext_vals[k]
        for i, adapt_ext_val in enumerate(adapt_ext_vals):
            print(i)
            ext_cond_adapt = lambda t: adapt_ext_val
            m_dict_adapt = cFBA.create_m_dict(mini, ext_cond_adapt, constants)
            out_adapt, mu_adapt = cFBA.RBA_like(m_dict_adapt, 0.0, del_t_RBA, verbosity_level=0, mumin=1.0, mumax=2.0, y_start=None, wvec=wvec)    
            yinit = np.fmax(out_adapt['y'], 0.0*out_adapt['y'])# make sure, none are negative
            ygoal = 2.0*yinit
            tast, out, _ = cFBA.opt_jump_from_to(mini, yinit, ygoal, ext_cond, constants, # J0=None, Jend=None, wvec=None, t0=0.0,
                               #est_min=1e-3, est_max=1e3, t_estimates=None,
                               n_steps=100, verbosity_level=0)
            # deFVA
            min_y, max_y = cFBA.low_level_deFVA(out, y_vec, verbosity_level=0, varphi=0.0, fva_level=1)
            #
            plt.plot(out['tgrid'], out['y_data'])
            plt.plot(out['tgrid'], min_y)
            plt.plot(out['tgrid'], max_y)
            # (c) "iterative" RBA
            out_it_RBA, mu_it_RBA = cFBA.RBA_like(m_dictRBA, 0.0, del_t_RBA, verbosity_level=0, mumin=1.0, mumax=2.0, y_start=yinit, wvec=None)
            lambda_it_RBA = np.log(mu_it_RBA)/del_t_RBA
            it_RBA_y = np.outer(yinit, np.exp(lambda_it_RBA*out['tgrid'])).T
            plt.plot(out['tgrid'], it_RBA_y)
            #
            plt.title(f"N = {extval}")
            plt.show()

    
def use_case_optimism_pessimism():
    '''
    cell doubling: the role of optimism/pessimism for storage
    the cell lives in a surrounding of ext_cond but is adapted to ext_cond_adapt_val
    '''
    constants = mini.create_constants(flag= 2)
    wvec = mini.create_weight_vector(y_vec, constants)
    ext_cond_adapt_vals = np.linspace(0.2, 2.0, n_ext:=45)
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
        # (i) compute RBA solution for adaptation value (utopia since the actual surrounding might not
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
        y_min, y_max, egal = cFBA.relative_tmin_deFVA(mini, yinit, ygoal, ext_cond, tast,
                                                              constants, wvec, verbosity_level=0,
                                                              t_extend_trials=(1.0e-7, 1.0e-5))
        # (iv) compute iterative RBA
        _, muRBA_it = cFBA.RBA_like(m_dictRBA_env, 0.0, del_t_RBA, verbosity_level=0,
                                    mumin=1.0, mumax=2.0, y_start=yinit, wvec=None)
        # (v) compute averages (plain so far)
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
        toa_disp.plot_ty_between(out['tgrid'], y_min, y_max, y_vec, col_dict)#, wvec=None, line_style='-'):
        toa_disp.plot_ty(out['tgrid'], out['y_data'], y_vec, col_dict)
        plt.title(f'N_adapt = {ext_cond_adapt_val}')
        plt.show()
    plt.subplot(3, 1, 1)
    plt.plot(ext_cond_adapt_vals, M_vals[:,[4,7]])
    plt.subplot(3, 1, 2)
    plt.plot(ext_cond_adapt_vals, tast_vals)
    plt.subplot(3, 1, 3)
    plt.plot(ext_cond_adapt_vals, mu_vals)
    #


def use_case_RBA_jumps():
    '''
    Figure 7
    '''    
    constants = mini.create_constants(flag= 2)
    wvec = mini.create_weight_vector(y_vec, constants)
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
        toa_disp.plot_ty_between(outtmin['tgrid'], y_min, y_max, y_vec, col_dict, line_style=':')#, wvec=None, line_style='-'):
        toa_disp.plot_ty(outtmin['tgrid'], outtmin['y_data'], y_vec, col_dict)
        plt.title(f'N from {init_N} to {goal_N}')
        plt.show()


def use_case_Pareto():
    '''
    Figure 8
    '''
    constants = mini.create_constants(flag=2)
    wvec = mini.create_weight_vector(y_vec, constants)
    n_givenvals = 26 
    n_steps_tmin = 100
    # adapted at start to these conditions -----------------------------------------------------------
    e0 = 1.0
    ext_adapt = lambda t: e0
    m_dict_adapt = cFBA.create_m_dict(mini, ext_adapt, constants)
    rba_adapt, mu_adapt = cFBA.RBA_like(m_dict_adapt, 0.0, del_t_RBA, verbosity_level=0,
                                        mumin=1.0, mumax=2.0, y_start=None, wvec=wvec)
    ygoal_normed = np.fmax(rba_adapt['y'], 0.0*rba_adapt['y'])
    # given external conditions --------------------------------------------------------------------
    ext_vals, col_string_vals = np.array([0.2, 1.0, 10.0]), ['green', 'blue', 'black']
    #
    all_mu_vals = []
    all_tast_vals = []
    #
    for n, e in enumerate(ext_vals):
        print(n)
        ext = lambda t: e
        m_dict = cFBA.create_m_dict(mini, ext, constants)
        rba_sol, mu_sol = cFBA.RBA_like(m_dict, 0.0, del_t_RBA, verbosity_level=0,
                                        mumin=1.0, mumax=2.0, y_start=None, wvec=wvec)
        yinit = np.fmax(rba_sol['y'], 0.0*rba_sol['y'])
        tast_vals = np.nan*np.ones(n_givenvals+1)
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
        mu_vals = np.hstack([[np.exp(i**2) for i in np.linspace(np.sqrt(np.log(mu_min)), np.sqrt(np.log(4.50)), n_givenvals)],[6.0]])
        tast_vals[0] = tast_min
        #
        for k, mu in enumerate(mu_vals[1:]):
            print('\t', k, '\tmu = ', mu)
            # fix the end configuration
            ygoal = mu*ygoal_normed
            # run tminopt
            tast, out, _ = cFBA.opt_jump_from_to(mini, yinit, ygoal, ext_adapt, constants,#J0=None, Jend=None, wvec=None, t0=0.0, 
                                                 #est_min=1e-3, est_max=1e3, # t_estimates=None, n_steps=50, verbosity_level=0)
                                                 n_steps=n_steps_tmin, verbosity_level=0)
            tast_vals[k+1] = tast
        print(tast, mu)
        #
        all_mu_vals.append(mu_vals)
        all_tast_vals.append(tast_vals)
        #
    print(40*'+')
    for mu_vals in all_mu_vals:
        print(mu_vals)
    print(40*'+')
    for tast_vals in all_tast_vals:
        print(tast_vals)
    for n, col_string in enumerate(col_string_vals):
        tast_vals = all_tast_vals[n]
        mu_vals = all_mu_vals[n]
        plt.semilogy(tast_vals, mu_vals, marker='.', color=col_string)





def use_case_Pareto_alt():
    '''
    adaptation of Figure 8: jump to different environments
    '''    
    constants = mini.create_constants(flag=2)
    wvec = mini.create_weight_vector(y_vec, constants)
    n_givenvals = 35
    n_steps_tmin = 100
    # adapted at start to these conditions -----------------------------------------------------------
    e0 = 1.0
    ext_adapt = lambda t: e0
    m_dict_adapt = cFBA.create_m_dict(mini, ext_adapt, constants)
    rba_adapt, mu_adapt = cFBA.RBA_like(m_dict_adapt, 0.0, del_t_RBA, verbosity_level=0,
                                        mumin=1.0, mumax=2.0, y_start=None, wvec=wvec)
    yinit = np.fmax(rba_adapt['y'], 0.0*rba_adapt['y'])
    # given external conditions --------------------------------------------------------------------
    ext_vals, col_string_vals = np.array([0.5, 1.0, 2.0]), ['green', 'blue', 'black']
    #
    all_mu_vals = []
    all_tast_vals = []
    #
    for n, e in enumerate(ext_vals):
        print(n)
        ext = lambda t: e
        m_dict = cFBA.create_m_dict(mini, ext, constants)
        rba_sol, mu_sol = cFBA.RBA_like(m_dict, 0.0, del_t_RBA, verbosity_level=0,
                                        mumin=1.0, mumax=2.0, y_start=None, wvec=wvec)
        ygoal_normed = np.fmax(rba_sol['y'], 0.0*rba_sol['y'])
        #
        tast_vals = np.nan*np.ones(n_givenvals)
        #
        if e != e0:
            tast_min, out, _ = cFBA.opt_jump_from_to(mini, yinit, ygoal_normed, ext, constants,#J0=None, Jend=None,
                                                     wvec=wvec,# t0=0.0, est_min=1e-3, est_max=1e3, t_estimates=None, n_steps=50, verbosity_level=0)
                                                     n_steps=n_steps_tmin, verbosity_level=0)
            mu_min = np.inner(wvec.flatten(), out['y_data'][-1,:].flatten())
        else:
            tast_min = 0.0
            mu_min = 1.0
            print(tast_min, mu_min)
        #
        mu_vals = np.linspace(mu_min, 1.65, n_givenvals)
        tast_vals[0] = tast_min
        #
        for k, mu in enumerate(mu_vals[1:]):
            print('\t', k)
            # fix the end configuration
            ygoal = mu*ygoal_normed
            # run tminopt
            tast, out, _ = cFBA.opt_jump_from_to(mini, yinit, ygoal, ext, constants,#J0=None, Jend=None, wvec=None, t0=0.0, 
                                                 #est_min=1e-3, est_max=1e3, # t_estimates=None, n_steps=50, verbosity_level=0)
                                                 n_steps=n_steps_tmin, verbosity_level=0)
            tast_vals[k+1] = tast
        all_mu_vals.append(mu_vals)
        all_tast_vals.append(tast_vals)
    print(40*'+')
    for mu_vals in all_mu_vals:
        print(mu_vals)
    print(40*'+')
    for tast_vals in all_tast_vals:
        print(tast_vals)
    for n, col_string in enumerate(col_string_vals):
        tast_vals = all_tast_vals[n]
        mu_vals = all_mu_vals[n]
        plt.semilogy(tast_vals, mu_vals, marker='.', color=col_string)
        

def use_case_optimism_pessimism_alt():
    '''
    cell doubling: the role of optimism/pessimism for storage
    the cell lives in a surrounding of ext_cond but is adapted to ext_cond_adapt_val
    This time, the former changes for the plots
    '''
    constants = mini.create_constants(flag= 2)
    wvec = mini.create_weight_vector(y_vec, constants)
    ext_cond_adapt = lambda t: 1.0 # always adapted to this external condition
    #
    n_int = 45 #45
    M_vals = np.zeros((n_int, 9))
    tast_vals = np.zeros((n_int, 1))
    mu_vals = np.zeros((n_int, 4))
    lambda_vals = np.zeros((n_int, 4))
    ext_cond_vals = np.linspace(0.2, 2.0, n_int)
    #
    m_dictRBA_adapt = cFBA.create_m_dict(mini, ext_cond_adapt, constants)
    outRBA_adapt, muRBA_adapt = cFBA.RBA_like(m_dictRBA_adapt, 0.0, del_t_RBA, verbosity_level=0,
                                              mumin=1.0, mumax=2.0, y_start=None, wvec=wvec)
    yinit = np.fmax(outRBA_adapt['y'], 0.0*outRBA_adapt['y'])
    ygoal = 2.0*yinit
    #
    for k, ext_cond_val in enumerate(ext_cond_vals):
        print(f'k = {k+1} of {n_int}')
        ext_cond = lambda t: ext_cond_val
        m_dictRBA_env = cFBA.create_m_dict(mini, ext_cond, constants)
        out_RBA_env, muRBA_env = cFBA.RBA_like(m_dictRBA_env, 0.0, del_t_RBA, verbosity_level=0,
                                               mumin=1.0, mumax=2.0, y_start=None, wvec=wvec)
        # (ii) compute tminopt
        #yinit = np.fmax(out_RBA_env['y'], 0.0*out_RBA_env['y'])# make sure, none are negative
        tast, out, _ = cFBA.opt_jump_from_to(mini, yinit, ygoal, ext_cond, constants, # J0=None, Jend=None, wvec=None, t0=0.0,
                                   #est_min=1e-3, est_max=1e3, t_estimates=None,
                                   n_steps=100, verbosity_level=0)
        # (iii) compute deFVA
        #y_min, y_max, egal = cFBA.relative_tmin_deFVA(mini, yinit, ygoal, ext_cond, tast,
        #                                                      constants, wvec, verbosity_level=0,
        #                                                      t_extend_trials=(1.0e-7, 1.0e-5))
        # (iv) compute iterative RBA
        _, muRBA_it = cFBA.RBA_like(m_dictRBA_env, 0.0, del_t_RBA, verbosity_level=0,
                                    mumin=1.0, mumax=2.0, y_start=yinit, wvec=None)
        # (v) compute averages (plain so far)
        min_M =   np.min( out['y_data'][:, y_vec.index('M')] )
        mean_M = np.mean( out['y_data'][:, y_vec.index('M')] )
        max_M =   np.max( out['y_data'][:, y_vec.index('M')] )
        #min_M_min =   np.min( y_min[:, y_vec.index('M')] )
        #mean_M_min = np.mean( y_min[:, y_vec.index('M')] )
        #max_M_min =   np.max( y_min[:, y_vec.index('M')] )
        #min_M_max =   np.min( y_max[:, y_vec.index('M')] )
        #mean_M_max = np.mean( y_max[:, y_vec.index('M')] )
        #max_M_max =   np.max( y_max[:, y_vec.index('M')] )
        #
        M_vals[k, 0] = min_M
        M_vals[k, 1] = mean_M
        M_vals[k, 2] = max_M
        #M_vals[k, 3] = min_M_min
        #M_vals[k, 4] = mean_M_min
        #M_vals[k, 5] = max_M_min
        #M_vals[k, 6] = min_M_max
        #M_vals[k, 7] = mean_M_max
        #M_vals[k, 8] = max_M_max
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
        #toa_disp.plot_ty_between(out['tgrid'], y_min, y_max, y_vec, col_dict)#, wvec=None, line_style='-'):
        #toa_disp.plot_ty(out['tgrid'], out['y_data'], y_vec, col_dict)
        #plt.title(f'N_adapt = {ext_cond_adapt_val}')
        #plt.show()
    #plt.subplot(3, 1, 1)
    #plt.plot(ext_cond_adapt_vals, M_vals[:,[4,7]])
    #plt.subplot(3, 1, 2)
    #plt.plot(ext_cond_vals, tast_vals)
    #plt.subplot(3, 1, 3)
    #plt.plot(ext_cond_vals, mu_vals)
    plt.subplot(1,2,1)
    plt.plot(ext_cond_vals, lambda_vals)
    plt.legend(['env.', 'adapt', 'it. RBA', 'TOA'])
    #
    print(lambda_vals)
    print('---------')
    print(lambda_vals[:,[2,3]]/lambda_vals[:,[0]])
    #
    plt.subplot(1,2,2)
    plt.plot(ext_cond_vals, lambda_vals[:,[2,3]]/lambda_vals[:,[0]])
    plt.legend(['it. RBA', 'TOA'])
    #


#use_case_RBA_curves()
#use_case_cell_doubling()
#use_case_different_cell_doublings()
#use_case_optimism_pessimism()
#use_case_RBA_jumps()
#use_case_Pareto()

use_case_optimism_pessimism_alt()

