#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 15:33:18 2021

@author: markukob
"""
import numpy as np
from scipy import sparse as sp
#from matplotlib import pyplot as plt
from pyrrrateFBA.matrrrices import Matrrrices
from pyrrrateFBA.util.runge_kutta import RungeKuttaPars
from pyrrrateFBA.optimization import oc
from scipy.optimize import bisect

#DEFAULT_RKM = RungeKuttaPars(s=3, family = 'Gauss')
DEFAULT_RKM = RungeKuttaPars(s=1, family = 'RadauIIA')
#DEFAULT_RKM = RungeKuttaPars(s=1, family = 'Explicit1')
#DEFAULT_RKM = RungeKuttaPars(s=2, family = 'LobattoIIIA')
#BISECT_TOL, BISECT_REL_TOL = 2e-12, 8.881784197001252e-16 # DEFAULT VALUES
BISECT_TOL, BISECT_REL_TOL = 1e-8, 1e-12
DEFVA_DEFAULT = np.nan # values to be set in deFVA (insane lavel) results if no solution was found
WRITE_INFEAS_TO_FILE = False# DEBUG (Write an lp file if no solution could be found in deFVA)



# TODO: Do we need to use m_dict all the time? Is it not better to use Matrrrices?

def create_m_dict(prob, ext_cond, constants, flag_objective=0):
    y_vec, m_vec, u_vec = prob.create_name_vectors()
    #n_y = len(y_vec)
    phi1, phi3 = prob.create_objective_vectors(flag_objective, constants)
    smat1, smat2 = prob.create_reaction_matrices(y_vec, m_vec, u_vec, constants)
    smat4 = prob.create_degredation(y_vec, constants)
    lbvec, ubvec = prob.create_flux_bounds(u_vec, constants)
    matrix_y, matrix_u, vec_h = prob.create_mixed_constraints(y_vec, u_vec, ext_cond, constants)
    #matrix_start = np.eye(n_y)
    #matrix_end   = np.zeros((n_y, n_y))
    #vec_bndry    = y0
    # BUILD THE MATRRRICES ###################################################
    m_dict = {'y_vec': y_vec,
              'u_vec': u_vec,
              'x_vec' : [],
              'phi1' : phi1,
              'phi2' : 0.0*phi1,
              'phi3':  phi3,
              'smat1': sp.csr_matrix(smat1),
              'smat2': sp.csr_matrix(smat2),
              'smat4': sp.csr_matrix(smat4),
              'lbvec': lbvec,
              'ubvec': ubvec,
              'matrix_y': matrix_y,
              'matrix_u': sp.csr_matrix(matrix_u),
              'vec_h': vec_h#,
              #'matrix_start': matrix_start,
              #'matrix_end': matrix_end,
              #'vec_bndry': vec_bndry
              }
    return m_dict    


def opt_jump_from_to(prob, y0, yend, ext_cond, constants, J0=None, Jend=None, wvec=None, t0=0.0,
                     est_min=1e-3, est_max=1e3, t_estimates=None, n_steps=50, verbosity_level=0):
    """
    Time optimal adaptation with optional filtering and/or relative weights included
    """
    m_dict = create_m_dict(prob, ext_cond, constants)
    n_y = y0.shape[0]
    if J0 is None:
        J0 = range(1, n_y+1)
    if Jend is None:
        Jend = range(1, n_y+1)
    matrix_start = np.zeros((len(J0) + len(Jend), n_y))
    matrix_end = np.zeros((len(J0) + len(Jend), n_y))
    vec_bndry = np.zeros((len(J0) + len(Jend), 1))
    # (1) implement y(0)[i] == y0[i] for all i in J0
    matrix_start[0:len(J0), :] = np.eye(n_y, n_y)[[i-1 for i in J0], :]
    matrix_end[0:len(J0), :] = np.zeros((len(J0), n_y))
    vec_bndry[0:len(J0), 0] = y0[[i-1 for i in J0], 0]
    # (2) implement end point constraint for all i in Jend
    if wvec is None: # formulate in absolute amounts (for end point only)
        matrix_start[len(J0):, :] = np.zeros((len(Jend), n_y))
        matrix_end[len(J0):, :] = np.eye(n_y, n_y)[[i-1 for i in Jend], :]
        vec_bndry[len(J0):, 0] = yend[[i-1 for i in Jend], 0]
    else:
        yendw = np.dot(wvec.T, yend)
        matrix_start[len(J0):, :] = np.zeros((len(Jend), n_y))
        matrix_end[len(J0):, :] = (np.kron(yend, wvec.T)- yendw*np.eye(n_y)) \
            [[i-1 for i in Jend], :]
        vec_bndry[len(J0):, 0] = np.zeros(len(Jend))
    m_dict['matrix_start'] = matrix_start
    m_dict['matrix_end'] = matrix_end
    m_dict['vec_bndry'] = vec_bndry
    #
    tast, out, feas_result = t_min_opt(m_dict, est_min=est_min, est_max=est_max,
                                       t_estimates=t_estimates, n_steps=n_steps, t0=t0,
                                       verbosity_level=verbosity_level)
    #
    return tast, out, feas_result


def t_min_opt(m_dict, est_min = 1e-3, est_max = 1e3, t_estimates=None, n_steps=50, t0=0.0, verbosity_level=0):
    """
    Find the smallest t such that the OC problem described by m_dict is feasible
    """
    tast = None
    out = None
    feas_result = []
    # The actual objective does not matter
    m_dict['phi1'] = 0.0*m_dict['phi1']
    m_dict['phi2'] = m_dict['phi1']
    m_dict['phi3'] = m_dict['phi1']
    #
    MM = Matrrrices(None, run_rdeFBA=False, **m_dict)
    #
    rkm = DEFAULT_RKM
    varphi  = 0.0
    def _bisect_fun(t):
        nonlocal out
        nonlocal tast
        tspan = np.linspace(t0, t, n_steps+1)
        if verbosity_level > 0:
            print('trying t = ', t)
        out_trial = oc.cp_rk_linprog_v(MM, rkm, tspan, varphi=varphi)
        #out_trial = oc.cp_rk_linprog(MM, rkm, tspan[0], tspan[-1], n_steps=n_steps, varphi=varphi)
        if out_trial['y_data'] is None:
            if verbosity_level > 1:
                print('no solutions')
            feas_result.append([t, -1.0])
            return -1.0
        else:
            if verbosity_level > 1:
                print('found solution')
            tast = t
            out = out_trial # This assumes last == best
            feas_result.append([t, 1.0])
            return 1.0
    if t_estimates is None:
        tast = bisect(_bisect_fun, est_min, est_max, xtol=BISECT_TOL, rtol=BISECT_REL_TOL)
    else:
        for t in t_estimates:
            f_r = _bisect_fun(t)
            feas_result.append([t, f_r])
    feas_result = sorted(feas_result, key=lambda t: t[0])
    # DEBUG
    #tast = tast - 10*np.nextafter(tast, 100000)
    #tspan = np.linspace(t0, tast, n_steps)
    #out = oc.cp_rk_linprog_v(MM, rkm, tspan, varphi=varphi)
    # DEBUG END
    return tast, out, np.array(feas_result)


def deFBA(m_dict, tspan, varphi=0.0):
    MM = Matrrrices(None, run_rdeFBA=False, **m_dict)
    rkm = DEFAULT_RKM
    out = oc.cp_rk_linprog_v(MM, rkm, tspan, varphi=varphi)
    return out


def RBA_like(m_dict, t0, del_t, verbosity_level=0, mumin=1.0, mumax=2.0, y_start=None, wvec=None):
    """
    solve min/max mu
          s.t. m_dict-problem is feas. and
          mu*y0 == yend 
    """
    tspan = np.array([t0, t0+del_t])
    n_y = len(m_dict['y_vec'])
    n_u = len(m_dict['u_vec'])
    out, mu = cFBA(m_dict, tspan, y_start=y_start, wvec=wvec,
                   verbosity_level=verbosity_level, mumin=mumin, mumax=mumax) 
    return {'y': np.reshape(out['y_data'][0, :], (n_y, 1)),
            'u': np.reshape(out['u_data'][0, :], (n_u, 1))}, mu



def cFBA(m_dict, tspan, verbosity_level=0, mumin=1.0, mumax=5.0, y_start=None, y_end_normed=None, wvec=None):
    """
    solve:      min/max mu 
          s.t. m_dict-problem is feasible with
          mu*matrix_start*y0 + matrix_end=y_end = vec_bndry
          [+ wvec^T*y0 == 1 if wvec is not None]
          [+ y(tspan[0]) == y_start if y_start is not None]
          [[+ y(tspan[-1]) == mu*y_end_normed if y_end_normed is not None]]
    #
    TODO: - include index sets j_start, j_end (or something similar)
    - if full start/final values are provided, there is no need for bisection (use Phi_3 and an
      extra variable, but deFBA cannot deal with this yet)
    """
    out = None
    muend = None
    murange = (mumin, mumax)
    m_dict_cFBA = m_dict.copy() # please no side effects here
    n_y = len(m_dict['y_vec'])
    eye_ny, zer_ny = np.eye(n_y), np.zeros((n_y, n_y))
    # set matrices (same in (a), (b), (c))
    m_dict_cFBA['matrix_start'] = np.vstack([eye_ny, zer_ny])
    m_dict_cFBA['matrix_end'] = np.vstack([zer_ny, eye_ny])
    if y_start is None:
        if y_end_normed is None:
            # case (d)
            if wvec is None:
                wvec = np.ones((n_y, 1))
            m_dict_cFBA['matrix_start'] = np.vstack([eye_ny, wvec.T])
            m_dict_cFBA['matrix_end'] = np.vstack([-eye_ny, np.zeros((1, n_y))])
            m_dict_cFBA['vec_bndry'] = np.vstack([np.zeros((n_y, 1)), 1.0])
        else:
            # We get y_N = mu*y_end_normed
            if wvec is not None:
                raise ValueError('Cannot provide both weight vector and final value in cFBA.')
            else:
                # case (c)
                m_dict_cFBA['vec_bndry'] = np.vstack([y_end_normed, y_end_normed])
    else:
        if wvec is not None:
            raise ValueError('Cannot provide both weight vector and initial value in cFBA.')
        else:
            if y_end_normed is None:
                # case (b)
                m_dict_cFBA['vec_bndry'] = np.vstack([y_start, y_start])
            else:
                # case (a)
                m_dict_cFBA['vec_bndry'] = np.vstack([y_start, y_end_normed])
    # Actual run of the algorithm ------------ 
    out, muend = _cFBA(m_dict_cFBA, tspan, verbosity_level, murange)
    return out, muend


def _cFBA(m_dict, tspan, verbosity_level, murange):
    """
    solve min/max mu
    s.t. m_dict problem is feasible with
         'matrix_start'*y0 0 1/mu*'matrix_end'*y_N == 'vec_bndry'
    """
    out = None
    muend = None
    m_dict['phi1'] *= 0.0
    m_dict['phi2'] *= 0.0
    m_dict['phi3'] *= 0.0
    #
    tmp_m_e = m_dict['matrix_end'].copy()
    #print(tmp_m_e)
    #
    def _bisect_fun_cFBA(mu):
        nonlocal muend
        nonlocal out
        if verbosity_level > 0:
            print('trying mu = ', mu)
        m_dict['matrix_end'] = 1/mu*tmp_m_e
        if verbosity_level > 9:
            print(m_dict['matrix_start'])
            print(m_dict['matrix_end'])
            print(m_dict['vec_bndry'])
        out_trial = deFBA(m_dict, tspan)
        if out_trial['y_data'] is None:
            if verbosity_level > 1:
                print('no solutions')
            return -1.0
        else:
            muend = mu
            out = out_trial.copy() # This assumes that the lase call is the best(!)
            if verbosity_level > 1:
                print('found solution')
            return 1.0
    # run bisection method
    bisect(_bisect_fun_cFBA, murange[0], murange[-1], xtol=BISECT_TOL, rtol=BISECT_REL_TOL)
    # ----------------------------------------
    if verbosity_level > 0:
        print('mu =', muend)
    return out, muend


def constraint_fulfillment(out, m_dict, verbosity_level=0, plot_kind = 'mixed'):
    """
    This needs some additional filters and/or output control later on
    """
#['y_vec', 'u_vec', 'x_vec',
# 'phi1', 'phi1u', 'phi2', 'phi3',
# 'smat1', 'smat2', 'smat3', 'smat4', 'f_1', 'f_2',
# 'lbvec', 'ubvec',
# 'matrix_u', 'matrix_y', 'vec_h', 'mixed_names',
# 'matrix_start', 'matrix_end', 'vec_bndry', 'matrix_u_start', 'matrix_u_end'
#]
    def _eval_callable(mat, t):
        if callable(mat):
            return mat(t)
        else:
            return mat

    # 
    plot_kinds = ['flux_bounds', 'mixed', 'positivity', 'dyn', 'qssa']
    #
    mats = Matrrrices(None, **m_dict)
    #
    out_names = []
    t_vals = out['tgrid']
    n_t = len(t_vals)
    t_u_vals = out['tgrid_u']
    y_vals = out['y_data']
    u_u_vals = out['u_data']
    # Interpolation of controls
    n_u = u_u_vals.shape[1]
    u_vals = np.zeros((n_t, n_u))
    for u_index in range(n_u):
        u_vals[:, u_index] = np.interp(x = t_vals, xp = t_u_vals, fp = u_u_vals[:, u_index])
    #

    if plot_kind not in plot_kinds:
        print('Unknown plot type: ', plot_kind)
        return None
    if plot_kind == 'flux_bounds':
        outval = np.zeros((n_t, 2*n_u))
        for i, t in enumerate(t_vals):
            outval[i, 0:n_u] = ( _eval_callable(m_dict['ubvec'], t).T ) - u_vals[i, :]
            outval[i, n_u:] = u_vals[i, :] - ( _eval_callable(m_dict['lbvec'], t).T )
        out_names = ['ub_'+u for u in m_dict['u_vec']] + ['lb_'+u for u in m_dict['u_vec']]
    elif plot_kind == 'mixed':
        outval = np.zeros((n_t, mats.n_mix))
        for i, t in enumerate(t_vals):
            tmp = _eval_callable(m_dict['vec_h'], t).T  - \
                  np.array(_eval_callable(m_dict['matrix_u'], t).todense()).dot( u_vals[i, :])  - \
                  np.array(_eval_callable(m_dict['matrix_y'], t).todense()).dot( y_vals[i, :])
            outval[i, :] = tmp
        out_names = ['mixed_'+str(i) for i in range(tmp.shape[1])]
            #0 <= 'vec_h' - matrix_u * u - matrix_y * y
    elif plot_kind == 'positivity':
        outval = out['y_data']
        out_names = m_dict['y_vec']
        #return 0
    elif plot_kind == 'dyn':
        return 0
    return outval, out_names



    
def relative_tmin_deFVA(prob, yinit, ygoal, ext_cond, tast, constants, wvec, n_steps=100,
                        verbosity_level=0, #J0=None, Jend=None,
                        t0=0.0, t_extend_trials=()):
    """
    Go through the elements of y_vec as defined by "prob", min/max the integral over this
    entry on the interval [t0, tast] and then compute the concentrations relative to biomass 
    wvec'*y.
    #
    TODO: - Formulate in terms of matrrrices and not m_dicts
    - tidy up, remove code duplication
    #
    QUESTION: filtering techniques? (e.g. not through all elements of the vector?)
    """
    #
    #def __int_rel_deFBA():    
    #    tgrid = np.linspace(t0, tast)
    #    MM = Matrrrices(None, run_rdeFBA=False, **m_dict)
    #    rkm = DEFAULT_RKM
    #    out = oc.cp_rk_linprog_v(MM, rkm, tgrid, varphi=0.0)
    #    return out
    y_vec, m_vec, u_vec = prob.create_name_vectors()
    n_y = len(y_vec)
    #
    m_dict = create_m_dict(prob, ext_cond, constants)
    m_dict['phi1'] *= 0.0
    m_dict['phi2'] *= 0.0
    m_dict['phi3'] *= 0.0
    #
    #tspan = np.linspace(t0, tast, n_steps+1)
    # yinit/ygoal
    m_dict['matrix_start'] = np.vstack([np.eye(n_y), np.zeros((n_y, n_y))])
    m_dict['matrix_end'] = np.vstack([np.zeros((n_y, n_y)), np.eye(n_y)])
    m_dict['vec_bndry'] = np.vstack([yinit, ygoal])
    # objective       
    obj_vec = np.zeros((n_y, 1))
    #
    y_min_rel, y_max_rel = np.zeros((n_steps+1, n_y)), np.zeros((n_steps+1, n_y))
    #
    # actual calculations
    rkm = DEFAULT_RKM
    t_extend_trials = np.unique(np.sort(np.hstack([0.0, np.array(t_extend_trials)])))
    for i in range(n_y):
        if verbosity_level > 1:
            print(f'in deFVA, i = {i} -------------')
        obj_vec *= 0.0
        # min int y_i(t) dt ---------------------------------------------
        obj_vec[i, 0] = 1.0
        m_dict['phi1'] = obj_vec
        if verbosity_level > 1:
            print(f'min y_{i}')
        for t_add in t_extend_trials:
            tgrid = np.linspace(t0, tast+t_add, n_steps+1)
            MM = Matrrrices(None, run_rdeFBA=False, **m_dict)
            out = oc.cp_rk_linprog_v(MM, rkm, tgrid, varphi=0.0)
            if out['tgrid'] is not None:
                if verbosity_level > 3:
                    print(f'out at t_add = {t_add}')
                break
        if out['tgrid'] is None:
            print('No solution found here')
            # TODO: Do something about this!!!
        # Now, add integral constraint and maximize biomass
        m_dict_biomass = m_dict.copy()
        m_dict_biomass['phi1'] = wvec
        # TODO: Here, we assume that there were no integral constraints before,
        # TODO: Work on matrrrices with according methods to add constraints 
        m_dict_biomass['bmati_int_y'] = obj_vec.T #': (('ni_bndry', 'n_y'), (np.ndarray, sp.csr_matrix)),
        m_dict_biomass['veci_bndry'] = np.array([[out['obj_result']]]) #': (('ni_bndry', 1), (np.ndarray, sp.csr_matrix))
        #
        if verbosity_level > 2:
            print('max biomass')
        for t_add in t_extend_trials:
            tgrid = np.linspace(t0, tast+t_add, n_steps+1)
            MM = Matrrrices(None, run_rdeFBA=False, **m_dict_biomass)
            out = oc.cp_rk_linprog_v(MM, rkm, tgrid, varphi=0.0)
            if out['tgrid'] is not None:
                if verbosity_level > 3:
                    print(f'out at t_add = {t_add}')
                break
        if out['tgrid'] is None:
            print('No solution found here')
            # TODO: Do something about this!!!
        #
        bio_out = out['y_data'].dot(wvec)
        y_min_rel[:, i] = wvec[i]*(out['y_data'][:, i]/bio_out[:, 0])
        # TODO: The next is code doubling
        # max int y_i(t) dt ------------------------
        obj_vec[i, 0] = -1.0
        m_dict['phi1'] = obj_vec
        if verbosity_level > 1:
            print(f'max y_{i}')
        for t_add in t_extend_trials:
            tgrid = np.linspace(t0, tast+t_add, n_steps+1)
            MM = Matrrrices(None, run_rdeFBA=False, **m_dict)
            out = oc.cp_rk_linprog_v(MM, rkm, tgrid, varphi=0.0)
            if out['tgrid'] is not None:
                if verbosity_level > 3:
                    print(f'out at t_add = {t_add}')
                break
        if out['tgrid'] is None:
            print('No solution found here')
            # TODO: Do something about this!!!
        # Now, add integral constraint and minimize biomass
        m_dict_biomass = m_dict.copy()
        m_dict_biomass['phi1'] = -wvec
        # TODO: Here, we assume that there were no integral constraints before,
        # TODO: Work on matrrrices with according methods to add constraints 
        m_dict_biomass['bmati_int_y'] = -obj_vec.T #': (('ni_bndry', 'n_y'), (np.ndarray, sp.csr_matrix)),
        m_dict_biomass['veci_bndry'] = -np.array([[out['obj_result']]]) #': (('ni_bndry', 1), (np.ndarray, sp.csr_matrix))
        #
        if verbosity_level > 2:
            print('max biomass')
        for t_add in t_extend_trials:
            tgrid = np.linspace(t0, tast+t_add, n_steps+1)
            MM = Matrrrices(None, run_rdeFBA=False, **m_dict_biomass)
            out = oc.cp_rk_linprog_v(MM, rkm, tgrid, varphi=0.0)
            if out['tgrid'] is not None:
                if verbosity_level > 3:
                    print(f'out at t_add = {t_add}')
                break
        if out['tgrid'] is None:
            print('No solution found here')
            # TODO: Do something about this!!!
        #
        bio_out = out['y_data'].dot(wvec)
        y_max_rel[:, i] = wvec[i]*(out['y_data'][:, i]/bio_out[:,0])
    #
    return y_min_rel, y_max_rel, {} # info_deFVA



def _medium_level_deFVA(m_dict, tspan, verbosity_level=0, varphi=0.0, fva_level=1, **kwargs):
    """
    deFVA with some more control than just playing with the plain LPs
    TODO: FINISH!!!!!
    """
    m_dict_medlevdeFVA = m_dict.copy()
    #
    #
    out_deFBA = deFBA(m_dict, tspan)
    print(out_deFBA['obj_result'])
    #
    return None



def low_level_deFVA(out, y_vec, verbosity_level=0, varphi=0.0, fva_level=3, **kwargs):
    """
    a very simple deFVA
    """
    lp_model = out['model']
    tgrid = out['tgrid']
    if verbosity_level > 3:
        print( lp_model )
    n_y = len(y_vec)
    n_steps = len(tgrid) - 1
    y_indices= kwargs.get('y_indices', range(n_y))
    minmax_choice = kwargs.get('minmax', ('min', 'max')) # min, max, TODO: mark invalid choices
    #n_ally = (n_steps + 1) * n_y
    #
    y_min_all, y_max_all = np.nan*np.ones((n_steps+1, n_y)), np.nan*np.ones((n_steps+1, n_y))
    y_ex_one, y_ex_two = np.nan*np.ones((n_steps+1, 1)), np.nan*np.ones((n_steps+1, 1))
    # define new constraint: Don't be worse than originally
    old_fvec = lp_model.get_objective_vector()
    old_obj_val = lp_model.get_objective_val()
    sense = '<'
    #lp_model.solver_model.write('pre.lp')
    relaxation_constants = (1e-6, 1e-6)
    #relaxation_constants = (0.0, 0.3)
    to_set_obj_val = old_obj_val*(1+np.sign(old_obj_val)*relaxation_constants[0]) + relaxation_constants[1]
    lp_model.add_constraints(sp.csr_matrix(old_fvec.T), np.array([[ to_set_obj_val]]), sense)
    #
    if fva_level == 3:
        # This is "INSANE" mode: Optimize each and every time point twice
        # TODO: Include the "new" filtering options
        for k in y_indices:
            if verbosity_level >= 2:
                print(f'In deFVA, max-/minimizing {y_vec[k]}')
            for m in range(n_steps+1):
                xfrakvec = np.zeros((len(lp_model.variable_names), 1))
                interessant_index = m*n_y+k
                if 'min' in minmax_choice:
                    xfrakvec[interessant_index, 0] = 1.0
                    lp_model.set_new_objective_vector(xfrakvec)
                    if verbosity_level >= 3:
                        print('t = ', tgrid[m])
                    lp_model.optimize()
                    y_min_all[m, k] = lp_model.get_solution()[interessant_index] if lp_model.get_solution() is not None else DEFVA_DEFAULT
                #
                if 'max' in minmax_choice:
                    xfrakvec[interessant_index, 0] = -1.0
                    lp_model.set_new_objective_vector(xfrakvec)
                    if verbosity_level >= 3:
                        print('t = ', tgrid[m], '(2)')
                    lp_model.optimize()
                    y_max_all[m, k] = lp_model.get_solution()[interessant_index] if lp_model.get_solution() is not None else DEFVA_DEFAULT
    elif fva_level == 2:
        # max/min int exp(+- varphi*t)*y dt
        for k in y_indices:
            if verbosity_level >= 2:
                print(f'In deFVA, max-/minimizing {y_vec[k]}')
            # setup minimization 
            xfrakvec = np.zeros((len(lp_model.variable_names), 1))
            use_y_indices = []
            for i, var_name in enumerate(lp_model.variable_names):
                if var_name.startswith('y_' + str(k+1)):
                    use_y_indices.append(i)
            if 'min' in minmax_choice:
                # (A) min int exp(-varphi*t)*y(t) dt
                xfrakvec[use_y_indices, 0] = np.exp(-varphi*tgrid)
                lp_model.set_new_objective_vector(xfrakvec)
                lp_model.optimize()
                #if WRITE_INFEAS_TO_FILE:# DEBUG
                #    lp_model.write_to_file(f'defva_{y_vec[k]}_A.lp')# DEBUG
                try:
                    y_ex_one = lp_model.get_solution()[use_y_indices] # FIXME: What if no sol?
                except:
                    print(f'no sol in deFVA (min exp(-phi)) {y_vec[k]}')
                    if WRITE_INFEAS_TO_FILE:
                        lp_model.write_to_file(f'defva_{y_vec[k]}_A.lp')
                    y_ex_one = [np.nan for _ in use_y_indices]
                # (B) min int exp(+varphi*t)*y(t) dt
                xfrakvec[use_y_indices, 0] = np.exp(+varphi*tgrid)
                lp_model.set_new_objective_vector(xfrakvec)
                lp_model.optimize()
                try:
                    y_ex_two = lp_model.get_solution()[use_y_indices] # FIXME: What if no sol?
                except:
                    print(f'no sol in deFVA (min exp(phi)) {y_vec[k]}')
                    if WRITE_INFEAS_TO_FILE:
                        lp_model.write_to_file(f'defva_{y_vec[k]}_B.lp')
                    y_ex_two = [np.nan for _ in use_y_indices]
            y_min_all[:, k] = [min(a, b) for a, b in zip(y_ex_one, y_ex_two)]
            # ------------------------------------
            if 'max' in minmax_choice:
                # (C) max int exp(-varphi*t)*y(t) dt
                xfrakvec[use_y_indices, 0] = -np.exp(-varphi*tgrid)
                lp_model.set_new_objective_vector(xfrakvec)
                lp_model.optimize()
                if WRITE_INFEAS_TO_FILE:# DEBUG
                    lp_model.write_to_file(f'defva_{y_vec[k]}_C.lp')# DEBUG
                try:
                    y_ex_one = lp_model.get_solution()[use_y_indices] # FIXME: What if no sol?
                except:
                    print(f'no sol in deFVA (max exp(-phi)) {y_vec[k]}')
                    if WRITE_INFEAS_TO_FILE:
                        lp_model.write_to_file(f'defva_{y_vec[k]}_C.lp')
                    y_ex_one = [np.nan for _ in use_y_indices]
                # (D) max int exp(+varphi*t)*y(t) dt
                xfrakvec[use_y_indices, 0] = -np.exp(varphi*tgrid)
                lp_model.set_new_objective_vector(xfrakvec)
                lp_model.optimize()
                try:
                    y_ex_two = lp_model.get_solution()[use_y_indices] # FIXME: What if no sol?
                except:
                    print(f'no sol in deFVA (max exp(phi)) {y_vec[k]}')
                    if WRITE_INFEAS_TO_FILE:
                        lp_model.write_to_file(f'defva_{y_vec[k]}_D.lp')
                    y_ex_two = [np.nan for _ in use_y_indices]
            y_max_all[:, k] = [max(a, b) for a, b in zip(y_ex_one, y_ex_two)]
    elif fva_level==1:
        # max/min int y_i dt
        for k in y_indices:
            if verbosity_level >= 2:
                print(f'In deFVA, max-/minimizing {y_vec[k]}')
            # setup minimization 
            xfrakvec = np.zeros((len(lp_model.variable_names), 1))
            use_y_indices = []
            for i, var_name in enumerate(lp_model.variable_names):
                if var_name.startswith('y_' + str(k+1)):
                    use_y_indices.append(i)
            if 'min' in minmax_choice:
                # (A) min int exp(-varphi*t)*y(t) dt
                xfrakvec[use_y_indices, 0] = 1.0
                lp_model.set_new_objective_vector(xfrakvec)
                lp_model.optimize()
                try:
                    y_min_all[:, k] = lp_model.get_solution()[use_y_indices] # FIXME: What if no sol?
                except:
                    print(f'no sol in deFVA (min int {y_vec[k]} dt)')
                    if WRITE_INFEAS_TO_FILE:
                        lp_model.write_to_file(f'defva_{y_vec[k]}_min.lp')
                    y_min_all[:, k] = [np.nan for _ in use_y_indices]
            # ------------------------------------
            if 'max' in minmax_choice:
                # max int y(t) dt
                xfrakvec[use_y_indices, 0] = -1.0
                lp_model.set_new_objective_vector(xfrakvec)
                lp_model.optimize()
                try:
                    y_max_all[:, k] = lp_model.get_solution()[use_y_indices] # FIXME: What if no sol?
                except:
                    print(f'no sol in deFVA (max int {y_vec[k]} dt)')
                    if WRITE_INFEAS_TO_FILE:
                        lp_model.write_to_file(f'defva_{y_vec[k]}_max.lp')
                    y_max_all[:, k] = [np.nan for _ in use_y_indices]
            '''
                    #xfrakvec[i, 0] = 1.0*discount_factors[j]
                    #xfrakvec[i, 0] = 1.0*np.exp(-varphi*tgrid[j])
                    #j += 1
            lp_model.set_new_objective_vector(xfrakvec)
            lp_model.solver_model.update()
            #lp_model.solver_model.write('post.lp')
            #return None
            lp_model.optimize()
            y_k_new = np.reshape(lp_model.get_solution()[:n_ally], (n_steps+1, n_y))[:, k]
            y_min_all[:, k] = y_k_new
            # setup maximization 
            xfrakvec = np.zeros((len(lp_model.variable_names), 1))
            j = 0
            for i, var_name in enumerate(lp_model.variable_names):
                if var_name.startswith('y_' + str(k)):
                    #xfrakvec[i, 0] = 1.0*discount_factors[j]
                    xfrakvec[i, 0] = -1.0*np.exp(varphi*tgrid[j])
                    j += 1
            print(xfrakvec.T)
            lp_model.set_new_objective_vector(xfrakvec)
            lp_model.solver_model.update()
            lp_model.optimize()
            y_k_new = np.reshape(lp_model.get_solution()[:n_ally], (n_steps+1, n_y))[:, k]
            y_max_all[:, k] = y_k_new
            '''
    return y_min_all, y_max_all








