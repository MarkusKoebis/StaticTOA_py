"""
Simple flux balance analysis
"""


# -*- coding: utf-8 -*-
import numpy as np
from copy import deepcopy
from scipy.optimize import bisect
from scipy.sparse import csr_matrix#, bmat
from ..simulation.results import Solutions
from .. import matrrrices as mat
from ..util import runge_kutta
from ..optimization.lp import LPModel
from ..optimization.oc import mi_cp_linprog, cp_rk_linprog, cp_rk_linprog_v


def perform_fba(model, **kwargs):
    """
    Classical Flux Balance Analysis by LP solution

    Parameters
    ----------
    model : PYRRRATEFBA Model
        main data structure of the underlying model.
    **kwargs :
        'objective': - None: The biomass reactions will be detected by name.
                     - vector of length #reactions

    Returns
    -------
    TODO
    """
    fvec = kwargs.get("objective", None)
    maximize = kwargs.get("maximize", True)
    #
    # Get QSSA stoichiometric matrix
    smat = model.stoich # FIXME: This is wrong if we have macromolecules in the model,
    # better: choose smat1 from the Matrrrices but then we probably need other flux
    # constraints or have to give an error message
    nrows, ncols = smat.shape
    high_reaction_flux = 1000.0
    lbvec = np.zeros(ncols)  # lower flux bounds
    ubvec = np.zeros(ncols)  # upper flux bounds
    for idx, rxn in enumerate(model.reactions_dict.values()):
        ubvec[idx] = rxn.get('upperFluxBound', high_reaction_flux)
        if rxn['reversible']:
            lbvec[idx] = rxn.get('lowerFluxBound', -high_reaction_flux)
        else:
            lbvec[idx] = rxn.get('lowerFluxBound', 0.0)
    fvec = np.zeros(ncols)

    if fvec is None:
        # biomass reaction indices
        brxns = [idx for idx, reac in enumerate(model.reactions_dict.keys())
                 if 'biomass' in reac.lower()]
        if not brxns:
            print('No biomass reaction found and no objective provided, exiting.')
            return None
        fvec[brxns] = -1.0
    if maximize:
        fvec = -fvec
    # set up the LP
    lp_model = LPModel()
    # fvec, amat, bvec, aeqmat, beq, lbvec, ubvec, variable_names
    lp_model.sparse_model_setup(fvec, csr_matrix((0, ncols)), np.zeros(0),
                                csr_matrix(smat), np.zeros(nrows), lbvec, ubvec,
                                list(model.reactions_dict.keys()))
    lp_model.optimize()
    sol = lp_model.get_solution()
    return sol


def perform_rdefba(model, **kwargs):
    """
    Use (r)deFBA to approximate the dynamic behavior of the model

    Parameters
    ----------
    model : PyrrrateFBAModel
        main biochemical model.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    sol : Solutions instance containing the time series information

    TODO : More options to "play around"
    """
    run_rdeFBA = kwargs.get('run_rdeFBA', True)
    if run_rdeFBA and not model.can_rdeFBA:
        raise ValueError('Cannot run an r-deFBA on this model.')
    t_0 = kwargs.get('t_0', 0.0)
    t_end = kwargs.get('t_end', 1.0)
    n_steps = kwargs.get('n_steps', 51)
    varphi = kwargs.get('varphi', 0.0)
    rkm = kwargs.get('runge_kutta', None)
    enforce_biomass = kwargs.get('enforce_biomass', False)
    #
    mtx = mat.Matrrrices(model, run_rdeFBA=run_rdeFBA)
    # adapt initial values if explicitly given
    y_0 = kwargs.get('set_y0', None)# FIXME: So far, y0 is acceted as row vector only(!?)
    if y_0 is not None:
        mtx.matrix_end = csr_matrix((y_0.size, y_0.size))
        mtx.matrix_start = csr_matrix(np.eye(y_0.size))
        mtx.matrix_u_start = csr_matrix((y_0.size, mtx.n_u))
        mtx.matrix_u_end = csr_matrix((y_0.size, mtx.n_u))
        mtx.vec_bndry = y_0.transpose()
    if enforce_biomass:
        mtx.enforce_biomass(model)
    # Call the OC routine
    if rkm is None:
        tgrid, tt_shift, sol_y, sol_u, sol_x = mi_cp_linprog(mtx, t_0, t_end, n_steps=n_steps,
                                                             varphi=varphi)
    else:
        if run_rdeFBA:
            print('Cannot (yet) run r-deFBA with arbitrary Runge-Kutta scheme. Fallback to deFBA')
        #tgrid, tt_shift, sol_y, sol_u = cp_rk_linprog(mtx, rkm, t_0, t_end, n_steps=n_steps,
        #                                              varphi=varphi)
        out = cp_rk_linprog_v(mtx, rkm, np.linspace(t_0, t_end, n_steps+1), varphi=varphi)
        tgrid, tt_shift, sol_y, sol_u = out['tgrid'], out['tgrid_u'], out['y_data'], out['u_data'] 
        #
        #sol_x = np.zeros((0, sol_u.shape[1]))
        sol_x = np.zeros((sol_u.shape[0], 0))

    sols = Solutions(tgrid, tt_shift, sol_y, sol_u, sol_x)

    return sols


def perform_soa_rdeFBA(model, **kwargs):
    """
    iterative process consisting of several (r)deFBA runs with a very crude one-step
    approximation in each step (quasi-_S_tatic _O_ptimization _A_pproach)
    # MAYBE: This could become a special case of short-term (r-)deFBA
    # QUESTION: Can it be a problem if set_y0 is set from the start?
       (quasi-recursive call of the algorithms...)
    """
    run_rdeFBA = kwargs.get('run_rdeFBA', True)
    n_steps = kwargs.get('n_steps', 51)
    tgrid = np.linspace(kwargs.get('t_0', 0.0), kwargs.get('t_end', 1.0), n_steps)
    varphi = kwargs.get('varphi', 0.0)
    kwargs.pop('varphi', kwargs)
    #
    mtx = mat.Matrrrices(model, run_rdeFBA=run_rdeFBA)
    y_0 = mtx.extract_initial_values()
    if y_0 is None:
        print('SOA (r)deFBA cannot be perforrrmed.')
        return None
    kwargs['set_y0'] = y_0
    kwargs['n_steps'] = 1
    #
    tslice = tgrid[0:2]
    sols = model.rdeFBA(tslice, varphi, do_soa=False, **kwargs)
    y_new = np.array(sols.dyndata.tail(n=1)) # row
    for k in range(1, n_steps-1):
        kwargs['set_y0'] = y_new # row
        tslice = tgrid[k:k+2]
        #print(tslice)# DEBUG
        try:
            sol_tmp = model.rdeFBA(tslice, varphi, do_soa=False, **kwargs)
        # TODO: Find a more elegant solution than try
            new_t_shift = sol_tmp.condata.index[-1]
            ux_new = np.array(sol_tmp.condata.tail(n=1))
            y_new = np.array(sol_tmp.dyndata.tail(n=1))
            sols.extend_y([tslice[-1]], y_new)
            sols.extend_ux([new_t_shift], ux_new)
        except:
            print(f'Could not extend SOA solution beyond t = {tslice[0]}')
            return sols
    #print(sols)
    return sols




# _EXPERIMENTAL ######################

BISECT_TOL, BISECT_REL_TOL = 2e-12, 8.881784197001252e-16 # DEFAULT VALUES
#BISECT_TOL, BISECT_REL_TOL = 1e-8, 1e-12


def deFBA(MM, tspan, varphi=0.0,rkm=runge_kutta.RungeKuttaPars(s=1,family='Explicit1')):
    out = cp_rk_linprog_v(MM, rkm, tspan, varphi=varphi)
    return out
  

def cFBA(MM, tspan, verbosity_level=0, mumin=1.0, mumax=5.0, y_start=None, wvec=None):
    """
    solve:      min/max mu 
          s.t. m_dict-problem is feasible with
          mu*matrix_start*y0 + matrix_end=y_end = vec_bndry
          + wvec^T*y0 == 1
          [+ y(tspan[0]) == y0 if y0 is not None]
    TODO: include index sets j_start, j_end (or something similar)
    """
    out = None
    muend = None
    MM_cFBA = deepcopy(MM) # avoid side effects on model
    n_y = MM_cFBA.n_y
    MM_cFBA.phi1 *= 0.0
    MM_cFBA.phi2 *= 0.0
    MM_cFBA.phi3 *= 0.0
    #
    if y_start is None:
        m_start = np.eye(n_y, n_y) # TODO: Sparse support
        m_end = -np.eye(n_y, n_y)
        v_bndry = np.zeros((n_y, 1))
        if wvec is None:
            wvec = np.ones((n_y, 1))
    else:# TODO: It is no longer necessary to use bisection if y_start is provided completely, can be dealt with using Phi_3 and an extra variable
        if wvec is not None:
            raise ValueError('Cannot provide both weight vector and initial value in cFBA.')
        else:
            wvec = np.zeros((n_y, 0))
            m_start = np.vstack([np.eye(n_y, n_y), np.eye(n_y, n_y)])
            m_end = np.vstack([-np.eye(n_y, n_y), np.zeros((n_y, n_y))])
            v_bndry = np.vstack([np.zeros((n_y, 1)), y_start])
    m_wvec= wvec.shape[1]
    tmp_m_s = np.vstack([m_start, wvec.T])
    MM_cFBA.matrix_start = tmp_m_s.copy() #np.vstack([m_start, wvec.T])
    #print('pre ---------')
    #print(m_dict_cFBA['matrix_start'], '\n', m_dict_cFBA['matrix_end'], '\n', m_dict_cFBA['vec_bndry'])
    MM_cFBA.matrix_end = np.vstack([m_end, np.zeros((m_wvec, n_y))])
    MM_cFBA.vec_bndry = np.vstack([v_bndry, np.ones((m_wvec, 1))])    
    # TODO: Check for entries in matrix_bndry_p first and provide warning if necessary
    MM_cFBA.matrix_bndry_p = np.zeros((MM_cFBA.matrix_start.shape[0], MM_cFBA.n_p))
    #
    #print('post ---------')
    #print(m_dict_cFBA['matrix_start'], '\n', m_dict_cFBA['matrix_end'], '\n', m_dict_cFBA['vec_bndry'])
    #
    def _bisect_fun_cFBA(mu):
        nonlocal muend
        nonlocal out
        if verbosity_level > 0:
            print('trying mu = ', mu)
        MM_cFBA.matrix_start[:n_y, :] = mu*tmp_m_s[:n_y, :]
        out_trial = deFBA(MM_cFBA, tspan)
        if out_trial['y_data'] is None:
            if verbosity_level > 1:
                print('no solutions')
            return -1.0
        else:
            muend = mu
            out = out_trial.copy() # This assumes that the lase call is the best(!)
            if verbosity_level > 1:
                print('found solution')
                #print(out['y_data'])
            return 1.0
    # Actual run of the algorithm ------------
    #muend = 
    bisect(_bisect_fun_cFBA, mumin, mumax, xtol=BISECT_TOL, rtol=BISECT_REL_TOL) # |
    # ----------------------------------------
    if verbosity_level > 1:
        print('mu =', muend)
    #out = deFBA(m_dict_cFBA, tspan)
    return out, muend


def RBA_like(MM, t0, del_t, verbosity_level=0, mumin=1.0, mumax=2.0, y_start=None, wvec=None):
    """
    solve min/max mu
          s.t. Matrrrices model MM is feas. and
          mu*y0 == yend 
    """
    tspan = np.array([t0, t0+del_t])
    n_y = MM.n_y
    n_u = MM.n_u
    out, mu = cFBA(MM, tspan, y_start=y_start, wvec=wvec,
                   verbosity_level=verbosity_level, mumin=mumin, mumax=mumax) 
    return {'y': np.reshape(out['y_data'][0, :], (n_y, 1)),
            'u': np.reshape(out['u_data'][0, :], (n_u, 1)),
            'last_LP': out['model']}, mu




'''
def perform_cFBA(model, runge_kutta, **kwargs):
    """
    conditional FBA (So far: quick'n'dirty)
    """
    run_rdeFBA = kwargs.get('run_rdeFBA', False)
    if run_rdeFBA:
        raise NotImplementedError('r-cFBA not available')
    t_0 = kwargs.get('t_0', 0.0)
    t_end = kwargs.get('t_end', 1.0)
    n_steps = kwargs.get('n_steps', 51)
    tspan = np.linspace(t_0, t_end, n_steps)
    #varphi = kwargs.get('varphi', 0.0)
    mumin = kwargs.get('mumin', 1.0)
    mumax = kwargs.get('mumax', 5.0)
    verbosity_level = kwargs.get('verbosity_level', 0)
    #
    mtx = mat.Matrrrices(model, run_rdeFBA=run_rdeFBA)
    #
    muend = None
    #
    # bisect(f,a,b,
    # args=(), xtol=2e-12, rtol=8.881784197001252e-16, maxiter=100, full_output=False, disp=True)
    #
    def _bisect_fun(mu):
        nonlocal muend
        if verbosity_level > 0:
            print('trying mu = ', mu)
        out = _internal_cFBA(mtx, tspan, mu)
        if out[0] is None:
            if verbosity_level > 1:
                print('no solutions')
            return -1.0
        else:
            muend = mu
            if verbosity_level > 1:
                print('found solution')
            return 1.0
    bisect(_bisect_fun, mumin, mumax)
    if verbosity_level > 1:
        print('mu =', muend)
    out = _internal_cFBA(mtx, tspan, muend)# TODO: This is double evaluation
    return out, muend


def _internal_cFBA(mats, tspan, mu):
    """
    """
    n_y, n_u = len(mats.y_vec), len(mats.u_vec)
    # CAUTION: SIDE EFFECT: This overwrites the actual values in the matrrrices instance
    mats.phi1 = np.zeros((n_y, 1))
    mats.phi2 = mats.phi1
    mats.phi3 = mats.phi1
    mats.phi1u = np.zeros((n_u, 1))
    # 
    # Append condition that sum of amounts at t0 is 1 -------------------START
    ee = np.ones((1, n_y))
    matrix_start = bmat([[csr_matrix(np.eye(n_y))], [ee]], format='csr')
    # FIXME: Continue here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    matrix_end   = np.vstack([m_dict['matrix_end'], np.zeros((1, n_y))])
    vec_bndry    = np.vstack([m_dict['vec_bndry'], 1.0])
    m_dict['matrix_start'] = matrix_start
    m_dict['matrix_end'] = matrix_end
    m_dict['vec_bndry'] = vec_bndry
    # ---------------------------------------------------------------------END
    #print(m_dict['matrix_start'],m_dict['matrix_end'],m_dict['vec_bndry'])
    MM = Matrrrices(None, run_rdeFBA=False, **m_dict)
    varphi  = 0.0
    out = oc.cp_rk_linprog_v(MM, rkm, tspan, varphi=varphi)
    return out
'''

#def perform_shortterm_rdeFBA(model, **kwargs):
#    """
#    short-term (r-)deFBA
#    """
#    t_0 = kwargs.get('t_0', 0.0)
