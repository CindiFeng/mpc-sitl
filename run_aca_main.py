# # # -*- coding: future_fstrings -*-

from acados_template import AcadosModel
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
import numpy as np
import casadi as ca

import aca_init as init
import aca_mpc_fcns as mpcFcn
import util_plotCtrl as plot2d


def gen_ocp_model() -> AcadosModel:

    ## Dynamics
    sym_x = ca.SX.sym('x',init.model['n_x']) # system states
    sym_u = ca.SX.sym('u',init.model['n_u']) # control inputs
    sym_xdot = ca.SX.sym('xdot', init.model['n_x'], 1)

    f_expl_expr = mpcFcn.sl_dr_dyn(sym_x,sym_u,init) # rhs of EOM
    f_impl_expr = f_expl_expr - sym_xdot

    ## Cost
    cost_expr_ext_cost = mpcFcn.cost(sym_x,sym_u,init)
    cost_expr_ext_cost_e = mpcFcn.cost_e(sym_x,init)

    ## Constraint 
    con_h_expr = mpcFcn.ineq(sym_x,init)

    # Jbx = np.zeros(init.model['nx']) # specify states affected by linear constraint
    # for i in init.idx['x']['uav_pos']:     
    #     Jbx[init.idx['x']['uav_pos'][i],init.idx['x']['uav_pos'][i]] = 1 

    ## populate structure
    model = AcadosModel()
    
    model.name = init.model['name']

    model.x = sym_x
    model.xdot = sym_xdot
    model.u = sym_u
    model.f_expl_expr = f_expl_expr
    model.f_impl_expr = f_impl_expr

    model.cost_expr_ext_cost = cost_expr_ext_cost
    model.cost_expr_ext_cost_e = cost_expr_ext_cost_e

    model.con_h_expr = con_h_expr
    model.con_h_expr_e = con_h_expr
    # model.Jbx = Jbx

    return model

def main():
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    model = gen_ocp_model()
    ocp.model = model

    N = init.params['control']['predictionHorizon'] # number of shooting intervals
    Ts = init.params['control']['sampleTime'] # sampling time 
    Tf = N * Ts; # prediction horizon time length

    nx = init.model['n_x']
    nu = init.model['n_u']

    # set dimensions
    ocp.dims.N = N

    # set cost module
    ocp.cost.cost_type = init.ocp['cost_type']
    ocp.cost.cost_type_e = init.ocp['cost_type_e']

    # set constraints
    lh = np.concatenate((np.ones((2,)),
                         init.sim['workspace'][0,:] + np.array([init.params['arm_len'],init.params['arm_len'],0]),
                         init.sim['workspace'][0,:],
                         np.array([-1])))
    uh = np.concatenate((30 * np.ones((2,)),
                         init.sim['workspace'][1,:] - np.array([init.params['arm_len'],init.params['arm_len'],0]),
                         init.sim['workspace'][1,:],
                         np.array([init.params['cable_len']^2])))
    ocp.constraints.constr_type = init.ocp['constr_type']
    ocp.constraints.lh = lh
    ocp.constraints.uh = uh
    ocp.constraints.constr_type_e = init.ocp['constr_type']
    ocp.constraints.lh_e = lh
    ocp.constraints.uh_e = uh 
    
    x0 = np.concatenate((init.ics['pld_rel_pos'],
                         init.ics['uav_pos'],
                         init.ics['pld_rel_vel'],
                         init.ics['uav_vel']))
    ocp.constraints.x0 = x0

    # solver options
    ocp.solver_options.integrator_type = init.ocp['integrator_type'] # dynamics
    ocp.solver_options.sim_method_num_stages = init.ocp['sim_method_num_stages']
    ocp.solver_options.tf = Tf # or: Ts, shooting_nodes, time_steps
    # time_steps = Ts * np.ones((N,))
    # ocp.solver_options.time_steps = time_steps
    ocp.solver_options.levenberg_marquardt = init.ocp['levenberg_marquardt']
    ocp.solver_options.hessian_approx = init.ocp['hessian_approx']

    ocp.solver_options.nlp_solver_type = init.ocp['nlp_solver_type']
    ocp.solver_options.nlp_solver_max_iter = init.ocp['nlp_solver_max_iter']
    ocp.solver_options.nlp_solver_tol_eq = init.ocp['nlp_solver_tol_eq']
    ocp.solver_options.nlp_solver_tol_ineq = init.ocp['nlp_solver_tol_ineq']
    ocp.solver_options.regularize_method = init.ocp['regularize_method']

    ocp.solver_options.qp_solver = init.ocp['qp_solver'] 
    if init.ocp['qp_solver'] == 'PARTIAL_CONDENSING_HPIPM':
        ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_iter_max = init.ocp['qp_solver_iter_max']

    # create solver class
    solver_json = 'acados_ocp_' + model.name + '.json'
    # ocp_solver = AcadosOcpSolver(ocp, json_file = solver_json, build=False, generate=False)
    ocp_solver = AcadosOcpSolver(ocp, json_file = solver_json)

    # create an integrator with the same settings as used in the OCP solver
    acados_integrator = AcadosSimSolver(ocp, json_file = solver_json)

    Nsim = int(init.sim['duration'] / Ts)
    simX = np.ndarray((Nsim+1, nx))
    simU = np.ndarray((Nsim, nu))
    simXplan = np.ndarray((N, nx, Nsim))

    x_cur = x0
    simX[0,:] = x_cur

    # ocp_solver.print_statistics() # encapsulates: stat = ocp_solver.get_stats("statistics")

    # initialize solver
    for stage in range(N + 1):
        ocp_solver.set(stage, "x", x_cur)

    u0 = -init.params["derived"]["sys_weight"]
    for stage in range(N):
        ocp_solver.set(stage, "u", u0)

    # closed loop
    for i in range(Nsim):
##################################################
        # status = ocp_solver.solve()
        # if status != 0:
        #     raise Exception(f'acados returned status {status}.')

        # # solve ocp and get next control input
        # # avoids the need to set lbx, ubx at node 0
        # simU[i,:] = ocp_solver.solve_for_x0(x0_bar = simX[i, :])

        ocp_solver.set(0, "lbx", x_cur)
        ocp_solver.set(0, "ubx", x_cur)
    
        status = ocp_solver.solve() 
        if status != 0:
            if i > 0: 
                plot2d.plot(simX[:i+1,:],simU[:i,:],init)
            raise Exception(f'acados returned status {status}.')

        
        simU[i,:] = ocp_solver.get(0,"u")
#################################################
        # simulate system
        x_cur = acados_integrator.simulate(x=simX[i, :], u=simU[i,:])
        simX[i+1, :] = x_cur

        # # simulate system and get planned states 
        # acados_integrator.set("x", x_cur)
        # acados_integrator.set("u", simU[i, :])

        # x_cur = acados_integrator.get("x")
        # sim[i+1, :] = x_cur

        # for i_plan in range(1,Nsim+1):
        #     simXplan[i_plan,:,i] = acados_integrator.get(i_plan,"x")

    # plot results
    plot2d.plot(simX,simU,init)

    print('total CPU time in the previous call:', ocp_solver.get_stats('time_tot'))

if __name__ == '__main__':
    main()