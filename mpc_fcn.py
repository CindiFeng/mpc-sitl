#!/usr/bin/env python2
import numpy as np
import casadi as ca
import init
import util_fcn as util

def slungLoadDyn(x,u):
    m_p = init.params["pld_mass"]
    m_q = init.params["uav_mass"]
    g = init.sim["g"]
    L = init.params["cable_len"]

    # translation states 
    r_L = x[init.idx["x"]["pld_rel_pos"][0]:init.idx["x"]["pld_rel_pos"][1]]
    x_q = x[init.idx["x"]["uav_pos"][0]:init.idx["x"]["uav_pos"][1]]
    v_L = x[init.idx["x"]["pld_rel_vel"][0]:init.idx["x"]["pld_rel_vel"][1]]
    v_q = x[init.idx["x"]["uav_vel"][0]:init.idx["x"]["uav_vel"][1]]
    v = ca.vertcat(v_L, v_q)

    # Forces and input vector
    F_I = u
    F = ca.vertcat(ca.SX.zeros(2,1), F_I)
    g_I= ca.vertcat(ca.SX.zeros(2,1),-g)

    # Geometric Relations
    B_hat = ca.vertcat(ca.SX.eye(2), r_L.T / ca.sqrt(L**2 - r_L.T @ r_L))
    B_hat_dot = ca.vertcat(ca.SX.zeros(2,2), -((v_L @ (r_L.T @ r_L-L**2)-r_L @ r_L.T @ v_L)/(L**2-r_L.T @ r_L)**(3/2)).T)

    # System Matrices
    G = ca.vertcat(m_p * B_hat.T @ g_I, (m_p+m_q)*g_I)
    C = ca.vertcat(ca.horzcat(m_p * B_hat.T @ B_hat_dot, ca.SX.zeros(2,3)),
                ca.horzcat(m_p * B_hat_dot, ca.SX.zeros(3,3)))
    M = ca.vertcat(ca.horzcat(m_p * (B_hat.T @ B_hat), m_p*B_hat.T),
                ca.horzcat(m_p * B_hat, (m_q + m_p) * ca.SX.eye(3)))

    # Nonlinear EOM 
    v_dot = ca.solve(M, (F + G - C @ v)) # in MATLAB it's A\b
    dxdt = ca.vertcat(v, v_dot)

    return dxdt

def ineqConFcn(x_cur):
    """ 
    Custom inequality constraints to supplements standard linear 
    constraints. Includes hard constraints on states to avoid obstacles. 
    
    INPUT: x_cur, column vector of
        - payload relative position (r_p)
        - uav position (x_q)
        - payload relative velocity (v_p)
        - uav velocity (v_q)
    """
    # distance to obstacle
    semi_prin_axis = 0.5 * ca.sqrt(3) * init.sim["obs_dim"]
    ellipsoid_weight = np.diag(1 / (semi_prin_axis + init.params["obstacle"]["bo"])**2)

    uav_pos = x_cur[init.idx["x"]["uav_pos"][0]:init.idx["x"]["uav_pos"][1]]
    pld_rel_pos = x_cur[init.idx["x"]["pld_rel_pos"][0]:init.idx["x"]["pld_rel_pos"][1]]
    pld_abs_pos = uav_pos + ca.vertcat(pld_rel_pos, 
                                       -ca.sqrt(init.params["cable_len"]**2 - \
                                                pld_rel_pos.T @ pld_rel_pos))

    uav_obs_vec = uav_pos - init.sim["obs_pos"]
    uav_obs_dist = uav_obs_vec.T @ ellipsoid_weight @ uav_obs_vec - 1
    pld_obs_vec = pld_abs_pos - init.sim["obs_pos"]
    pld_obs_dist = pld_obs_vec.T @ ellipsoid_weight @ pld_obs_vec - 1

    # payload distance
    pld_rel_dist = pld_rel_pos.T @ pld_rel_pos

    cineq = ca.vertcat(pld_obs_dist, uav_obs_dist, pld_abs_pos, pld_rel_dist)

    return cineq 

def costFcn(x_cur,u_cur,u_prev,P):
    """ 
    Abridged custom cost function based on Potdar 2020 paper
    INPUT:  x_cur, column state vector
            - payload relative position (r_p)
            - uav position (x_q)
            - payload relative velocity (v_p)
            - uav velocity (v_q)

            u_cur, column input vector
            - 3x lift force (F_L) vector in NED frame

            P, column x0 and xref vector
    """

    # assign variables
    uav_pos = x_cur[init.idx["x"]["uav_pos"][0]:init.idx["x"]["uav_pos"][1]]
    pld_rel_pos = x_cur[init.idx["x"]["pld_rel_pos"][0]:init.idx["x"]["pld_rel_pos"][1]]
    x_goal = P[init.idx["p"]["x_goal"][0]:init.idx["p"]["x_goal"][1]]
    uav_goal = x_goal[init.idx["x"]["uav_pos"][0]:init.idx["x"]["uav_pos"][1]]
    x_start = P[init.idx["p"]["x0"][0]:init.idx["p"]["x0"][1]]
    uav_start = x_start[init.idx["x"]["uav_pos"][0]:init.idx["x"]["uav_pos"][1]]

    denom = util.ca_sq_norm(uav_goal - uav_start) # cannot start at goal!!!
    cost_nav = init.params["control"]["cost_nav"] * util.ca_sq_norm(uav_goal-uav_pos) / denom

    cost_nav_k = init.params["control"]["cost_nav_k"] * util.ca_sq_norm(uav_goal - uav_pos)

    cost_swing = init.params["control"]["cost_swing"] * util.ca_sq_norm(pld_rel_pos)

    cost_in = init.params["control"]["cost_in"] * util.ca_sq_norm(u_cur - init.params["derived"]["sys_weight"])

    # cost_inRate = init.params["control"]["cost_inRate"] * np.sum((u_cur - u_prev)**2)

    J = cost_nav + cost_nav_k + cost_in + cost_swing

    return J

def _genSolver():
    """
    Uses CasADi symbolics framework to interface with IPOPT solver. 
    Objective function and inequality equations are built up using CasADi 
    symbolics and passed to the IPOPT solver generation class.
    
    Input: none
    Return: args <dict>
            solver <IPOPT solver class>
    """

    # Design Nonlinear init.model Predictive Controller
    N = init.params["control"]["predictionHorizon"]
    Nu = init.params["control"]["controlHorizon"]
    Ts = init.params["control"]["sampleTime"]

    x = ca.SX.sym('x',init.model["n_x"]) # system states
    u = ca.SX.sym('u',init.model["n_u"]) # control inputs
    dxdt = slungLoadDyn(x,u) # rhs of EOM

    sys_dyn = ca.Function('sys_dyn',[x,u],[dxdt]) # nonlinear mapping function f(x,u)
    U = ca.SX.sym('U',init.model["n_u"],Nu) # manipulative variables
    P = ca.SX.sym('P',init.model["n_p"]) # parameters incl x0 and xref
    X = ca.SX.sym('X',init.model["n_x"],(N + 1)) # states over the prediction horizon

    n_X = init.model["n_x"] * (N + 1) # number of predicted states
    n_U = init.model["n_u"] * Nu # number of predicted inputs

    # Make decision variable one column vector with X and U over N horizon
    decision_var = ca.vertcat(ca.reshape(X,n_X,1), ca.reshape(U,n_U,1))
    n_XU = n_X + n_U

    # iniitialize objective function and equality/inequality vectors
    obj_fcn = 0

    con_ws = []
    con_obs = []
    con_pld_d = []
    con_eq = X[:,0]-P[init.idx['p']['x0'][0]:init.idx['p']['x0'][1]] # start equality constraints

    u_prev = U[:,0] # for input rate cost
    for k in range(0, N):
        x_cur = X[:,k] # actual states
        
        # optimize input up to Nu
        if k < Nu:
            u_cur = U[:,k] # controls 
        else:
            u_cur = U[:,Nu-1]
        
        x_next = X[:,k+1] # evaluated states at next time step f(x,u)
        xdot = sys_dyn(x_cur,u_cur) # dxdt
        x_next_actual = util.RK4(sys_dyn,Ts,x_cur,u_cur) # discretized via RK4
        
        # constraints
        con_eq = ca.vertcat(con_eq, x_next-x_next_actual) # multiple shooting equality
        ineq_fcn = ineqConFcn(x_cur) 
        con_ws = ca.vertcat(con_ws, ineq_fcn[init.idx["g"]["ineq_ws"][0]:init.idx["g"]["ineq_ws"][1]])
        con_obs = ca.vertcat(con_obs, ineq_fcn[init.idx["g"]["ineq_obs"][0]:init.idx["g"]["ineq_obs"][1]])
        con_pld_d = ca.vertcat(con_pld_d, ineq_fcn[init.idx["g"]["ineq_pld_d"]])
        
        # objective function
        obj_fcn = obj_fcn + costFcn(x_cur,u_cur,u_prev,P)       
        u_prev = u_cur
        
        if( k == (N - 1)): 
            # terminal objective function
            init.params["control"]["cost_nav"] = init.params["control"]["cost_nav_N"]
            obj_fcn = obj_fcn + costFcn(x_next,u_cur,u_prev,P)
            
            ineq_fcn = ineqConFcn(x_cur)
            con_ws = ca.vertcat(con_ws, ineq_fcn[init.idx["g"]["ineq_ws"][0]:init.idx["g"]["ineq_ws"][1]])
            con_obs = ca.vertcat(con_obs, ineq_fcn[init.idx["g"]["ineq_obs"][0]:init.idx["g"]["ineq_obs"][1]])
            con_pld_d = ca.vertcat(con_pld_d, ineq_fcn[init.idx["g"]["ineq_pld_d"]])

    # con_fcn = ca.vertcat(con_eq, con_ws, con_obs, con_pld_d) # inequality constraints
    con_fcn = ca.vertcat(con_eq, con_ws, con_obs)
    # con_fcn = con_eq

    # Set equality and inequality constraints
    # upper/lower function bounds lb <= g <= ub
    lbg = np.zeros((con_fcn.shape[0],1))
    ubg = np.zeros((con_fcn.shape[0],1))

    # bounds on equality constraints
    lbg[init.idx["g"]["eq"][0]:init.idx["g"]["eq"][1]] = 0 
    ubg[init.idx["g"]["eq"][0]:init.idx["g"]["eq"][1]] = 0

    # bounds on workspace constraints
    n_repeat = int((init.idx["g"]["ws"][1] - init.idx["g"]["ws"][0]) / 3)
    lbg[init.idx["g"]["ws"][0]:init.idx["g"]["ws"][1]] = \
        np.tile(init.sim["workspace"][0,:].reshape(3,1), (n_repeat,1)) 
    ubg[init.idx["g"]["ws"][0]:init.idx["g"]["ws"][1]] = \
        np.tile(init.sim["workspace"][1,:].reshape(3,1), (n_repeat,1))

    # bounds on collision avoidance constraints
    lbg[init.idx["g"]["obs"][0]:init.idx["g"]["obs"][1]] = 0 
    ubg[init.idx["g"]["obs"][0]:init.idx["g"]["obs"][1]] = ca.inf

    # Set hard constraints on states and input
    # upper/lower variable bounds lb <= x <= ub
    lbx = -ca.inf*np.ones((n_XU,1))
    ubx = ca.inf*np.ones((n_XU,1))

    state_min = -ca.inf*np.ones((init.model["n_x"],1))
    state_max = ca.inf*np.ones((init.model["n_x"],1))
    state_min[init.idx["x"]["uav_pos"][0]:init.idx["x"]["uav_pos"][1]] = \
        init.sim["workspace"][0].reshape(3,1) + np.array([init.params["arm_len"], \
                                                        init.params["arm_len"], \
                                                            0]).reshape(3,1)
    state_max[init.idx["x"]["uav_pos"][0]:init.idx["x"]["uav_pos"][1]] = \
        init.sim["workspace"][1].reshape(3,1) - np.array([init.params["arm_len"], \
                                                        init.params["arm_len"], \
                                                            0]).reshape(3,1)
    for i in range(init.model["n_x"]):
        lbx[i:n_X:init.model["n_x"]] = state_min[i] # state lower limit
        ubx[i:n_X:init.model["n_x"]] = state_max[i] # state upper limit

    # input_min = -ca.inf*np.ones((init.model["n_u"],1))
    # input_max = ca.inf*np.ones((init.model["n_u"],1))
    input_min = np.array([-15,-15,0])
    input_max = np.array([15,15,45])
    for i in range(init.model["n_u"]):
        lbx[n_X+i:n_XU:init.model["n_u"]] = input_min[i] #input lower limit
        ubx[n_X+i:n_XU:init.model["n_u"]] = input_max[i] #input upper limit

    # inequality constraint arguments
    args = {
        'lbg' : lbg,
        'ubg' : ubg,
        'lbx' : lbx,
        'ubx' : ubx
    }

    # set up solver
    nlp_prob = {
        'f': obj_fcn,
        'x': decision_var,
        'g': con_fcn,
        'p': P
    }

    codeopts = {
        'ipopt': {
            'max_iter': 1000,
            'print_level': 0,
            'acceptable_tol': 1e-8,
            'acceptable_obj_change_tol': 1e-4
        },
        'print_time': 0
    }

    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, codeopts)

    return(args, solver)
