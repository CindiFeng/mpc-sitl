import numpy as np
import casadi as ca

def sl_dr_dyn(x,u,init):
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
    g_I= ca.vertcat(ca.SX.zeros(2,1),g)

    # Geometric Relations
    B_hat = ca.vertcat(ca.SX.eye(2), -r_L.T / ca.sqrt(L**2 - r_L.T @ r_L))
    B_hat_dot = ca.vertcat(ca.SX.zeros(2,2), ((v_L @ (r_L.T @ r_L-L**2)-r_L @ r_L.T @ v_L)/(L**2-r_L.T @ r_L)**(3/2)).T)

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

def util_RK4(f,Ts,x_current,con):
    """ Functionality: Discretize continuous state function xdot = f(x,u) 
    using the Runge-Kutta 4th order method

    Inputs: Ts - discretization time step 
            st - states
            con - control inputs

    Outputs: params struct
    """

    k1 = f(x_current, con)
    k2 = f(x_current + Ts/2*k1, con)
    k3 = f(x_current + Ts/2*k2, con)
    k4 = f(x_current + Ts*k3, con)
    x_next = x_current + Ts/6 * (k1 +2*k2 +2*k3 +k4)

    return x_next

def ineq(x_cur,init):
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
                                       ca.sqrt(init.params["cable_len"]**2 - \
                                                pld_rel_pos.T @ pld_rel_pos))

    uav_obs_vec = uav_pos - init.sim["obs_pos"]
    uav_obs_dist = uav_obs_vec.T @ ellipsoid_weight @ uav_obs_vec - 1
    pld_obs_vec = pld_abs_pos - init.sim["obs_pos"]
    pld_obs_dist = pld_obs_vec.T @ ellipsoid_weight @ pld_obs_vec - 1

    # payload distance
    pld_rel_dist = pld_rel_pos.T @ pld_rel_pos

    # cineq = ca.vertcat(pld_obs_dist, uav_obs_dist, pld_abs_pos, pld_rel_dist)
    cineq = sq_norm(x_cur[init.idx["x"]["uav_vel"][0]:init.idx["x"]["uav_vel"][1]])
    
    uav_speed = -sq_norm(x_cur[init.idx['x']['uav_vel'][0]:init.idx['x']['uav_vel'][1]])
    cineq = uav_speed
    return cineq 

def cost(x_cur,u_cur,init):
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
    uav_goal = np.array(init.params['mission']['uav_pos']).reshape(3,1)

    cost_nav_k = init.params["control"]["cost_nav_k"] * sq_norm(uav_goal - uav_pos)

    cost_swing = init.params["control"]["cost_swing"] * sq_norm(pld_rel_pos)

    cost_in = init.params["control"]["cost_in"] * sq_norm(u_cur + init.params["derived"]["sys_weight"])

    J = cost_in + cost_swing

    return J

def cost_e(x_cur,init):
    """ 
    Abridged custom cost function based on Potdar 2020 paper
    INPUT:  x_cur, column state vector
            - payload relative position (r_p)
            - uav position (x_q)
            - payload relative velocity (v_p)
            - uav velocity (v_q)

            u_cur, column input vector
            - 3x lift force (F_L) vector in NED frame
    """

    # assign variables
    uav_pos = x_cur[init.idx["x"]["uav_pos"][0]:init.idx["x"]["uav_pos"][1]]
    pld_rel_pos = x_cur[init.idx["x"]["pld_rel_pos"][0]:init.idx["x"]["pld_rel_pos"][1]]
    uav_goal = np.array(init.params['mission']['uav_pos']).reshape(3,1)
    uav_start = np.array(init.ics['uav_pos']).reshape(3,1)

    denom = sq_norm(uav_goal - uav_start) # cannot start at goal!!!
    cost_nav = init.params["control"]["cost_nav_N"] * sq_norm(uav_goal-uav_pos) / denom

    cost_swing = init.params["control"]["cost_swing"] * sq_norm(pld_rel_pos)

    J = cost_nav + cost_swing

    return J

def sq_norm(a): 
    """
    returns equivalent to MATLAB: sum(a.^2)
    """
    r = a.shape[0]
    a = ca.reshape(a,r,1)

    sq_norm = a.T @ a

    return sq_norm
