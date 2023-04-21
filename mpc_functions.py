import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

def slungLoadDyn(x,u,init):
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

def ineqConFcn(x_cur,init):
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

    cineq = ca.vertcat(pld_obs_dist, uav_obs_dist, pld_abs_pos, pld_rel_dist)

    return cineq 

def costFcn(x_cur,u_cur,u_prev,P,init):
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

    denom = util_sq_norm(uav_goal - uav_start) # cannot start at goal!!!
    cost_nav = init.params["control"]["cost_nav"] * util_sq_norm(uav_goal-uav_pos) / denom

    cost_nav_k = init.params["control"]["cost_nav_k"] * util_sq_norm(uav_goal - uav_pos)

    cost_swing = init.params["control"]["cost_swing"] * util_sq_norm(pld_rel_pos)

    cost_in = init.params["control"]["cost_in"] * util_sq_norm(u_cur + init.params["derived"]["sys_weight"])

    # cost_inRate = init.params["control"]["cost_inRate"] * np.sum((u_cur - u_prev)**2)

    J = cost_nav + cost_nav_k + cost_in + cost_swing

    return J

def att_extract(f_ctrl): 
    """
    Takes control input from mpc solver to derive the desired angular velocities
    Input: F_ctrl (np.array, 1x3) 
    Output: omega_d (np.array, 3x1)
    """
    f_ctrl = f_ctrl.reshape((3,1))
    psi_des = 0 # command yaw angle 
    n_z = -f_ctrl / util_sq_norm(f_ctrl)**0.5 # assume lift along the -z axis
    n_x = np.array([np.cos(psi_des), np.sin(psi_des), 
                    -(np.cos(psi_des) * n_z[0] + np.sin(psi_des) * 
                      n_z[1]) / n_z[2]]).reshape((3,1))
    n_x = n_x / util_sq_norm(n_x)**0.5
    n_y = util_hat(n_z) @ n_x / util_sq_norm(util_hat(n_z) @ n_x)**0.5
    R_des = np.hstack((n_x, n_y, n_z))

    Rd_des = 0 # TODO: derivative of rotation matrix 
    ang_vel_d = util_vee(R_des.T @ Rd_des)
    

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

def util_vee(ss): 
    """
    vee function to map a skew-symmetric matrix to a vector
    """
    if (ss.shape != (3,3) or ss[2,1] != -ss[1,2] or ss[0,2] != -ss[2,0] 
        or ss[0,1] != -ss[1,0]): 
        raise Exception("The provided matrix is not skew symmetric.")
    
    vec = np.array([ss[2,1], ss[0,2], ss[0,1]]).reshape(3,1)

    return vec

def util_hat(v):
    """
    create skew symmetric matrix from vector
    """
    ss = np.array([0,-v[2],v[1],v[2],0,-v[0],-v[1],v[0],0]).reshape(3,3)
    return ss

def util_sq_norm(a): 
    """
    returns equivalent to MATLAB: sum(a.^2)
    """
    r = a.shape[0]
    a = ca.reshape(a,r,1)

    sq_norm = a.T @ a

    return sq_norm

def util_DM2Arr(dm):
    return np.array(dm.full())

def util_plot2d(xHistory,uHistory,init):

    simT = np.linspace(0,
                       init.params["control"]["sampleTime"]*(len(xHistory)-1),
                       len(xHistory)) # simulation time step

    fig1 = plt.figure('Pld Motion') # plot payload motion

    plt.subplot(2,2,1)
    plt.plot(simT,xHistory[:,0],'b-',linewidth=1)
    plt.xlabel('t (s)')
    plt.ylabel('x (m)')
    plt.title('Pld Relative x Pos ($r_{x}$)')

    plt.subplot(2,2,2)
    plt.plot(simT,xHistory[:,1],'b-',linewidth=1)
    plt.xlabel('t (s)')
    plt.ylabel('y (m)')
    plt.title('Pld Relative y Pos ($r_{y}$)')

    plt.subplot(2,2,3)
    plt.plot(xHistory[:,0],xHistory[:,1],'b-',linewidth=1)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Pld Relative Pos')

    plt.subplot(2,2,4)
    pld_planar_z = np.sqrt(xHistory[:,0]**2 + xHistory[:,1]**2)
    plt.plot(simT,pld_planar_z,'b-',linewidth=1)
    plt.axhline(np.linalg.norm(init.params["mission"]["pld_rel_pos"]),linestyle='--') 
    plt.xlabel('t (s)')
    plt.ylabel('$r_{p}$ (m)')
    plt.title('Planar Pld Relative Dist (radius)')

    # plt.show()

    fig2 = plt.figure('Drone Motion') # plot quad motion

    plt.subplot(2,2,1)
    plt.plot(simT,xHistory[:,2],'b-',linewidth=1) 
    plt.axhline(init.params["mission"]["uav_pos"][0],linestyle='--')
    plt.axhline(init.sim["workspace"][0,0] + init.params["arm_len"],color='#A2142F')
    plt.axhline(init.sim["workspace"][1,0] - init.params["arm_len"],color='#A2142F')
    plt.ylim(init.sim["workspace"][:,0])
    plt.xlabel('t (s)')
    plt.ylabel('x (m)')
    plt.title('x Pos')

    plt.subplot(2,2,2)
    plt.plot(simT,xHistory[:,3],'b-',linewidth=1) 
    plt.axhline(init.params["mission"]["uav_pos"][1],linestyle='--')
    plt.xlabel('t (s)')
    plt.ylabel('y (m)')
    plt.title('y Pos')

    plt.subplot(2,2,3)
    plt.plot(simT,-xHistory[:,4],'b-',linewidth=1)
    plt.axhline(-init.params["mission"]["uav_pos"][2],linestyle='--')
    plt.xlabel('t (s)')
    plt.ylabel('z (m)')
    plt.title('z Pos')

    plt.subplot(2,2,4)
    plt.plot(simT[1:],np.array([1, 1, -1])*uHistory,linewidth=1)
    plt.legend(['$f_{x}$','$f_{y}$','$f_{z}$'],loc='upper right')
    plt.xlabel('t (s)')
    plt.ylabel('Force (N)')
    plt.title('Input')

    fig3 = plt.figure('x-position') # highlight need for non swing-minimizing 
    plt.plot(simT,xHistory[:,0]+xHistory[:,2],'b-',linewidth=1)
    plt.xlabel('time (s)')
    plt.ylabel('x (m)')
    plt.title('Payload Absolute Position')
    plt.axhline(init.sim["workspace"][0,0] + init.params["arm_len"], 
                linestyle='-', color='#A2142F')
    plt.axhline(init.sim["workspace"][1,0] - init.params["arm_len"], 
                linestyle='-', color='#A2142F')
    plt.ylim(init.sim["workspace"][:,0])

    plt.show()