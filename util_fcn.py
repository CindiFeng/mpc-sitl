#!/usr/bin/env python
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as scipyR

def ca_sq_norm(a): 
    """
    returns equivalent to MATLAB: sum(a.^2)
    """
    r = a.shape[0]
    a = ca.reshape(a,r,1)

    sq_norm = a.T @ a

    return sq_norm

def RK4(f,Ts,x_current,con):
    """ 
    Functionality: Discretize continuous state function xdot = f(x,u) 
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

# def att_extract(u_NED): 
#         # Takes control input (in NED) from mpc solver to derive the desired angular velocities

#         f_ctrl = np.array([u_NED[1],u_NED[0],-u_NED[2]])
#         f_mag = norm(f_ctrl)
#         thrust_norm = (f_mag + 6)/50
#         if thrust_norm > 1: 
#              thrust_norm = 1
        
#         psi_des = 0 # command yaw angle 
        
#         # build rotation matrix using unit vectors
#         n_z = (f_ctrl / f_mag).reshape((3,1)) # assume lift along the z axis
#         n_x_tilde = np.array([np.cos(psi_des), np.sin(psi_des), 
#                         -(np.cos(psi_des) * n_z[0,0] + np.sin(psi_des) * 
#                         n_z[1,0]) / n_z[2,0]]).reshape((3,1))
#         n_x = n_x_tilde / norm(n_x_tilde)
#         n_y = hat(n_z) @ n_x / norm(hat(n_z) @ n_x)
#         R_des = np.hstack((n_x, n_y, n_z))

#         r = scipyR.from_matrix(R_des.T)
#         quat_des = r.as_quat()
#         return (thrust_norm, quat_des)

# # def att_extract(mpc, u_NED): 
# #         # Takes control input (in NED) from mpc solver to derive the desired angular velocities

# #         f_ctrl = np.array([u_NED[1],u_NED[0],-u_NED[2]])
# #         f_mag = norm(f_ctrl)
# #         thrust_norm = (f_mag + 6)/50
# #         if thrust_norm > 1: 
# #              thrust_norm = 1
        
# #         psi_des = 0 # command yaw angle 
        
# #         # build rotation matrix using unit vectors
# #         n_z = (f_ctrl / f_mag).reshape((3,1)) # assume lift along the z axis
# #         n_x_tilde = np.array([np.cos(psi_des), np.sin(psi_des), 
# #                         -(np.cos(psi_des) * n_z[0,0] + np.sin(psi_des) * 
# #                         n_z[1,0]) / n_z[2,0]]).reshape((3,1))
# #         n_x = n_x_tilde / norm(n_x_tilde)
# #         n_y = hat(n_z) @ n_x / norm(hat(n_z) @ n_x)
# #         R_des = np.hstack((n_x, n_y, n_z))

# #         Rd_des = (R_des - mpc.R_des_prev)/0.05
# #         mpc.R_des_prev = R_des

# #         ang_vel_des = vee(R_des.T @ Rd_des)
# #         return (thrust_norm, ang_vel_des)

# def vee(ss): 
#     """
#     vee function to map a skew-symmetric matrix to a vector
#     """
#     # if (ss.shape != (3,3) or ss[2,1] != -ss[1,2] or ss[0,2] != -ss[2,0] 
#     #     or ss[0,1] != -ss[1,0]): 
#     #     raise Exception("The provided matrix is not skew symmetric.")
    
#     vec = np.array([ss[2,1], ss[0,2], ss[0,1]]).reshape(3,1)

#     return vec

# def hat(v):
#     """
#     create skew symmetric matrix from vector
#     """
#     v = v.reshape(3,)
#     ss = np.array([0,-v[2],v[1],v[2],0,-v[0],-v[1],v[0],0]).reshape(3,3)
#     return ss

def norm(a):
    r = max(a.shape)
    a = a.reshape((r,))
    return ((a.T @ a)**0.5)

def DM2Arr(dm):
    return np.array(dm.full())

def plot2d(xHistory,uHistory,tHistory,init):

    # simT = np.linspace(0,
    #                    init.params["control"]["sampleTime"]*(len(xHistory)-1),
    #                    len(xHistory)) # simulation time step
    simT = tHistory

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
    plt.plot(simT,xHistory[:,4],'b-',linewidth=1)
    plt.axhline(init.params["mission"]["uav_pos"][2],linestyle='--')
    plt.xlabel('t (s)')
    plt.ylabel('z (m)')
    plt.title('z Pos')

    plt.subplot(2,2,4)
    plt.plot(simT[1:],uHistory,linewidth=1)
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