#!/usr/bin/env python
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

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

def norm(a):
    r = max(a.shape)
    a = a.reshape((r,))
    return ((a.T @ a)**0.5)

def DM2Arr(dm):
    return np.array(dm.full())

def plot(xHistory,uHistory,uHistory_ude,init):
    endT = 0.05 * (len(xHistory)-1) 
    simT = np.linspace(0,
                       endT, #init.params["control"]["sampleTime"]*(len(xHistory)-1),
                       len(xHistory)) # simulation time step

    fig1 = plt.figure('Pld Motion') # plot payload motion
    plt.plot(simT,xHistory[:,0],'b-',linewidth=1,label='$r_{x}$')
    plt.plot(simT,xHistory[:,1],'g-',linewidth=1,label='$r_{y}$')
    pld_planar_z = np.sqrt(xHistory[:,0]**2 + xHistory[:,1]**2)
    plt.plot(simT,pld_planar_z,'r-',linewidth=1,label='||r||')
    plt.axhline(np.linalg.norm(init.params["mission"]["pld_rel_pos"]),color='k',linestyle='--',label='goal')
    plt.xlim([0,endT]) 
    plt.xlabel('time (s)')
    plt.ylabel('distance (m)')
    plt.title('Pld Relative Pos')
    plt.legend(loc='upper right')
    plt.grid(color='#ADAAAB',linewidth=0.5)

    fig2 = plt.figure('Pld Planar Motion') # plot payload motion
    plt.plot(xHistory[:,0],xHistory[:,1],'b-',linewidth=1)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Pld Relative Pos')

    # plt.show()

    fig3 = plt.figure('Drone Motion') # plot quad motion

    plt.subplot(3,1,1)
    plt.plot(simT,xHistory[:,2],'b-',linewidth=1) 
    plt.axhline(init.params["mission"]["uav_pos"][0],linestyle='--',label='goal')
    # plt.axhline(init.sim["workspace"][0,0] + init.params["arm_len"],color='#A2142F')
    # plt.axhline(init.sim["workspace"][1,0] - init.params["arm_len"],color='#A2142F')
    plt.ylim(init.sim["workspace"][:,0] + init.params["arm_len"] * np.array([1,-1]))
    plt.xlim([0,endT]) 
    plt.xlabel('t (s)')
    plt.ylabel('x (m)')
    plt.title('x Pos')
    plt.legend(loc='upper right')
    plt.grid(color='#ADAAAB',linewidth=0.5)

    plt.subplot(3,1,2)
    plt.plot(simT,xHistory[:,3],'b-',linewidth=1) 
    plt.axhline(init.params["mission"]["uav_pos"][1],linestyle='--')
    plt.ylim(init.sim["workspace"][:,1] + init.params["arm_len"] * np.array([1,-1]))
    plt.xlim([0,endT]) 
    plt.xlabel('t (s)')
    plt.ylabel('y (m)')
    plt.title('y Pos')
    plt.grid(color='#ADAAAB',linewidth=0.5)

    plt.subplot(3,1,3)
    plt.plot(simT,xHistory[:,4],'b-',linewidth=1)
    plt.axhline(init.params["mission"]["uav_pos"][2],linestyle='--')
    plt.ylim(init.sim["workspace"][:,2])
    plt.xlim([0,endT]) 
    plt.xlabel('t (s)')
    plt.ylabel('z (m)')
    plt.title('z Pos')
    plt.grid(color='#ADAAAB',linewidth=0.5)

    fig4 = plt.figure('x-position') # highlight need for non swing-minimizing 
    plt.plot(simT,xHistory[:,0]+xHistory[:,2],'b-',linewidth=1)
    plt.xlabel('time (s)')
    plt.ylabel('x (m)')
    plt.title('Payload Absolute x-Position')
    plt.axhline(init.sim["workspace"][0,0] + init.params["arm_len"], 
                linestyle='-', color='#A2142F', label='drone pos limits')
    plt.axhline(init.sim["workspace"][1,0] - init.params["arm_len"], 
                linestyle='-', color='#A2142F')
    plt.ylim(init.sim["workspace"][:,0])
    plt.xlim([0,endT]) 
    plt.legend()
    plt.grid(color='#ADAAAB',linewidth=0.5)

    fig5 = plt.figure('Input')
    plt.subplot(3,1,1)
    plt.plot(simT[:-1],uHistory[1:,:],linewidth=1)
    plt.legend(['$f_{x}$','$f_{y}$','$f_{z}$'],loc='upper right')
    plt.xlim([0,endT]) 
    plt.xlabel('t (s)')
    plt.ylabel('Force (N)')
    plt.title('MPC Input')
    plt.grid(color='#ADAAAB',linewidth=0.5)

    plt.subplot(3,1,2)
    plt.plot(simT[:-1],uHistory_ude[1:,:],linewidth=1)
    plt.legend(['$f_{x}$','$f_{y}$','$f_{z}$'],loc='upper right')
    plt.xlim([0,endT]) 
    plt.xlabel('t (s)')
    plt.ylabel('Force (N)')
    plt.title('UDE Input')
    plt.grid(color='#ADAAAB',linewidth=0.5)

    plt.subplot(3,1,3)
    plt.plot(simT[:-1],uHistory[1:,:] - uHistory_ude[1:,:],linewidth=1)
    plt.legend(['$f_{x}$','$f_{y}$','$f_{z}$'],loc='upper right')
    plt.xlim([0,endT]) 
    plt.xlabel('t (s)')
    plt.ylabel('Force (N)')
    plt.title('Actual Input')
    plt.grid(color='#ADAAAB',linewidth=0.5)

    plt.show()