import matplotlib.pyplot as plt
import numpy as np

def plot(xHistory,uHistory,init):

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