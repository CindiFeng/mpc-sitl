#!/usr/bin/env python3
import rospy
from std_msgs.msg import String

import time
import numpy as np
import pandas as pd

# import init
import init_setting as init
from rosInterface import runROSNode
import attitude_conversion as att
import util_fcn as util 

WS_X_MAX = 4
WS_Y_MAX = 1
WS_Z_MAX = 7
WS_X_MIN = -1
WS_Y_MIN = -1
WS_Z_MIN = 0
TRACKING = False
TESTING = False

def go2start(irisDrone):
    """
    Fly drone to a specified initial position 
    """
    error = 1
    while not rospy.is_shutdown() and error > 0.1:
        irisDrone.mpc.get_odom(irisDrone._mocap_uav, irisDrone._mocap_pld)
        # if safetyCheckFail(irisDrone):
        #     irisDrone._pub_land()
        # else:
        
        # irisDrone._pub_start_pos() # fly uav to start position
        irisDrone._pub_ff_start()
        irisDrone.rate.sleep()   

        error = (np.vstack((irisDrone.mpc.pld_rel_pos, irisDrone.mpc.uav_pos)) - 
                np.vstack((init.ics["pld_rel_pos"], init.ics["uav_pos"])))
        error = np.linalg.norm(error)

def main_loop(irisDrone):
    """
    Run MPC to derive optimized inputs to be published to PX4 Mavros topics.
    MPCController and PIDController -> rosInterface -> run_main
    Types of mission include online planning and offline planning with tracking. 
    """
    go2start(irisDrone)

    rospy.loginfo("Reached Start point. Starting MPC solver loop.")
    # status_msg = String()
    # status_msg.data = "Reached Start point. Starting MPC solver loop."
    # irisDrone.pub_status.publish(status_msg)

    irisDrone.sec_prev = irisDrone._mocap_uav.header.stamp.secs
    irisDrone.tHist = np.array([[irisDrone._mocap_uav.header.stamp.secs + 
                                 irisDrone._mocap_uav.header.stamp.nsecs * 1e-9]])

    i = 0
    i_max = init.x_preGen.shape[0]

    while not rospy.is_shutdown():
        # if safetyCheckFail(irisDrone): 
        #     irisDrone._pub_land()
        # else:
        #     irisDrone._run_mpc()
        if TRACKING:
            # PX4 position control to follow pre-generated waypoints
            if i < i_max:
                xref = init.x_preGen[i,2:5]
                # irisDrone.mpc._solve_mpc(init.x_preGen[i,:], irisDrone._mocap_uav, irisDrone._mocap_pld)
                irisDrone.mpc._solve_mpc(np.vstack((init.params["mission"]["pld_rel_pos"],
                              init.params["mission"]["uav_pos"], 
                              init.params["mission"]["pld_rel_vel"], 
                              init.params["mission"]["uav_vel"])), irisDrone._mocap_uav, irisDrone._mocap_pld)
                i += 1

            irisDrone._pub_ff_hold_pos(xref)
            # irisDrone.mpc._solve_mpc(xref, irisDrone.mpc._mocap_uav, irisDrone.mpc._mocap_pld)
        else:
            # MPC for point navigation tasks
            xref = np.vstack((init.params["mission"]["pld_rel_pos"],
                              init.params["mission"]["uav_pos"], 
                              init.params["mission"]["pld_rel_vel"], 
                              init.params["mission"]["uav_vel"]))
            irisDrone._run_mpc(xref)

            # pd.DataFrame(irisDrone.mpc.uHist).to_csv("uHist.csv")
            # pd.DataFrame(irisDrone.tHist).to_csv("tHist.csv")
            # pd.DataFrame(irisDrone.mpc.uHist_ude).to_csv("uHist_ude.csv")
            # pd.DataFrame(irisDrone.mpc.xHist).to_csv("xHist.csv")

            # util.plot(irisDrone.mpc.xHist,irisDrone.mpc.uHist,irisDrone.mpc.uHist_ude,init)
        irisDrone.rate.sleep()

def safetyCheckFail(irisDrone):
    # Return True if any safety violation occurs
    msg = String()

    dr_pos = np.array(irisDrone._mocap_uav.position)
    pld_pos = np.array(irisDrone._mocap_pld.position)

    # check geofence
    if (dr_pos[0] > WS_X_MAX or dr_pos[0] < WS_X_MIN or
        dr_pos[1] > WS_Y_MAX or dr_pos[0] < WS_Y_MIN or 
        dr_pos[2] > WS_Z_MAX):
        msg.data = "Activate failsafe...vehicle out of geofence."
        irisDrone.pub_status.publish(msg)
        return True
    
    # check cable connection
    if np.linalg.norm(dr_pos-pld_pos) > (init.params["cable_len"] + 0.05):
        msg.data = "Activate failsafe...payload disconnected from body."
        irisDrone.pub_status.publish(msg)
        return True
    
    # check user defined initial and goal pos are within workspace
    # user defined values are in NED frame 
    if (init.ics["uav_pos"][1,0] > WS_X_MAX or init.ics["uav_pos"][1,0] < WS_X_MIN
        or init.ics["uav_pos"][0,0] > WS_Y_MAX or init.ics["uav_pos"][0,0] < WS_Y_MIN
        or -init.ics["uav_pos"][2,0] > WS_Z_MAX):
        msg.data = "Activate failsafe...dangerous start position chosen."
        irisDrone.pub_status.publish(msg)
        return True

    if (init.params["mission"]["uav_pos"][1,0] > WS_X_MAX or 
        init.params["mission"]["uav_pos"][1,0] < WS_X_MIN
        or init.params["mission"]["uav_pos"][0,0] > WS_Y_MAX or 
        init.params["mission"]["uav_pos"][0,0] < WS_Y_MIN
        or -init.params["mission"]["uav_pos"][2,0] > WS_Z_MAX):
        msg.data = "Activate failsafe...dangerous goal position chosen."
        irisDrone.pub_status.publish(msg)
        return True
    return False

# def testing(irisDrone): 

#     go2start(irisDrone)
#     rospy.loginfo("Reached Start point. Starting MPC solver loop.")

#     i = 0
#     i_max = init.u_preGen.shape[0]
#     t0 = rospy.Time.now().nsecs * 10e-9
#     while not rospy.is_shutdown():
#         t_elapsed = rospy.Time.now().nsecs * 10e-9 - t0
#         if i < i_max:
#             thrust, quat = att.att_extract(init.u_preGen[i,:])
#             # thrust,quat = att.att_extract(np.array([0.3258203,-0.41344817,-19.99157252]))
#             irisDrone._pub_mpc_cmd(thrust, quat)
#             last_pos = irisDrone._mocap_uav.position
            
#             print(i, ' thrust is: ', thrust)
#             i += 1
            
#             # if t_elapsed > 0.01:
#             #     print(i, ' thrust is: ', thrust)
#             #     i += 1
#             #     t0 = rospy.Time.now().nsecs * 10e-9

#             if i == i_max: 
#                 print('last pos: ', last_pos)
#         else:
#             irisDrone._pub_hold_pos(last_pos)
        
#         irisDrone.rate.sleep()

def testing(irisDrone):
    xref = np.vstack((init.params["mission"]["pld_rel_pos"],
                              init.params["mission"]["uav_pos"], 
                              init.params["mission"]["pld_rel_vel"], 
                              init.params["mission"]["uav_vel"]))
    
    i = 0
    i_max = init.x_preGen.shape[0]

    for i in range(0,i_max):
        irisDrone.mpc._solve_mpc(xref,init.x_preGen[i,:])
        irisDrone.mpc.xHist = np.vstack((irisDrone.mpc.xHist,
                                np.vstack((irisDrone.mpc.pld_rel_pos,
                                            irisDrone.mpc.uav_pos, 
                                            irisDrone.mpc.pld_rel_vel,
                                            irisDrone.mpc.uav_vel)).T))
    util.plot(irisDrone.mpc.xHist,irisDrone.mpc.uHist,irisDrone.mpc.uHist_ude,init)    
    
if __name__ == '__main__': 
    try: 
        rospy.init_node('mpc_SITL', anonymous=True)
        irisDrone = runROSNode()
        # status_msg = String()
        
        # fly uav to start position
        # status_msg.data = "Waiting for 3 seconds to connect to subscriber."
        # irisDrone.pub_status.publish("Waiting for 3 seconds to connect to subscriber.")
        # status_msg.data = "Wait is over."
        # irisDrone.pub_status.publish(status_msg)
        rospy.loginfo("Waiting for 2 seconds to connect to subscriber.")
        time.sleep(2)
        rospy.loginfo("Wait is over.")
        
        if TESTING: 
            testing(irisDrone)
        else:        
            main_loop(irisDrone)

    except rospy.ROSInterruptException: 
        rospy.loginfo('Interrupted')
        # irisDrone.pub_status.publish("Interrupted")
        pass