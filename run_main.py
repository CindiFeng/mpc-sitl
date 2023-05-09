#!/usr/bin/env python
import rospy
from std_msgs.msg import String

import time
import numpy as np

import init
import util_fcn as util
import mpc_fcn as mpc
from rosInterface import runROSNode

WS_X_MAX = 4
WS_Y_MAX = 1
WS_Z_MAX = 7
WS_X_MIN = -1
WS_Y_MIN = -1
WS_Z_MIN = 0
# def go2start(self):
    #     error = 1
    #     while not rospy.is_shutdown() and error > 0.05:
    #         self.mpc.get_odom(self._mocap_uav, self._mocap_pld)
    #         self.rate.sleep()
    #         self.pub_ff_start()
    #         error = (np.vstack((self.mpc.pld_rel_pos, self.mpc.uav_pos)) - 
    #                  np.vstack((init.ics["pld_rel_pos"], init.ics["uav_pos"])))
    #         error = util.norm(error)
    #     last_pos = self.mpc.uav_pos
    #     self.pub_ff_hold_pos(last_pos)

def go2start(irisDrone):
    error = 1
    while not rospy.is_shutdown() and error > 0.05:
        irisDrone.mpc.get_odom(irisDrone._mocap_uav, irisDrone._mocap_pld)
        if safetyCheckFail(irisDrone):
            irisDrone._pub_land()
        else:
            irisDrone._pub_start_pos() # fly uav to start position
            irisDrone.rate.sleep()        
        error = (np.vstack((irisDrone.mpc.pld_rel_pos, irisDrone.mpc.uav_pos)) - 
                np.vstack((init.ics["pld_rel_pos"], init.ics["uav_pos"])))
        error = util.norm(error)
        irisDrone.rate.sleep()

    # print('Testing: holding position')
    # last_pos = irisDrone._mocap_uav.position
    # while not rospy.is_shutdown():
    #     # irisDrone._pub_hold_pos(last_pos)
    #     irisDrone._pub_ff_hold_pos(last_pos)
    #     irisDrone.rate.sleep()

def main_loop(irisDrone):
    go2start(irisDrone)

    rospy.loginfo("Reached Start point. Starting MPC solver loop.")
    # status_msg = String()
    # status_msg.data = "Reached Start point. Starting MPC solver loop."
    # irisDrone.pub_status.publish(status_msg)

    irisDrone.sec_prev = irisDrone._mocap_uav.header.stamp.secs
    irisDrone.tHist = np.array([[irisDrone._mocap_uav.header.stamp.secs + 
                                    irisDrone._mocap_uav.header.stamp.nsecs * 1e-9]])
    
    while not rospy.is_shutdown():
        # if safetyCheckFail(irisDrone): 
        #     irisDrone._pub_land()
        # else:
        #     irisDrone._run_mpc()
        irisDrone._run_mpc()
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
    if util.norm(dr_pos-pld_pos) > (init.params["cable_len"] + 0.05):
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

if __name__ == '__main__': 
    try: 
        rospy.init_node('mpc_SITL', anonymous=True)
        args, solver = mpc._genSolver()
        irisDrone = runROSNode(args,solver)
        # status_msg = String()
        
        # fly uav to start position
        # status_msg.data = "Waiting for 3 seconds to connect to subscriber."
        # irisDrone.pub_status.publish("Waiting for 3 seconds to connect to subscriber.")
        # status_msg.data = "Wait is over."
        # irisDrone.pub_status.publish(status_msg)
        rospy.loginfo("Waiting for 3 seconds to connect to subscriber.")
        time.sleep(3)
        rospy.loginfo("Wait is over.")        
        main_loop(irisDrone)

    except rospy.ROSInterruptException: 
        rospy.loginfo('Interrupted')
        # irisDrone.pub_status.publish("Interrupted")
        pass