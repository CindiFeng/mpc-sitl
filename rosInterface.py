#!/usr/bin/env python
import time
import rospy
import numpy as np
import pandas as pd
import init
import util_fcn as util
from MPCController import runMPC
from PIDController import runPID
# from PIDTest import runPID

import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import Thrust, PositionTarget, AttitudeTarget
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode, SetModeRequest
from optitrack_broadcast.msg import Mocap 
from std_msgs.msg import String

class runROSNode(object):
    def __init__(self,args,solver):

        # publisher
        pub_freq = 20
        self.rate = rospy.Rate(pub_freq)
        # self.pub_thrust = rospy.Publisher("/mavros/setpoint_attitude/thrust", Thrust, queue_size=1)
        # self.pub_quat = rospy.Publisher("/mavros/setpoint_attitude/attitude", PoseStamped, queue_size=1)
        self.pub_pos_raw = rospy.Publisher("/mavros/setpoint_raw/local", PositionTarget, queue_size=1)
        # self.pub_att = rospy.Publisher("/mavros/setpoint_raw/local", PositionTarget, queue_size=1)
        self.pub_att = rospy.Publisher("/mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=1)
        self.pub_status = rospy.Publisher("/safetyStatus", String, queue_size=1)

        # subscriber
        self.uav_odom = rospy.Subscriber('/mocap/uav', Mocap, self.uav_callback)
        self.pld_odom = rospy.Subscriber('/mocap/payload', Mocap, self.pld_callback)

        # Controllers
        self.mpc = runMPC(args,solver)
        self.pid = runPID()

        self.sec_prev = 0
        self.tHist = 0

    def uav_callback(self,msg):
        self._mocap_uav = msg

    def pld_callback(self,msg):
        self._mocap_pld = msg

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
    
    def _pub_ff_start(self):
        msg = PositionTarget()
        msg.coordinate_frame = 1

        msg.type_mask = 0b0000100111111000
        msg.position.x = init.ics["uav_pos"][1] #init.params["mission"]["uav_pos"][1]
        msg.position.y = init.ics["uav_pos"][0] #init.params["mission"]["uav_pos"][0]
        msg.position.z = -init.ics["uav_pos"][2] #-init.params["mission"]["uav_pos"][2]

        self.pub_pos_raw.publish(msg)   
    
    def _pub_ff_hold_pos(self,last_pos):
        msg = PositionTarget()
        msg.coordinate_frame = 1

        msg.type_mask = 0b0000100111111000
        msg.position.x = last_pos[0] #init.params["mission"]["uav_pos"][1]
        msg.position.y = last_pos[1] #init.params["mission"]["uav_pos"][0]
        msg.position.z = last_pos[2] #-init.params["mission"]["uav_pos"][2]

        self.pub_pos_raw.publish(msg)
        # try: 
        #     last_pos = last_pos.reshape((3,))
        # except:
        #     print("hold position is not in standard form.")
        
    def _pub_start_pos(self):
        # moves drone to user-defined initial position given in NED 
        des_pos = np.array([init.ics["uav_pos"][1],
                            init.ics["uav_pos"][0],
                            -init.ics["uav_pos"][2]]).reshape(3,)
        msg = self.pid.publish_cmd(self._mocap_uav, des_pos)
        self.pub_att.publish(msg)
        
    def _pub_hold_pos(self,des_pos):
        # moves drone to des_pos: 1D (3,) array in ENU frame
        try: 
            msg = self.pid.publish_cmd(self._mocap_uav,des_pos)
            self.pub_att.publish(msg) 
        except: 
            print("hold position is not in standard form.")
        
    def _pub_land(self,target_pos = np.array([0,0,0])):
        while self._mocap_uav.position[2] > 0.12:
            msg = self.pid.publish_cmd(self._mocap_uav,target_pos)
            self.pub_att.publish(msg)
        self._setDisarm()

    def _setDisarm(self):
        rospy.wait_for_service('/mavros/cmd/arming')
        try:
            armService = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
            armService(False)
        except rospy.ServiceException:
            print ("Service disarm call failed.")  

    def _setLandMode(self):
        rospy.wait_for_service('/mavros/cmd/land')
        try:
            landService = rospy.ServiceProxy('/mavros/cmd/land', CommandTOL)
            isLanding = landService(altitude = 0, latitude = 0, longitude = 0, min_pitch = 0, yaw = 0)
        except rospy.ServiceException:
            print ("service land call failed. The vehicle cannot land.")
    
    def _setArm(self):
        rospy.wait_for_service('/mavros/cmd/arming')
        try:
            armService = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
            armService(True)
        except rospy.ServiceException:
            print ("Service arm call failed.")

    def pub_mpc_cmd(self): 
        # thrust, quat_des = util.att_extract(self.mpc.u[0,:].T)

        msg_force = PositionTarget()
        msg_force.coordinate_frame = 1
        msg_force.type_mask = 0b0000100111000000
        pos_des = self.mpc.sol_x[1,2:5]
        vel_des = self.mpc.sol_x[1,7:]
        msg_force.position.x = pos_des[1]
        msg_force.position.y = pos_des[0]
        msg_force.position.z = -pos_des[2]
        msg_force.velocity.x = vel_des[1]
        msg_force.velocity.y = vel_des[0]
        msg_force.velocity.z = -vel_des[2]
        self.pub_pos_raw.publish(msg_force)

        # msg_quat = PoseStamped()
        # msg_quat.pose.orientation.x = quat_des[0]
        # msg_quat.pose.orientation.y = quat_des[1]
        # msg_quat.pose.orientation.z = quat_des[2]
        # msg_quat.pose.orientation.w = quat_des[3]
        # self.pub_quat.publish(msg_quat)

        # msg_thrust = Thrust()
        # msg_thrust.thrust = thrust
        # self.pub_thrust.publish(msg_thrust)

    def _run_mpc(self):
        mpc = self.mpc
        mpc.xHist = np.vstack((mpc.xHist,
                               np.hstack((mpc.pld_rel_pos.T,
                                          mpc.uav_pos.T, 
                                          mpc.pld_rel_vel.T,
                                          mpc.uav_vel.T))))
        new_t = np.array([[(self._mocap_uav.header.stamp.secs + 
                            self._mocap_uav.header.stamp.nsecs * 1e-9 - 
                            self.tHist[0,0])]])
        self.tHist = np.vstack((self.tHist,
                                new_t))
        self.sec_prev = self._mocap_uav.header.stamp.secs
        # if goal is reached, hold uav at goal 
        # else, run MPC solver to get next commands
        error = mpc.uav_pos - init.params["mission"]["uav_pos"].reshape((3,1))
        error = util.norm(error)
        if error < 0.1: # or mpciter > 100: 
            rospy.loginfo("Reached goal position.")
            DF = pd.DataFrame(mpc.xHist)
            DF.to_csv("xHist.csv")
            DF = pd.DataFrame(mpc.uHist)
            DF.to_csv("uHist.csv")
            DF = pd.DataFrame(self.tHist)
            DF.to_csv("tHist.csv")
            # util.plot2d(mpc.xHist[1:,:],mpc.uHist[1:,:],self.tHist[1:,:],init)
            # last_pos = np.array([mpc.uav_pos[1,0],mpc.uav_pos[0,0],-mpc.uav_pos[2,0]])
            last_pos = self._mocap_uav.position
            while not rospy.is_shutdown():
                self._pub_hold_pos(last_pos)
                # self._pub_ff_hold_pos(last_pos)
        else:
            self.mpc._solve_mpc(self._mocap_uav, self._mocap_pld) # update ctrl input
            self.pub_mpc_cmd()