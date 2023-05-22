#!/usr/bin/env python3
import time
import rospy
import numpy as np
import pandas as pd
import init
import util_fcn as util
from MPCController import runMPC
from PIDController import runPID
# from PIDTest import runPID
import attitude_conversion as att

import rospy
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from mavros_msgs.msg import Thrust, PositionTarget, AttitudeTarget
from mavros_msgs.srv import CommandBool, CommandTOL, SetMode, SetModeRequest
from optitrack_broadcast.msg import Mocap
from std_msgs.msg import String

class runROSNode(object):
    def __init__(self,args,solver):

        # publisher
        pub_freq = 20 # 30
        self.rate = rospy.Rate(pub_freq)
        self.pub_pos_raw = rospy.Publisher("/mavros/setpoint_raw/local", PositionTarget, queue_size=1)
        # self.pub_att = rospy.Publisher("/mavros/setpoint_raw/local", PositionTarget, queue_size=1)
        self.pub_att = rospy.Publisher("/mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=1)
        self.pub_mpc_activity = rospy.Publisher("/controller/mpc_activity", String, queue_size=1)
        self.pub_mpc_output = rospy.Publisher("/controller/mpc_output", Vector3Stamped, queue_size=1)
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
    
    def _pub_ff_start(self):
        msg = PositionTarget()
        msg.coordinate_frame = 1

        msg.type_mask = 0b0000100111111000
        msg.position.x = init.ics["uav_pos"][0] #init.params["mission"]["uav_pos"][1]
        msg.position.y = init.ics["uav_pos"][1] #init.params["mission"]["uav_pos"][0]
        msg.position.z = init.ics["uav_pos"][2] #-init.params["mission"]["uav_pos"][2]

        self.pub_pos_raw.publish(msg)   
    
    def _pub_ff_hold_pos(self,des_pos):
        # Publish position coordinates to setpoint_raw/positionTarget 

        msg = PositionTarget()
        msg.coordinate_frame = 1

        msg.type_mask = 0b0000100111111000
        msg.position.x = des_pos[0] 
        msg.position.y = des_pos[1] 
        msg.position.z = des_pos[2]

        self.pub_pos_raw.publish(msg)

    def _pub_start_pos(self):
        # moves drone to user-defined initial position given in NED 
        des_pos = np.array([init.ics["uav_pos"][0],
                            init.ics["uav_pos"][1],
                            init.ics["uav_pos"][2]]).reshape(3,)
        msg = self.pid.publish_cmd(self._mocap_uav, des_pos)
        msg.header.stamp = rospy.Time.now()
        self.pub_mpc_activity.publish(String(data="start_pos")
)
        self.pub_att.publish(msg)
        
    def _pub_hold_pos(self,des_pos):
        # moves drone to des_pos: 1D (3,) array in ENU frame
        try: 
            msg = self.pid.publish_cmd(self._mocap_uav,des_pos)
            msg.header.stamp = rospy.Time.now()
            self.pub_mpc_activity.publish(String(data="hold_pos"))
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

    def _pub_mpc_cmd(self, thrust, quat_des): 
        mpc_output_msg = Vector3Stamped()
        mpc_output_msg.header.stamp = rospy.Time.now()
        for idx, it in enumerate("xyz"):
            setattr(mpc_output_msg.vector, it, self.mpc.u[0, idx])
        self.pub_mpc_output.publish(mpc_output_msg)

        msg_att = AttitudeTarget()
        msg_att.header.stamp = rospy.Time.now()
        msg_att.orientation.x = quat_des[0]
        msg_att.orientation.y = quat_des[1]
        msg_att.orientation.z = quat_des[2]
        msg_att.orientation.w = quat_des[3]
        msg_att.thrust = thrust
        self.pub_att.publish(msg_att)

        self.pub_mpc_activity.publish(String(data="mpc_cmd"))
    
    def _run_mpc(self, xref):
        mpc = self.mpc
        mpc.xHist = np.vstack((mpc.xHist,
                               np.vstack((mpc.pld_rel_pos,
                                          mpc.uav_pos, 
                                          mpc.pld_rel_vel,
                                          mpc.uav_vel)).T))
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
            pd.DataFrame(mpc.xHist).to_csv("xHist.csv")
            pd.DataFrame(mpc.uHist).to_csv("uHist.csv")
            pd.DataFrame(mpc.uHist_ude).to_csv("uHist_ude.csv")
            pd.DataFrame(self.tHist).to_csv("tHist.csv")
            last_pos = self._mocap_uav.position
            while not rospy.is_shutdown():
                self._pub_ff_hold_pos(last_pos)
        else:
            self.mpc._solve_mpc(xref, self._mocap_uav, self._mocap_pld) # update ctrl input
            thrust, quat_des = att.att_extract(mpc.F_act)
            # thrust, quat_des = att.att_extract(mpc.u[0,:])
            self._pub_mpc_cmd(thrust, quat_des)
