#!/usr/bin/env python

# Import ROS libraries
import rospy
import numpy as np

# Import class 
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import TwistStamped
from mavros_msgs.msg import PositionTarget

class runPID(object):
    """Backup PID controller."""
    # input:    current states (1D arrays of pos, vel, ang vel, quat)
    #           desired states (1D array (3,))
    # output:   PID velocity cmds 

    def __init__(self):

        # PD controller gains for x, y, z, psi errors
        self.kpx, self.kdx, self.kix = 0.9, 0.05, 0
        self.kpy, self.kdy, self.kiy = self.kpx, self.kdx, self.kix
        self.kpz, self.kdz =  1, 0.1
        self.kppsi, self.kdpsi = 1, 0
        
        # integration
        self.x_int_err = 0
        self.y_int_err = 0

        # init for numerical differentiating
        self.x_last_err = 0
        self.y_last_err = 0
        self.z_last_err = 0
        self.psi_last_err = 0
        self.last_time = rospy.Time.now().to_sec()

        self.z_last = 0
        self.zd_last = 0
    
    def get_odom(self,msg):
        # drone position and velocity in ENU frame 

        self.mocap_header = msg.header
        self.x = msg.position[0]
        self.y = msg.position[1]
        self.z = msg.position[2]
        self.qw = msg.quaternion[0]
        self.qx = msg.quaternion[1]
        self.qy = msg.quaternion[2]
        self.qz = msg.quaternion[3]
        self.vx = msg.velocity[0]
        self.vy = msg.velocity[1]
        self.vz = msg.velocity[2]

    def saturation(self, x):
        if x > 1.0:
            x = 1.0
        elif x < -1.0:
            x = -1.0
        return x

    def publish_cmd(self, _mocap_uav, des_pos):
        # des_pos is a 1D array 

        self.get_odom(_mocap_uav) # get current vehicle ground truth

        g = 9.8

        time = self.mocap_header.stamp.secs + (self.mocap_header.stamp.nsecs) * 10**-9
        dt = time - self.last_time
        if dt == 0.0:
            dt = 0.01

        # desired position 
        x_des = des_pos[0]
        y_des = des_pos[1]
        z_des = des_pos[2]

        # current error
        x_err = x_des - self.x
        y_err = y_des - self.y
        z_err = z_des - self.z

        # integration error
        self.x_int_err += x_err * dt
        self.y_int_err += y_err * dt
        
        # numerical differentiator to obtain velocity error
        xd_err = (x_err - self.x_last_err) / dt
        yd_err = (y_err - self.y_last_err) / dt
        zd_err = (z_err - self.z_last_err) / dt

        # determine control command
        xdd_des = 0 # feedforward term
        ydd_des = 0
        zdd_des = 0
        xdd_c = self.kdx * xd_err + self.kpx * x_err + self.kix * self.x_int_err + xdd_des
        ydd_c = self.kdy * yd_err + self.kpy * y_err + self.kiy * self.y_int_err + ydd_des
        zdd_c = self.kdz * zd_err + self.kpz * z_err + zdd_des

        zd = (self.z-self.z_last) / dt
        zdd = (zd - self.zd_last) / dt

        pub_cmd = PositionTarget()
        pub_cmd.coordinate_frame = 1
        pub_cmd.type_mask = 7
        pub_cmd.velocity.x = self.saturation(xdd_c) 
        pub_cmd.velocity.y = self.saturation(ydd_c) 
        pub_cmd.velocity.z = self.saturation(zdd_c) 

        # update
        self.last_time = time
        self.x_last_err = x_err
        self.y_last_err = y_err
        self.z_last_err = z_err

        self.z_last = self.z
        self.zd_last = zd

        return pub_cmd

    # def hold_pos(self, _mocap_uav, last_pos): 
    #     self.publish_cmd(_mocap_uav, last_pos)

        


