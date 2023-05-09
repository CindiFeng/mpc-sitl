#!/usr/bin/env python

"""Class for writing position controller."""

from __future__ import division, print_function, absolute_import

from scipy.spatial.transform import Rotation as scipyR

# Import ROS libraries
import rospy
import numpy as np

# Import class 
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import PositionTarget, AttitudeTarget

class runPID(object):
    """ROS interface for controlling the Parrot ARDrone in the Vicon Lab."""
    # input:    current states (TransformStamped)
    #           desired states (TransformStamped)
    # output:   control inputs (Twist)

    def __init__(self):

        # PD controller gains for x, y, z, psi errors
        self.kpx, self.kdx, self.kix = 0.8, 2, 0.1
        self.kpy, self.kdy, self.kiy = self.kpx, self.kdx, self.kix
        self.kpz, self.kdz, self.kiz =  1, 0.6, 0.035 
        self.kppsi, self.kdpsi = 1, 0
        
        # integration
        self.x_int_err = 0
        self.y_int_err = 0
        self.z_int_err = 0

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

    def heading_compensator(self, psi_err):
        if psi_err > np.pi:
            return psi_err - 2 * np.pi
        elif psi_err < - np.pi:
            return psi_err + 2 * np.pi
        else:
            return psi_err

    def publish_cmd(self, _mocap_uav, des_pos):
        # des_pos is a 1D array 

        self.get_odom(_mocap_uav) # get current vehicle ground truth

        g = 9.8

        time = self.mocap_header.stamp.secs + (self.mocap_header.stamp.nsecs) * 10**-9
        dt = time - self.last_time
        if dt == 0.0:
            dt = 0.01

        (phi, theta, psi) = euler_from_quaternion([self.qx, self.qy, self.qz, self.qw])

        # desired position 
        x_des = des_pos[0]
        y_des = des_pos[1]
        z_des = des_pos[2]
        psi_des = 0

        # current error
        x_err = x_des - self.x
        y_err = y_des - self.y
        z_err = z_des - self.z
        psi_err = self.heading_compensator(psi_des - psi)

        # integration error
        self.x_int_err += x_err * dt
        self.y_int_err += y_err * dt
        self.z_int_err += z_err * dt
        
        # numerical differentiator to obtain velocity error
        xd_err = (x_err - self.x_last_err) / dt
        yd_err = (y_err - self.y_last_err) / dt
        zd_err = (z_err - self.z_last_err) / dt
        psid_err = (psi_err - self.psi_last_err) / dt

        # determine control command
        xdd_des = 0 # feedforward term
        ydd_des = 0
        zdd_des = 0
        psi_des = 0
        xdd_c = self.kdx * xd_err + self.kpx * x_err + self.kix * self.x_int_err + xdd_des
        ydd_c = self.kdy * yd_err + self.kpy * y_err + self.kiy * self.y_int_err + ydd_des
        zdd_c = self.kdz * zd_err + self.kpz * z_err + self.kiz * self.z_int_err + zdd_des
        psi_c = self.kdpsi * psid_err + self.kppsi * psi_err + zdd_des

        zd = (self.z-self.z_last) / dt
        zdd = (zd - self.zd_last) / dt

        # f_z = (zdd + g)/(np.cos(theta)*np.cos(phi))

        # calculated commands: roll, pitch, and force in z-axis
        r_c = np.arctan(-ydd_c * np.cos(theta) / (zdd + g))
        p_c = np.arctan(xdd_c / (zdd + g))
    
        phi_c = r_c * np.cos(psi) + p_c * np.sin(psi)
        theta_c = -r_c * np.sin(psi) + p_c * np.cos(psi)

        r = scipyR.from_euler('xyz',[self.saturation(phi_c),self.saturation(theta_c),0],degrees=False)
        quat_c = r.as_quat()
        pub_cmd = AttitudeTarget()
        pub_cmd.orientation.x = quat_c[0]
        pub_cmd.orientation.y = quat_c[1]
        pub_cmd.orientation.z = quat_c[2]
        pub_cmd.orientation.w = quat_c[3]
        pub_cmd.thrust = self.saturation(zdd_c)
        # pub_cmd.type_mask = 192
        # pub_cmd.body_rate.x = self.saturation(phi_c)
        # pub_cmd.body_rate.y = self.saturation(theta_c)
        # pub_cmd.body_rate.z = self.saturation(psi_c)

        # pub_cmd = PositionTarget()
        # pub_cmd.coordinate_frame = 1
        # pub_cmd.type_mask = 7
        # pub_cmd.velocity.x = self.saturation(phi_c) # commanded roll angle
        # pub_cmd.velocity.y = self.saturation(theta_c) # commanded pitch angle
        # pub_cmd.velocity.z = self.saturation(zdd_c) # commanded z-velocity
        # pub_cmd.yaw_rate = self.saturation(psi_c) # commanded yaw rate

        # update
        self.last_time = time
        self.x_last_err = x_err
        self.y_last_err = y_err
        self.z_last_err = z_err
        self.psi_last_err = psi_err

        self.z_last = self.z
        self.zd_last = zd

        return pub_cmd

    # def hold_pos(self, _mocap_uav, last_pos): 
    #     self.publish_cmd(_mocap_uav, last_pos)

        


