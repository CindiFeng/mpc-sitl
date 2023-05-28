import numpy as np
from scipy.spatial.transform import Rotation as scipyR

def vee(ss): 
    """
    vee function to map a skew-symmetric matrix to a vector
    """
    # if (ss.shape != (3,3) or ss[2,1] != -ss[1,2] or ss[0,2] != -ss[2,0] 
    #     or ss[0,1] != -ss[1,0]): 
    #     raise Exception("The provided matrix is not skew symmetric.")
    
    vec = np.array([ss[2,1], ss[0,2], ss[0,1]]).reshape(3,1)

    return vec

def hat(v):
    """
    create skew symmetric matrix from vector
    """
    v = v.reshape(3,)
    ss = np.array([0,-v[2],v[1],v[2],0,-v[0],-v[1],v[0],0]).reshape(3,3)
    return ss

def norm(a):
    r = max(a.shape)
    a = a.reshape((r,))
    return ((a.T @ a)**0.5)

def eul2rot(eul): 
    """
    Input:
        eul: 1D array of euler angles in radians. Uses 'xyz' for extrinsic rotation.
    Output:
        rotation matrix with size (3,3)
    """
    r = scipyR.from_euler('xyz',eul,degrees=False)
    return r.as_matrix().reshape((3,3))

def rotm2quat(rotm): return scipyR.from_matrix(rotm).as_quat()

def eul2quat(eul): return scipyR.from_euler('xyz',eul,degrees=False).as_quat()

def force2thrust(f_des):
    """
    Input: 3D desired force vector
    Return: magnitude and normalized thrust
    """
    f_mag = np.linalg.norm(f_des)

    # thrust must be between 0 and 1
    thrust = (f_mag + 6)/36 
    if thrust > 1: 
        thrust = 1

    return f_mag, thrust

def att_extract(f_ctrl): 
        """
        Takes control input (in ENU) from mpc solver to derive desired quaternion
        """

        f_mag, thrust = force2thrust(f_ctrl)
        
        psi_des = 0 # command yaw angle 
        
        # build rotation matrix using unit vectors
        n_z = (f_ctrl / f_mag).reshape((3,1)) # assume lift along the z axis
        n_x_tilde = np.array([np.cos(psi_des), np.sin(psi_des), 
                        -(np.cos(psi_des) * n_z[0,0] + np.sin(psi_des) * 
                        n_z[1,0]) / n_z[2,0]]).reshape((3,1))
        n_x = n_x_tilde / np.linalg.norm(n_x_tilde)
        n_y = hat(n_z) @ n_x / np.linalg.norm(hat(n_z) @ n_x)
        R_des = np.hstack((n_x, n_y, n_z))

        quat_des = rotm2quat(R_des)
        return (thrust, quat_des)