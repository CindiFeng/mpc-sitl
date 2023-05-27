import sys
sys.path.insert(0, '/home/fmncindi/Research/scripts/mpc/')

import numpy as np
import init_setting as init
import attitude_conversion as att
import matplotlib.pyplot as plt

# This class provides UDE algorithm to correct the input received by robot
class UDE():
    
    def __init__(self, f_init):
        """
        Input: initial input to the robot (f_init); 3x1 vector
        """
        self.f_ude = 0 # estimated disturbance force using UDE algorithm
        # keep track of integral portion of UDE
        self.ude_int = (init.params["derived"]["sys_mass"] *
                        np.array([0,0,-init.sim["g"]]).reshape((3,1)) + 
                        f_init + 
                        self.f_ude)

    def estimateDisturbance(self, f_des, pld_rel_pos, pld_rel_vel, uav_vel):
        """
        Input:  actual force for robot (f_des); column vector
                uav and payload positions; column vectors
        Return: none

        Update the integration portion of UDE and the disturbance estimated 
        with the UDE algorithm
        """
        f_mag, thrust = att.force2thrust(f_des)
        n_z = (f_des / f_mag).reshape((3,1))
        f_act = (thrust*36 - 6)*n_z
    
        ude_lambda = init.params["control"]["ude_lambda"]
        m_p = init.params["pld_mass"]
        m_q = init.params["uav_mass"]
        m_tot = init.params["derived"]["sys_mass"]
        L = init.params["cable_len"]
        g_I=np.array([0, 0, -init.sim["g"]]).reshape((3,1))
        Ts = init.params["control"]["sampleTime"]
        
        r_L = pld_rel_pos
        B = np.vstack((np.eye(2), r_L.T / np.sqrt(L**2-np.linalg.norm(r_L)**2 )))
        v_L = pld_rel_vel
        v_q = uav_vel
        
        self.ude_int = self.ude_int - (m_tot * g_I + f_act + self.f_ude) * Ts

        self.f_ude = 1/ude_lambda * (m_p * B @ v_L + m_tot * v_q + self.ude_int)

# Uncomment below to test out UDE class
# if __name__ == '__main__': 
#     uRef = np.vstack((init.params["derived"]["sys_mass"] *
#                         np.array([0,0,init.sim["g"]]).reshape((1,3)),init.u_preGen[:-2,:]))
#     xHist = init.x_preGen
#     uHist_ude = np.zeros(uRef.shape)
#     ude = UDE(uRef[0,:].reshape((3,1)))
#     for ct, u_i in enumerate(uRef): 
#         pld_rel_pos = xHist[ct,0:2].reshape((2,1))
#         pld_rel_vel = xHist[ct,5:7].reshape((2,1))
#         uav_vel = xHist[ct,7:].reshape((3,1))
#         ude.estimateDisturbance(u_i.reshape((3,1)), 
#                                 pld_rel_pos, pld_rel_vel, uav_vel)
#         uHist_ude[ct,:] = ude.f_ude.T
    
#     fig1 = plt.figure('UDE Input') 
#     simT = np.linspace(0,
#                        init.params["control"]["sampleTime"]*(len(uHist_ude)-1),
#                        len(uHist_ude)) # simulation time step
#     plt.plot(simT,uHist_ude,linewidth=1)
#     plt.xlabel('time (s)')
#     plt.ylabel('force (N)')
#     plt.title('UDE Input')
#     plt.grid(color='#ADAAAB',linewidth=0.5)
#     plt.show()

