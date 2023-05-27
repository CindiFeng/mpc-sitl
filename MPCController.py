#!/usr/bin/env python3

import casadi as ca
import init_setting as init
import util_fcn as util
import numpy as np
from ude_module import UDE  
import mpc_fcn 

class runMPC(): 
    
    def __init__(self):
        
        # initialize solver parameters

        self.args, self.solver = mpc_fcn._genSolver()

        N = init.params["control"]["predictionHorizon"]
        Nu = init.params["control"]["controlHorizon"]
        u0 = init.params["derived"]["sys_weight"] # controls from mpc solver
        x0 = np.vstack((init.ics["pld_rel_pos"],
                        init.ics["uav_pos"],
                        init.ics["pld_rel_vel"],
                        init.ics["uav_vel"]))
        self.u = np.tile(u0,(1,Nu)).T
        self.sol_x = np.tile(x0,(1,N+1)).T

        self.uHist = u0.reshape((1,init.model["n_u"]))
        self.xHist = x0.T

        self.uHist_ude = u0.reshape((1,init.model["n_u"]))
        self.f_des = u0.reshape((init.model["n_u"],1))
        self.ude = UDE(self.f_des)

        self.uav_pos = x0[init.idx["x"]["uav_pos"][0]:init.idx["x"]["uav_pos"][1]]
        self.uav_vel = x0[init.idx["x"]["uav_vel"][0]:init.idx["x"]["uav_vel"][1]]
        self.pld_rel_pos = x0[init.idx["x"]["pld_rel_pos"][0]:init.idx["x"]["pld_rel_pos"][1]]
        self.pld_rel_vel = x0[init.idx["x"]["pld_rel_vel"][0]:init.idx["x"]["pld_rel_vel"][1]]

    # def _solve_mpc(self, xref):  # for testing in MPCController
    # def _solve_mpc(self, xref, x_preGen): # for TESTING in run_main
    #     self.get_odom(x_preGen)
    def _solve_mpc(self, xref, _mocap_uav, _mocap_pld): 
        self.get_odom(_mocap_uav, _mocap_pld)      

        # update MPC solver arguments and solve for control input
        N = init.params["control"]["predictionHorizon"]
        Nu = init.params["control"]["controlHorizon"]
        n_X = init.model["n_x"] * (N + 1) 
        n_U = init.model["n_u"] * Nu
        
        x0 = np.vstack((self.pld_rel_pos,
                        self.uav_pos,
                        self.pld_rel_vel,
                        self.uav_vel)) 
        self.args['p'] = ca.vertcat(x0, xref)
        
        X0 = self.sol_x
        X0 = np.vstack((X0[1:,:], X0[-1,:]))
        U0 = np.vstack((self.u[1:,:], self.u[-1,:]))
        self.args['x0'] = ca.vertcat(X0.reshape((n_X,1)),
                                     U0.reshape((n_U,1)))
        
        # solve MPC by passing in necessary arguments for IPOPT
        sol = self.solver(
            x0=self.args['x0'],
            lbx=self.args['lbx'],
            ubx=self.args['ubx'],
            lbg=self.args['lbg'],
            ubg=self.args['ubg'],
            p=self.args['p']
        )

        sol_x = util.DM2Arr(sol['x'][0:n_X]).T 
        self.sol_x = sol_x.reshape((N + 1, init.model['n_x'])) # N x n_x planned states

        self.u = util.DM2Arr(ca.reshape(sol['x'][n_X:], 
                                        init.model["n_u"], Nu).T) # Nu x n_u optimal input
        self.uHist = np.vstack((self.uHist,self.u[0,:]))

        # get UDE compensated controls
        self.ude.estimateDisturbance(self.f_des, self.pld_rel_pos, 
                                     self.pld_rel_vel, self.uav_vel)
        f_ude = self.ude.f_ude
        f_mpc = self.u[0,:].reshape((3,1))
        self.f_des = f_mpc - f_ude
        self.uHist_ude= np.vstack((self.uHist_ude,f_ude.T))
  
    def get_odom(self, uav_msg, pld_msg):
        """
        Record uav and relative payload positions to match equations of motion.
        """
        self.uav_pos = np.array([[uav_msg.position[0]],
                                 [uav_msg.position[1]],
                                 [uav_msg.position[2]]])
        self.uav_vel = np.array([[uav_msg.velocity[0]],
                                 [uav_msg.velocity[1]],
                                 [uav_msg.velocity[2]]])
        pld_pos = np.array([[pld_msg.position[0]],
                            [pld_msg.position[1]],
                            [pld_msg.position[2]]])
        pld_vel = np.array([[pld_msg.velocity[0]],
                            [pld_msg.velocity[1]],
                            [pld_msg.velocity[2]]])
        pld_rel_pos = pld_pos - self.uav_pos
        self.pld_rel_pos = pld_rel_pos[0:2]
        pld_rel_vel = pld_vel - self.uav_vel
        self.pld_rel_vel = pld_rel_vel[0:2]

        # self.uav_quat = uav_msg.quaternion

############################### TESTING BELOW ##################################
    # def get_odom(self, msg):  # for TESTING in run_main   
    #     self.uav_pos = msg[2:5].reshape((3,1))
    #     self.uav_vel = msg[7:].reshape((3,1))
    #     self.pld_rel_pos = msg[0:2].reshape((2,1))
    #     self.pld_rel_vel = msg[5:7].reshape((2,1))


############################### TESTING BELOW ##################################
    # Uncomment below to test out runMPC class
    # def get_new_state(self):
    #     x = ca.SX.sym('x',init.model["n_x"]) # system states
    #     u = ca.SX.sym('u',init.model["n_u"]) # control inputs
    #     dxdt = mpc_fcn.slungLoadDyn(x,u) # rhs of EOM
    #     Ts = init.params["control"]["sampleTime"]

    #     f = ca.Function('sys_dyn',[x,u],[dxdt]) # nonlinear mapping function f(x,u)
    #     x_current = np.vstack((self.pld_rel_pos, self.uav_pos, self.pld_rel_vel, self.uav_vel))
    #     x_next = util.DM2Arr(util.RK4(f,Ts,x_current,self.u[0,:]))
    #     self.uav_pos = x_next[init.idx["x"]["uav_pos"][0]:init.idx["x"]["uav_pos"][1]]
    #     self.uav_vel = x_next[init.idx["x"]["uav_vel"][0]:init.idx["x"]["uav_vel"][1]]
    #     self.pld_rel_pos = x_next[init.idx["x"]["pld_rel_pos"][0]:init.idx["x"]["pld_rel_pos"][1]]
    #     self.pld_rel_vel = x_next[init.idx["x"]["pld_rel_vel"][0]:init.idx["x"]["pld_rel_vel"][1]]


# Uncomment below to test out runMPC class
# if __name__ == '__main__':
#     mpc = runMPC()
#     mpc_maxiter = int(init.sim["duration"]/init.params["control"]["sampleTime"])
#     mpciter = 0

#     xref = np.vstack((init.params["mission"]["pld_rel_pos"], 
#                  init.params["mission"]["uav_pos"], 
#                  init.params["mission"]["pld_rel_vel"], 
#                  init.params["mission"]["uav_vel"]))

#     while mpciter < mpc_maxiter: 
#         mpc._solve_mpc(xref)
#         mpc.get_new_state()
#         x_next = np.vstack((mpc.pld_rel_pos,
#                             mpc.uav_pos,
#                             mpc.pld_rel_vel,
#                             mpc.uav_vel)) 
#         mpc.xHist = np.vstack((mpc.xHist, x_next.T))
#         mpciter += 1

#     util.plot(mpc.xHist,mpc.uHist,mpc.uHist_ude,init)