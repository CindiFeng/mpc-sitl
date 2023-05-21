#!/usr/bin/env python3

import casadi as ca
import init
import util_fcn as util
import numpy as np

class runMPC(): 
    
    def __init__(self,args,solver):
        
        # initialize solver parameters
        self.args = args
        self.solver = solver
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
        # self.xHist = x0[0:5].T
        self.xHist = x0.T

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
        
        sol = self.solver(
            x0=self.args['x0'],
            lbx=self.args['lbx'],
            ubx=self.args['ubx'],
            lbg=self.args['lbg'],
            ubg=self.args['ubg'],
            p=self.args['p']
        )

        sol_x = util.DM2Arr(sol['x'][0:n_X]).T
        self.sol_x = sol_x.reshape((N + 1, init.model['n_x']))

        self.u = util.DM2Arr(ca.reshape(sol['x'][n_X:], 
                                        init.model["n_u"], Nu).T)
        self.uHist = np.vstack((self.uHist,self.u[0,:]))
    
    def get_odom(self, uav_msg, pld_msg):
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