#!/usr/bin/env python

import casadi as ca
import init
import mpc_functions as mpcFunc
import numpy as np
import time

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import Thrust
from nav_msgs.msg import Odometry

def gen_solver():
    """
    Uses CasADi symbolics framework to interface with IPOPT solver. 
    Objective function and inequality equations are built up using CasADi 
    symbolics and passed to the IPOPT solver generation class.
    
    Input: none
    Return: args <dict>
            solver <IPOPT solver class>
    """

    # Design Nonlinear init.model Predictive Controller
    N = init.params["control"]["predictionHorizon"]
    Nu = init.params["control"]["controlHorizon"]
    Ts = init.params["control"]["sampleTime"]

    x = ca.SX.sym('x',init.model["n_x"]) # system states
    u = ca.SX.sym('u',init.model["n_u"]) # control inputs
    dxdt = mpcFunc.slungLoadDyn(x,u,init) # rhs of EOM

    sys_dyn = ca.Function('sys_dyn',[x,u],[dxdt]) # nonlinear mapping function f(x,u)
    U = ca.SX.sym('U',init.model["n_u"],Nu) # manipulative variables
    P = ca.SX.sym('P',init.model["n_p"]) # parameters incl x0 and xref
    X = ca.SX.sym('X',init.model["n_x"],(N + 1)) # states over the prediction horizon

    n_X = init.model["n_x"] * (N + 1) # number of predicted states
    n_U = init.model["n_u"] * Nu # number of predicted inputs

    # Make decision variable one column vector with X and U over N horizon
    decision_var = ca.vertcat(ca.reshape(X,n_X,1), ca.reshape(U,n_U,1))
    n_XU = n_X + n_U

    # iniitialize objective function and equality/inequality vectors
    obj_fcn = 0

    con_ws = []
    con_obs = []
    con_pld_d = []
    con_eq = X[:,0]-P[init.idx['p']['x0'][0]:init.idx['p']['x0'][1]] # start equality constraints

    u_prev = U[:,0] # for input rate cost
    for k in range(0, N):
        x_cur = X[:,k] # actual states
        
        # optimize input up to Nu
        if k < Nu:
            u_cur = U[:,k] # controls 
        else:
            u_cur = U[:,Nu-1]
        
        x_next = X[:,k+1] # evaluated states at next time step f(x,u)
        xdot = sys_dyn(x_cur,u_cur) # dxdt
        x_next_actual = mpcFunc.util_RK4(sys_dyn,Ts,x_cur,u_cur) # discretized via RK4
        
        # constraints
        con_eq = ca.vertcat(con_eq, x_next-x_next_actual) # multiple shooting equality
        ineq_fcn = mpcFunc.ineqConFcn(x_cur,init) 
        con_ws = ca.vertcat(con_ws, ineq_fcn[init.idx["g"]["ineq_ws"][0]:init.idx["g"]["ineq_ws"][1]])
        con_obs = ca.vertcat(con_obs, ineq_fcn[init.idx["g"]["ineq_obs"][0]:init.idx["g"]["ineq_obs"][1]])
        con_pld_d = ca.vertcat(con_pld_d, ineq_fcn[init.idx["g"]["ineq_pld_d"]])
        
        # objective function
        obj_fcn = obj_fcn + mpcFunc.costFcn(x_cur,u_cur,u_prev,P,init)       
        u_prev = u_cur
        
        if( k == (N - 1)): 
            # terminal objective function
            init.params["control"]["cost_nav"] = init.params["control"]["cost_nav_N"]
            obj_fcn = obj_fcn + mpcFunc.costFcn(x_next,u_cur,u_prev,P,init)
            
            ineq_fcn = mpcFunc.ineqConFcn(x_cur,init)
            con_ws = ca.vertcat(con_ws, ineq_fcn[init.idx["g"]["ineq_ws"][0]:init.idx["g"]["ineq_ws"][1]])
            con_obs = ca.vertcat(con_obs, ineq_fcn[init.idx["g"]["ineq_obs"][0]:init.idx["g"]["ineq_obs"][1]])
            con_pld_d = ca.vertcat(con_pld_d, ineq_fcn[init.idx["g"]["ineq_pld_d"]])

    # con_fcn = ca.vertcat(con_eq, con_ws, con_obs, con_pld_d) # inequality constraints
    # con_fcn = ca.vertcat(con_eq, con_ws, con_obs)
    con_fcn = con_eq

    # Set equality and inequality constraints
    # upper/lower function bounds lb <= g <= ub
    lbg = np.zeros((con_fcn.shape[0],1))
    ubg = np.zeros((con_fcn.shape[0],1))

    # bounds on equality constraints
    lbg[init.idx["g"]["eq"][0]:init.idx["g"]["eq"][1]] = 0 
    ubg[init.idx["g"]["eq"][0]:init.idx["g"]["eq"][1]] = 0

    # # bounds on workspace constraints
    # n_repeat = int((init.idx["g"]["ws"][1] - init.idx["g"]["ws"][0]) / 3)
    # lbg[init.idx["g"]["ws"][0]:init.idx["g"]["ws"][1]] = \
    #     np.tile(init.sim["workspace"][0,:].reshape(3,1), (n_repeat,1)) 
    # ubg[init.idx["g"]["ws"][0]:init.idx["g"]["ws"][1]] = \
    #     np.tile(init.sim["workspace"][1,:].reshape(3,1), (n_repeat,1))

    # # bounds on collision avoidance constraints
    # lbg[init.idx["g"]["obs"][0]:init.idx["g"]["obs"][1]] = 0 
    # ubg[init.idx["g"]["obs"][0]:init.idx["g"]["obs"][1]] = ca.inf

    # # bounds on payload swing radius
    # lbg[init.idx["g"]["pld_d"][0]:init.idx["g"]["pld_d"][1]] = 0 
    # ubg[init.idx["g"]["pld_d"][0]:init.idx["g"]["pld_d"][1]] = \
    #     init.params["cable_len"]**2

    # Set hard constraints on states and input
    # upper/lower variable bounds lb <= x <= ub
    lbx = -ca.inf*np.ones((n_XU,1))
    ubx = ca.inf*np.ones((n_XU,1))

    state_min = -ca.inf*np.ones((init.model["n_x"],1))
    state_max = ca.inf*np.ones((init.model["n_x"],1))
    state_min[init.idx["x"]["uav_pos"][0]:init.idx["x"]["uav_pos"][1]] = \
        init.sim["workspace"][0].reshape(3,1) + np.array([init.params["arm_len"], \
                                                        init.params["arm_len"], \
                                                            0]).reshape(3,1)
    state_max[init.idx["x"]["uav_pos"][0]:init.idx["x"]["uav_pos"][1]] = \
        init.sim["workspace"][1].reshape(3,1) - np.array([init.params["arm_len"], \
                                                        init.params["arm_len"], \
                                                            0]).reshape(3,1)
    for i in range(init.model["n_x"]):
        lbx[i:n_X:init.model["n_x"]] = state_min[i] # state lower limit
        ubx[i:n_X:init.model["n_x"]] = state_max[i] # state upper limit

    input_min = -ca.inf*np.ones((init.model["n_u"],1))
    input_max = ca.inf*np.ones((init.model["n_u"],1))
    for i in range(init.model["n_u"]):
        lbx[n_X+i:n_XU:init.model["n_u"]] = input_min[i] #input lower limit
        ubx[n_X+i:n_XU:init.model["n_u"]] = input_max[i] #input upper limit

    # inequality constraint arguments
    args = {
        'lbg' : lbg,
        'ubg' : ubg,
        'lbx' : lbx,
        'ubx' : ubx
    }

    # set up solver
    nlp_prob = {
        'f': obj_fcn,
        'x': decision_var,
        'g': con_fcn,
        'p': P
    }

    codeopts = {
        'ipopt': {
            'max_iter': 1000,
            'print_level': 0,
            'acceptable_tol': 1e-8,
            'acceptable_obj_change_tol': 1e-4
        },
        'print_time': 0
    }

    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, codeopts)

    return(args, solver)

class runMPCNode(): 
    
    def __init__(self,args,solver): 
        # publisher
        pub_freq = 10
        self.rate = rospy.Rate(pub_freq)
        self.pub_thrust = rospy.Publisher("/mavros/setpoint_attitude/thrust", Thrust, queue_size=1)
        self.pub_ang_vel = rospy.Publisher("/mavros/setpoint_attitude/cmd_vel", TwistStamped, queue_size=1)
        self.pub_hold = rospy.Publisher("/mavros/setpoint_position/local", PoseStamped, queue_size=1)
        self.pub_start = rospy.Publisher("/mavros/setpoint_position/local", PoseStamped, queue_size=1)
        
        # subscriber
        # self.uav_pos = init.ics["uav_pos"] + 10
        # self.uav_vel = init.ics["uav_vel"]
        # self.pld_pos = 0
        # self.pld_vel = 0
        # self.pld_rel_pos = init.ics["pld_rel_pos"]
        # self.pld_rel_vel = init.ics["pld_rel_vel"]
        # self.uav_msg = Odometry()
        # self.pld_msg = Odometry() 
        self.uav_odom = rospy.Subscriber('/ground_truth/uav/pose', Odometry, self.uav_callback)
        self.pld_odom = rospy.Subscriber('/ground_truth/payload/pose', Odometry, self.pld_callback)

        # initialize solver parameters
        self.args = args
        self.solver = solver
        N = init.params["control"]["predictionHorizon"]
        Nu = init.params["control"]["controlHorizon"]
        u0 = -init.params["derived"]["sys_weight"] # controls from mpc solver
        x0 = np.vstack((init.ics["pld_rel_pos"],
                        init.ics["uav_pos"],
                        init.ics["pld_rel_vel"],
                        init.ics["uav_vel"]))
        self.u = np.tile(u0,(1,Nu)).T
        self.sol_x = np.tile(x0,(1,N+1)).T
        # print('initial starting states:', self.sol_x)
        self.prev_R_des = 0

    def uav_callback(self,msg):
        # self.uav_msg = msg
        self.uav_pos = np.array([[msg.pose.pose.position.x],
                                 [msg.pose.pose.position.y],
                                 [-msg.pose.pose.position.z]])
        self.uav_vel = np.array([[msg.twist.twist.linear.x],
                                 [msg.twist.twist.linear.y],
                                 [-msg.twist.twist.linear.z]])

    def pld_callback(self,msg):
        # self.pld_msg = msg
        self.pld_pos = np.array([[msg.pose.pose.position.x],
                                 [msg.pose.pose.position.y],
                                 [-msg.pose.pose.position.z]])
        self.pld_vel = np.array([[msg.twist.twist.linear.x],
                                 [msg.twist.twist.linear.y],
                                 [-msg.twist.twist.linear.z]])
        pld_rel_pos = self.pld_pos - self.uav_pos
        self.pld_rel_pos = pld_rel_pos[0:2]
        pld_rel_vel = self.pld_vel - self.uav_vel
        self.pld_rel_vel = pld_rel_vel[0:2]

    def publish_cmd(self): 
        # derive control inputs to publish as thrust and angular velocities

        self.solve_mpc() # update ctrl input 
        thrust, ang_vel = self.att_extract()
        if thrust < 0 or thrust > 1: 
            raise Exception("thrust cmd is out of bounds.")

        msg_thrust = Thrust()
        msg_ang_vel = TwistStamped()
        msg_thrust.thrust = thrust
        msg_ang_vel.twist.angular.x = ang_vel[0]
        msg_ang_vel.twist.angular.y = ang_vel[1]
        msg_ang_vel.twist.angular.z = -ang_vel[2]
        
        # rospy.loginfo("publishing thrust %s and angular vel %s")
        self.pub_thrust.publish(msg_thrust)
        self.pub_ang_vel.publish(msg_ang_vel)

    def hold_pos(self): 
        # publish command to hover at current position

        sp = PoseStamped()
        sp.pose.position.x = self.uav_pos[0]
        sp.pose.position.y = self.uav_pos[1]
        sp.pose.position.z = -self.uav_pos[2]
        self.pub_hold.publish(sp)

    def start_pos(self):
        # publish cmd to hover at initial position

        sp = PoseStamped()
        sp.pose.position.x = init.ics["uav_pos"][0]
        sp.pose.position.y = init.ics["uav_pos"][1]
        sp.pose.position.z = -init.ics["uav_pos"][2]
        self.pub_start.publish(sp)

    def solve_mpc(self): 
        # update MPC solver arguments and solve for control input

        N = init.params["control"]["predictionHorizon"]
        Nu = init.params["control"]["controlHorizon"]
        n_X = init.model["n_x"] * (N + 1) 
        n_U = init.model["n_u"] * Nu
        
        xref = np.vstack((init.params["mission"]["pld_rel_pos"],
                          init.params["mission"]["uav_pos"], 
                          init.params["mission"]["pld_rel_vel"], 
                          init.params["mission"]["uav_vel"]))
        
        x0 = np.vstack((self.pld_rel_pos,
                        self.uav_pos,
                        self.pld_rel_vel,
                        self.uav_vel)) 
        self.args['p'] = ca.vertcat(x0, xref)
        
        X0 = self.sol_x.reshape((N + 1, init.model['n_x']))
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

        self.sol_x = mpcFunc.util_DM2Arr(sol['x'][0:n_X]).T
        self.u = ca.reshape(sol['x'][n_X:], init.model["n_u"], Nu).T
    

    def att_extract(self): 
        # Takes control input from mpc solver to derive the desired angular velocities

        f_ctrl = mpcFunc.util_DM2Arr(self.u[0,:].T) # extract the first control to be applied
        psi_des = 0 # command yaw angle 
        
        # build rotation matrix using unit vectors
        n_z = -f_ctrl / mpcFunc.util_norm(f_ctrl) # assume lift along the -z axis
        n_x_tilde = np.array([np.cos(psi_des), np.sin(psi_des), 
                        -(np.cos(psi_des) * n_z[0,0] + np.sin(psi_des) * 
                        n_z[1,0]) / n_z[2,0]]).reshape((3,1))
        n_x = n_x_tilde / mpcFunc.util_norm(n_x_tilde)
        n_y = mpcFunc.util_hat(n_z) @ n_x / mpcFunc.util_norm(mpcFunc.util_hat(n_z) @ n_x)
        R_des = np.hstack((n_x, n_y, n_z))

        Rd_des = (R_des - self.prev_R_des)/init.params["control"]["sampleTime"] # TODO: derivative of rotation matrix 
        self.prev_R_des = R_des

        ang_vel_d = mpcFunc.util_vee(R_des.T @ Rd_des)

        return (n_z[2], ang_vel_d)


if __name__ == '__main__': 
    try: 
        args, solver = gen_solver()
        rospy.init_node('mpc_SITL', anonymous=True)
        run_mpc = runMPCNode(args,solver)
        
        # fly uav to start position
        print("Waiting for 5 seconds to connect to subscriber.")
        time.sleep(5)
        print("Wait is over.")
        error = 1
        while error > 0.4:
            run_mpc.start_pos() # fly uav to start position
            run_mpc.rate.sleep()
            error = run_mpc.uav_pos - init.ics["uav_pos"].reshape((3,1))
            error = (error.T @ error)**0.5
        print("Starting MPC solver loop.")

        while not rospy.is_shutdown(): 
            # if goal is reached, hold uav at goal 
            # else, run MPC solver to get next commands
            # error = run_mpc.uav_pos - init.params["mission"]["uav_pos"].reshape((3,1))
            # error = mpcFunc.util_ca_sq_norm(error)**0.5
            # if error < 0.05: 
            #     print("Reached goal position.")
            #     run_mpc.hold_pos()
            # else:
            #     run_mpc.publish_cmd()
            run_mpc.publish_cmd()

            run_mpc.rate.sleep()

    except rospy.ROSInterruptException: 
        pass