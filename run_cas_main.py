#!/usr/bin/env python

import casadi as ca
import init
import mpc_functions as mpcFunc
import numpy as np 
import time
import rospy

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
con_fcn = ca.vertcat(con_eq, con_ws, con_obs)

# Set equality and inequality constraints
# upper/lower function bounds lb <= g <= ub
lbg = np.zeros((con_fcn.shape[0],1))
ubg = np.zeros((con_fcn.shape[0],1))

# bounds on equality constraints
lbg[init.idx["g"]["eq"][0]:init.idx["g"]["eq"][1]] = 0 
ubg[init.idx["g"]["eq"][0]:init.idx["g"]["eq"][1]] = 0

# bounds on workspace constraints
n_repeat = int((init.idx["g"]["ws"][1] - init.idx["g"]["ws"][0]) / 3)
lbg[init.idx["g"]["ws"][0]:init.idx["g"]["ws"][1]] = \
    np.tile(init.sim["workspace"][0,:].reshape(3,1), (n_repeat,1)) 
ubg[init.idx["g"]["ws"][0]:init.idx["g"]["ws"][1]] = \
    np.tile(init.sim["workspace"][1,:].reshape(3,1), (n_repeat,1))

# bounds on collision avoidance constraints
lbg[init.idx["g"]["obs"][0]:init.idx["g"]["obs"][1]] = 0 
ubg[init.idx["g"]["obs"][0]:init.idx["g"]["obs"][1]] = ca.inf

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

################################################################################
# Simulation starts here
x0 = np.vstack((init.ics["pld_rel_pos"], init.ics["uav_pos"], 
               init.ics["pld_rel_vel"], init.ics["uav_vel"]))
u0 = -init.params["derived"]["sys_weight"]
xref = np.vstack((init.params["mission"]["pld_rel_pos"], 
                 init.params["mission"]["uav_pos"], 
                 init.params["mission"]["pld_rel_vel"], 
                 init.params["mission"]["uav_vel"]))

X0 = np.tile(x0,(1,N+1)).T # init states decision variables
U0 = np.tile(u0,(1,Nu)).T

# Store history of states and input
mpc_maxiter = int(init.sim["duration"]/Ts)
xHistory = np.zeros((mpc_maxiter+1, init.model["n_x"]))
xHistory[0,:] = x0.T 
xPlanned = np.zeros((N, init.model["n_x"], mpc_maxiter))
uHistory = np.zeros((mpc_maxiter, init.model["n_u"]))

# Start MPC
mpciter = 0
t_start = time.time()

t = np.zeros((mpc_maxiter+1,1)) # Store history time
t[0] = 0

# the main simulaton loop... it works as long as the error is greater
# than 0.2 and the number of mpc steps is less than its maximum
# value
while(mpcFunc.util_sq_norm(x0-xref) > 0.01 and mpciter < mpc_maxiter):
    # set values of the parameters vector
    args['p'] = ca.vertcat(x0,
                           xref) 
    # initial value of the optimization variables
    args['x0'] = ca.vertcat(X0.reshape((n_X,1)),
                            U0.reshape((n_U,1))) 

    # pass on arguments values to solver Function made earlier 
    sol = solver(
        x0=args['x0'],
        lbx=args['lbx'],
        ubx=args['ubx'],
        lbg=args['lbg'],
        ubg=args['ubg'],
        p=args['p']
    )

    
    # get controls only from the solution
    u = ca.reshape(sol['x'][n_X:], init.model["n_u"], Nu).T

    # solution trajectory (planned trajecotry)
    sol_x = mpcFunc.util_DM2Arr(sol['x'][0:n_X]).T
    
    uHistory[mpciter,:] = u[0,:]
    
    # apply the control and shift the solution
    sol_x_next = mpcFunc.util_RK4(sys_dyn,Ts,x0,u[0,:].T)
    x0 = mpcFunc.util_DM2Arr(sol_x_next)
    U0 = np.vstack((u[1:,:], u[-1,:]))
    xHistory[mpciter+1,:] = x0.T # actual trajectory at current time step
    
    # shift trajectory to initialize the next step
    # X0 = sol_x.reshape((init.model["n_x"], N + 1),order='F').T
    X0 = sol_x.reshape((N + 1, init.model['n_x']))
    xPlanned[:,0:init.model["n_x"],mpciter] = X0[1:,:]
    X0 = np.vstack((X0[1:,:], X0[-1,:]))

    mpciter += 1
    t[mpciter] = t[mpciter-1] + Ts

tot_t = time.time() - t_start
avg_t = tot_t/mpciter
print('Average time per loop: ', avg_t)
ss_error = mpcFunc.util_sq_norm(x0-xref)
print('Steady state error: ', ss_error)

mpcFunc.util_plot2d(xHistory,uHistory,init)