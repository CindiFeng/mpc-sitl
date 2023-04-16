# -*- coding: utf-8 -*-
import json as js 
import os 
import numpy as np

# gives the path of this file
path = os.path.realpath(__file__)
dir = dir = os.path.dirname(path)
print('current path is:', path)

# load all simulation parameters
params_filepath = dir + '/params.json' 
ics_filepath = dir + '/ics.json' 
# sim_filepath = dir + '\init_settings\generated_data\sim_.mat' 
with open (params_filepath, "r") as f:
    params = js.loads(f.read()) # params[missions] waypoints are stored as  
                                # nested row vectors in lists

for i in params["mission"]: 
    n = len(params["mission"][i][0])
    params["mission"][i] = np.array(params["mission"][i]).reshape(n,)
                                
with open (ics_filepath, "r") as f:
    ics = js.loads(f.read())

# turn lists into column vectors
for i in ics: 
    n = len(ics[i])
    ics[i] = np.array(ics[i]).reshape(n,)

# with open (sim_filepath, "r") as f:
#     sim_ = js.loads(f.read())  

# simulation environment parameters
sim = {
    "g" : 9.80665,
    "grav" : np.array([0, 0, 9.80665]).reshape(3,1),
    "workspace" : np.array([[-0.6, -1, -2],   # min x, y, z
                            [0.3,  10,  0]]),  # max x, y, z
    "obs_pos" : np.array([-0.4, 1.5, -0.7]).reshape(3,1),
    "obs_dim" : np.array([0.68, 0.55, 0.9]),
    "duration" : 6,
    }

derived = {
    "sys_mass" : params["uav_mass"] + params["pld_mass"],
    "uav_weight" : params["uav_mass"] * sim["grav"],
    "pld_weight" : params["pld_mass"] * sim["grav"],
    "sys_weight" : (params["uav_mass"] + params["pld_mass"]) * sim["grav"]
    }
params["derived"] = derived

# nlobj information 
model = {
    "n_x" : 10,               # number of states
    "n_u" : 3,                 # number of input
    "name": 'sl_drone'
}
model["n_p"] = model["n_x"] * 2 # number of parameters incl x0 and xref

# nlobj index splices
g = {
    "ineq_obs" : (0,2),
    "ineq_ws" : (2,5),
    "ineq_pld_d" : 5,
    "eq" : (0,model["n_x"] * (params["control"]["predictionHorizon"] + 1))
}
i_start = g["eq"][1]
i_end = (g["eq"][1] + (g["ineq_ws"][1] - g["ineq_ws"][0]) * 
        (params["control"]["predictionHorizon"] + 1))
g["ws"] = (i_start,i_end)
i_start = g["ws"][1]
i_end = (g["ws"][1] + (g["ineq_obs"][1] - g["ineq_obs"][0]) * 
        (params["control"]["predictionHorizon"] + 1))
g["obs"] = (i_start,i_end)
i_start = g["obs"][1]
i_end = g["obs"][1] + (params["control"]["predictionHorizon"] + 1)
g["pld_d"] = (i_start,i_end)

p = {
    "x0" : (0,model["n_x"]), 
    "x_goal" : (model["n_x"],model["n_p"])
}

x = {
    "pld_rel_pos" : (0,2),
    "uav_pos" : (2,5),
    "pld_rel_vel" : (5,7),
    "uav_vel" : (7,10)
}

idx = {
    "g" : g,
    "x" : x,
    "p" : p
}

ocp = {
    'cost_type' : 'EXTERNAL',
    'cost_type_e' : 'EXTERNAL',
    'constr_type' : 'BGH',
    'hessian_approx' : 'EXACT',
    'regularize_method' : 'NO_REGULARIZE',
    'integrator_type' : 'ERK',
    'nlp_solver_type' : 'SQP',
    'nlp_solver_max_iter' : 100, 
    'nlp_solver_tol_stat' : 1e-6,
    'nlp_solver_tol_eq': 1e-6, 
    'nlp_solver_tol_ineq' : 1e-6, 
    'nlp_solver_tol_comp' : 1e-6,
    'nlp_solver_ext_qp_res' : 1, 
    'qp_solver' : 'PARTIAL_CONDENSING_HPIPM',
    'qp_solver_max_iter' : 50,
    'qp_solver_cond_ric_alg' : 0, 
    'qp_solver_ric_alg' : 0, 
    'qp_solver_warm_start' : 2
}
