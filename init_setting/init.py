# -*- coding: utf-8 -*-

### TODO: FIX PATHS SO THIS INSERT IS NOT NECESSARY ###
import sys
sys.path.insert(0, '/home/fmncindi/Research/scripts/mpc/init_setting')
#######################################################

import os 
import json as js 
import numpy as np
import para_list

# gives the path of this file
path = os.path.realpath(__file__)
dir = os.path.dirname(path)

# load all simulation parameters
params_filepath = dir + '/params.json' 
ics_filepath = dir + '/ics.json' 
with open (params_filepath, "r") as f:
    params = js.loads(f.read()) # params[missions] waypoints are stored as  
                                # nested row vectors in lists

for i in params["mission"]: 
    n = len(params["mission"][i][0]) 
    params["mission"][i] = np.array(params["mission"][i]).reshape(n,1) 
                                
with open (ics_filepath, "r") as f:
    ics = js.loads(f.read())

# turn lists into column vectors
for i in ics: 
    n = len(ics[i]) # size of list
    ics[i] = np.array(ics[i]).reshape(n,1) 

x_preGen = np.loadtxt(dir+"/xRef.csv", delimiter=",")
x_track1 = np.loadtxt(dir+"/x_track.csv", delimiter=",")
x_track2 = np.loadtxt(dir+"/x_track2.csv", delimiter=",")
# u_preGen = np.loadtxt(dir+"/uRef.csv",delimiter=",")

# simulation environment parameters

sim = para_list.sim_list[-1]

pub_rate = 20

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
    "n_u" : 3                 # number of input
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

# loading mat files
# from scipy import io
# def load_from_mat(filename=None, data={}, loaded=None):
#     if filename:
#         vrs = io.whosmat(filename)
#         name = vrs[0][0]
#         loaded = io.loadmat(filename,struct_as_record=True)
#         loaded = loaded[name]
#     whats_inside = loaded.dtype.fields
#     fields = list(whats_inside.keys())
#     for field in fields:
#         if len(loaded[0,0][field].dtype) > 0: # it's a struct
#             data[field] = {}
#             data[field] = load_from_mat(data=data[field], loaded=loaded[0,0][field])
#         else: # it's a variable
#             data[field] = loaded[0,0][field]
#     return data

# # and then just call the function
# with open (sim_filepath, "r") as f:
#     data = load_from_mat(filename=f.read()) # Don't worry about the other input vars (data, loaded), there are used in the recursion.
