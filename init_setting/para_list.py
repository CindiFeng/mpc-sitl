import numpy as np
sim_list = [
    {
    "g" : 9.80665,
    "grav" : np.array([0, 0, 9.80665]).reshape(3,1),
    "workspace" : np.array([[-0.7,  -1, 0],   # min x, y, z
                            [ 0.7,  10, 6]]),  # max x, y, z
    "obs_pos" : np.array([-0.4, 1.5, 3.9]).reshape(3,1),
    "obs_dim" : np.array([0.68, 0.55, 0.9]),
    "duration" : 6,
    },
    {
    "g" : 9.80665,
    "grav" : np.array([0, 0, 9.80665]).reshape(3,1),
    "workspace" : np.array([[-7,  -1, 0],   # min x, y, z
                            [ 7,  10, 60]]),  # max x, y, z
    "obs_pos" : np.array([-4, 15, 3.9]).reshape(3,1),
    "obs_dim" : np.array([3.8, 0.55, 0.9]),
    "duration" : 6,
    },
    {
    "g" : 9.80665,
    "grav" : np.array([0, 0, 9.80665]).reshape(3,1),
    "workspace" : np.array([[-3.5,  -1, 0],   # min x, y, z
                            [ 3.5,  10, 60]]),  # max x, y, z
    "obs_pos" : np.array([-2, 7.5, 3.9]).reshape(3,1),
    "obs_dim" : np.array([3, 0.55, 0.9]),
    "duration" : 6,
    },
    {
    "g" : 9.80665,
    "grav" : np.array([0, 0, 9.80665]).reshape(3,1),
    "workspace" : np.array([[-1.4,  -1, 0],   # min x, y, z
                            [ 1.4,  10, 60]]),  # max x, y, z
    "obs_pos" : np.array([-0.8, 3, 3.9]).reshape(3,1),
    "obs_dim" : np.array([1.34, 0.55, 0.9]),
    "duration" : 6,
    },
    {
    "g" : 9.80665,
    "grav" : np.array([0, 0, 9.80665]).reshape(3,1),
    "workspace" : np.array([[-1.4,  -1, 0],   # min x, y, z
                            [ 1.4,  10, 60]]),  # max x, y, z
    "obs_pos" : np.array([-0.8, 3, 3.9]).reshape(3,1),
    "obs_dim" : np.array([0.9, 0.55, 0.9]),
    "duration" : 6,
    },
    {
    "g" : 9.80665,
    "grav" : np.array([0, 0, 9.80665]).reshape(3,1),
    "workspace" : np.array([[-0.7,  -1, 0],   # min x, y, z
                            [ 2.7,  100, 6]]),  # max x, y, z
    "obs_pos" : np.array([-0.4, 1.5, 3.9]).reshape(3,1),
    "obs_dim" : np.array([0.68, 0.55, 4.9]),
    "duration" : 6,
    },
    {
    "g" : 9.80665,
    "grav" : np.array([0, 0, 9.80665]).reshape(3,1),
    "workspace" : np.array([[-0.7,  -1, 0],   # min x, y, z
                            [0.6,  10, 6]]),  # max x, y, z
    "obs_pos" : np.array([-0.4, 3.5, 3.4]).reshape(3,1),
    "obs_dim" : np.array([0.68, 0.55, 0.8]),
    "duration" : 6,
    }
]