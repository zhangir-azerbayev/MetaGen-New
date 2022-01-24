import jax.numpy as np 
from jax import grad, jit, vmap 
import random
from scipy.stats import multivariate_normal
from pyinference.inference import *
import numpy 
import matplotlib.pyplot as plt
import os 
import jax.random as jrandom
import multiprocessing as mp 
from itertools import repeat
import time


os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

random.seed(12)
numpy.random.seed(12)
key = jrandom.PRNGKey(12)

def sample_baseline(num_objects, 
                    num_categories, 
                    num_observations, 
                    sigma
                    ): 
    cov = sigma * np.identity(3)
    #num_objects = random.randrange(1, max_objects+1)
    
    object_locations = [np.array([0.0, 0, 0])]
    for _ in range(num_objects): 
        x = float(random.uniform(-5, 5))
        y = random.uniform(0, 3)
        z = random.uniform(-5, 5)
        object_locations.append(np.array([x, y, z]))

    object_locations = np.stack(object_locations)

    object_categories = np.array([0] + [random.randrange(1, num_categories+1) 
        for _ in range(num_objects)])
    
    camera_locations = []
    directions = []
    obs_categories = []
    obs_objects = []
    for _ in range(num_observations): 
        x = float(random.uniform(-5, 5))
        y = random.uniform(0, 3)
        z = random.uniform(-5, 5)
        camera_location = np.array([x, y, z])

        obs_obj = random.randrange(1, num_objects+1)
        obj_location = object_locations[obs_obj]

        detection_location = np.array([multivariate_normal.rvs(obj_location, cov)])

        long_direction = detection_location-camera_location
        direction = long_direction/np.linalg.norm(long_direction)

        camera_locations.append(camera_location)
        directions.append(direction)
        obs_categories.append(object_categories[obs_obj])
        obs_objects.append(obs_obj)


    camera_locations = np.stack(camera_locations)
    directions = np.stack(directions)
    obs_categories = np.array(obs_categories)

    return object_locations, object_categories, camera_locations, directions, obs_categories, obs_objects


def test_baseline(): 
    K = 2
    sigma = 0.1
    num_categories = 2
    gt_object_locations, gt_object_categories, camera_locations, directions, obs_categories, obs_objects = sample_baseline(K, num_categories, 500, sigma)
    print(gt_object_categories)


    num_em_steps = 5
    num_gd_steps = 3000

    init_displacement = np.array([[0.0, 0, 0], [-1, 1, 2], [-3, -.5, .2]])
        #[.1, -1, .1], [2, .3, -1], [2, .3, -.2]])

    #object_locations = gt_object_locations + init_displacement
    object_locations = init_displacement
    object_categories = np.array([0, 1, 1])
    print(object_categories)

    #v_matrix = np.array([[0, .2, .2, .2, .2, .2], [0, 1, 0, 0, 0, 0], 
        #[0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], 
        #[0, 0, 0, 0, 0, 1]])
    v_matrix = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]])

    for _ in range(num_em_steps): 
        object_locations, object_categories = em_step(camera_locations, 
                                                      directions, 
                                                      obs_categories, 
                                                      sigma, 
                                                      v_matrix, 
                                                      object_locations, 
                                                      object_categories, 
                                                      num_categories, 
                                                      num_gd_steps=num_gd_steps
                                                      )
    




    print("ground truth: \n", gt_object_locations)
    print(gt_object_categories)
    print("inference: \n", object_locations)
    print(object_categories)


K = 3
sigma = 0.1
num_categories = 5
gt_object_locations, gt_object_categories, camera_locations, directions, obs_categories, obs_objects = sample_baseline(K, num_categories, 500, sigma)

v_matrix = np.array([[0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], 
        [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], 
        [0, 0, 0, 0, 0, 1]])



"""
with mp.Pool(processes=4) as pool: 
    everything = pool.starmap(do_em_inference, zip(repeat(camera_locations), 
                                                   repeat(directions), 
                                                   repeat(obs_categories), 
                                                   repeat(sigma),
                                                   repeat(v_matrix), 
                                                   range(1, 5+1), 
                                                   repeat(num_categories), 
                                                   repeat(5), 
                                                   repeat(3000), 
                                                   repeat(10000), 
                                                   repeat(key),
                                                   ))

print(everything)
"""
nlls = [None]
start = time.time()


for k in range(3, 4): 
    print(f"################### K = {k}")
    object_locations, object_categories, resps, nll = do_em_inference(camera_locations, 
                                                          directions, 
                                                          obs_categories, 
                                                          sigma, 
                                                          v_matrix, 
                                                          k, 
                                                          num_categories, 
                                                          5, 
                                                          3000, 
                                                          10000, 
                                                          key,
                                                          )
                                                      

    print("ground truth\n", gt_object_locations, "\n", gt_object_categories)
    print("inferred\n", object_locations, "\n", object_categories)
    print(resps, nll)
    nlls.append(nll)
end = time.time()
print(nlls)

print("time: ", end-start)


