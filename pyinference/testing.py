import jax.numpy as np 
from jax import grad, jit, vmap 
import random
from scipy.stats import multivariate_normal
from pyinference.inference import *
import numpy 
import matplotlib.pyplot as plt

random.seed(11)
numpy.random.seed(11)

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
    K = 5
    sigma = 0.1
    num_categories = 2
    gt_object_locations, gt_object_categories, camera_locations, directions, obs_categories, obs_objects = sample_baseline(K, num_categories, 500, sigma)
    print(gt_object_categories)


    num_em_steps = 10
    num_gd_steps = 5000

    init_displacement = np.array([[0.0, 0, 0], [-.5, .4, .2], [-.3, -.5, .2],
        [1, -1, 1], [2, 3, -1], [2, 3, -2]])

    object_locations = gt_object_locations + init_displacement
    object_categories = np.array([0, 1, 1, 2, 2, 2])

    v_matrix = np.array([[0, .5, .5], [0, 1, 0], [0, 0, 1]])

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


test_baseline()
