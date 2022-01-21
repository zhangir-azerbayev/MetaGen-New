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


gt_object_locations, gt_object_categories, camera_locations, directions, obs_categories, obs_objects = sample_baseline(1, 1, 10, 0.1)


num_em_steps = 5
num_gd_steps = 10000

init_displacement = np.array([[0.0, 0, 0], [-4, 2.7, 4.8]])

object_locations = gt_object_locations + init_displacement
object_categories = np.array([0, 1])


for _ in range(num_em_steps): 
    # E step 
    resps = compute_resps(camera_locations, directions, obs_categories, 0.1, 
            object_locations, object_categories)
    print(resps)

    # M step 
    optimized, losses, _ = optimize_location(resps[:, 1],
                                                  camera_locations, 
                                                  directions, 
                                                  0.1,
                                                  object_locations[1], 
                                                  num_gd_steps, 
                                                  save_losses=True)
    object_locations = np.stack([np.array([0.0, 0, 0]), optimized])

    plt.plot(losses)
    plt.show()
    plt.close()


print("ground truth: \n", gt_object_locations)
print("inference: \n", object_locations)


