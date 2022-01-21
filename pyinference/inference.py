import jax.numpy as np 
from jax.scipy.special import erfc as erfc
from jax import grad, jit, vmap
import math 
from functools import partial
from tqdm import tqdm

eps = 1e-10

"""Correct up to a multiplicative constant"""
def likelihood(camera_location, 
               direction, 
               obs_category,
               sigma, 
               object_location, 
               object_category): 
    is_same_category = np.heaviside(-np.abs(object_category-obs_category).astype(int), 1)

    mean = object_location - camera_location 
    product_term = np.dot(direction, mean)

    exp_factor = np.exp(-(np.dot(mean, mean) - np.square(product_term))/(2*sigma))
    erf_factor = erfc(-product_term/np.sqrt(2*sigma))

    lhood = is_same_category * exp_factor * erf_factor
    return lhood

def compute_row_responsibilities(camera_location, 
                                 direction, 
                                 obs_category,
                                 sigma, 
                                 object_locations, 
                                 object_categories): 
    in_axes = (None, None, None, None, 0, 0)
    likelihood_vec = vmap(likelihood, in_axes=in_axes)(camera_location, 
                                  direction,
                                  obs_category,
                                  sigma,
                                  object_locations, 
                                  object_categories
                                  )
    likelihood_vec = (likelihood_vec.flatten() + eps).at[0].set(0.0)
    resps = likelihood_vec/np.sum(likelihood_vec)

    return resps

def compute_resps(camera_locations, 
                             directions, 
                             obs_categories,
                             sigma, 
                             object_locations, 
                             object_categories): 
    in_axes = (0, 0, 0, None, None, None)
    
    return vmap(compute_row_responsibilities, in_axes=in_axes)(camera_locations, 
                                                              directions, 
                                                              obs_categories,
                                                              sigma, 
                                                              object_locations, 
                                                              object_categories)



def per_element_m_step_loss(responsibility, 
                 camera_location, 
                 direction, 
                 sigma, 
                 object_location): 
    mean = object_location - camera_location 

    product_term = np.dot(direction, mean)

    inside_log = erfc(-product_term / np.sqrt(2*sigma)) + eps
    erf_term = -2 * sigma * np.log(inside_log)

    unweighted_loss = np.dot(mean, mean) - np.square(product_term) + erf_term

    return responsibility * unweighted_loss


def m_step_loss(responsibilities, 
                camera_locations, 
                directions, 
                sigma, 
                object_location):
    normalizer = np.sum(responsibilities)
    losses_vec = vmap(per_element_m_step_loss, in_axes=(0,0,0,None,None))(responsibilities, 
                                                               camera_locations, 
                                                               directions, 
                                                               sigma, 
                                                               object_location)

    return  1/normalizer * np.sum(losses_vec)


def optimize_location(responsibilities, 
                      camera_locations, 
                      directions, 
                      sigma, 
                      init_location, 
                      num_steps,
                      lr=1e-3, 
                      clipping_threshold=1,
                      save_losses = False, 
                      save_locations = False 
                      ):
    loss_fn = partial(m_step_loss, responsibilities, camera_locations, directions, 
            sigma)

    grad_loss = jit(grad(loss_fn))

    losses = []
    locations = []
    
    object_location = init_location
    for i in tqdm(range(num_steps)): 
        loss = loss_fn(object_location)
        
        if save_losses: 
            losses.append(loss)

        location_grad = grad_loss(object_location)
        
        size = np.linalg.norm(location_grad) 
        if np.linalg.norm(location_grad) > clipping_threshold: 
            object_grad = clipping_threshold * location_grad/size
            print(i, "clipped")
        
        object_location = object_location - lr * location_grad

        if save_locations:
            locations.append(object_location)

    return object_location, losses, locations
