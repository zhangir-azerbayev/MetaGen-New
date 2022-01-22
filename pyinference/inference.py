import jax.numpy as np 
from jax.scipy.special import erfc as erfc
from jax import grad, jit, vmap 
import math 
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt

eps = 1e-10

"""Correct up to a multiplicative constant"""
def likelihood(camera_location, 
               direction, 
               obs_category,
               sigma, 
               v_matrix,
               object_location, 
               object_category): 
    v_matrix_weight = v_matrix[object_category, obs_category]

    mean = object_location - camera_location 
    product_term = np.dot(direction, mean)

    exp_factor = np.exp(-(np.dot(mean, mean) - np.square(product_term))/(2*sigma))
    erf_factor = erfc(-product_term/np.sqrt(2*sigma))

    lhood = v_matrix_weight * exp_factor * erf_factor
    return lhood


def compute_row_responsibilities(camera_location, 
                                 direction, 
                                 obs_category,
                                 sigma, 
                                 v_matrix, 
                                 object_locations, 
                                 object_categories): 
    in_axes = (None, None, None, None, None, 0, 0)
    likelihood_vec = vmap(likelihood, in_axes=in_axes)(camera_location, 
                                  direction,
                                  obs_category,
                                  sigma,
                                  v_matrix,
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
                             v_matrix, 
                             object_locations, 
                             object_categories): 
    in_axes = (0, 0, 0, None, None, None, None)
    
    return vmap(compute_row_responsibilities, in_axes=in_axes)(camera_locations, 
                                                              directions, 
                                                              obs_categories,
                                                              sigma, 
                                                              v_matrix, 
                                                              object_locations, 
                                                              object_categories)




def location_nll(camera_location, 
                 direction, 
                 sigma, 
                 object_location): 
    mean = object_location - camera_location 

    product_term = np.dot(direction, mean)

    inside_log = erfc(-product_term / np.sqrt(2*sigma)) + eps
    erf_term = -2 * sigma * np.log(inside_log)

    unweighted_loss = np.dot(mean, mean) - np.square(product_term) + erf_term

    return unweighted_loss


def m_step_loss(resps, 
                camera_locations, 
                directions, 
                sigma, 
                object_location):
    normalizer = np.sum(resps)
    nll_vec = vmap(location_nll, in_axes=(0,0,None,None))(camera_locations, 
                                                               directions, 
                                                               sigma, 
                                                               object_location).flatten()
    
    losses_vec = resps * nll_vec

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
        
        object_location = object_location - lr * location_grad

        if save_locations:
            locations.append(object_location)

    return object_location, losses, locations


"""correct up to additive constant"""
def nll(camera_location, 
                  direction, 
                  obs_category, 
                  sigma, 
                  v_matrix, 
                  object_location, 
                  object_category): 

    v_matrix_weight = v_matrix[object_category, obs_category]

    location_only = location_nll(camera_location,
                                direction, 
                                sigma, 
                                object_location)
    
    nl_lhood = -np.log(v_matrix_weight+eps) + location_only
    return nl_lhood

def compute_component_nll(resps, 
                                    camera_locations, 
                                    directions, 
                                    obs_categories, 
                                    sigma, 
                                    v_matrix, 
                                    object_location, 
                                    object_category): 
    normalizer = np.sum(resps)
    in_axes = (0, 0, 0, None, None, None, None)
    nll_vec = vmap(nll, in_axes=in_axes)(camera_locations, 
                                                directions, 
                                                obs_categories, 
                                                sigma, 
                                                v_matrix, 
                                                object_location, 
                                                object_category).flatten()



    return 1/normalizer * np.sum(resps * nll_vec)


def em_step(camera_locations, 
            directions, 
            obs_categories, 
            sigma, 
            v_matrix, 
            object_locations, 
            object_categories, 
            num_categories, 
            num_gd_steps=100):

    K = np.shape(object_locations)[0]-1
    print("K: ", K)
    
    # E-step
    resps = compute_resps(camera_locations, 
                          directions, 
                          obs_categories, 
                          sigma, 
                          v_matrix, 
                          object_locations, 
                          object_categories
                          )

    print(resps)
    
    # M-step locations 
    new_object_locations = [np.array([0.0, 0, 0])] + [None for _ in range(1,K+1)]
    new_object_categories = [0] + [None for _ in range(1, K+1)]
    for k in range(1, K+1): 
        #M-step locations
        new_object_locations[k], losses, _ = optimize_location(resps[:, k], 
                                                 camera_locations, 
                                                 directions, 
                                                 sigma, 
                                                 object_locations[k], 
                                                 num_gd_steps, 
                                                 save_losses=True 
                                                 )  
        #M-step categories 
        categories = np.arange(1, num_categories+1)

        in_axes = (None, None, None, None, None, None, None, 0)
        per_category_nll = vmap(compute_component_nll, in_axes=in_axes)(resps[:, k], 
                                        camera_locations, 
                                        directions, 
                                        obs_categories, 
                                        sigma, 
                                        v_matrix, 
                                        new_object_locations[k], 
                                        categories)
    
        print("per category nll: ", per_category_nll)
        best_cat = np.argmin(per_category_nll) + 1
        print("best cat: ", best_cat)
        new_object_categories[k] = best_cat 

    new_object_locations = np.stack(new_object_locations)
    new_object_categories = np.stack(new_object_categories)

    return new_object_locations, new_object_categories
                   




        


    


    
