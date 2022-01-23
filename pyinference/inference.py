import jax.numpy as np 
from jax.scipy.special import erfc as erfc
from jax import grad, jit, vmap, pmap
import math 
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt

eps = 1e-10
def heaviside(x): 
    return np.heaviside(x, 0)

def alt_heaviside(x): 
    return 1-np.heaviside(x, 0)

"""Correct up to a multiplicative constant"""
def likelihood(camera_location, 
               direction, 
               obs_category,
               sigma, 
               v_matrix,
               object_location, 
               object_category): 
    const = 1/(4*np.pi*sigma)
    v_matrix_weight = v_matrix[object_category, obs_category]

    mean = object_location - camera_location 
    product_term = np.dot(direction, mean)

    exp_factor = np.exp(-(np.dot(mean, mean) - np.square(product_term))/(2*sigma))
    erf_factor = erfc(-product_term/np.sqrt(2*sigma))

    cat_zero_lhood = alt_heaviside(object_category) * v_matrix_weight/600 
    
    cat_nonzero_lhood = heaviside(object_category) * const * v_matrix_weight * exp_factor * erf_factor
    return cat_nonzero_lhood + cat_zero_lhood


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
    const = np.log(4*np.pi*sigma)

    mean = object_location - camera_location 

    product_term = np.dot(direction, mean)

    inside_log = erfc(-product_term / np.sqrt(2*sigma)) + eps
    erf_term = -np.log(inside_log)

    unweighted_loss = const + (np.dot(mean, mean) - np.square(product_term))/(2*sigma) + erf_term

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

    cat_zero_nll = alt_heaviside(object_category) * (-np.log(v_matrix_weight+eps) + np.log(600))
    
    cat_nonzero_nll = heaviside(object_category) * (-np.log(v_matrix_weight+eps) + location_only)
    return cat_zero_nll + cat_nonzero_nll

def compute_component_nll(resps, camera_locations, 
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

def optimize_location(resps, 
                      camera_locations, 
                      directions, 
                      obs_categories, 
                      sigma, 
                      v_matrix,
                      init_location, 
                      object_category,
                      num_steps,
                      lr, 
                      print_losses = False, 
                      ):
    loss_fn = lambda loc: compute_component_nll(resps, 
                                                camera_locations, 
                                                directions, 
                                                obs_categories, 
                                                sigma, 
                                                v_matrix,
                                                loc, 
                                                object_category
                                                )

    grad_loss = jit(grad(loss_fn))

    losses = []
    
    object_location = init_location
    for i in range(num_steps): 
        loss = loss_fn(object_location)
        
        if print_losses: 
            losses.append(loss)

        location_grad = grad_loss(object_location)
        
        
        object_location = object_location - lr * location_grad

    if print_losses:
        plt.plot(losses)
        plt.show()
        plt.close()

    
    return object_location, loss

def optimize_location_and_category(resps, 
                      camera_locations, 
                      directions, 
                      obs_categories,
                      sigma, 
                      v_matrix,
                      init_location, 
                      num_categories, 
                      num_steps,
                      lr=1e-3, 
                      clip_threshold=1,
                      ):
    """
    locations = [None for _ in range(num_categories)]
    nlls = [None for _ in range(num_categories)]
    
    for c in range(1, num_categories+1): 
        locations[c-1], nlls[c-1] = optimize_location(resps, 
                                                      camera_locations, 
                                                      directions, 
                                                      obs_categories, 
                                                      sigma, 
                                                      v_matrix, 
                                                      init_location, 
                                                      c, 
                                                      num_steps, 
                                                      lr=lr, 
                                                      clip_threshold=clip_threshold)
    
    nlls = np.array(nlls)
    """
    print("doing component with all categories")
    categories = np.arange(1, num_categories+1)
    in_axes = (None, None, None, None, None, None, None, 0, None, None)
    locations, nlls = vmap(optimize_location, in_axes=in_axes)(resps, 
                                                               camera_locations, 
                                                               directions, 
                                                               obs_categories, 
                                                               sigma, 
                                                               v_matrix, 
                                                               init_location, 
                                                               categories, 
                                                               num_steps, 
                                                               lr)

    print("nlls: ", nlls)
    best_cat = np.argmin(nlls) + 1
    print("best category: ", best_cat)
    best_location = locations[best_cat-1]

    return best_location, best_cat

def em_step(camera_locations, 
            directions, 
            obs_categories, 
            sigma, 
            v_matrix, 
            object_locations, 
            object_categories, 
            num_categories, 
            num_gd_steps=100, 
            lr=1e-3):

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

    """ 
    # M-step locations 
    new_locations = [np.array([0.0, 0, 0])] + [None for _ in range(1,K+1)]
    new_categories = [0] + [None for _ in range(1, K+1)]
    
    for k in range(1, K+1): 
        #M-step locations
        print("running m step")
        new_locations[k], new_categories[k]= optimize_location_and_category(resps[:, k], 
                                                 camera_locations, 
                                                 directions, 
                                                 obs_categories, 
                                                 sigma,
                                                 v_matrix,
                                                 object_locations[k], 
                                                 num_categories, 
                                                 num_gd_steps, 
                                                 lr
                                                 )  

    
    new_locations = np.stack(new_locations)
    new_categories = np.stack(new_categories)
    """
    in_axes = (1, None, None, None, None, None, 0, None, None, None)
    inferred_locations, inferred_categories = vmap(optimize_location_and_category, 
            in_axes=in_axes)(resps[:, 1:], 
                             camera_locations, 
                             directions, 
                             obs_categories, 
                             sigma,
                             v_matrix,
                             object_locations[1:], 
                             num_categories, 
                             num_gd_steps, 
                             lr
                            )
    new_locations = np.concatenate((np.array([[0.0, 0, 0]]), inferred_locations), 
            axis=0)
    new_categories = np.concatenate((np.array([0]), inferred_categories))
    print("new locations:\n", new_locations) 
    print("new categories:\n", new_categories)
    return new_locations, new_categories 
