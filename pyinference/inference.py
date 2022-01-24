import jax.numpy as np 
from jax.scipy.special import erfc as erfc
from jax import grad, jit, vmap, pmap
import math 
from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt
import jax.random as jrandom

eps=1e-10

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

    cat_zero_lhood = v_matrix_weight/300
    
    cat_nonzero_lhood = const * v_matrix_weight * exp_factor * erf_factor
    return np.where(object_category!=0, cat_nonzero_lhood, cat_zero_lhood)


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

    likelihood_vec = likelihood_vec.flatten() 
    safe_likelihood_vec = np.where(np.sum(likelihood_vec)!=0, 
            likelihood_vec, (likelihood_vec+eps).at[0].set(0))
    resps = safe_likelihood_vec/np.sum(safe_likelihood_vec)

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

    cat_zero_nll = -np.log(v_matrix_weight+eps) + np.log(300)
    
    cat_nonzero_nll = -np.log(v_matrix_weight+eps) + location_only
    return np.where(object_category!=0, cat_nonzero_nll, cat_zero_nll)


def compute_component_q(resps, camera_locations, 
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
    # what happens if we delete the normalizer? 
    return np.where(normalizer!=0, np.sum(resps * nll_vec)/normalizer, 0)

def compute_example_ll(camera_location, 
                       direction, 
                       obs_category, 
                       sigma, 
                       v_matrix,
                       object_locations, 
                       object_categories, 
                       ): 
    in_axes = (None, None, None, None, None, 0, 0)
    sum_component_l = np.sum(vmap(likelihood, in_axes=in_axes)(camera_location, 
                                                        direction,
                                                        obs_category, 
                                                        sigma, 
                                                        v_matrix, 
                                                        object_locations, 
                                                        object_categories,
                                                        ))

    return sum_component_l


def compute_model_nll(camera_locations, 
                         directions, 
                         obs_categories, 
                         sigma, 
                         v_matrix, 
                         object_locations, 
                         object_categories, 
                         ): 
    in_axes = (0, 0, 0, None, None, None, None)

    per_example_ll = vmap(compute_example_ll, in_axes=in_axes)(camera_locations, 
                                                      directions, 
                                                      obs_categories, 
                                                      sigma, 
                                                      v_matrix, 
                                                      object_locations, 
                                                      object_categories,
                                                      )

    nll = -np.mean(np.log(per_example_ll+eps))

    return nll

                                        


"""
if print_losses=True, all upstream vmaps must be disabled
"""
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
                      clip_threshold=1,
                      print_losses = False,
                      ):
    loss_fn = lambda loc: compute_component_q(resps, 
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

        size = np.linalg.norm(location_grad)
        need_clipping = size > clip_threshold
        clip_factor = np.where(need_clipping, clip_threshold/size, 1)
        location_grad = clip_factor * location_grad
        
        
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
                      categories, 
                      num_steps,
                      lr=1e-3, 
                      clip_threshold=1,
                      parallel=True
                      ):

    if not parallel: 
        locations = [None for _ in categories[1:]]
        nlls = [None for _ in categories[1:]]
        
        for c in categories[1:]: 
            locations[c-1], nlls[c-1] = optimize_location(resps, 
                                                          camera_locations, 
                                                          directions, 
                                                          obs_categories, 
                                                          sigma, 
                                                          v_matrix, 
                                                          init_location, 
                                                          categories[c], 
                                                          num_steps, 
                                                          lr=lr
                                                          )

        locations = np.stack(locations)
        nlls = np.array(nlls)
    else: 
        print("doing component with all categories")
        in_axes = (None, None, None, None, None, None, None, 0, None, None)
        locations, nlls = vmap(optimize_location, in_axes=in_axes)(resps, 
                                                                   camera_locations, 
                                                                   directions, 
                                                                   obs_categories, 
                                                                   sigma, 
                                                                   v_matrix, 
                                                                   init_location, 
                                                                   categories[1:], 
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
            categories, 
            num_gd_steps=100, 
            lr=1e-3, 
            parallel=True):

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
    
    if not parallel: 
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
                                                     categories, 
                                                     num_gd_steps, 
                                                     lr
                                                     )  

        
        new_locations = np.stack(new_locations)
        new_categories = np.stack(new_categories)
    else: 
        in_axes = (1, None, None, None, None, None, 0, None, None, None)
        inferred_locations, inferred_categories = vmap(optimize_location_and_category, 
                in_axes=in_axes)(resps[:, 1:], 
                                 camera_locations, 
                                 directions, 
                                 obs_categories, 
                                 sigma,
                                 v_matrix,
                                 object_locations[1:], 
                                 categories, 
                                 num_gd_steps, 
                                 lr
                                )
        new_locations = np.concatenate((np.array([[0.0, 0, 0]]), inferred_locations), 
                axis=0)
        new_categories = np.concatenate((np.array([0]), inferred_categories))
    print("new locations:\n", new_locations) 
    print("new categories:\n", new_categories)
    return new_locations, new_categories 

def init_random_search(camera_locations, 
                                  directions, 
                                  obs_categories, 
                                  sigma, 
                                  v_matrix, 
                                  num_objects, 
                                  num_categories, 
                                  num_inits, 
                                  key,
                                  ):
    key, subkey = jrandom.split(key)
    unscaled_candidate_object_locations = jrandom.uniform(subkey, (num_inits,
        num_objects, 3), minval=0, maxval=1)

    scale = np.array([[[10, 3, 10]]])
    shift = np.array([[[-5, 0, -5]]])
    candidate_object_locations = unscaled_candidate_object_locations*scale+shift

    zero_locations = np.zeros((num_inits, 1, 3))

    all_candidate_object_locations = np.concatenate((zero_locations, 
        candidate_object_locations), axis=1)
    
    subkey, subsubkey = jrandom.split(subkey)
    candidate_object_categories = jrandom.randint(subsubkey, 
            (num_inits, num_objects), 1, num_categories+1)
    
    print("dtype: ", np.dtype(candidate_object_categories))
    zero_categories = np.zeros((num_inits, 1), dtype=np.int64)

    all_candidate_object_categories = np.concatenate((zero_categories, 
        candidate_object_categories), axis=1)
    
    in_axes = (None, None, None, None, None, 0, 0)
    candidate_resps = vmap(compute_resps, in_axes=in_axes)(camera_locations, 
                          directions, 
                          obs_categories, 
                          sigma, 
                          v_matrix, 
                          all_candidate_object_locations, 
                          all_candidate_object_categories,
                          )


    in_axes = (None, None, None, None, None, 0, 0)
    candidate_nlls = vmap(compute_model_nll, 
            in_axes=in_axes)(camera_locations, 
                             directions, 
                             obs_categories, 
                             sigma, 
                             v_matrix, 
                             all_candidate_object_locations, 
                             all_candidate_object_categories, 
                             )
    
    best_index = np.argmin(candidate_nlls)

    best_locations = all_candidate_object_locations[best_index]
    best_categories = all_candidate_object_categories[best_index]

    return best_locations, best_categories

def do_em_inference(camera_locations, 
                    directions, 
                    obs_categories, 
                    sigma, 
                    v_matrix, 
                    num_objects,
                    num_categories,
                    num_em_steps,
                    num_gd_steps,
                    num_inits,
                    key,
                    ): 
    
    object_locations, object_categories = init_random_search(camera_locations, 
                                                             directions, 
                                                             obs_categories, 
                                                             sigma, 
                                                             v_matrix, 
                                                             num_objects, 
                                                             num_categories, 
                                                             num_inits, 
                                                             key,
                                                             )
    print("initialization: ", object_locations, object_categories)
    categories = np.arange(num_categories+1)
    for _ in range(num_em_steps): 
        object_locations, object_categories = em_step(camera_locations, 
                                                      directions, 
                                                      obs_categories, 
                                                      sigma, 
                                                      v_matrix, 
                                                      object_locations, 
                                                      object_categories, 
                                                      categories, 
                                                      num_gd_steps=num_gd_steps
                                                      )
    # computes final likelihood 
    resps_final = compute_resps(camera_locations, 
                                directions, 
                                obs_categories, 
                                sigma, 
                                v_matrix, 
                                object_locations, 
                                object_categories, 
                                )
    
    # implement nll
    nll_final = compute_model_nll(camera_locations,
                                  directions, 
                                  obs_categories, 
                                  sigma, 
                                  v_matrix, 
                                  object_locations, 
                                  object_categories, 
                                  )


    return object_locations, object_categories, resps_final, nll_final


