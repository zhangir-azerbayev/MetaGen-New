import jax.numpy as np 
from jax.scipy.special import erfc as erfc
from jax import grad, jit, vmap

eps = 0

def _m_step_loss(responsibility, 
                 camera_location, 
                 direction, 
                 sigma, 
                 object_location): 
    mean = object_location - camera_location 

    product_term = np.dot(direction, mean)

    inside_log = erfc(-product_term / np.sqrt(2*sigma)) 
    erf_term = -2 * sigma * np.log(inside_log)

    unweighted_loss = np.linalg.norm(mean) - np.square(product_term) + erf_term

    return responsibility * unweighted_loss


def m_step_loss(responsibilities, 
                camera_locations, 
                directions, 
                sigma, 
                object_location):
    normalizer = np.sum(responsibilities)
    losses_vec = vmap(_m_step_loss, in_axes=(0,0,0,None,None))(responsibilities, 
                                                               camera_locations, 
                                                               directions, 
                                                               sigma, 
                                                               object_location)

    return  1/normalizer * np.sum(losses_vec)


print("good fit: ", _m_step_loss(1, np.array([0.0, 0, 0]), np.array([1.0, 0, 0]), 0.05, np.array([1.0, 0, 0])))


print("ok fit: ", _m_step_loss(1, np.array([0.0, 0, 0]), np.array([1.0, 0, 0]), 0.05, np.array([0, 1, 0])))

print("bad fit: ", _m_step_loss(1, np.array([0.0, 0, 0]), np.array([-1.0, 0, 0]), 0.05, np.array([1.0, 0, 0])))
    
