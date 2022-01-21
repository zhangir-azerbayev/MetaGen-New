import jax.numpy as np
from jax import grad, jit, vmap 

class Scene: 
    def __init__(self, 
                 object_locations, 
                 object_categories, 
                 camera_locations, 
                 directions, 
                 obs_categories, 
                 # use `obs_component` only if simulating generative model
                 obs_component = None
                 ): 
        self.locations = locations
        self.categories = categories
        self.camera_locations=camera_locatoins
        self.directions=directions
        self.obs_categories=obs_categories, 
        self.obs_component=obs_component



