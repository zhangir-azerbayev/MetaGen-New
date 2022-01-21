from functools import partial
from tqdm import tqdm
from pyinference.inference import m_step_loss as m_step_loss
from pyinference.inference import per_element_m_step_loss as per_element_m_step_loss
import jax.numpy as np 
import jax.random as jrandom
from jax.scipy.special import erf as erf
from jax import grad, jit, vmap
import matplotlib.pyplot as plt
import random 

N = 60
sigma = 0.5
lr = 1e-3
clipping_threshold = 1

seed = 1701

key = jrandom.PRNGKey(seed)

responsibilities = np.ones(N)
camera_locations = []
directions = []
for _ in range(N): 
    old_key, key = jrandom.split(key)
    x = random.uniform(-5, 5)
    y = random.uniform(-5, 5)
    z = random.uniform(-5, 5)
    camera_location = np.array([x, y, z])
    detection_location = jrandom.multivariate_normal(key, np.zeros(3), sigma*np.identity(3))
    print(detection_location)
    camera_locations.append(camera_location)
    directions.append((detection_location-camera_location/np.linalg.norm(detection_location-camera_location)))

camera_locations = np.stack(camera_locations)

directions = np.stack(directions)


"""
some_samples = jrandom.normal(key, (N, 3))

camera_locations = np.stack([vec / np.linalg.norm(vec) for vec in some_samples])

directions = np.stack([-vec for vec in camera_locations])

camera_locations = 3 * camera_locations
camera_locations = np.array([[1.0, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
directions = np.array([[-1.0, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]])
"""
print("camera locations: ", camera_locations)

print("directions: ", directions)

object_location = np.array([3.0, 5, -4])

loss_fn = partial(m_step_loss, responsibilities, camera_locations, directions, sigma)

print("building and compiling gradient function")
grad_loss = jit(grad(loss_fn))


"""
locus = np.array([1.0, 1, 1])/np.sqrt(3)
rs = np.arange(0, 10, 1)
locations = np.einsum('i,j->ij', rs, locus) 
print(locations)
losses = vmap(loss_fn)(locations)

plt.plot(losses)
plt.show()
"""



losses = []
locations = []
for i in tqdm(range(20000)): 
    loss = loss_fn(object_location)
    losses.append(loss)
    
    object_grad = grad_loss(object_location)

    if np.linalg.norm(object_grad) > clipping_threshold: 
        object_grad = clipping_threshold * object_grad / np.linalg.norm(object_grad)
        print(i, " clipped")
    
    print(object_grad)
    object_location = object_location - lr * object_grad 
    locations.append(object_location)


fig, axs = plt.subplots(2)
axs[0].plot(losses)
axs[0].set_title('loss')
axs[1].plot([np.linalg.norm(x) for x in locations])
plt.show()

print(object_location)


