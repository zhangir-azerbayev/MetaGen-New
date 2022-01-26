import sys
import json
import jax.numpy as np
import jax.random as jrandom
import jax 
from tqdm import tqdm
from pyinference.inference import do_em_inference
import os 


jax.config.update('jax_platform_name', 'cpu')

def get_scene(x): 
    fle_num = 1 + (x//50)
    index = x % 50 

    with open(f"/home/zaa7/project/metagen_data/01_23/vectorized_data_labelled_retinanet_{fle_num}.json.json", "r") as fle: 
        lst = json.load(fle)
    
    scene = lst[index]

    return scene 

def retinanet_catmap(x): return {62:1, 64:2, 72:3, 28:4, 51:5}[x]
def retinanet_semantic_to_model(x): return {"chair":1, 
        "plant":2, "tv":3, "umbrella": 4, "bowl": 5}[x]

def get_array_data(i):
    scene = get_scene(i)

    object_locations = [np.array([.0, 0, 0])]
    object_categories = [0]
    for gt_component in scene["labels"]: 
        object_locations.append(np.array(gt_component["position"]))
        cat = retinanet_semantic_to_model(gt_component["category_name"])
        object_categories.append(cat)

    object_locations = np.stack(object_locations)
    object_categories = np.array(object_categories)


    camera_locations = []
    directions = []
    obs_categories = []
    for view in scene["views"]: 
        camera = view["camera"]
        camera_location = np.array([camera["x"], camera["y"], camera["z"]])
        detections = view["detections"]
        for category, direction in zip(detections["labels"], detections["vector"]): 
            camera_locations.append(camera_location)
            directions.append(np.array(direction))
            obs_categories.append(retinanet_catmap(category))


    camera_locations = np.stack(camera_locations)
    directions = np.stack(directions)
    obs_categories = np.array(obs_categories)

    return camera_locations, directions, obs_categories, object_locations, object_categories

idx = int(os.environ['SLURM_ARRAY_TASK_ID'])

camera_locations, directions, obs_categories, gt_object_locations, gt_object_categories = get_array_data(idx)

v_matrix = np.array([[.0, .09, .0, .03, .01, 0], 
                      [0, .97, .02, 0, .01, 0], 
                      [0, .04, .92, 0, .02, .02], 
                      [0, .52, .03, .4, .05, 0], 
                      [0, .06, 0, .01, .92, 0], 
                      [0, .13, .01, 0, .03, .83],
                      ])
"""
v_matrix = np.array([[.0, 0, 0, 0, 0, 0], 
                     [0, 1, 0, 0, 0, 0], 
                     [0, 0, 1, 0, 0, 0], 
                     [0, 0, 0, 1, 0, 0], 
                     [0, 0, 0, 0, 1, 0], 
                     [0, 0, 0, 0, 0, 1], 
                     ])
"""

sigma=1
key = jrandom.PRNGKey(idx)

for k in range(4, 5): 
    object_locations, object_categories, resps, nll = do_em_inference(camera_locations, 
            directions, 
            obs_categories, 
            sigma, 
            v_matrix, 
            k, 
            5, 
            12, 
            4000, 
            10000, 
            key, 
            )

    np.savez(f"results/gtv_retinanet/sigma{sigma}_scene{idx}_objects{k}", 
             object_locations=object_locations, 
             object_categories=object_categories, 
             resps=resps, 
             nll=nll, 
             )
        

    print("ground truth: ", gt_object_locations, gt_object_categories)

    print("inference: ", object_locations, object_categories)


