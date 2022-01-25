import json
import jax.numpy as np
import jax.random as jrandom
from tqdm import tqdm
from pyinference.inference import do_em_inference



def get_scene(x): 
    fle_num = 1 + (x//50)
    index = x % 50 

    with open(f"/home/zaa7/project/metagen_data/01_23/vectorized_data_labelled_retinanet_{fle_num}.json.json", "r") as fle: 
        lst = json.load(fle)
    
    scene = lst[index]

    return scene 

def retinanet_catmap(x): return {28:1, 51:2, 62:3, 64:4, 72:5}[x]
def retinanet_semantic_to_model(x): return {"umbrella":1, 
        "bowl":2, "chair":3, "potted plant": 4, "tv": 5}[x]

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

idx = 0 
k = 3

camera_locations, directions, obs_categories, gt_object_locations, gt_object_categories = get_array_data(idx)


v_matrix = np.array([[.0, 0, 0, 0, 0, 0], 
                      [0, 1, 0, 0, 0, 0], 
                      [0, 0, 1, 0, 0, 0], 
                      [0, 0, 0, 1, 0, 0], 
                      [0, 0, 0, 0, 1, 0], 
                      [0, 0, 0, 0, 0, 1],
                      ])

sigma=.2
key = jrandom.PRNGKey(idx)



object_locations, object_categories, resps, nll = do_em_inference(camera_locations, 
        directions, 
        obs_categories, 
        sigma, 
        v_matrix, 
        k, 
        5, 
        10, 
        3000, 
        10000, 
        key, 
        )

np.savez(f"results/baseline_retinanet/sigma{sigma}_scene{idx}_objects{k}", 
         object_locations=object_locations, 
         object_categories=object_categories, 
         resps=resps, 
         nll=nll, 
         )
    

print("ground truth: ", gt_object_locations, gt_object_categories)

print("inference: ", object_locations, object_categories)



