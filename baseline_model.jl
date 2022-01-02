using Gen
using Distributions
using LinearAlgebra
include("structs.jl")

@gen function gen_scene_and_observations(
    max_objects::Int64,
    num_object_categories::Int64,
    num_observations::Int64,
    room::RoomParams,
    camera::CameraParams,
    detection_sd::Float64
    )

    # Generates scene
    num_objects = @trace(rand(1:max_objects), :num_objects)
    categories = [rand(1:num_object_categories) for _ in num_objects]
    locations = [Coordinate(Uniform(room.x_lim...),
                            Uniform(room.y_lim...),
                            Uniform(room.z_lim...)) for _ in num_objects]

    objects = [Object(categories[i], locations[i]) for i in 1:num_objects]

    scene = SceneState(num_objects, objects)

    @trace(scene, :scene_state)

    # Generates observations
    observations::Vector{Observation}

    for i in 1:num_observations
        loc_x = Uniform(room.x_lim...)
        loc_y = Uniform(room.y_lim...)
        loc_z = Uniform(room.z_lim...)
        camera_position = Coordinate(loc_x, loc_y, loc_z)

        detected_object = rand(scene.objects)
        category = detected_object.category
        d = detected_object.location - camera_position
        direction = Direction(normalize([d.x, d.y, d.z])...)

        observation = Observation(camera_position, category, direction)
        @trace(observation, (:observation, i))
        observations.append(observation)
    end

    return scene, observations
