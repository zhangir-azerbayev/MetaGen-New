using Gen
using Distributions
using LinearAlgebra
include("structs.jl")
include("distributions.jl")

@gen function gen_scene_and_observations(
    max_objects::Int64,
    num_object_categories::Int64,
    num_observations::Int64,
    room::RoomParams,
    camera::CameraParams,
    detection_sd::Float64
    )

    # Generates scene
    num_objects = @trace(uniform_discrete(1, max_objects), :num_objects)
    println("num_objects: ", num_objects)
    categories = [@trace(uniform_discrete(1, num_object_categories),
        :objects=>i=>:category) for i in 1:num_objects]
    locations = [@trace(mvuniform(room.bottom_left_lim, room.top_right_lim),
        :objects=>i=>:location) for i in 1:num_objects]
    # Generates observations
    observations = []
    for i in 1:num_observations
        camera_position = @trace(mvuniform(room.bottom_left_lim, room.top_right_lim),
            :observations=>i=>:camera_location)

        object = @trace(uniform_discrete(1, num_objects), :observations=>i=>:object)

        category_dist = [k == object ? 1 : 0 for k in 1:num_objects]

        category = @trace(categorical(category_dist), :observations=>i=>:category)

        location = locations[object]

        direction = @trace(choose_direction(camera_position, location, detection_sd), :observations=>i=>:direction)

        append!(observations, Dict(:camera_location => camera_position,
            :category => category,
            :direction => direction,
            :object => object))
    end
    return num_objects, categories, locations, observations
end

args = (5, 10, 20, RoomParams(), CameraParams(), 0.5)

(trace, _) = Gen.generate(gen_scene_and_observations, args)

Gen.get_choices(trace)
