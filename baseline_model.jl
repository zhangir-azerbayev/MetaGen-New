using Gen
using Distributions
using LinearAlgebra
include("structs.jl")
include("distributions.jl")

@dist choose_category(categories) = categories[uniform(1, length(categories))]

@dist function choose_direction(camera_location, object_index, object_location,
                                detection_sd))
    detection_location = mvnormal()

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
    categories = [@trace(uniform_discrete(1, num_object_categories),
        :objects=>i=>:category) for i in 1:num_objects]
    locations = vcat([@trace(mvuniform([room.bottom_left_lim, room.top_right_lim),
        :objects=>i=>:location) for i in 1:num_objects]...)
    # Generates observations
    observations = []
    for i in 1:num_observations
        camera_location = [@trace(mvuniform([room.bottom_left_lim, room.top_right_lim),
            :observations=>i=>:camera_location)

        object_index = @trace(uniform_discrete(1:max_objects),
            :observations=>i=>:object_index)

        direction =

    return
end

args = (5, 10, 20, RoomParams(), CameraParams(), 0.5)

(trace, _) = Gen.generate(gen_scene_and_observations, args)

Gen.get_choices(trace)
