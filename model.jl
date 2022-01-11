using Gen
using Distributions
using LinearAlgebra
using OffsetArrays
include("structs.jl")
include("distributions.jl")

"""
`confusion_matrix` must be a 0:num_object_categories x 0:num_object_categories array
where V_{ij} = P(j observed | object i). The 0th object category
represents a hallucination. The column V_{:, 0} is 0
and the row V_{0, :} is a probability distribution with
support 1, ..., num_object categories.
"""
@gen function gen_scene_and_observations(
    max_objects::Int64,
    num_object_categories::Int64,
    num_observations::Int64,
    room::RoomParams,
    camera::CameraParams,
    detection_sd::Float64,
    confusion_matrix::OffsetArray{Float64, 2, Array{Float64, 2}},
    p_hallucination:: Float64
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
    object_dist = OffsetArray([k==0 ? p_hallucination : (1-p_hallucination)/num_objects
        for k in 0:num_objects], 0:num_objects)
    for i in 1:num_observations
        camera_position = @trace(mvuniform(room.bottom_left_lim, room.top_right_lim),
            :observations=>i=>:camera_location)

        object = @trace(zero_categorical(object_dist), :observations=>i=>:object)

        object_category = object == 0 ? 0 : categories[object]

        category_dist = confusion_matrix[object_category, :]

        category = @trace(zero_categorical(category_dist), :observations=>i=>:category)

        if object == 0
            # using this mvuniform as a placeholder until I switch to spherical coordinates
            # note this placeholder would break inference.
            direction = @trace(mvuniform([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]), :observations=>i=>:direction)
        else
            location = locations[object]
            direction = @trace(choose_direction(camera_position, location, detection_sd), :observations=>i=>:direction)
        end

        append!(observations, Dict(:camera_location => camera_position,
            :category => category, :direction => direction,
            :object => object))
    end
    return num_objects, categories, locations, observations
end

confusion_matrix = OffsetArray([0 0.5 0.5; 0 0.9 0.1;0 0.1 0.9], 0:2, 0:2)
args = (5, 2, 20, RoomParams(), CameraParams(), 0.5, confusion_matrix, 0.2)

(trace, _) = Gen.generate(gen_scene_and_observations, args)

Gen.get_choices(trace)
