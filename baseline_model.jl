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

    categories = [@trace(uniform_discrete(1, num_object_categories), :objects=>i=>:category) for i in num_objects]
    xs = [@trace(uniform(room.x_lim...), :objects=>i=>:x)
        for i in num_objects]
    ys = [@trace(uniform(room.y_lim...), :objects=>i=>:y)
        for i in num_objects]
    zs = [@trace(uniform(room.z_lim...), :objects=>i=>:z)
        for i in num_objects]

    # Generates observations
    return
end

args = (5, 10, 20, RoomParams(), CameraParams(), 0.5)

(trace, _) = Gen.generate(gen_scene_and_observations, args)

println(Gen.get_choices(trace))
