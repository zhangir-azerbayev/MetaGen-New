import Base.-

"""
Parametrizes a scene
"""
Base.@kwdef struct RoomParams
    max_allowed_objects = 8
    x_lim::Vector{Float64, 2} = [-5, 5]
    y_lim::Vector{Float64, 2} = [-5, 5]
    z_lim::Vector{Float64, 2} = [0, 3]
end

Base.@kwdef struct CameraParams
    image_dim_x = 256
    image_dim_y = 256
    horizontal_fov::Float64 = 55
    vertical_fov::Float64 = 55

Base.@kwdef struct LineSegment
    start::Coordinate
    endpoint::Coordinate
    a::Float64
    b::Float64
    c::Float64

Base.@kwdef struct Coordinate
    x::Float64
    y::Float64
    z::Float64

a::Coordinate - b::Coordinate = Direction(a.x-b.x, a.y-b.y, a.z-b.z)


Base.@kwdef struct SceneState
    num_objects::Int64
    objects::Vector{Object, num_objects}

Base.@kwdef struct Object
    category::Int64
    location::Coordinate

Base.@kwdef struct Direction
    x::Float64
    y::Float64
    z::Float64

Base.@kwdef struct Observation
    camera_position::Coordinate
    category::Int64
    direction::Direction

export RoomParams
export CameraParams
export LineSegment
export Coordinate
export SceneState
export Object
export Direction
export Observation
