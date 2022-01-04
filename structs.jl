Base.@kwdef struct RoomParams
    max_allowed_objects = 8
    bottom_left_lim::Vector{Float64} = [-5, -5, 0]
    top_right_lim::Vector{Float64} = [5, 5, 3]
end

Base.@kwdef struct CameraParams
    image_dim_x = 256
    image_dim_y = 256
    horizontal_fov::Float64 = 55
    vertical_fov::Float64 = 55
end

Base.@kwdef struct LineSegment
    start::Coordinate
    endpoint::Coordinate
    a::Float64
    b::Float64
    c::Float64
end

export RoomParams
export CameraParams
export LineSegment
