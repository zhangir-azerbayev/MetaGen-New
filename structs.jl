Base.@kwdef struct RoomParams
    max_allowed_objects = 8
    x_lim::Vector{Float64} = [-5, 5]
    y_lim::Vector{Float64} = [-5, 5]
    z_lim::Vector{Float64} = [0, 3]
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
