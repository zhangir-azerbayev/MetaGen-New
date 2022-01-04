import LinearAlgebra

### MvUniform
struct MvUniform <: Gen.Distribution{Vector{Float64}} end

const mvuniform = MvUniform()

function Gen.random(::MvUniform, lows::AbstractVector{T},
            highs::AbstractVector{U}) where {T<:Real, U<:Real}
    return [Gen.random(uniform, lows[i], highs[i]) for i in 1:length(lows)]
end

function Gen.logpdf(::MvUniform, x::Array{Float64, 1}, lows::Array{Float64, 1},
            highs::Array{Float64,1})
    if any(x < lows || x > highs)
        return -Inf
    else
        return sum([Gen.logpdf(uniform, x[i], lows[i], highs[i]) for i in 1:length(lows)])
    end
end

function Gen.logpdf_grad(::MvUniform, x::AbstractVector{T},
            lows::AbstractVector{U},
            highs::AbstractVector{V}) where {T<:Real, U<:Real, V<:Real}
    (nothing, nothing, nothing)
end

(::MvUniform)(lows, highs) = Gen.random(MvUniform(), lows, highs)

has_output_grad(::MvUniform) = false
has_argument_grads(::MvUniform) = (false, false)

export mvuniform




### CategoryAndDirection

struct CategoryAndDirection <: Gen.Distribution{Tuple{Int64, Vector{Float64}}} end

const category_and_direction = CategoryAndDirection()

function Gen.random(::CategoryAndDirection, camera_position::AbstracVector{S}, categories::AbstractVector{T},
            locations::Matrix{U}, detection_sd::V) where {S<:Real, T<:Integer,
            U<:Real, V<:Real}
    i = uniform_discrete(1, length(categories))

    location = locations[i, :]
    detection_location = mvnormal(location, LinearAlgebra.Diagonal([detection_sd for _ in 1:length(location)]))

    direction = (detection_location-camera_position)/LinearAlgebra.norm(detection_location-camera_position)

    return categories[i], direction
end

function _per_category_logpdf(direction::AbstractVector{T}, camera_position:: AbstractVector{U}, location::AbstractVector{V},
            detection_sd::W) where {T<:Real, U<:Real, V<:Real, W<:Real}
    k = length(direction)

    constant = -0.5 * (k-1) * log(2 * pi * detection_sd)

    rest = -0.5 / detection_sd * (LinearAlgebra.norm(mean)^2 - LinearAlgebra.dot(mean, location)/2)

    return constant + rest
end

function Gen.logpdf(::CategoryAndDirection, x::Tuple{S, AbstractVector{T}},
            camera_position::AbstractVector{U}, categories::AbstractVector{V}, locations::Matrix{W},
            detection_sd::X) where {S<:Integer, T<:Real, U<:Real, V<:Integer, W<:Real, X<:Real}
    category, direction = x

    obj_idxs = [c for c in categories if c==category]

    logpdfs = []

    for i in obj_idxs
        this_category = _per_category_logpdf(direction, camera_position, locations[i, :],
            detection_sd)
        logpdfs.append(this_category)
    end
    return sum(logpdfs) / length(categories)
end


function Gen.logpdf_grad(::CategoryAndDirection, x::Tuple{S, AbstractVector{T}},
            camera_position::AbstractVector{U}, categories::AbstractVector{V}, locations::Matrix{W},
            detection_sd::X) where {S<:Integer, T<:Real, U<:Real, V<:Integer, W<:Real, X<:Real}
    return (nothing, nothing, nothing, nothing)
end


(::CategoryAndDirection)(position, categories, locations, sd) = Gen.random(CategoryAndDirection(), position, categories, locations, sd)

has_output_grad(::CategoryAndDirection) = false
has_argument_grads(::CategoryAndDirection) = false

export category_and_direction


print(category_and_direction([0, 0, 0], [1, 2], [1.2 3 4; 7 9 1], 0.05))
