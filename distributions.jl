import LinearAlgebra
using StatsFuns
using SpecialFunctions
using OffsetArrays

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


### Direction
struct ChooseDirection <: Gen.Distribution{Vector{Float64}} end

const choose_direction = ChooseDirection()

function Gen.random(::ChooseDirection, camera_position::AbstractVector{T},
            location::AbstractVector{U}, detection_sd::V) where {T<:Real,
            U<:Real, V<:Real}
    detection_location = mvnormal(location, LinearAlgebra.Diagonal([detection_sd for _ in 1:length(location)]))

    direction = (detection_location-camera_position)/LinearAlgebra.norm(detection_location-camera_position)

    return direction
end

function Gen.logpdf(::ChooseDirection, x::AbstractVector{T}, camera_position::AbstractVector{U},
            location::AbstractVector{V}, detection_sd::W) where {T<:Real,
            U<:Real, V<:Real, W<:Real}
    k = length(x)
    constant = -0.5 * (k-1) * log(2 * pi * detection_sd)

    term_1 = 0.5 / detection_sd * (LinearAlgebra.dot(location-camera_position, x)/2 - LinearAlgebra.norm(location-camera_position)^2)

    term_2 = log(SpecialFunctions.erfc(-LinearAlgebra.dot(location - camera_position, x) / (2 * sqrt(2*detection_sd))))

    return constant + term_1 + term_2
end

function Gen.logpdf_grad(::ChooseDirection, x::AbstractVector{T}, camera_position::AbstractVector{U},
            location::AbstractVector{V}, detection_sd::W) where {T<:Real,
            U<:Real, V<:Real, W<:Real}
    return (nothing, nothing, nothing, nothing)
end

(::ChooseDirection)(position, location, sd) = Gen.random(ChooseDirection(), position, location, sd)

has_output_grad(::ChooseDirection) = false
has_argument_grads(::ChooseDirection) = false

export choose_direction


###ZeroCategorical
struct ZeroCategorical <: Gen.Distribution{Int} end

"""
    zero_categorical(probs::AbstractArray{U, 1}) where {U <: Real}
Given a vector of probabilities `probs` where `sum(probs) = 1`, sample an `Int` `i` from the set {0, 1, .., `length(probs)-1`} with probability `probs[i]`.
"""
const zero_categorical = ZeroCategorical()

function Gen.logpdf(::ZeroCategorical, x::Int, probs::AbstractArray{U,1}) where {U <: Real}
    (x >= 0 && x < length(probs)) ? log(probs[x]) : -Inf
end

function Gen.logpdf_grad(::ZeroCategorical, x::Int, probs::AbstractArray{U,1})  where {U <: Real}
    (nothing, nothing)
end

function Gen.random(::ZeroCategorical, probs::AbstractArray{U,1}) where {U <: Real}
    one_indexed_probs = OffsetArray(probs, 1:length(probs))
    rand(Distributions.Categorical(one_indexed_probs))-1
end
is_discrete(::ZeroCategorical) = true

(::ZeroCategorical)(probs) = random(ZeroCategorical(), probs)

has_output_grad(::ZeroCategorical) = false
has_argument_grads(::ZeroCategorical) = false

export zero_categorical
