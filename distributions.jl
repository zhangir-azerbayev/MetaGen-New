using Gen
import Gen.random
import Gen.logpdf_grad
import Gen.has_output_grad
import Gen.has_argument_grads

struct MvUniform <: Gen.Distribution{Vector{Float64}} end

const mvuniform = MvUniform()

function random(::MvUniform, lows::AbstractVector{T},
            highs::AbstractVector{U}) where {T<:Real, U<:Real}
    return [rand(Uniform(lows[i], highs[i])) for i in 1:length(lows)]
end

function logpdf(::MvUniform, x::AbstractVector{T}, lows::AbstractVector{U},
            highs::AbstractVector{V}) where {T<:Real, U<:Real, V<:Real}
    if any(x < lows || x > highs)
        return -Inf
    else
        return sum([-log(highs[i]-lows[i]) for i in 1:length(lows)])
    end
end

function logpdf_grad(::MvUniform, x::AbstractVector{T},
            lows::AbstractVector{U},
            highs::AbstractVector{V}) where {T<:Real, U<:Real, V<:Real}
    (nothing, nothing, nothing)
end

(::MvUniform)(lows, highs) = random(MvUniform(), lows, highs)

has_output_grad(::MvUniform) = false
has_argument_grads(::MvUniform) = (false, false)

export mvuniform
