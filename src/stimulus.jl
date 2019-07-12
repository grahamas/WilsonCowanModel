

function distance(x,y) where {T <: Real}
    sqrt(sum((x .- y).^2))
end

struct GaussianNoiseStimulus{T,N} <: AbstractStimulus{T,N}
    mean::T
    sd::T
end
function GaussianNoiseStimulus{T,N}(; SNR::T=0.0, mean::T=0.0) where {T,N}
    sd = sqrt(1/10 ^ (SNR / 10))
    GaussianNoiseStimulus{T,N}(mean, sd)
end
function gaussian_noise!(val::AT, mean::T, sd::T) where {T, AT<:AbstractArray{T}} # assumes signal power is 0db
    val .+= randn(size(val))
    val .*= sd
    val .+= mean
end
function NeuralModels.make_stimulus(wns::GaussianNoiseStimulus{T}, space::AbstractSpace{T}) where {T}
    (val,t) -> gaussian_noise!(val, wns.mean, wns.sd) # Not actually time dependent
end

abstract type TransientBumpStimulus{T,N} <: AbstractStimulus{T,N} end
function NeuralModels.make_stimulus(bump::TBS, space::AbstractSpace{T,N}) where {T, N, TBS<:TransientBumpStimulus{T,N}}
    bump_frame = on_frame(bump, space)
    function stimulus!(val, t)
        for window in bump.time_windows
            if window[1] <= t < window[2]
                val .+= bump_frame
            end
        end
    end
end

struct SharpBumpStimulus{T,N} <: TransientBumpStimulus{T,N}
    width::T
    strength::T
    time_windows::Array{Tuple{T,T},1}
    center::NTuple{N,T}
end

function SharpBumpStimulus{T,N}(; strength::T=nothing, width::T=nothing,
        duration=nothing, time_windows=nothing, center=NTuple{N,T}(zero(T) for i in 1:N)) where {T,N}
    if time_windows == nothing
        return SharpBumpStimulus{T,N}(width, strength, [(zero(T), duration)], center)
    else
        @assert duration == nothing
        return SharpBumpStimulus{T,N}(width, strength, time_windows, center)
    end
end

function on_frame(sbs::SharpBumpStimulus{T,N}, space::AbstractSpace{T,N}) where {T,N}
    coords = coordinates(space)
    frame = zero(space)
    half_width = sbs.width / 2.0
    frame[distance.(coords, Ref(sbs.center)) .<= half_width] .= sbs.strength
    return frame
end


struct NoisyStimulus{T,N,STIM} <: AbstractStimulus{T,N}
    noise::GaussianNoiseStimulus{T,N}
    stimulus::STIM
end
function NoisyStimulus{T,N}(; stim_type::Type=SharpBumpStimulus{T,N}, SNR::T, mean::T=0.0, kwargs...) where {T,N}
    NoisyStimulus{T,N,stim_type}(
            GaussianNoiseStimulus{T,N}(SNR = SNR, mean=mean),
            stim_type(; kwargs...)
        )
end
@memoize Dict function NeuralModels.make_stimulus(noisy_stimulus::NoisyStimulus{T}, space::AbstractSpace) where {T}
    noise_mutator = NeuralModels.make_stimulus(noisy_stimulus.noise, space)
    stim_mutator = NeuralModels.make_stimulus(noisy_stimulus.stimulus, space)
    (val, t) -> (noise_mutator(val,t); stim_mutator(val,t))
end
