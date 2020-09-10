# Thanks to Tamas Papp
function view_slice_last(arr::AbstractArray{T,N}, dx::Int) where {T,N}
    view(arr, ntuple(_ -> Colon(), N - 1)..., dx)
end

function view_slice_last(arr::AbstractArray{T,N}, dx::CartesianIndex{DX}) where {T,N,DX}
    view(arr, ntuple(_ -> Colon(), N - DX)..., dx)
end

@inline function slice_first(arr::AbstractArray{T,N}, dx::Int) where {T,N}
    arr[dx, ntuple(_ -> Colon(), N - 1)...]
end

@inline function slice_first(arr::AbstractArray{T,N}, dx::CartesianIndex{DX}) where {T,N,DX}
    arr[dx, ntuple(_ -> Colon(), N - DX)...]
end
