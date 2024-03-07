struct FakeGPU{Host} <: DeviceManager
    host::Host
end

struct DeviceArray{T,N, Data<:AbstractArray{T,N}} <: AbstractArray{T,N}
    data :: Data
end
Base.eltype(dev::DeviceArray)        = eltype(dev.data)
Base.eachindex(dev::DeviceArray)     = eachindex(dev.data)
Base.size(dev::DeviceArray, args...) = size(dev.data, args...)
Base.axes(dev::DeviceArray, args...) = axes(dev.data, args...)
Base.view(dev::DeviceArray, args...) = DeviceArray(view(dev.data, args...))
Base.similar(dev::DeviceArray, dims::Union{Integer, AbstractUnitRange}...) = DeviceArray(similar(dev.data, dims...))
Base.similar(dev::DeviceArray, F::Type, dims::Union{Integer, AbstractUnitRange}...) = DeviceArray(similar(dev.data, F, dims...))

function Base.copy!(a::DeviceArray{F, 1, <:AbstractVector{F}}, b::AbstractVector{F}) where F
    copy!(a.data, b)
    return a
end

function Base.getindex(a::DeviceArray, index...)
    error("""Elements of a device array are accessible only from offloaded code. If this error is triggered from offloaded code,
this means that the array has not been properly managed by the device backend. This may be because it is
part of a struct, or captured by a closure. A possible fix is to pass this array explicitly as an argument to the
offloaded function/closure. This is the role of the extra arguments of `offload` and `@offload`. """)
end

@inline unwrap(x)=x
@inline unwrap(x::Tuple)=Tuple(map(unwrap,x))
@inline unwrap(ddata::DeviceArray)=ddata.data
unwrap(x::AbstractArray) = error(
    "$(typeof(x)) is not on the device. You must use `to_device` to
    transfer array arguments to the device before calling `offload` with `backend::FakeGPU`")

to_host(x::Array)=copy(x)
to_host(x::DeviceArray, ::FakeGPU) = copy(x.data)
to_device(x::AbstractArray, ::FakeGPU) = DeviceArray(copy(x))

@inline function forall(fun::Fun, backend::FakeGPU, range, args::VArgs{NA}) where {Fun<:Function, NA}
    check_boxed_variables(fun)
    @inline forall(fun, backend.host, range, unwrap(args)...)
end

@inline function offload(fun::Fun, backend::FakeGPU, range, args::VArgs{NA}) where {Fun<:Function, NA}
    check_boxed_variables(fun)
    @inline offload(fun, backend.host, range, unwrap(args)...)
end

function map_closure(f, closure::Function)
    # replaces by `f(x)` each object `x` captured by `closure`
    captured = map(n->f(getproperty(closure, n)), propertynames(closure))
    return replace_captured(closure, captured)
end

@generated function replace_captured(closure::Closure, args) where {Closure<:Function}
    # here closure and args are Types : typeof(f), Tuple{Type1, ...} and
    # in the returned expression, they are the actual arguments
    basetype = closure.name.wrapper
    types = args.parameters
    N = length(types)
    if N>0
        return Expr(:new, basetype{types...}, ( :(args[$i]) for i=1:N)... )
    else
        return :(closure)
    end
end
