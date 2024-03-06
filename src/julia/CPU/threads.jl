"""
    manager = MultiThread(b=PlainCPU(), nt=Threads.nthreads())

Returns a multithread manager derived from `cpu_manager`, with a fork-join pattern.
When `manager` is passed to [`offload`](@ref), `manager.nthreads` threads are spawn (fork).
They each work on a subset of indices. Progress continues only after all threads have finished (join),
so that `barrier` is not needed between two uses of `offload` and does nothing.

!!! tip
    It is highly recommended to pin the Julia threads to specific cores.
    The simplest way is probably to set `JULIA_EXCLUSIVE=1` before launching Julia.
    See also [Julia Discourse](https://discourse.julialang.org/t/compact-vs-scattered-pinning/69722)
"""
struct MultiThread{Manager<:SingleCPU} <: HostManager
    b::Manager
    nthreads::Int
    MultiThread(b::B=PlainCPU(), nt=Threads.nthreads()) where B = new{B}(b, nt)
end

# Wraps arguments in order to avoid conversion to PtrArray by Polyester
struct Args{T}
    args::T
end

function offload(fun::Fun, manager::MultiThread, range, args::VArgs{NA}) where {Fun<:Function, NA}
    check_boxed_variables(fun)
    args = Args(args)
    @Polyester.batch for id=1:manager.nthreads
        offload_single(fun, manager.b, range, args.args, manager.nthreads, id)
    end
end

struct ConditionBarrier
    barrier :: Threads.Condition # condition for Barrier()
    arrived::Ref{Tuple{Int, Int, Symbol}} # ref to (arrived, id, there) where arrived = number of threads having reached the barrier
    ConditionBarrier() = new(Threads.Condition(), Ref((0,0,:open)))
end

"""
    manager = MainThread(cpu_manager=PlainCPU())

Returns a multithread manager derived from `cpu_manager`, initially in sequential mode.
In this mode, `manager` behaves exactly like `cpu_manager`.
When `manager` is passed to [`parallel`](@ref), `Threads.nthreads()` threads are spawn.
`manager` switches to parallel mode while the threads are running.
When passed to `offload` by these threads, `manager` behaves like `cpu_manager`,
except that the outer loop is distributed among threads.
Furthermore [`barrier`](@ref) and [`share`](@ref)
allow synchronisation and data-sharing across threads.

```julia
multi = MultiThread()
parallel(multi) do manager
    x = share(randn, manager)
    println("Thread \$(Threads.threadid()) has drawn \$x.")
end
```
"""
struct MainThread{Manager} <: HostManager
    cbarrier  :: ConditionBarrier
    manager :: Manager
end
MainThread(manager=PlainCPU()) = MainThread(ConditionBarrier(), manager)

@inline function no_simd(thread::MainThread)
    (; cbarrier, manager) = thread
    return MainThread(cbarrier, no_simd(manager))
end

# It is crucial to store the thread id in the WorkThread because
# there is no guarantee that Threads.threadid() remains the same over the lifetime
# of the thread. When two successive loops have the same loop range,
# relying on Threadid() can lead to the same thread computing over different parts
# of the range. In the absence of a barrier between the loops
# (which should not be necessary), incorrect data may be read in the second loop.

struct WorkThread{Manager} <: HostManager
    cbarrier  :: ConditionBarrier
    manager :: Manager
    N :: Int
    id :: Int
end

@inline function no_simd(thread::WorkThread)
    (; cbarrier, manager, N, id) = thread
    return WorkThread(cbarrier, no_simd(manager), N,id)
end

@inline function offload(fun::Fun, thread::MainThread, ranges, args::Vararg{Any,N}) where {Fun<:Function, N}
    @inline offload_single(fun, thread.manager, ranges, args, 1, 1)
end

@inline function offload(fun::Fun, thread::WorkThread, ranges, args::Vararg{Any,N}) where {Fun<:Function, N}
    @inline offload_single(fun, thread.manager, ranges, args, thread.N, thread.id)
end

#============== parallel, barrier, share ==============#

parallel(::Any, thread::WorkThread) = error("Nested used of ManagedLoops.parallel is forbidden.")

function parallel(main::Fun, thread::MainThread, args::Vararg{Any,N}) where {Fun,N}
    thread.cbarrier.arrived[]=(0,0,:open) # reset barrier
    Threads.@threads for id=1:Threads.nthreads()
        (; cbarrier, manager) = thread
        thread = WorkThread(cbarrier, manager, Threads.nthreads(), id)
        main(thread, args...)
    end
    return nothing
end

function barrier(thread::WorkThread, there::Symbol=:unknown)
    (; cbarrier, N, id) = thread
    wait_condition_barrier(cbarrier, N, id, there)
end


function master(fun::Fun, thread::WorkThread, args::Vararg{Any,N}) where {Fun,N}
    barrier(thread)
    if thread.id == 1
        manager = thread.manager
        fun(manager, args...)
    end
    barrier(thread)
    return nothing
end

function share(fun, thread::WorkThread, args::Vararg{Any,N}) where N
    b = thread.barrier
    # wait for other threads finish their work before master thread calls fun
    barrier(thread)
    id = threads.id
    # only master thread is allowed to write to barrier.shared
    id==1 && ( b.shared = fun(args...) )
    # barrier ensures that other threads read *after* the master has written
    barrier(thread)
    shared = b.shared # ::Any => type instability
    # barrier ensures that master waits until others have read the result
    barrier(thread)
    # now all threads have read, do not keep the result alive longer than needed
    id==1 && ( b.shared = nothing )
    return shared
end

function wait_condition_barrier(cb::ConditionBarrier, size, id, there)
    (; barrier, arrived) = cb
#    @info "Enter condition_barrier" there id size

    lock(barrier)
    try
        (arrived_old, id_old, there_old) = arrived[]
        arrived_new = arrived_old+1
        if (arrived_new>1) && (there_old != there)
            err = """
            Race condition detected. Thread $id is waiting at :
                $there
            while thread $id_old is waiting at :
                $there_old
            """
            error(err)
        end
        if arrived_new == size
            arrived[] = (0, 0, :open)
            notify(barrier)
        else
            arrived[] = (arrived_new, id, there)
            wait(barrier)
        end
    finally
        unlock(barrier)
    end
#    @info "Leave condition_barrier" there id
    return nothing
end
