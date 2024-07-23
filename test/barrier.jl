mutable struct Barrier{Lock}
    const lock::Lock
    const size::Int
    left::Int # number of threads not having reached the barrier
end

function Base.wait(barrier::Barrier)
    start = time()
    left = lock(barrier.lock) do
        barrier.left = barrier.left-1
    end

    if left==0 # we are the last thread arriving at the barrier
        lock(barrier.lock) do
            barrier.left = barrier.size
        end
    else # wait until barrier.left == barrier.size
        while left<barrier.size
            sleep(0.0)
            left = lock(barrier.lock) do
                barrier.left
            end
        end
    end
    return time()-start
end

function run_thread(id, barrier)
    @info "run_thread" id barrier.size barrier.left
    for i=1:20
        sleep(0.05*rand())
        wait(barrier)
    end
    @info "run_thread" id barrier.size barrier.left
end

function main(N)
    barrier = Barrier(Threads.ReentrantLock(), N, N)
    @info "main" barrier.size barrier.left

    @sync begin
        for id=1:N
            Threads.@spawn run_thread(id, barrier)
        end
    end

    @info "main" barrier.size barrier.left
    println()
end

# main(4)
# main(5)
map(main, (1,4,5,10,100))
