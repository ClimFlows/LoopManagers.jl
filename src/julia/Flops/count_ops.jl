# borrowed from GFlops.jl by François Févotte  https://github.com/triscale-innov/GFlops.jl (MIT expat license)

macro count_ops(funcall)
    return count_ops(funcall)
end

function count_ops(funcall)
    v = []
    e = prepare_call!(v, funcall)
    quote
        let
            ctx = CounterCtx(; metadata=Counter())
            $(v...)
            Cassette.overdub(ctx, () -> begin
                                 $e
                             end)
            ctx.metadata
        end
    end
end

prepare_call!(vars, expr) = expr
prepare_call!(vars, s::Symbol) = esc(s)

function prepare_call!(vars, e::Expr)
    if e.head == :$
        var = gensym()
        push!(vars, :($var = $(prepare_call!(vars, e.args[1]))))
        return var
    else
        return Expr(e.head, map(x -> prepare_call!(vars, x), e.args)...)
    end
end
