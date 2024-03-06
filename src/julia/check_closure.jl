@generated boxed_variables(::F) where {F} = filter(n -> fieldtype(F,n) <: Core.Box, fieldnames(F))

function check_boxed_variables(fun::Fun) where Fun
    boxed = boxed_variables(fun)
    if !isempty(boxed)
        error("""
        Your use of either 
            * the `@offload` macro 
            * the `forall(...) do ... end` construct
            * the `offload(...) do ... end` construct
        has produced closure `$fun` (=`do ... end` block) which captures variable(s) `$(boxed...)` in a `Core.Box`. This is a serious performance issue and is forbidden.
        To avoid this, you may either :
            * check if the offending variable is redefined outside of the closure ; if so, use another name rather than reusing the same name.
            * explicitly pass this variable to the closure via `@offload` / `offload` / `forall`.
            * enclose the whole construct in a `let ... end` block (see 'Performance of captured variables' in the Julia manual).
        """)
    end
end

function check_closure(fun::Fun) where Fun
    err = false
    msg=""
    names = propertynames(fun)
    if ! isbits(fun)
        msg*= """
        It seems your use of the forall(...) do ... end construct results in closure
            $fun capturing variables $names, some of which are not isbits.
        This is discouraged since some backends require that the compute kernels have only isbits arguments.
        """
        for name in names
            var = getproperty(fun, name)
            if ! isbits(var) 
                msg*= "   Variable $name captured by $fun has type $(typeof(var)) which is not isbits.\n"
            end
        end
        err = true
    end
    if sizeof(fun)>0
        msg*= """It seems your use of the forall(...) do ... end construct results in closure
            $fun capturing variables $names, some of which have size>0.
        This is discouraged since this may severely affect the performance of some backends.
        """
        for name in names
            s = sizeof(getproperty(fun, name))
            if s>0
                msg*= "   Variable $name captured by $fun has size $s>0.\n"
            end
        end
        err = true
    end
    err && error(msg*"Cannot execute forall statement.")
end

