macro tcgtrace()
    quote
        if tracing
            dt = Dict()
            if o.extended_trace
                dt["x"] = copy(x)
                dt["g(x)"] = copy(g)
                dt["Current step size"] = alpha
            end
            g_norm = vecnorm(g, Inf)
            update!(tr,
                    iteration,
                    f_x,
                    g_norm,
                    dt,
                    o.store_trace,
                    o.show_trace,
                    o.show_every,
                    o.callback)
        end
    end
end

immutable Tcg <: Optimizer
    linesearch!::Function
end

Tcg(; linesearch!::Function = armoji_linesearch!) =
  Tcg(linesearch!)
function optimize{T}(df::DifferentiableFunction,
                     initial_x::Vector{T},
                     mo::Tcg,
                     o::OptimizationOptions)
    # Print header if show_trace is set
    print_header(o)

    # Maintain current state in x and previous state in x_previous
    x, x_previous = copy(initial_x), copy(initial_x)

    # Count the total number of iterations
    iteration = 0

    # Track calls to function and gradient
    f_calls, g_calls = 0, 0

    # Count number of parameters
    n = length(x)

    # Maintain current gradient in g and previous gradient in g_previous
    g, g_previous = similar(x), similar(x)


    # The current search direction
    s = similar(x)
    s1 = similar(x)
    # Buffers for use in line search
    x_ls, g_ls = similar(x), similar(x)

    # Intermediate value in CG calculation
    y = similar(x)

    # Store f(x) in f_x
    f_x = df.fg!(x, g)
    @assert typeof(f_x) == T
    f_x_previous = convert(T, NaN)
    f_calls, g_calls = f_calls + 1, g_calls + 1
    copy!(g_previous, g)


    mu =0.8
    # TODO: How should this flag be set?
    mayterminate = false

    # Maintain a cache for line search results
    lsr = LineSearchResults(T)

    # Trace the history of states visited
    tr = OptimizationTrace{typeof(mo)}()
    tracing = o.store_trace || o.show_trace || o.extended_trace || o.callback != nothing
    @tcgtrace

    # Output messages
    if !isfinite(f_x)
        error("Must have finite starting value")
    end
    if !all(isfinite(g))
        @show g
        @show find(!isfinite(g))
        error("Gradient must have all finite values at starting point")
    end

    # Determine the intial search direction
    scale!(copy!(s, g), -1)
    alpha = 1.0
    # Assess multiple types of convergence
    x_converged, f_converged = false, false
    g_converged = vecnorm(g, Inf) < o.g_tol

    # Iterate until convergence
    converged = g_converged
    while !converged && iteration < o.iterations
        # Increment the number of steps we've had to perform
        iteration += 1


        # Determine the distance of movement along the search line
        alpha, f_update, g_update =
          mo.linesearch!(df, x,s, g, lsr, alpha, mayterminate)
        f_calls, g_calls = f_calls + f_update, g_calls + g_update

        # Maintain a record of previous position
        copy!(x_previous, x)

        # Update current position # x = x + alpha * s
        LinAlg.axpy!(alpha, s, x)

        # Maintain a record of the previous gradient
        copy!(g_previous, g)

        # Update the function value and gradient
        f_x_previous, f_x = f_x, df.fg!(x, g)
        f_calls, g_calls = f_calls + 1, g_calls + 1

        x_converged,
        f_converged,
        g_converged,
        converged = assess_convergence(x,
                                       x_previous,
                                       f_x,
                                       f_x_previous,
                                       g,
                                       o.x_tol,
                                       o.f_tol,
                                       o.g_tol)

        # Check sanity of function and gradient
        if !isfinite(f_x)
            error("Function must finite function values")
        end

        #TCG rules
        thetak::T = -vecdot(g,s)/vecdot(s,g_previous)
        @simd for i in 1:n
            @inbounds y[i] = g[i] - g_previous[i]
        end
        @simd for i in 1:n
            @inbounds s1[i] = x[i] - x_previous[i]
        end
        Ak = zeros(n,n)
        @simd for i in 1:n
            @inbounds Ak[i,i]=(2*(f_x_previous - f_x)+vecdot(g_previous+g,s1))/vecnorm(s1,Inf)^2
        end
        y = y + Ak*s1
        betak::T = -vecdot(g,y)/vecdot(s,g_previous) - vecnorm(y,Inf)^2*vecdot(g,s)/(mu*vecdot(s,g_previous)^2)
        @simd for i in 1:n
            @inbounds s[i] = betak * s[i] - g[i] + thetak * (y[i]-s[i])
        end

        @tcgtrace
    end

    return MultivariateOptimizationResults("Triple Conjugate Gradient",
                                           initial_x,
                                           x,
                                           Float64(f_x),
                                           iteration,
                                           iteration == o.iterations,
                                           x_converged,
                                           o.x_tol,
                                           f_converged,
                                           o.f_tol,
                                           g_converged,
                                           o.g_tol,
                                           tr,
                                           f_calls,
                                           g_calls)
end
