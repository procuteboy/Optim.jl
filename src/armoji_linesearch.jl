function armoji_linesearch!{T}(d::Union{DifferentiableFunction,
                                        TwiceDifferentiableFunction},
                               x::Vector{T},
                               s::Vector,
                               g::Vector,
                               lsr::LineSearchResults,
                               alpha::Real,
                               mayterminate::Bool = false,
                               delta::Real = 1e-3,
                               rho::Real = 0.5,
                               tau::Real =1.0,
                               iterations::Integer = 1000000)

    # Count the total number of iterations
    iteration = 0

    # Track calls to function and gradient
    f_calls = 0
    g_calls = 0

    # Count number of parameters
    n = length(x)

    # Store f(x) in f_x
    f_x = d.fg!(x, g)
    f_calls += 1
    g_calls += 1
    x_lr = similar(x)
    # Store angle between search direction and gradient
    gxp = abs(vecdot(g, s))
    alpha = tau*gxp/vecnorm(s,Inf)^2
    # Tentatively move a distance of alpha in the direction of s
    @simd for i in 1:n
        @inbounds x_lr[i] = x[i] + alpha * s[i]
    end

    # Backtrack until we satisfy sufficient decrease condition
    f_x_lr = d.f(x_lr)
    f_calls += 1
    while f_x_lr > f_x + delta * alpha * gxp
        # Increment the number of steps we've had to perform
        iteration += 1

        # Ensure termination
        if iteration > iterations
            error("Too many iterations in armoji_linesearch!")
        end

        # Shrink proposed step-size
        alpha *= rho

        # Update proposed position
        @simd for i in 1:n
            @inbounds x_lr[i] = x[i] + alpha * s[i]
        end

        # Evaluate f(x) at proposed position
        f_x_lr = d.f(x_lr)
        f_calls += 1
    end

    return alpha, f_calls, g_calls
end
