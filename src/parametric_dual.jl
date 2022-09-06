function cosmo_optimizer()
	  cache = MOIU.UniversalFallback(MOIU.Model{Float64}());
    optimizer =  COSMO.Optimizer(check_termination = 1, verbose = false);
    cached = MOIU.CachingOptimizer(cache, optimizer)
    return MOI.Bridges.full_bridge_optimizer(cached, Float64)
end

function setup_dual(
    n, # number of variables,
    K, # number of objectives,
    M, # number of constraints,
    j_max, # number of stored subgradients,
)
    opt = POI.Optimizer(cosmo_optimizer())

    _lambda_p = [
        MOI.add_constrained_variable(opt, MOI.GreaterThan(0.0))[1] for i = 1:K
    ]
    _lambda = [
        MOI.add_constrained_variable(opt, MOI.GreaterThan(0.0))[1] for i = 1:K, j = 1:j_max
    ]
    _mu = [
        MOI.add_constrained_variable(opt, MOI.GreaterThan(0.0))[1] for i = 1:M, j = 1:j_max
    ]
    _mu_p = [
        MOI.add_constrained_variable(opt, MOI.GreaterThan(0.0))[1] for i = 1:M
    ]

    # add convexity constraint
    _constr_l = first.(MOI.add_constrained_variable.(opt, POI.Parameter.(zeros(K, j_max))))
    _constr_m = first.(MOI.add_constrained_variable.(opt, POI.Parameter.(zeros(M, j_max))))

    MOI.add_constraint(
        opt,
        MOI.ScalarQuadraticFunction{Float64}(
            MOI.ScalarQuadraticTerm{Float64}[
                (MOI.ScalarQuadraticTerm{Float64}.(1.0, _lambda, _constr_l))[:];
                (MOI.ScalarQuadraticTerm{Float64}.(1.0, _mu, _constr_m))[:];
           ],
            MOI.ScalarAffineTerm{Float64}[
                MOI.ScalarAffineTerm{Float64}.(1.0,_lambda_p);
                MOI.ScalarAffineTerm{Float64}.(1.0,_mu_p);
            ],
            0.0
        ),
        MOI.EqualTo(1.0)
    )

    # setup linear part of objective function
    MOI.set(opt, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    # coefficients for the linear part of the objective
    # for each j, each vector in _lambda adds K terms and each vector in _mu adds M terms:
    _cl_l_p = first.(MOI.add_constrained_variable.(opt, POI.Parameter.(zeros(K))))
    _cl_m_p = first.(MOI.add_constrained_variable.(opt, POI.Parameter.(zeros(M))))
    _cl_l = first.(MOI.add_constrained_variable.(opt, POI.Parameter.(zeros(K, j_max))))
    _cl_m = first.(MOI.add_constrained_variable.(opt, POI.Parameter.(zeros(M, j_max))))

    MOI.set(
        opt, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
        MOI.ScalarQuadraticFunction{Float64}(
            MOI.ScalarQuadraticTerm{Float64}[
                MOI.ScalarQuadraticTerm{Float64}.(1.0, _cl_l_p, _lambda_p);
                MOI.ScalarQuadraticTerm{Float64}.(1.0, _cl_m_p, _mu_p);
                (MOI.ScalarQuadraticTerm{Float64}.(1.0, _cl_l, _lambda))[:];
                (MOI.ScalarQuadraticTerm{Float64}.(1.0, _cl_m, _mu))[:];
            ],
            MOI.ScalarAffineTerm{Float64}[],
            0.0
        )
    )

    # setup quadratic terms of objective function
    num_total = (j_max + 1) * (K + M)
    # parameters for terms involving _lambda_p
    _c_l_p = [
        MOI.add_constrained_variable(opt, POI.Parameter(0.0))[1] for l=1:K, r=1:num_total
    ]
    # … involving _lambda
    _c_l = [
        MOI.add_constrained_variable(opt, POI.Parameter(0.0))[1] for l=1:K, r=1:num_total, j=1:j_max
    ]
    # … involving _mu_p
    _c_m_p = [
        MOI.add_constrained_variable(opt, POI.Parameter(0.0))[1] for l= 1:M, r =1:num_total
    ]
    # … involving _mu
    _c_m = [
        MOI.add_constrained_variable(opt, POI.Parameter(0.0))[1] for l= 1:M, r=1:num_total, j=1:j_max
    ]

    # check: K*num_total + K*j_max*num_total + M*num_total + M*j_max*num_total
    # =  ( j_max + 1 ) (K * num_total + M * num_total) = num_total^2

    add_quad_coeff!(opt, _c_l_p, _lambda_p, K, M, j_max, _lambda_p, _lambda, _mu_p, _mu)
    add_quad_coeff!(opt, _c_m_p, _mu_p, K, M, j_max, _lambda_p, _lambda, _mu_p, _mu)
    for j=1:j_max
        add_quad_coeff!(opt, _c_l[:,:,j], _lambda[:,j], K, M, j_max, _lambda_p, _lambda, _mu_p, _mu)
        add_quad_coeff!(opt, _c_m[:,:,j], _mu[:,j], K, M, j_max, _lambda_p, _lambda, _mu_p, _mu)
    end

    return (
        opt,
        _cl_l_p, _cl_l, _cl_m_p, _cl_m, _c_l_p, _c_l, _c_m_p, _c_m,
        _lambda_p, _lambda, _mu_p, _mu,
        _constr_l, _constr_m
    )
end

function add_quad_coeff!(
    opt, dest_arr, lhs_arr,
    K, M, j_max,
    _lambda_p, _lambda, _mu_p, _mu
    )

    kappa = size(dest_arr, 1) # either M or K

    PQOC = POI.QuadraticObjectiveCoef()

    # “multiply” variables in `lhs_array` with every other variable
    for l1=1:kappa, l2=1:K
        MOI.set(opt, PQOC, (lhs_arr[l1], _lambda_p[l2]), dest_arr[l1,l2])
    end
    offset = K
    for j=1:j_max
        for l1=1:kappa, l2=1:K
            MOI.set(opt, PQOC, (lhs_arr[l1], _lambda[l2, j]), dest_arr[l1,l2+offset])
        end
        offset += K
    end

    for l1=1:kappa, l2=1:M
        MOI.set(opt, PQOC, (lhs_arr[l1], _mu_p[l2]), dest_arr[l1,l2+offset])
    end
    offset += M

    for j=1:j_max
        for l1=1:kappa, l2=1:M
            MOI.set(opt, PQOC, (lhs_arr[l1], _mu[l2, j]), dest_arr[l1,l2+offset])
        end
        offset += M
    end
    return nothing
end

function set_quad_coeff!(
    opt,
    dest_arr,      # coefficients to be set
    lhs_arr,       # jacobian corresponding to dest_arr
    n, K, M, j_max, J,
    factor,
    _c_l_p, _c_l, _c_m_p, _c_m,
    sg_f_p, sg_f, sg_g_p, sg_g
    )

    kappa = size(dest_arr, 1)
    PV = POI.ParameterValue()

    # (i) terms resulting from multiplication with _lambda_p:
    for l1=1:kappa, l2=1:K
        MOI.set(
            opt, PV, dest_arr[l1, l2],
            factor * sum( lhs_arr[l1, vi] * sg_f_p[l2,vi] for vi=1:n )
        )
    end
    offset = K
    # (ii) terms resulting from multiplication with columns in _lambda:
    for j=1:j_max
        if J[j]
            for l1=1:kappa, l2=1:K
                MOI.set(
                    opt, PV, dest_arr[l1,l2+offset],
                    factor * sum( lhs_arr[l1, vi] * sg_f[l2,vi,j] for vi=1:n )
                )
            end
        else
            MOI.set.(opt, PV, dest_arr[:,1+offset:offset+K], zeros(K,K))
        end
        offset += K
    end

    # (iii) multiplication with _mu_p:
    for l1=1:kappa, l2=1:M
        MOI.set(
            opt, PV, dest_arr[l1,l2+offset],
            factor * sum( lhs_arr[l1, vi] * sg_g_p[l2,vi] for vi=1:n )
        )
    end
    offset += M

    # (iv) multiplication with _mu:
    for j=1:j_max
        if J[j]
            for l1=1:kappa, l2=1:M
                MOI.set(
                    opt, PV, dest_arr[l1,l2+offset],
                    factor * sum( lhs_arr[l1, vi] * sg_g[l2,vi,j] for vi=1:n )
                )
            end
        else
            MOI.set.(opt, PV, dest_arr[:,1+offset:offset+M], zeros(M,M))
        end
        offset += M
    end
    return nothing
end
function set_parameters_and_solve!(
    opt,
    n, K, M, j_max, J,
    weight,
    _cl_l_p, _cl_l, _cl_m_p, _cl_m,
    _c_l_p, _c_l, _c_m_p, _c_m,
    _constr_l, _constr_m,
    err_f_p, err_f, err_g_p, err_g,
    sg_f_p, sg_f, sg_g_p, sg_g
)

    # linear parameters are easy:
    PV = POI.ParameterValue()
    MOI.set.(opt, PV, _cl_l_p, err_f_p)
    MOI.set.(opt, PV, _cl_m_p, err_g_p)
    for j=1:j_max
        if J[j]
            for i=1:K
                MOI.set(opt, PV, _cl_l[i,j], err_f[i,j])
                MOI.set(opt, PV, _constr_l[i,j], 1.0)
            end
        else
            MOI.set.(opt, PV, _cl_l[:,j], zeros(K))
            MOI.set.(opt, PV, _constr_l[:,j], zeros(K))
        end
    end
    for j=1:j_max
        if J[j]
            for i=1:M
                MOI.set(opt, PV, _cl_m[i,j], err_g[i,j])
                MOI.set(opt, PV, _constr_m[i,j], 1.0)
            end
        else
            MOI.set.(opt, PV, _cl_m[:,j], zeros(M))
            MOI.set.(opt, PV, _constr_m[:,j], zeros(M))
        end
    end

    factor = 1/(2*weight)
    # for quadratic terms we have to aggregate
    set_quad_coeff!(opt, _c_l_p, sg_f_p,
                    n, K, M, j_max, J, factor, _c_l_p, _c_l, _c_m_p, _c_m, sg_f_p, sg_f, sg_g_p, sg_g)
    set_quad_coeff!(opt, _c_m_p, sg_g_p,
                    n, K, M, j_max, J, factor, _c_l_p, _c_l, _c_m_p, _c_m, sg_f_p, sg_f, sg_g_p, sg_g)
    for j=1:j_max
        set_quad_coeff!(opt, _c_l[:,:,j], sg_f[:,:,j],
                        n, K, M, j_max, J, factor, _c_l_p, _c_l, _c_m_p, _c_m, sg_f_p, sg_f, sg_g_p, sg_g)
        set_quad_coeff!(opt, _c_m[:,:,j], sg_g[:,:,j],
                        n, K, M, j_max, J, factor, _c_l_p, _c_l, _c_m_p, _c_m, sg_f_p, sg_f, sg_g_p, sg_g)
    end

    MOI.optimize!(opt)
    return nothing
end

function scale_solutions!(
    # modified:
	  lambda_p, lambda, mu_p, mu, 	# lagrangian multipliers
	  sum_lambda, sum_mu, 				  # sum of multipliers
	  z_f, z_g, nz_f, nz_g, 				# zero index vectors
	  d, p,								          # descent steps arrays
	  sg_f_p, sg_g_p, lin_f_p, lin_g_p, dist_f_p, dist_g_p, # aggregation arrays
	  _err_f_p, _err_g_p, 				# linearization error with scaled multipliers
	  # not modified:
    opt, _lambda_p, _lambda, _mu_p, _mu,
	  J, weight, fx, gamma_f, gamma_g,
	  sg_f, sg_g, lin_f, lin_g, dists,
)
    # read solutions:
    nJ = sum(J)

    VP = MOI.VariablePrimal()
    lambda_p .= MOI.get.( opt, VP, _lambda_p)
    lambda .= MOI.get.( opt, VP, _lambda )
    mu_p .= MOI.get.( opt, VP, _mu_p)
    mu .= MOI.get.( opt, VP, _mu )

    # scale multipliers (per objective/constraint):
	  sum_lambda = lambda_p .+ sum(lambda[:, J]; dims=2)[:] # K-vector
	  sum_mu = mu_p .+ sum(mu[:, J]; dims=2)[:]

	  z_f = iszero.(sum_lambda)
	  z_g = iszero.(sum_mu)
	  nz_f = .!z_f
	  nz_g = .!z_g

    mult_fallback = 1/(nJ + 1)

    ## row-wise scaling:
    ## In the book [^3], these parameters are denoted with a tilde.
    ## We save the memory and instead mutate the original arrays.
    lambda[ nz_f, : ] ./= sum_lambda[ nz_f ]
    lambda[ z_f, : ] .= mult_fallback

    mu[ nz_g, : ] ./= sum_mu[ nz_g ]
    mu[ z_g, : ] .= mult_fallback

    lambda_p[nz_f] ./= sum_lambda[ nz_f ]
    lambda_p[z_f] .= mult_fallback

    mu_p[ nz_g ] ./= sum_mu[ nz_g ]
    mu_p[ z_g ] .= mult_fallback

    # aggregate and begin update of the subgradients
    # in accordance with the book, we would have to
    # denote the vectors by `pₖᶠ, f̃ₖᵖ,s̃ₖᵖ` etc.
    # As we don't need `sg_f_p, ...` anymore, we
    # instead modify them in place:
    sg_f_p .*= lambda_p 		# pₖᶠ = λ̃ₖᵖ .* sg_f_p
    sg_g_p .*= mu_p 		# pₖᵍ = μ̃ₖᵖ .* sg_g_p
    lin_f_p .*= lambda_p 		# f̃ₖᵖ = λ̃ₖᵖ .* lin_f_p
    lin_g_p .*= mu_p 		# g̃ₖᵖ = μ̃ₖᵖ .* lin_g_p
    dist_f_p[] *= sum(lambda_p)	# s̃ₖᶠ = λ̃ₖᵖ * dist_f_p # TODO does `sum(lambda_p)` make sense??
    dist_g_p[] *= sum(mu_p) 	# s̃ₖᵍ = μ̃ₖᵖ * dist_g_p
    for (j,_j) = enumerate(J)
        if _j
            sg_f_p .+= lambda[:,j] .* sg_f[:,:,j]
            sg_g_p .+= mu[:,j] .* sg_g[:,:,j]
            lin_f_p .+= lambda[:,j] .* lin_f[:,j]
            lin_g_p .+= mu[:,j] .* lin_g[:,j]
            dist_f_p[] += sum(lambda[:,j]) * dists[j] 	# TODO as above: `sum()` sensible?
            dist_g_p[] += sum(mu[:,j]) * dists[j]
        end
    end
    p .= sg_f_p'sum_lambda .+ sg_g_p'sum_mu
    _err_f_p .= max.( abs.( fx .- lin_f_p ), dist_f_p[] .* gamma_f  ) # we do not really need extra arrays here, just use err_f_p...
    _err_g_p .= max.( abs.(lin_g_p), dist_g_p[] .* gamma_g )
    _err_p = sum_lambda'_err_f_p + sum_mu'_err_g_p

    d .= -1/weight .* p

    pp = p'p
    p_norm = sqrt(pp)

    crit_val1 = 0.5 * pp + _err_p
    var_est_p = p_norm + _err_p
    crit_val2 = -( pp/weight + _err_p )		# TODO this is a guess...

	  return crit_val1, var_est_p, crit_val2
end

