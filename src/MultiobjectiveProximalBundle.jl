module MultiobjectiveProximalBundle

using Parameters: @with_kw
using LinearAlgebra: norm

import MathOptInterface as MOI
import ParametricOptInterface as POI

const MOIU = MOI.Utilities

using COSMO

using DocStringExtensions

export mpb_optimize
export MPBOptions

include("parametric_dual.jl")

"""
    MPBOptions(; kwargs...)

Options to configure the algorithm. An instance of type
`MPBOptions` can be passed to optimization methods
with the keyword argument `options`.

Possible `kwargs` for `MPBOptions` are:
$(FIELDS)
"""
@with_kw struct MPBOptions
	  "Maximum number of iterations (default: 1000)."
	  max_iter :: Int = 1000 			# max_iter=1000 was used in [^2]

	  "Maximum number of function evaluations (default: 1000)."
	  max_fun_calls :: Int = 1000 	# max_fun_calls=100 was used in [^2]

	  "Line-search parameter for determination of ``tL ∈ [0,1]`` (default: 1e-2)."
	  mL :: Float64 = 1e-2 	# mL=1e-2 was used in [^1], sec. 5.1
	  "Line-search parameter for determination of ``tR`` (default: 0.5)."
	  mR :: Float64 = 0.5 	# mR=0.5 was used in [^1]
	  "Line-search parameter for classification of long or short steps (default: 1e-2)."
	  t_bar :: Float64 = 1e-2 	# t_bar=1e-2 was used in [^1]
	  "Minimum stepsize tested in line-search (default: `eps(Float64)*1e2`)."
	  t_min :: Float64 = eps(Float64) * 1e2

	  "Maximum number of function calls in line-search (default: 100)."
	  max_fun_calls_ls :: Int = 100 	# 100 was used in [^2]

	  "Fallback non-convexity constant if none are provided (default: 0.5)."
	  gamma_default :: Float64 = 0.5 	# γ_default=0.5 was used in [^2]

	  #"Feasibility tolerance."
	  #tol_feas :: Float64 = 1e-9 	# tol_feas=1e-9 was used in [^2]

	  "Final accuracy tolerance (default: 1e-5)."
	  tol_acc :: Float64 = 1e-5 			# εₛ=1e-5 was used in [^2]

	  "Initial weight ``u^1`` (default: NaN)."
	  initial_weight :: Float64 = NaN # NaN leads to computation according to [^2]

	  "Maximum number of stored subgradients (default: 10)."
	  max_subgrads :: Int = 10   # max_subgrads=10 used in [^2]

	  @assert 0 < mL < 0.5 "`mL` must be in (0, 0.5)."
	  @assert mL < mR < 1 "`mR` must be in `(m_x, 1)`."
	  @assert 0 < t_bar <= 1 "`t_bar` must be in (0,1]."
	  @assert gamma_default >= 0
	  @assert max_fun_calls > 1
	  @assert max_fun_calls_ls > 1
	  @assert max_iter > 0
	  @assert max_subgrads >= 2
	  #@assert tol_feas * tol_objf > 0
	  @assert tol_acc > 0
	  @assert t_min > 0
end

"""
    check_functions(_objectives, _constraints)

Return the number of scalar objectives functions,
the number of constraint functions and the vectors
of objective functions and constraint functions.

`_objectives` and `_constraints` are provided by
the user. In case of a single function we allow to
pass a `Function` instead of a `Vector{<:Function}`.
"""
function check_functions(
	  _objective_functions, _constraint_functions;
    )
	  ## ensure vector-of-functions:
	  objective_functions = if isa(_objective_functions, Function)
		    [_objective_functions,]
	  elseif isa(_objective_functions, AbstractVector{<:Function})
		    _objective_functions
	  else
		    error("Provide a vector of objective functions.")
	  end
	  num_objectives = length(objective_functions)
	  @assert num_objectives > 0 "Provide at least one objective function."

	  constraint_functions = if isa(_constraint_functions, Function)
		    [_constraint_functions,]
	  elseif isa(_constraint_functions, AbstractVector{<:Function})
		    _constraint_functions
	  else
		    error("Provide a vector of constraint functions.")
	  end
	  num_constraints = length(constraint_functions)

	  return num_objectives, num_constraints, objective_functions, constraint_functions
end

"""
    gamma_vectors(
	    gammas_objectives, gammas_constraints,
	    num_objectives, num_constraints;
		options
    )

    Return vectors of nonlocality constants for objectives
    and constraints. `gammas_objectives` and `gammas_constraints`
    are provided by the user.
"""
function gamma_vectors(
	  gammas_objectives, gammas_constraints,
	  num_objectives, num_constraints;
	  options
    )
	  gamma_f = if gammas_objectives isa Real
		    fill( gammas_objectives, num_objectives )
	  elseif gammas_objectives isa Vector{Real} && length(gammas_objectives) == num_objectives
		    gammas_objectives
	  else
		    fill( options.gamma_default, num_objectives )
	  end

	  gamma_g = if gammas_constraints isa Real
		    fill( gammas_constraints, num_constraints )
	  elseif gammas_constraints isa Vector{Real} && length(gammas_constraints) == num_constraints
		    gammas_constraints
	  else
		    fill( options.gamma_default, num_constraints )
	  end

	  return gamma_f, gamma_g
end

function prealloc_iter_arrays(x0,n,K,M,j_max; options)
	  x = copy(x0)
	  # evaluation value vectors at current iterate
	  fx = Vector{Float64}(undef, K)
	  gx = Vector{Float64}(undef, M)

	  return x, fx, gx
end

function prealloc_diff_arrays(x0,n,K,M,j_max; options)
	  # generalized Jacobian arrays
	  # sg_f[:,:,j] is the jacobian of f at yʲ etc.
	  # sg_f[i,:,j] is the gradient of fᵢ at yʲ
	  sg_f = Array{Float64}(undef, K, n, j_max)
	  sg_g = Array{Float64}(undef, M, n, j_max)

	  # matrices to store the linearization values:
	  # lin_f[i,j] = linearization of objective i at *current* x around yʲ, i.e.,
	  # lin_f[:,j] =̂ lin_f(x) = f(yʲ) + sg_fʲ(x - yʲ)
	  lin_f = Matrix{Float64}(undef, K, j_max)
	  lin_g = Matrix{Float64}(undef, M, j_max)

	  # linearization errors
	  # err_f[i,j] = linearization error of objective i at *current* x around yʲ, i.e.,
	  # err_f[:, j] =̂ max.( abs.( fx .- lin_f[:,j] ), gamma_f .* (sⱼᵏ)² )
	  # err_g[:, j] =̂ max.( abs.( lin_g[:,j] ), gamma_g .* (sⱼᵏ)² )
	  err_f = Matrix{Float64}(undef, K, j_max)
	  err_g = Matrix{Float64}(undef, M, j_max)

	  # array of aggregate distances measures of x to yʲ
	  dists = Vector{Float64}(undef, j_max)

	  return sg_f, sg_g, lin_f, lin_g, err_f, err_g, dists
end

function prealloc_ls_res_arrays(x0,n,K,M,j_max; options)
	  # LINE SEARCH ARRAYS
	  # we denote the *n*ext iterate as `x_n`
	  # instead of only calculating a step-size `t_x`,
	  # we directly set and store `x_n` and the values
	  # `fx_n` and `gx_n`.
	  x_n = Vector{Float64}(undef, n)
	  fx_n = Vector{Float64}(undef, K)
	  gx_n = Vector{Float64}(undef, M)

	  # likewise, we do not only calculate `t_y`
	  # but directly store the derivative values for the
	  sg_f_n = Matrix{Float64}(undef, K, n)
	  sg_g_n = Matrix{Float64}(undef, M, n)

	  # additionally, we can already calculate the
	  # linearization around `yn` evaluated at `x_n`
	  # we store these values as `lin_f_n` and `lin_g_n`.
	  # Lastly, the linearization errors are `err_fx_n` and `err_gx_n`.
	  lin_f_n = Vector{Float64}(undef, K)
	  lin_g_n = Vector{Float64}(undef, M)
	  err_f_n = Vector{Float64}(undef, K)
	  err_g_n = Vector{Float64}(undef, M)

	  return x_n, fx_n, gx_n, sg_f_n, sg_g_n, lin_f_n, lin_g_n, err_f_n, err_g_n
end

function prealloc_ls_w_arrays(x0,n,K,M,j_max; options)
	  # working arrays during line-search, at x_t = x + t*d
	  x_t = Vector{Float64}(undef, n)
	  f_x_t = Vector{Float64}(undef, K)
	  g_x_t = Vector{Float64}(undef, M)
	  sg_f_t = Matrix{Float64}(undef, K, n)
	  sg_g_t = Matrix{Float64}(undef, M, n)
	  lin_f_t = similar(f_x_t)
	  err_f_t = similar(f_x_t)
	  diff_t = similar(f_x_t) 	# stores jacobian-vector product
	  return x_t, f_x_t, g_x_t, sg_f_t, sg_g_t, lin_f_t, err_f_t, diff_t
end

function prealloc_step_arrays(x0,n,K,M,j_max; options)
	  # descent step array(s)
	  p = similar(x0)
	  d = similar(x0)
	  return p, d
end

function prealloc_aggregation_arrays(x0,n,K,M,j_max; options)
	  # aggregate subgradients:
	  sg_f_p = Matrix{Float64}(undef,K,n)
	  sg_g_p = Matrix{Float64}(undef,M,n)
	  # aggregate linearizations
	  lin_f_p = Vector{Float64}(undef,K)
	  lin_g_p = Vector{Float64}(undef,M)
	  # aggregate errors
	  err_f_p = similar(lin_f_p)
	  err_g_p = similar(lin_g_p)
	  # aggregate distance
	  dist_f_p = Ref(0.0)
	  dist_g_p = Ref(0.0)

	  # errors computed from scaled lagrange coefficients:
	  _err_f_p = similar(err_f_p)
	  _err_g_p = similar(err_g_p)

	  return sg_f_p, sg_g_p, lin_f_p, lin_g_p, err_f_p, err_g_p, dist_f_p, dist_g_p,_err_f_p, _err_g_p
end

function prealloc_lagrange_arrays(x0,n,K,M,j_max; options)
	  lambda_p=Vector{Float64}(undef, K) # Lagrange multipliers for aggregated jacobian of f
	  mu_p=Vector{Float64}(undef, M) # Lagrange multipliers for aggregated jacobian of g
	  lambda=Matrix{Float64}(undef,K,j_max) # multipliers for stored jacobians of f
	  mu=Matrix{Float64}(undef,M,j_max) # multipliers for stored jacobians of g
	  sum_lambda=similar(lambda_p) # column sum of λ multipliers for f
	  sum_mu=similar(mu_p) # column sum of μ mulitpliers for g
	  z_f=Vector{Bool}(undef, K) 	# zero rows of sum_lambda
	  z_g=Vector{Bool}(undef, M) 	# zero rows of sum_mu
	  nz_f=similar(z_f)  # elementwise ¬ of `z_f`
	  nz_g=similar(z_g)  # elementwise ¬ of `z_g`
	  return lambda_p, lambda, mu_p, mu, sum_lambda, sum_mu, z_f, z_g, nz_f, nz_g
end

"""
    eval_and_jac!(y, J, funcs, x)

Evaluate all scalar functions in `funcs`.
Mutate `y` to contain the primals and
`J` to have the gradients as rows.
"""
function eval_and_jac!(y, J, funcs, x)
	  for (i,f)=enumerate(funcs)
		    fi, dfi = f(x)
		    y[i] = fi
		    J[i,:] .= dfi
	  end
	  return nothing
end

function mpb_optimize(
	  x0, objectives, constraints,
	  gammas_f = nothing, gammas_g = nothing;
	  options=MPBOptions()
    )

    # Check dimensions
	  n = length(x0)
	  @assert n > 0 "`x0` must not be empty!."

	  K, M, f_funcs, g_funcs = check_functions(objectives, constraints)

	  # Pre-allocation of working arrays:
	  j_max = options.max_subgrads
	  x, fx, gx = prealloc_iter_arrays(x0, n, K, M, j_max; options)
	  (
		    sg_f, sg_g, lin_f, lin_g, err_f, err_g, dists
	  ) = prealloc_diff_arrays(x0, n, K, M, j_max; options)
	  (
		    x_n, fx_n, gx_n, sg_f_n, sg_g_n, lin_f_n, lin_g_n, err_f_n, err_g_n
	  ) = prealloc_ls_res_arrays(x0, n, K, M, j_max; options)
	  (
		    x_t, f_x_t, g_x_t, sg_f_t, sg_g_t, lin_f_t, err_f_t, diff_t
	  ) = prealloc_ls_w_arrays(x0, n, K, M, j_max; options)
	  p, d = prealloc_step_arrays(x0, n, K, M, j_max; options)
	  (
		    sg_f_p, sg_g_p, lin_f_p, lin_g_p, err_f_p, err_g_p,
		    dist_f_p, dist_g_p,_err_f_p, _err_g_p
	  ) = prealloc_aggregation_arrays(x0, n, K, M, j_max; options)
	  (
		    lambda_p, lambda, mu_p, mu, sum_lambda, sum_mu, z_f, z_g, nz_f, nz_g
	  ) = prealloc_lagrange_arrays(x0, n, K, M, j_max; options)

	  # vectors of nonlocality constants
	  gamma_f, gamma_g = gamma_vectors(gammas_f,gammas_g, K, M; options)

	  # Initialization of first iterate
	  x .= x0
	  j_x = 1

	  dists[j_x] = 0.0 	# yₖ == x

	  # evaluate and set subgradients/jacobians
	  _sg_f = view(sg_f, :, :, j_x)
	  _sg_g = view(sg_g, :, :, j_x)
	  eval_and_jac!(fx, _sg_f, f_funcs, x)
	  eval_and_jac!(gx, _sg_g, g_funcs, x)
	  num_fun_calls = Ref(1)

	  # initial aggregrate directions
	  sg_f_p .= _sg_f
	  sg_g_p .= _sg_g

	  # initial linearizations `lin_f(x) = f(x) + (x - y)^T ξ` = f(x):
	  lin_f[:,j_x] .= fx
	  lin_g[:,j_x] .= gx
	  lin_f_p .= fx
	  lin_g_p .= gx

	  # initial locality measures (y == x):
	  err_f[:,j_x] .= 0.0
	  err_g[:,j_x] .= 0.0
	  err_f_p .= 0.0
	  err_g_p .= 0.0

	  dist_f_p[] = 0.0
	  dist_g_p[] = 0.0

	  # subgradient indices
	  J = zeros(Bool, j_max) 	# index of linearizations used in step calculation
	  J[j_x] = true

	  # inital weight for descent step calculation
	  weight = if isnan(options.initial_weight)
		    sum(norm.(eachrow(_sg_f)))/K
	  else
		    options.initial_weight
	  end

	  var_est = Inf	 # variation estimate εᵥᵏ in [^3], EPSV in [^2]
	  weight_counter = 0  	# iteration counter for changes of `u`, iᵤᵏ in [^3], IUK in [^2]
	  weight_min = 1e-10 * weight
	  zeta = 1 - 0.5 * 1/(1-options.mL)


    # pre-allocate the dual problem structure, coefficient and solution arrays:
    (
        opt,
        _cl_l_p, _cl_l, _cl_m_p, _cl_m, _c_l_p, _c_l, _c_m_p, _c_m,
        _lambda_p, _lambda, _mu_p, _mu,
        _constr_l, _constr_m,
    ) = setup_dual(n,K,M,j_max)

	  for iter=1:options.max_iter

        set_parameters_and_solve!(
            opt,
            n, K, M, j_max, J,
            weight,
            _cl_l_p, _cl_l, _cl_m_p, _cl_m,
            _c_l_p, _c_l, _c_m_p, _c_m,
            _constr_l, _constr_m,
            err_f_p, err_f, err_g_p, err_g,
            sg_f_p, sg_f, sg_g_p, sg_g
        )

		    crit_val1, var_est_p, crit_val2 = scale_solutions!(
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

		    if crit_val1 <= options.tol_acc
			      @info "Criticality is $(crit_val1) <= $(options.tol_acc)."
			      break
		    end

		    LS_success, weight_n, weight_counter_n, var_est_n, tL, tR, dist_n = linesearch!(
			      # modified:
			      x_n, fx_n, gx_n, sg_f_n, sg_g_n,
			      lin_f_n, lin_g_n, err_f_n, err_g_n,
			      x_t, f_x_t, g_x_t, sg_f_t, sg_g_t, lin_f_t, err_f_t, diff_t,
			      num_fun_calls,
			      # not_modified
			      j_x, zeta, gamma_f, gamma_g,
			      weight, weight_counter, var_est, var_est_p, weight_min,
			      crit_val2, d, x, fx, gx, sg_f, sg_g,
			      f_funcs, g_funcs
			      ;
			      options
		    )

		    @info """
		    k      = $(iter)
		    x     = $(x)
		    f(x)  = $(fx)
		    g(x)  = $(gx)
		    weight = $(weight)
		    crit_val1     = $(crit_val1)
		    d      = $(d)
		    tL     = $(tL)
		    tR     = $(tR)
		    """

		    if !LS_success
			      break
		    end

		    # updates:
		    weight, weight_counter, var_est = weight_n, weight_counter_n, var_est_n

		    # set next iterate and values
		    x .= x_n
		    fx .= fx_n
		    gx .= gx_n

		    # calculate index for derivative storage
		    j_x_n = mod( j_x + 1, j_max )
		    if j_x_n == 0
			      j_x_n = j_max
		    end

		    # update previous linearizations to hold at x_n
		    if tL != 0
			      d .*= tL  # so now: `x_n - x == d` and for updating we only have to add ξ*d

			      for (j,_j)=enumerate(J)
				        if _j && j != j_x_n
					          lin_f[:,j] .+= sg_f[:,:,j]*d
					          lin_g[:,j] .+= sg_g[:,:,j]*d

					          dists[j] += dist_n
					          err_f[:,j] .= max.( abs.(fx_n .- lin_f[:,j]), dists[j]^2 .* gamma_f )
					          err_g[:,j] .= max.( abs.(lin_g[:,j]), dists[j]^2 .* gamma_g )
				        end
			      end
		    end
		    dist_f_p[] += dist_n
		    dist_g_p[] += dist_n
		    lin_f_p .+= tL .* sg_f_p*d
		    lin_g_p .+= tL .* sg_g_p*d
		    err_f_p .=  max.( abs.(fx_n .- lin_f_p), dist_n^2 .* gamma_f )
		    err_g_p .=  max.( abs.(lin_g_p), dist_n^2 .* gamma_g )

		    dists[j_x_n] = dist_n
		    # store new derivatives
		    sg_f[:,:,j_x_n] .= sg_f_n
		    sg_g[:,:,j_x_n] .= sg_g_n
		    # store new linearizations and errors
		    lin_f[:, j_x_n] .= lin_f_n
		    lin_g[:, j_x_n] .= lin_g_n
		    err_f[:, j_x_n] .= err_f_n
		    err_g[:, j_x_n] .= err_g_n

		    J[j_x_n] = true
		    j_x = j_x_n
	  end

	  return x, fx
end

"Perform two point linesearch and also calculate the next weight."
function linesearch!(
	  # modified:
	  x_n, fx_n, gx_n, sg_f_n, sg_g_n,
	  lin_f_n, lin_g_n, err_f_n, err_g_n,
	  x_t, f_x_t, g_x_t, sg_f_t, sg_g_t, lin_f_t, err_f_t, diff_t,
	  num_fun_calls,
	  # not_modified
	  j_x, zeta, gamma_f, gamma_g,
	  weight, weight_counter, var_est, var_est_p, weight_min,
	  crit_val2, d, x, fx, gx, sg_f, sg_g,
	  f_funcs, g_funcs
	  ;
	  options
    )
	  # line search like in Algorithm 3.2.2 in [^3], implemented in [^2]
	  mL = options.mL
	  mR = options.mR

	  tL = 0.0
	  t = tU = 1.0
	  t_bar = options.t_bar
	  tR = NaN

	  # set values for tL = 0
	  x_n .= x
	  fx_n .= fx
	  gx_n .= gx
	  sg_f_n .= sg_f[:,:,j_x]
	  sg_g_n .= sg_g[:,:,j_x]

	  Fx_t = 0.0
	  FxU = NaN

	  dist_n = d_norm = norm(d)
	  sigma = 1.0

	  rhsL = mL * crit_val2
	  rhsR = mR * crit_val2

	  max_fc = options.max_fun_calls
	  max_fc_ls = options.max_fun_calls_ls
	  t_min = options.t_min
	  fc_ls = 0
	  while t >= t_min && fc_ls <= max_fc_ls && num_fun_calls[] <= max_fc
		    x_t .= x .+ t .* d
		    # evaluate:
		    eval_and_jac!(f_x_t, sg_f_t, f_funcs, x_t)

		    eval_and_jac!(g_x_t, sg_g_t, g_funcs, x_t)
		    fc_ls += 1
		    num_fun_calls[] += 1

		    # Step (ii) in [^3]
		    Fx_t, lm = findmax(f_x_t .- fx)
		    if Fx_t <= t * rhsL && maximum(g_x_t) <= 0
			      # accept stepsize for x and make it next lower bound
			      tL = t

			      if t == tU FxU = Fx_t end

			      x_n .= x_t
			      fx_n .= f_x_t
			      gx_n .= gx
			      sg_f_n .= sg_f_t
			      sg_g_n .= sg_g_t
		    else
			      # make stepsize next upper bound
			      tU = t
			      fxU = Fx_t
		    end

		    sigma = t-tL
		    # Step (iii) in [^3]
		    if tL >= t_bar
			      tR = tL
			      lin_f_t .= f_x_t
			      err_f_t .= 0.0
			      dist_n = 0.0
			      break
		    end

		    # calculate lineariztion and error assuming yₖ₊₁ = x + t*d
		    dist_n = sigma * d_norm

		    diff_t .= sg_f_t * d
		    lin_f_t .= f_x_t .- sigma .* diff_t
		    err_f_t .= max.( abs.(fx_n .- lin_f_t ), dist_n^2 .* gamma_f )
		    if -err_f_t[lm] + weight * diff_t[lm] .>= rhsR
			      tR = t
			      break
		    end

		    # Step (iv) in [^3]
		    if tL <= 0
			      t = zeta * tU
			      if tU*crit_val2 > FxU
				        t = max( t,	0.5 * tU^2 * crit_val2 /(tU*crit_val2 - FxU) )
			      end
		    else
			      t = 0.5 * (tL + tU)
		    end
	  end

	  weight_n = weight
	  weight_counter_n= weight_counter
	  var_est_n = var_est

	  if !isnan(tR)
		    # line search successful, set remaining values for next iteration
		    # @info "Successful Line Search with tL=$(tL) and tR=$(tR)."
		    lin_f_n .= lin_f_t
		    err_f_n .= err_f_t
		    if tL == tR
			      lin_g_n .= gx_n
			      err_g_n .= abs.(lin_g_n)
		    else
			      lin_g_n .= g_x_t .- sigma * sg_g_t * d
			      err_g_n .= max.( abs.(lin_g_n), dist_n^2 .* gamma_g )
		    end
		    LS_success = true

		    # determine next weight:
		    u = weight
		    weight_interp = 2 * weight * (1 - Fx_t/crit_val2)
		    if tL == 0
			      var_est_n = min( var_est, var_est_p )
			      if weight_counter < - 3 && all(err_f_n .> max(var_est_n, -10*crit_val2))
				        u = weight_interp
			      end
			      weight_n = min(u, 10*weight)
			      weight_counter_n = min( weight_counter-1, -1)
			      if weight_n != weight
				        weight_counter_n = - 1
			      end
		    elseif tL == 1
			      var_est_n = min( var_est, -2*crit_val2)
			      if  weight_counter > 0
				        if Fx_t <= rhsR
					          u = weight_interp
				        end
				        if weight_counter > 3
					          u = weight/2
				        end
			      end
			      weight_n = max( u, weight/10, weight_min )
			      weight_counter_n = max( weight_counter + 1, 1 )
			      if weight_n != weight
				        weight_counter_n = 1
			      end
		    end
	  else
		    @warn "Line Search was not successful!"
		    LS_success = false
	  end

	  return LS_success, weight_n, weight_counter_n, var_est_n, tL, tR, dist_n
end

end
