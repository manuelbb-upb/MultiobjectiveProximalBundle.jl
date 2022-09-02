# MultiobjectiveProximalBundle

This an implementation of the Multiobjective Proximal Bundle Method presented in [^1].
Many important details can be found in [^2].
I used this for personal testing purposes only and cannot promise to keep this repository up to date.

Features:
* Subgradient aggregation to save memory.
* Support for multiple non-smooth objective and constraint functions.
* Pure Julia implementation, no wrapper of Fortran code.

## Installation
The package is not yet registered.
You have to do
```julia
using Pkg
Pkg.add(;url="https://github.com/manuelbb-upb/MultiobjectiveProximalBundle.jl.git")
```

## Usage
Importing the module via
```julia
using MultiobjectiveProximalBundle
```
provides `MPBOptions` (see docstring) and `mpb_optimize`.
The latter has the following siganture:
```
x, fx = mpb_optimize(
    x0, objective_funcs, constraint_funcs,
    gammas_objectives=nothing, gammas_constraints=nothing;
    options = MPBOptions()
)
```
Here, `x0` is the initial **feasible** start vector and
`objective_funcs` is a vector of functions.
Each functions must return a tuple of its scalar primal value
and its gradient. Likewise for the constraints.
`gammas_objectives` are constants to compensate non-convexity.
The default value of `nothing` results in `0.5` being used.

### Example
```julia
using MultiobjectiveProximalBundle
using LinearAlgebra: norm

function f1(x)
    __x = norm(x)
    _x = __x + 2
    y = sqrt(_x)
    dy = if __x == 0
        zeros(length(x))
    else
        1/(2 * __x * y) .* x
    end
    return y, dy
end

function f2(x)
    _f21 = -sum(x)
    _f22 = _f21 + sum(x.^2) - 1
    _df21 = -ones(length(x))
    y, dy = if _f22 >= _f21
        _df22 = _df21 .+ 2 .* x
        if _f22==_f21
            _f22, 0.5 .* (_df21 .+ _df22)
        else
            _f22, _df22
        end
    else
        _f21, _df21
    end
    return y, dy
end

function g(x)
    _g1 = sum(x.^2) - 10.0
    _g2 = 3 * x[1] + sum(x[2:end]) + 1.5

    _dg1 = 2 .* x
    y, dy = if _g2 >= _g1
        _dg2 = ones(length(x))
        _dg2[1] = 3
        if _g2 == _g1
            _g2, 0.5 .* (_dg1 .+ _dg2)
        else
            _g2, _dg2
        end
    else
        _g1, _dg1
    end
    return y, dy
end

x0 = fill(-0.5, 2)

x, fx = mpb_optimize(x0, [f1, f2], [g])
```

## Notes

ToDo's:
* Support linear constraints.
* Remove unnecessary arrays.
* “Pre-allocate” model of sub-problem via parametrization.
* Enable choice of sub-problem solver.
* More verbosity, informative return codes.


[^1]: M. M. Mäkelä, N. Karmitsa, and O. Wilppu, “Proximal Bundle Method for Nonsmooth and Nonconvex Multiobjective Optimization,” in Mathematical Modeling and Optimization of Complex Structures, vol. 40, P. Neittaanmäki, S. Repin, and T. Tuovinen, Eds. Cham: Springer International Publishing, 2016, pp. 191–204. doi: 10.1007/978-3-319-23564-6_12.

[^2]: M. M. Mäkelä, “Nonsmooth Optimization”
