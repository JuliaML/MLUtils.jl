
"""
    make_sin(n, start, stop; noise = 0.3, f_rand = randn) -> x, y

Generates `n` noisy equally spaces samples of a sinus from `start` to `stop`
by adding `noise .* f_rand(length(x))` to the result of `fun(x)`.
"""
function make_sin(n::Int = 50, start::Real = 0, stop::Real = 2π; noise::Real = 0.3, f_rand::Function = randn)
    x = collect(range(start, stop=stop, length=n))
    y = sin.(x) .+ noise  .* f_rand.()
    return x, y
end

"""
    make_poly(coef, x; noise = 0.01, f_rand = randn) -> x, y

Generates a noisy response for a polynomial of degree `length(coef)`
using the vector `x` as input and adding `noise .* f_randn(length(x))` to the result.
The vector `coef` contains the coefficients for the terms of the polynome.
The first element of `coef` denotes the coefficient for the term with
the highest degree, while the last element of `coef` denotes the intercept.
"""
function make_poly(coef::AbstractVector{R}, x::AbstractVector{T}; 
                noise::Real = 0.1, f_rand::Function = randn) where {T<:Real,R<:Real}
    n = length(x)
    m = length(coef)
    x_vec = collect(x)
    y = zeros(n)
    @inbounds for i = 1:n
        for k = 1:m
            y[i] += coef[k] * x_vec[i]^(m-k)
        end
    end
    y .+= noise .* f_rand.()
    x_vec, y
end


"""
    make_spiral(n, a, theta, b; noise = 0.01, f_rand = randn) -> x, y

Generates `n` noisy responses for a spiral with two labels. Uses the radius, angle
and scaling arguments to space the points in 2D space and adding `noise .* f_randn(n)`
to the response.
"""
function make_spiral(n::Int = 97, a::Real = 6.5, theta::Real = 16.0, b::Real=104.0; 
                noise::Real = 0.1, f_rand::Function = randn)
    x = zeros(Float64, (2, 2*n))
    y = zeros(Int, 2*n)
    index = 0:1.0:(n-1)
    for i = 1:n
        _angle = index[i]*pi/theta
    	_radius = a * (b-index[i]) / b
    	x_coord = _radius * sin(_angle)
    	y_coord = _radius * cos(_angle)
    	x[1, i] = x_coord
    	x[2, i] = y_coord
        x[1, n+i] = -(x_coord)
    	x[2, n+i] = -(y_coord)
        y[i] = 1
        y[n+i] = 0
    end
    x[1, :] .+= noise .* f_rand.()
    x[2, :] .+= noise .* f_rand.()
    x, y
end

"""
    make_regression(n, n_features; noise = 0.01, ground_truth = false) -> X, y, ground_truth

Generates `n` noisy samples of `n_features` dimensions for regression tasks.
ground_truth is a boolean that indicates whether the ground truth is returned. If `true`, the ground truth is returned as a Dict.
"""
function make_regression(n::Int = 100, n_features::Int = 1; noise::Float64 = 0.01, ground_truth::Bool = false)
    X = randn(n, n_features)
    coef = rand(1:100)*rand(Float64, n_features)
    intercept = rand(1:100)*rand(Float64)
    ϵ = noise * randn(n)
    y = X*coef .+ intercept + noise*randn(n)

    if ground_truth == false
        return X, y
    else
        return X, y, (; coef,  intercept)
    end
end