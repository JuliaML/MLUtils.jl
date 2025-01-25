
"""
    make_sin(n, start, stop; noise = 0.3, f_rand = randn) -> x, y

Generates `n` noisy equally spaces samples of a sinus from `start` to `stop`
by adding `noise .* f_rand(length(x))` to the result of `sin(x)`.

Returns the vector `x` with the samples and the noisy response `y`.
"""
function make_sin(n::Int = 50, start::Real = 0, stop::Real = 2π; noise::Real = 0.3, f_rand::Function = randn)
    x = collect(range(start, stop=stop, length=n))
    y = sin.(x) .+ noise  .* f_rand.()
    return x, y
end

"""
    make_poly(coef, x; noise = 0.01, f_rand = randn) -> x, y

Generates a noisy response for a polynomial of degree `length(coef)`
and with the coefficients given by `coef`. The response is generated
by elmentwise computation of the polynome on the elements of `x`
and adding `noise .* f_randn(length(x))` to the result.

The vector `coef` contains the coefficients for the terms of the polynome.
The first element of `coef` denotes the coefficient for the term with
the highest degree, while the last element of `coef` denotes the intercept.

Return the input `x` and the noisy response `y`.
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
    return x_vec, y
end

"""
    make_spiral(n, a, theta, b; noise = 0.01, f_rand = randn) -> x, y

Generates `n` noisy responses for a spiral with two labels. Uses the radius, angle
and scaling arguments to space the points in 2D space and adding `noise .* f_randn(n)`
to the response.

Returns the 2 x n matrix `x` with the coordinates of the samples and the
vector `y` with the labels.
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
    return x, y
end


"""
    make_moons(n; noise=0.0, f_rand=randn, shuffle=true) -> x, y

Generate a dataset with two interleaving half circles. 

If `n` is an integer, the number of samples is `n` and the number of samples
for each half circle is `n ÷ 2`. If `n` is a tuple, the first element of the tuple
denotes the number of samples for the first half circle and the second element
denotes the number of samples for the second half circle.

The noise level can be controlled by the `noise` argument.

Set `shuffle=false` to keep the order of the samples.

Returns a 2 x n matrix with the the samples. 
"""
function make_moons(n_samples::Union{Int, Tuple{Int, Int}} = 100;
        noise::Number = 0.0, f_rand = randn, shuffle::Bool = true)
    
    rng = Random.default_rng()
    if n_samples isa Tuple
        @assert length(n_samples) == 2
        n_samples_1, n_samples_2 = n_samples
    else
        n_samples_1 = n_samples ÷ 2
        n_samples_2 = n_samples - n_samples_1
    end
    t_min, t_max = 0.0, π
    t_inner = rand(rng, n_samples_1) * (t_max - t_min) .+ t_min
    t_outer = rand(rng, n_samples_2) * (t_max - t_min) .+ t_min
    outer_circ_x = cos.(t_outer)
    outer_circ_y = sin.(t_outer)
    inner_circ_x = 1 .- cos.(t_inner)
    inner_circ_y = 0.5 .- sin.(t_inner)

    x = [outer_circ_x outer_circ_y; inner_circ_x inner_circ_y]
    x = permutedims(x, (2, 1))
    x .+= noise .* f_rand(rng, size(x))
    y = [fill(1, n_samples_1); fill(2, n_samples_2)] 
    if shuffle
        x, y = getobs(shuffleobs((x, y)))
    end
    return x, y
end
