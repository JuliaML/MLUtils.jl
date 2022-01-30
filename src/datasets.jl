DATAPATH = joinpath(@__DIR__, "..", "data")

"""
    load_iris() -> X, y, names

Loads the first 150 observations from the
Iris flower data set introduced by Ronald Fisher (1936).
The 4 by 150 matrix `X` contains the numeric measurements,
in which each individual column denotes an observation.
The vector `y` contains the class labels as strings.
The vector `names` contains the names of the features (i.e. rows of `X`)

[1] Fisher, Ronald A. "The use of multiple measurements in taxonomic problems." Annals of eugenics 7.2 (1936): 179-188.
"""
function load_iris()
    path = joinpath(DATAPATH, "iris.csv")
    raw_csv = readdlm(path, ',')
    X = convert(Matrix{Float64}, raw_csv[:, 1:4]')
    y = convert(Vector{String}, raw_csv[:, 5])
    vars = ["Sepal length", "Sepal width", "Petal length", "Petal width"]
    X, y, vars
end

"""
    load_sin() -> x, y, names

Loads an artificial example dataset for a noisy sin.
It is particularly useful to explain under- and overfitting.
The vector `x` contains equally spaced points between 0 and 2π
The vector `y` contains `sin(x)` plus some gaussian noise
The vector `names` contains descriptive names for `x` and `y`
"""
function load_sin()
    path = joinpath(DATAPATH, "sin.csv")
    raw_csv = readdlm(path, ',')
    x = convert(Vector{Float64}, raw_csv[:, 1])
    y = convert(Vector{Float64}, raw_csv[:, 2])
    x, y, ["X", "sin(X) + ɛ"]
end

"""
    load_line() -> x, y, names

Loads an artificial example dataset for a noisy line.
It is particularly useful to explain under- and overfitting.
The vector `x` contains 11 equally spaced points between 0 and 1
The vector `y` contains `x ./ 2 + 1` plus some gaussian noise
The vector `names` contains descriptive names for `x` and `y`
"""
function load_line()
    path = joinpath(DATAPATH, "line.csv")
    raw_csv = readdlm(path, ',')
    x = convert(Vector{Float64}, raw_csv[:, 1])
    y = convert(Vector{Float64}, raw_csv[:, 2])
    x, y, ["x", "0.5 x + 1 + ɛ"]
end

"""
    load_poly() -> x, y, names

Loads an artificial example dataset for a noisy quadratic function.
It is particularly useful to explain under- and overfitting.
The vector `x` contains 50 points between 0 and 4
The vector `y` contains `2.6 * x^2 + .8 * x` plus some gaussian noise
The vector `names` contains descriptive names for `x` and `y`
"""
function load_poly()
    path = joinpath(DATAPATH, "poly.csv")
    raw_csv = readdlm(path, ',')
    x = convert(Vector{Float64}, raw_csv[:, 1])
    y = convert(Vector{Float64}, raw_csv[:, 2])
    x, y, ["x", "2.6 x² + .8 x + ɛ"]
end


"""
    load_spiral() -> x, y, names

Loads an artificial example dataset for a noisy spiral function.
It is particularly useful to explain representation learning and nonlinearity.
The matrix `x` contains 194 points between 0 and 6.5 lying on the spiral.
The vector `y` contains the corresponding labels, i.e "ones" or "zeros".
The vector `names` contains descriptive names for `x` and `y`.
"""
function load_spiral()
    path = joinpath(DATAPATH, "spiral.csv")
    raw_csv = readdlm(path, ',')
    X = convert(Matrix{Float64}, raw_csv[:, 1:2])
    y = convert(Vector{Int}, raw_csv[:, 3])
    X, y, ["r*sin(θ)", "r*cos(θ)", "class"]
end