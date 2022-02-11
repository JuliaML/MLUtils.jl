DATAPATH = joinpath(@__DIR__, "data")

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
