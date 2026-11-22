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
    rows = [split(l, ',') for l in eachline(joinpath(DATAPATH, "iris.csv")) if !isempty(l)]
    X = [parse(Float64, r[i]) for i in 1:4, r in rows]
    y = [String(r[5]) for r in rows]
    vars = ["Sepal length", "Sepal width", "Petal length", "Petal width"]
    return X, y, vars
end
