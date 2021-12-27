X = rand(3, 4)
x = randobs(X)
@test size(x) == (3,)
@test any(X[:,i] == x for i in 1:4) 

x = randobs(X, 2) 
@test size(x) == (3, 2)
@test any(X[:,i] == x[:,1] for i in 1:4) 
@test any(X[:,i] == x[:,2] for i in 1:4)
