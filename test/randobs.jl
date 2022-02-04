x = randobs(X[:,1:4])
@test size(x) == (4,)
@test any(X[:,i] == x for i in 1:4) 

x = randobs(X[:,1:4], 2) 
@test size(x) == (4, 2)
@test any(X[:,i] == x[:,1] for i in 1:4) 
@test any(X[:,i] == x[:,2] for i in 1:4)
