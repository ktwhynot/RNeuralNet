A = matrix(c(3,0.5,1,2,3,9,0.6,4,5,6,8,0.7,7,8,9,7,0.3,1,2,3), nrow=4, ncol=5, byrow=TRUE);
B = c(5,15,7);
dimnames(A) = list(c("row1","row2","row3","row4"), c("Value","Error","Weight","Weight","Weight"))
A;

#Forward
A[,3:ncol(A)];
W = t(A[,3:ncol(A)]);
WI = B*W;
WI;
V = apply(WI, MARGIN=2, FUN=function(x){
	#return(activationFN(sum(x)))
	return(sum(x)/2);
});
V;
A[,"Value"] = V;
A;

#Backward
WE = A[,"Error"]*A[,3:ncol(A)];
#Calculate error for previous layer
E = apply(t(WE), MARGIN=1, sum);
E;

D = sapply(A[,"Value"], FUN=function(x){
	#return(backFn(x))
	return(x / 100);
});
D;
LR = 0.1;

#Scalars for change in weights in each node = learningrate * error * derivativeofactivation
SDELTAW = LR*E*D;
SDELTAW;

#Weight update matrix is cross product of scalar and inputs
DELTAW = tcrossprod(SDELTAW, B);

#Update the weights
newW = A[,3:ncol(A)] + DELTAW;
A[,3:ncol(A)] = newW;
A;

#Reset the error
A[,"Error"] = 0;