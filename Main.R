source("OO/ANNet.R");
source("Matrix/MNet.R");


testinputs = matrix(c(2.7810836, 2.550537003, 
				1.465489372, 2.362125076, 
				3.396561688, 4.400293529, 
				1.38807019, 1.850220317, 
				3.06407232, 3.005305973, 
				7.627531214, 2.759262235, 
				5.332441248, 2.088626775, 
				6.922596716, 1.77106367, 
				8.675418651, -0.242068655, 
				7.673756466, 3.508563011), ncol = 2, nrow = 10, byrow = TRUE);
testoutputs = matrix(c(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0), ncol = 2, nrow = 10, byrow = TRUE);

xorin = matrix(c(0, 0, 1, 1, 1, 0, 0, 1), ncol = 2, nrow = 4, byrow = TRUE);
xorout = matrix(c(0, 0, 1, 1), ncol = 1, nrow = 4, byrow = TRUE);

nodevec = c(2,3,2);
biasvec = c(-1, 0);
weightedvectorsum = function(w, i){
	return(sum(w*i));
};

sigmoidFn = function(a){
	return(1 / (1 + exp(-a)));
};

sigmoidDeriv = function(x){
	return(x*(1-x));
};

testNet = ANNet$new();
testNet$createCompleteNetwork(nodevec, biasvec, weightedvectorsum, sigmoidFn, sigmoidDeriv, 0.1);

#testNet$train(xorin, xorout, 20000);
#testNet$test(xorin, xorout);



testMNet = MNet$new();


testMNet$create(nodevec, biasvec, sigmoidFn, sigmoidDeriv);

testMNet$train(testinputs, testoutputs, 200, 0.5);
testMNet$test(testinputs, testoutputs);

for(layer in testMNet$Layers){
	print(layer);
}