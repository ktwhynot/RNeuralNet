#A network that uses matrices rather than an OO approach
#Layers: A list of matrices, representing the hidden and output layers of the network
#Biases: A vector of bias values for each layer in Layers. The value will be added to the end of the input for the layer
#eg. Input of 1 2 3 and bias of -1 leads to the layer getting [1 2 3 -1] as an input vector
#this also leads to each layer needing n+1 weights per node, where n = the amount of nodes in the previous layer
#AFUN: activation function
#AFUNPRIME: derivative of the activation function to be used for backpropagation
MNet = setRefClass("MNet", fields = c("Layers", "Biases", "AFUN", "AFUNPRIME"));

#Initialization function
MNet$methods(initialize = function(...){
	#Do the usual assignment of fields with the given values
	callSuper(...);
});

#Create a network with the given parameters.
#LAYERVEC: a vector outlining the number of inputs and nodes in each layer
#eg. [2 5 1] is a network with 2 inputs, 1 hidden layer of 5 nodes, and 1 output layer of 1 node
#BIASVEC: a vector outlining the value of the bias for each layer. Should be of length 1 less than LAYERVEC
#ACTIVATIONFUN: The activation function to be used
#BACKPROPFUN: Derivative of the activation function
MNet$methods(create = function(LAYERVEC, BIASVEC, ACTIVATIONFUN, BACKPROPFUN){
	.self$Layers = vector("list", length(LAYERVEC)-1);
	.self$Biases = BIASVEC;
	.self$AFUN = ACTIVATIONFUN;
	.self$AFUNPRIME = BACKPROPFUN;

	for(i in 2:length(LAYERVEC)){
		#Create a 0 matrix for the layer
		#The first 2 columns are for Value and Error, respectively
		#The 3rd to second-last are input weights, and the last the bias weight
		tempmat = matrix(0, nrow=LAYERVEC[[i]], ncol=LAYERVEC[[i-1]]+3);
		weightnames = rep("Weight", LAYERVEC[[i-1]]+1);
		colnames(tempmat) = c("Value", "Error", weightnames);

		#Randomize the weights
		weightamount = LAYERVEC[[i-1]] + 1;
		random_weight_matrix = matrix(runif(n = LAYERVEC[[i]]*weightamount, min = -sqrt(2/weightamount), max = sqrt(2/weightamount)), nrow = LAYERVEC[[i]], ncol = weightamount);
		
		tempmat[,3:ncol(tempmat)] = random_weight_matrix;

		.self$Layers[[i-1]] = tempmat;
	}
});


#Compute the output for a given vector of inputs via forward propagation
#inputs: a vector of inputs for the network. Should have length equal to one less than the number of weights in the first layer
MNet$methods(compute = function(inputs){
	newin = inputs;
	for(i in 1:length(Layers)){
		newin = c(newin, Biases[[i]]);
		weights = t(Layers[[i]][,3:ncol(Layers[[i]])]);
		
		weighted_inputs = newin*weights;
		
		themargin = 2;

		#Special case for layers with 1 node
		if(nrow(Layers[[i]]) == 1){
			themargin = 1;
		}

		new_values = apply(weighted_inputs, MARGIN=themargin, FUN=function(x){
			return(AFUN(sum(x)));
		});
		
		.self$Layers[[i]][,"Value"] = new_values;
		newin = new_values;
	}
});

#Wrapper for compute that returns the output
MNet$methods(computeValue = function(inputs){
	compute(inputs);
	return(Layers[[length(Layers)]][,"Value"]);
});

#Compute the errors of a network for a given series of outputs via backpropagation
#outputs: a vector of expected outputs. Should have length equal to the number of nodes in the last layer
MNet$methods(backpropagate = function(outputs){
	#Calculate and set the error for the output layer
	calculated_values = Layers[[length(Layers)]][,"Value"];
	output_error = outputs - calculated_values;
	.self$Layers[[length(Layers)]][,"Error"] = output_error;

	#Calculate and set errors for hidden layers
	if(length(Layers) > 1){
		#No need to go any further for perceptrons
		for(i in length(Layers):2){
			weighted_errors = Layers[[i]][,"Error"]*Layers[[i]][,3:(ncol(Layers[[i]])-1)];
			prev_layer_errors = apply(t(weighted_errors), MARGIN=1, sum);
			.self$Layers[[i-1]][,"Error"] = prev_layer_errors;
		}
	}
});

#Update the weights of a network. Assumes that the error has already been calculated for each layer.
#lrate: learning rate to be used in weight update
#inputs: the inputs for the current training example
MNet$methods(updateWeights = function(lrate, inputs){
	theinputs = inputs;
	for(i in 1:length(Layers)){
		layer = Layers[[i]];
		theinputs = c(theinputs, Biases[[i]]);

		derivatives = sapply(layer[,"Value"], FUN=function(x){
			return(AFUNPRIME(x));
		});
		
		#vector of values to be multiplied against the layer's inputs
		tempvals = lrate*layer[,"Error"]*derivatives;

		weight_update_matrix = tcrossprod(tempvals, theinputs);

		#update weights
		new_weights = layer[,3:ncol(layer)] + weight_update_matrix;
		.self$Layers[[i]][,3:ncol(layer)] = new_weights;
		
		theinputs = layer[,"Value"];
	}
});

#Reset the error of all layers to 0
MNet$methods(resetError = function(){
	for(i in 1:length(Layers)){
		.self$Layers[[i]][,"Error"] = 0;
	}
});

#Train the network over a certain number of epochs with the given inputs and outputs
#inputs and outputs should both have the same depth (number of rows), and should both be 2-dimensional matrices
#Example:
# Inputs  Outputs
# 1 6 3  |  2 1
# 8 4 0  |  3 9
# 1 1 1  |  2 2
#Inputs would be matrix(c(1, 6, 3, 8, 4, 0, 1, 1, 1), ncol = 3, nrow = 3, byrow = TRUE) 
#and outputs matrix(c(2, 1, 3, 9, 2, 2), ncol = 2, nrow = 3, byrow = TRUE)
#epochs: the amount of times to train on the given data
#rate: the learning rate for the network
MNet$methods(train = function(inputs, outputs, epochs, rate){
	for(e in 1:epochs){
		for(row in 1:nrow(inputs)){
			#Forward
			compute(inputs[row,]);
			#Backward
			backpropagate(outputs[row,]);
			#Weight update
			updateWeights(rate, inputs[row,]);
			#Reset Error
			resetError();
		}
	}
});

#Tests the outputs of the network against the outputs given for a given series of inputs
MNet$methods(test = function(inputs, outputs){
	for(row in 1:nrow(inputs)){
		print("expected: ");
		print(outputs[row,]);
		print("got: ");
		print(computeValue(inputs[row,]));
	}
});
