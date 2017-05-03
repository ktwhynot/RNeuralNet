#Simplified node for inputs and biases
#Since input nodes only need a value, cutting out the methods and fields of a normal node should save space
#Also allows for more intuitive checking of node type for certain methods
iNode = setRefClass("iNode", fields = c("Value"));

#Initialization function. Will be automatically called upon object creation.
#Really, we just want to make sure any references to value return a number
iNode$methods(initialize = function(...){
	#Do the usual assignment of fields with the given values
	callSuper(...);
	
	#Initialize value if an initital value has not been given
	if(is(Value, "uninitializedField")){
		.self$Value = 0;
	}
});

#Value: The node's numerical value to be passed to children or output
#Error: The node's error to be used in weight update
#LearningRate: The node's learning rate
#Parents: A vector of the node's parents from which it gets values. Has the same length as weights. Input nodes have no parents.
#Weights: A vector of the node's weights associated with each parent. Has the same length as parents.
#Inputs: A vector of the parents' values to be used in the input function.
#inputFn: Input function. Takes weights and input vector and outputs a numerical value to be used in the activation function.
#activationFn: Activation function. Takes a numerical value and compute's the node's value.
#backFn: Backpropagation function. Should be the derivative of the activation function w.r.t. the node's value. Takes a numerical value.
Node = setRefClass("Node", fields = c("Value", "Error", "LearningRate", "Parents", "Weights", "Inputs", "inputFn", "activationFn", "backFn"));

#Randomize the weights of the node. Meant for initialization of the node.
Node$methods(randomizeWeights = function(){
	vector_size = length(Parents);
	.self$Weights = runif(n = vector_size, min = -sqrt(2/vector_size), max = sqrt(2/vector_size));
});

#Retrieve the values of parents and set the node's inputs field accordingly
Node$methods(getInputVector = function(){
	.self$Inputs = sapply(Parents, FUN = function(n){return(n$Value)});
});

#Compute the value of the node via the weights, inputs, input function and activation function
#Set the node's value accordingly
Node$methods(computeValue = function(){
	getInputVector();
	activation_in = inputFn(Weights, Inputs);
	.self$Value = activationFn(activation_in);
});

#Add the given error to the current error
Node$methods(addError = function(e){
	.self$Error = Error + e;
});

#Reset the error to 0
Node$methods(resetError = function(){
	.self$Error = 0;
});

#Calculate the error to be used in updating the node's weights
#Can optionally take the expected value to calculate the error (mainly for output nodes)
Node$methods(computeError = function(expected){
	if(!missing(expected)){
		.self$Error = (expected - Value);
	}
	counter = 1;
	for(p in Parents){
		#Check to ensure parent is not an input node
		if(!is(p, "iNode")){
			p$addError(Error*Weights[[counter]]);
		}
		counter = counter + 1;
	}
});

#Update the weights for a node. Expects the error to have been calculated beforehand.
#Also resets the error to prepare for future training
Node$methods(updateWeights = function(){
	scalarVal = LearningRate * Error * backFn(Value);
	weightChange = scalarVal*Inputs;
	.self$Weights = Weights + weightChange;
	resetError();
});

#Initialization function. Will be automatically called upon object creation.
Node$methods(initialize = function(...){
	#Do the usual assignment of fields with the given values
	callSuper(...);
	
	#Initialize weights for nodes without specified weights
	if(!is(Parents, "uninitializedField") & is(Weights, "uninitializedField")){
		randomizeWeights();
	}

	#Initialize numeric variables
	if(is(Value, "uninitializedField")){
		.self$Value = 0;
	}	
	if(is(Error, "uninitializedField")){
		.self$Error = 0;
	}
	if(is(LearningRate, "uninitializedField")){
		.self$LearningRate = 0;
	}
});
