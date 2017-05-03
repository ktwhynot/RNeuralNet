source("Nodes.R");

#Container to hold the nodes in a network
#Essentially a 2-dimensional array/matrix, but each layer/row's length may be different
#Layers: The list containing the layers in the network, each of which is a list of nodes
#InsertionPoint: Number used to denote where a new layer should be inserted in the network
ANNet = setRefClass("ANNet", fields = c("Layers", "InsertionPoint", "Bias"));

#Add a list of nodes as a layer to the network
#If the current Layers list is full, create a new list with double the capacity (double leads to an amortized time of O(n))
ANNet$methods(addLayer = function(layer){
	if(InsertionPoint == length(Layers)){
		newLayers = vector("list", length(Layers)*2);
		counter = 1;
		for(l in Layers){
			newLayers[[counter]] = l;
			counter = counter + 1;
		}
		.self$Layers = newLayers;
	}	

	.self$Layers[[InsertionPoint]] = layer;
	.self$InsertionPoint = InsertionPoint + 1;
});

#a as in access
#Get the node at a specified location in the network
#Works like a 2-dimensional array in other languages, taking a length and depth which start at 0
#Error if out of bounds
#Layer 0 is always the input layer
#l - layer, p - position in layer
ANNet$methods(a = function(l, p){
	return(Layers[[l + 1]][[p + 1]]);
});

#Compute the output of the network for a given series of inputs via forward propagation
#inputs should be the same length as the first layer
ANNet$methods(compute = function(inputs){
	if(length(inputs) != length(Layers[[1]])){
		stop("Given inputs do not fit input layer");
	} else{
		for(i in 1:(InsertionPoint-1)){
			if(is(Layers[[i]][[1]], "iNode")){
				counter = 1;
				for(inode in Layers[[i]]){
					inode$Value = inputs[[counter]];
					counter = counter + 1;
				}
			} else{
				for(node in Layers[[i]]){
					node$computeValue();
				}
			}
		}
	}
});

#Wrapper for compute that returns the output vector
ANNet$methods(computeValue = function(inputs){
	compute(inputs);
	temp = c()
	for(node in Layers[[InsertionPoint - 1]]){
		temp = c(temp, node$Value);
	}
	return(temp);
});

#Compute the errors of a network for a given series of outputs via backpropagation
#outputs should be the same length as the last layer
ANNet$methods(backpropagate = function(outputs){
	if(length(outputs) != length(Layers[[InsertionPoint - 1]])){
		stop("Given outputs do not match output layer");
	} else{
		counter = 1;
		for(node in Layers[[InsertionPoint - 1]]){
			node$computeError(outputs[[counter]]);
			counter = counter + 1;
		}
		secondLastLayer = InsertionPoint - 2;
		if(secondLastLayer > 1){
			for(i in secondLastLayer:2){
				for(node in Layers[[i]]){
					node$computeError();
				}
			}
		}
	}
});

#Update the weights of a network. Assumes that errors have already been calculated
ANNet$methods(updateWeights = function(){
	for(i in (InsertionPoint - 1):2){
		for(node in Layers[[i]]){
			node$updateWeights();
		}
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
ANNet$methods(train = function(inputs, outputs, epochs){
	if(nrow(inputs) != nrow(outputs)){
		stop("Mismatched inputs and outputs");
	}
	for(e in 1:epochs){
		for(row in 1:nrow(inputs)){
			#Forward pass
			compute(inputs[row,]);
			#Backward pass and weight update
			backpropagate(outputs[row,]);
			#Weight Update
			updateWeights();
		}
	}
});

#Tests the outputs of the network against the outputs given for a given series of inputs
ANNet$methods(test = function(inputs, outputs){
	for(row in 1:nrow(inputs)){
		print("expected: ");
		print(outputs[row,]);
		print("got: ");
		print(computeValue(inputs[row,]));
	}
});

#Create a "complete" network. "Complete" in the sense that each node shares the same functions and learning rate, and is connected to all nodes in the previous layer
#nodes should be a vector outlining the amount of nodes in each layer, including inputs. The length of the vector is the amount of layers. Expects a length of at least 2.
#Eg. [8 5 2] would mean a network with 8 inputs, 1 layer of 5 hidden nodes, and 2 outputs. [3 1] would mean 3 inputs, and 1 output (a perceptron)
#biases: a vector outlining the value of each bias for each layer; a value of 0 means no bias for that layer. Length should be one less that that of nodes (since the input layer doesn't have biases)..
#Eg. [-1 0] would mean a bias of -1 for the hidden layer and no bias for the output layer.
#inputFUN: the input function each node should use
#activationFUN: the activation function each node should use
#backFUN: the backpropagation function each node should use (probably the derivative of the activation function)
#lrate: the learning rate 
ANNet$methods(createCompleteNetwork = function(nodes, biases, inputFUN, activationFUN, backFUN, lrate){
	if(length(nodes) < 2){
		stop("Network needs at least two layers");
	}
	#Initialize Bias container
	.self$Bias = vector("list", length(biases));
	for(i in 1:length(nodes)){
		templayer = vector("list", nodes[[i]]);
		if(i > 1){
			#initialize the bias for this layer as an empty vector so that it can be added to parents regardless of if a bias exists for the layer
			this_layers_bias = c();
			if(biases[[i-1]] != 0){
				this_layers_bias = iNode$new(Value = biases[[i-1]]);
				.self$Bias[[i-1]] = this_layers_bias;
			}
		}
		for(j in 1:nodes[[i]]){
			if(i == 1){
				#first layer is comprised of inodes
				templayer[[j]] = iNode$new();
			} else{
				templayer[[j]] = Node$new(Parents = c(unlist(Layers[[i-1]]), this_layers_bias), inputFn = inputFUN, activationFn = activationFUN, backFn = backFUN, LearningRate = lrate);
			}
		}
		addLayer(templayer);
	}
});

#Initialization function. Start layers off as a length-2 list, it will grow as more layers are added
ANNet$methods(initialize = function(...){
	#Do the usual assignment of fields with the given values
	callSuper(...);

	#Initialize Layers, which should be a list of lists of Nodes
	.self$Layers = vector("list", 2);

	#Insertion point should be 1 when initialized
	.self$InsertionPoint = 1;
});
