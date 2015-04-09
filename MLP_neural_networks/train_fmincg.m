function [ out ] = train_fmincg(imdb, opts)
                             
X = imdb.data;
input_layer_size  = size(X, 2);  % input layer is a single row vector
m = size(X, 1);
X_T = imdb.test;
y = imdb.labels;
y_t = imdb.test_labels;
lambda = opts.lambda;
hidden_layer_size = opts.hidden_layer_size;
iterations = opts.iterations;
num_labels = opts.num_labels;

% =============  Visualization ==================
% first use displayData to look at 100 examples of images
rand_indices = randperm(m);
displayData(X(rand_indices(1:100), :));
fprintf('Visualizing subset of data.\n');

% =============  Check gradients ==================
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
%  Check gradients by running checkNNGradients
checkNNGradients(lambda, @nnCostFunction);

% =============  Training ==================
fprintf('\nTraining Neural Network... \n')
options = optimset('MaxIter', iterations);
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);
% using a more advanced optimization algorithm than gradient descent 
[nn_params, ~] = fmincg(costFunction, initial_nn_params, options);

% =============  Testing ==================
% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
             
% visualize what was learned in the hidden layer
fprintf('\nVisualizing hidden layer of the Neural Network... \n')
displayData(Theta1(:, 2:end));  % removes bias term

% Get training and test set accuracy
pred = predict(Theta1, Theta2, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
pred = predict(Theta1, Theta2, X_T);
fprintf('\nTest Set Accuracy: %f\n', mean(double(pred == y_t)) * 100);

out.Theta1 = Theta1;
out.Theta2 = Theta2;

end
% =========================================================================


% =========================================================================
function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given 
%   the trained weights of a neural network (Theta1, Theta2)

m = size(X, 1);
X = [ones(m, 1) X];  % add the bias
Layer2 = sigmoid(Theta1*X'); 
Layer2 = [ ones(1, size(Layer2,2)); Layer2 ];
Layer3 = sigmoid(Theta2*Layer2)'; 
%  p is set to a vector containing labels between 1 to size(Theta2, 1).
[~, p] = max(Layer3, [], 2); 

end
% =========================================================================

% =========================================================================
function W = randInitializeWeights(L_in, L_out)
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
%incoming connections and L_out outgoing connections
%   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
%   of a layer with L_in incoming connections and L_out outgoing 
%   connections so that we break the symmetry while training a NN.
%
%   Note that W sets a matrix of size(L_out, 1 + L_in) as
%   the first column of W handles the "bias" terms

epsilon_init = sqrt(6) / sqrt(L_in + L_out);
W = rand(L_out, 1+L_in) * 2*epsilon_init - epsilon_init;

end
% =========================================================================

function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
m = size(X, 1);
       
%FEEDFORWARD NN
% Add ones to the X data matrix
X = [ones(m, 1) X];
% Make the y vector into [1 0 0 ... 0] for each training example, where 1 is in the
% index of what class it is (10 is Class 0) 
yVector = zeros(m, num_labels);
yVector(sub2ind(size(yVector), 1:length(y), y')) = 1;
%Feed forward every training example
Layer2 = sigmoid(Theta1*X'); 
Layer2 = [ ones(1, size(Layer2,2)); Layer2 ];
Layer3 = sigmoid(Theta2*Layer2);  
%Each sample x is now a column
%Sum over each value of k (column-wise) then sum over every training example (row) 
J = (1/m)*sum(sum((-yVector.*log(Layer3'))-((1-yVector).*log(1-Layer3')), 2 )); 
%Get rid of the bias terms when regularizing and unroll into single vector
RegTheta1 = Theta1; 
RegTheta1(1:hidden_layer_size) = 0;
RegTheta2 = Theta2; 
RegTheta2(1:num_labels) = 0;
RegTheta = [RegTheta1(:) ; RegTheta2(:)];
%The final cost adds the square of every parameter except the bias
J = J + (lambda/(2*m))*(RegTheta'*RegTheta);

%BACKPROPAGATION
%First, get the output layer error for each training example
Delta3 = Layer3 - yVector';    
Delta2 = ( (Theta2' * Delta3) .* Layer2 .* (1 - Layer2) );  
Delta2 = Delta2(2:end,:);  %get rid of bias node
%Need three dimension matrix to get rid of for loop
for i=1:m,
	Theta2_grad = Theta2_grad + Delta3(:,i)*Layer2(:,i)';
	Theta1_grad = Theta1_grad + Delta2(:,i)*X(i,:); %X doesn't need to be transposed
end;

%REGULARIZATION
Theta2_grad = (1/m)*(Theta2_grad + lambda*[zeros(size(Theta2,1),1) Theta2(:,2:end)]);
Theta1_grad = (1/m)*(Theta1_grad + lambda*[zeros(size(Theta1,1),1) Theta1(:,2:end)]);
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
% =========================================================================

% =========================================================================
function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

g = 1.0 ./ (1.0 + exp(-z));

end
% =========================================================================

% =========================================================================
function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This works regardless if z is a matrix, vector or scalar.

g = sigmoid(z).*(1-sigmoid(z));

end
% =========================================================================