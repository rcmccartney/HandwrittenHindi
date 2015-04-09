function [model] = train_backprop_crossEntropy(data, opts)
% This function trains a neural network language model.
% Inputs:
%   data: Contains data.trainData, data.validData, 
%		  and data.testData.  Each should be formatted as 
%         a single example per row, with each column a different 
%         feature of that instance.  Then target output is in last column
%   out_size: the number of possible output classes
%   epochs: Number of epochs to run.
%   batch_size: default = 100 
%   learning_rate: default = 0.1.
%   momentum:  default = 0.9
%   numhid: Number of units in hidden layer; default = 200.
%   init_wt: Standard deviation of the normal distribution
%            which is sampled to get the initial weights; default = 0.01
% Output:
%   model: A struct containing the learned weights and biases and vocabulary.

% save output 

if size(ver('Octave'),1)
  OctaveMode = 1;
  warning('error', 'Octave:broadcast');
  start_time = time;
else
  OctaveMode = 0;
  start_time = clock;
end

out_size = opts.num_labels;
epochs = opts.iterations;
batchsize = opts.batchsize;
learning_rate = opts.learning_rate;
momentum = opts.momentum;
numhid = opts.hidden_layer_size;
init_wt = opts.init_wt;

% SPLIT DATA INTO MINIBATCHES
train = data.trainData';
valid = data.validData';
test = data.testData';
numdims = size(train, 1);
D = numdims - 1;
numbatches = floor(size(train, 2) / batchsize);
train_input = reshape(train(1:D, 1:batchsize * numbatches), D, batchsize, numbatches);
train_target = reshape(train(D + 1, 1:batchsize * numbatches), 1, batchsize, numbatches);
valid_input = valid(1:D, :);
valid_target = valid(D + 1, :);
test_input = test(1:D, :);
test_target = test(D + 1, :);
input_size = D;  

% =============  Visualization ==================
% first use displayData to look at one minibatch
t = train_input(:,:,1)';
rand_indices = randperm(size(t, 1));
fprintf('Visualizing subset of data.\n');
displayData(t(rand_indices(1:100), :));
savefig( strcat('output\', opts.name, '_subset.fig') ); 

% VARIABLES FOR TRACKING TRAINING PROGRESS.
show_training_CE_after = 1000;
show_validation_CE_after = 1000;

% INITIALIZE WEIGHTS AND BIASES.
input_to_hid_weights = init_wt * randn(input_size, numhid);
hid_to_output_weights = init_wt * randn(numhid, out_size);
hid_bias = zeros(numhid, 1);
output_bias = zeros(out_size, 1);

input_to_hid_weights_delta = zeros(input_size, numhid);
hid_to_output_weights_delta = zeros(numhid, out_size);
hid_bias_delta = zeros(numhid, 1);
output_bias_delta = zeros(out_size, 1);
expansion_matrix = eye(out_size);
count = 0;
tiny = exp(-30);

CE_plot = [];
% TRAIN.
for epoch = 1:epochs
  fprintf(1, 'Epoch %d\n', epoch);
  this_chunk_CE = 0;
  trainset_CE = 0;
  % LOOP OVER MINI-BATCHES.
  for m = 1:numbatches
    input_batch = train_input(:, :, m);
    target_batch = train_target(:, :, m);

    % FORWARD PROPAGATE.
    % Compute the state of each layer in the network given the input batch
    % and all weights and biases
    [hidden_layer_state, output_layer_state] = ...
      fprop(input_batch, input_to_hid_weights, ...
            hid_to_output_weights, hid_bias, output_bias);

    % COMPUTE DERIVATIVE.
    %% Expand the target to a sparse 1-of-K vector.
    expanded_target_batch = expansion_matrix(:, target_batch);
    %% Compute derivative of cross-entropy loss function (which is just (Y - T)).
    error_deriv = output_layer_state - expanded_target_batch;

    % MEASURE LOSS FUNCTION.
    % Cross entropy is entropy, but using the true distribution times the log of the estimated distr. 
    CE = -sum(sum(...
      expanded_target_batch .* log(output_layer_state + tiny))) / batchsize;
    count =  count + 1;
    % GENIUS - this is a running average
    this_chunk_CE = this_chunk_CE + (CE - this_chunk_CE) / count;
    trainset_CE = trainset_CE + (CE - trainset_CE) / m;
    if mod(m, show_training_CE_after) == 0
      fprintf(1, '\rBatch %d Train CE %.3f', m, this_chunk_CE);
      count = 0;
      this_chunk_CE = 0;
    end
    if OctaveMode
      fflush(1);
    end

    % BACK PROPAGATE.
    %% OUTPUT LAYER.
    % this calculates  SUM_n yi * (yj - tj) for each neuron, so it ends
    % with a row having every entry as a sum over every training case of a particular weight Wij
    % so one row is all weights for one hidden neuron.  Each row are the
    % different weights of a different hidden neuron
    % this will be used to update the weights of the hidden layer
    hid_to_output_weights_gradient =  hidden_layer_state * error_deriv';
    % this sums up the rows into a column vector, each item is the sum of
    % all the errors made at that output neuron for every training example
    output_bias_gradient = sum(error_deriv, 2);
    % this gives you SUM Wij*Err for a given hidden neuron, so that one
    % entry is that and one row is that calculated for every training
    % example.  It is then multiplied by activation*(1 - activation)
    % for every training example.  So we have 
    % yj * (1 - yj) * SUM( Wjk * dE/dZk)
    % this is the delta at a given neuron for each training example (which
    % is a row)
    back_propagated_deriv_1 = (hid_to_output_weights * error_deriv) ...
      .* hidden_layer_state .* (1 - hidden_layer_state);

    %% HIDDEN LAYER.
    % take all your activations at the first neuron, multiply them by all
    % the deltas seen at the (hidden) outputs you connect to. So one entry
    % is a single weight update summed over all training examples and a row
    % is the connections from neuron i to each neuron j
    % this will be used to update weights
    input_to_hid_weights_gradient = input_batch * back_propagated_deriv_1';

    % Bias is mutliplied by 1 as the activation so it is just the sum of
    % the gradient over all training examples in the batch
    hid_bias_gradient = sum(back_propagated_deriv_1, 2);

   % UPDATE WEIGHTS AND BIASES.
   input_to_hid_weights_delta = ...
      momentum .* input_to_hid_weights_delta + ...
      input_to_hid_weights_gradient ./ batchsize;
    input_to_hid_weights = input_to_hid_weights...
      - learning_rate * input_to_hid_weights_delta;

    hid_to_output_weights_delta = ...
      momentum .* hid_to_output_weights_delta + ...
      hid_to_output_weights_gradient ./ batchsize;
    hid_to_output_weights = hid_to_output_weights...
      - learning_rate * hid_to_output_weights_delta;

    hid_bias_delta = momentum .* hid_bias_delta + ...
      hid_bias_gradient ./ batchsize;
    hid_bias = hid_bias - learning_rate * hid_bias_delta;

    output_bias_delta = momentum .* output_bias_delta + ...
      output_bias_gradient ./ batchsize;
    output_bias = output_bias - learning_rate * output_bias_delta;

    % VALIDATE.
    if mod(m, show_validation_CE_after) == 0
      fprintf(1, '\rRunning validation ...');
      if OctaveMode
        fflush(1);
      end
      [~, output_layer_state] = fprop(valid_input, ... 
          input_to_hid_weights, hid_to_output_weights, hid_bias, output_bias);
      datasetsize = size(valid_input, 2);
      expanded_valid_target = expansion_matrix(:, valid_target);
      CE = -sum(sum(...
        expanded_valid_target .* log(output_layer_state + tiny))) /datasetsize;
      fprintf(1, ' Validation CE %.3f\n', CE);
      if OctaveMode
        fflush(1);
      end
    end
  end
  fprintf(1, '\rAverage Training CE %.3f\n\n', trainset_CE);
  CE_plot = [ CE_plot; trainset_CE ];
end
fprintf(1, 'Finished Training.\n');
if OctaveMode
  fflush(1);
end
fprintf(1, 'Final Training CE %.3f\n', trainset_CE);

% EVALUATE ON VALIDATION SET.
fprintf(1, '\rRunning validation ...');
if OctaveMode
  fflush(1);
end
[~, output_layer_state] = fprop(valid_input, ...
    input_to_hid_weights, hid_to_output_weights, hid_bias, output_bias);
datasetsize = size(valid_input, 2);
expanded_valid_target = expansion_matrix(:, valid_target);
CE = -sum(sum(...
  expanded_valid_target .* log(output_layer_state + tiny))) / datasetsize;
fprintf(1, '\rFinal Validation CE %.3f\n', CE);
[~, indices] = max(output_layer_state);
fprintf('\nValidation Set Accuracy: %f\n', mean(double(indices == valid_target)) * 100);
if OctaveMode
  fflush(1);
end

% EVALUATE ON TEST SET.
fprintf(1, '\rRunning test ...');
if OctaveMode
  fflush(1);
end
[~, output_layer_state] = fprop(test_input, ...
    input_to_hid_weights, hid_to_output_weights, hid_bias, output_bias);
datasetsize = size(test_input, 2);
expanded_test_target = expansion_matrix(:, test_target);
CE = -sum(sum(...
  expanded_test_target .* log(output_layer_state + tiny))) / datasetsize;
fprintf(1, '\rFinal Test CE %.3f\n', CE);
[~, indices] = max(output_layer_state);
fprintf('\nTest Set Accuracy: %f\n', mean(double(indices == test_target)) * 100);
if OctaveMode
  fflush(1);
end

fprintf('Confusion matrix for test data:\n');
for i=1:out_size,
    for j=1:out_size,
        tmp = indices(test_target == j);
        fprintf('%5d ', sum(tmp == i));
    end;
    fprintf('\n');
end;

if OctaveMode
  end_time = time;
  diff = end_time - start_time;
else  % In MATLAB
  end_time = clock;
  diff = etime(end_time, start_time);
end
fprintf(1, 'Training took %.2f seconds\n', diff);

% VISUALIZE TRAINING
title('Cross Entropy Error')
xlabel('Epochs')
ylabel('CE on training set')
plot(CE_plot)
savefig( strcat('output\', opts.name, '_TrainingError.fig') ); 

% VISUALIZE THE LEARNED NETWORK
fprintf('\nVisualizing hidden layer of the Neural Network... \n')
displayData(input_to_hid_weights');  
savefig( strcat('output\', opts.name, '_learnedWeights.fig') ); 

model.input_to_hid_weights = input_to_hid_weights;
model.hid_to_output_weights = hid_to_output_weights;
model.hid_bias = hid_bias;
model.output_bias = output_bias;

end
% =================================================================================

% =================================================================================
function [hidden_layer_state, output_layer_state] = ...
  fprop(input_batch, in_to_hid_weights, hid_to_output_weights, hid_bias, output_bias)
% This method forward propagates through a neural network.
% Inputs:
%   input_batch: The input data as a matrix of size num_input X batchsize where,
%     num_input is the number of input features, batchsize is the number of data points.
%
%   in_to_hid_weights: Weights between the input layer and hidden
%     layer as a matrix of size input_size x numhid
%
%   hid_to_output_weights: Weights between the hidden layer and output softmax
%               unit as a matrix of size numhid2 X vocab_size
%
%   hid_bias: Bias of the hidden layer as a matrix of size numhid2 X 1.
%
%   output_bias: Bias of the output layer as a matrix of size vocab_size X 1.
%
% Outputs:
%   embedding_layer_state: State of units in the embedding layer as a matrix of
%     size input_size X batchsize
%
%   hidden_layer_state: State of units in the hidden layer as a matrix of size
%     numhid X batchsize
%
%   output_layer_state: State of units in the output layer as a matrix of size
%     out_size X batchsize
%

batchsize = size(input_batch, 2);
out_size = size(hid_to_output_weights, 2);

%% COMPUTE STATE OF HIDDEN LAYER.
% Compute inputs to hidden units.
% could have just added column of 1's to input instead of external bias...
% this gives every training example activation at the hidden neurons
% each training example is a column
inputs_to_hidden_units = in_to_hid_weights' * input_batch + ...
  repmat(hid_bias, 1, batchsize);

% Apply logistic activation function.
hidden_layer_state = 1 ./ (1 + exp(-inputs_to_hidden_units));

%% COMPUTE STATE OF OUTPUT LAYER.
% Compute inputs to softmax.
% each training example is still a column
inputs_to_softmax = hid_to_output_weights' * hidden_layer_state +  repmat(output_bias, 1, batchsize);

% Subtract maximum. 
% Remember that adding or subtracting the same constant from each input to a
% softmax unit does not affect the outputs. Here we are subtracting maximum to
% make all inputs <= 0. This prevents overflows when computing their
% exponents.
% max gives the max by column
inputs_to_softmax = inputs_to_softmax...
  - repmat(max(inputs_to_softmax), out_size, 1);

% Compute exp.
output_layer_state = exp(inputs_to_softmax);

% Normalize to get probability distribution.
output_layer_state = output_layer_state ./ repmat(...
  sum(output_layer_state, 1), out_size, 1);

end

% =================================================================================