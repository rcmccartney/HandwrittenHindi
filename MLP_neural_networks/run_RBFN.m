function [ output ] = run_RBFN( imdb, opts )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

disp('Training the RBFN...');

m = size(imdb.data, 1);

% Train the RBFN using k centers per category.
[Centers, betas, Theta] = trainRBFN(imdb.data, imdb.labels, ...
    opts.clusters, true);

disp('Measuring training accuracy...');
numRight = 0;
% For each training sample...
for i=1:m,
    % Compute the scores for both categories.
    scores = evaluateRBFN(Centers, betas, Theta, imdb.data(i, :));    
	[~, category] = max(scores);
    % Validate the result.
    if (category == imdb.labels(i))
        numRight = numRight + 1;
    end
end
accuracy = numRight / m * 100;
fprintf('Training accuracy: %d / %d, %.1f%%\n', numRight, m, accuracy);
if exist('OCTAVE_VERSION'), 
    fflush(stdout); 
end;

disp('Measuring testing accuracy...');
numRight = 0;
% make a confusion matrix
confusion = zeros(opts.num_labels, opts.num_labels);
m = size(imdb.test, 1);
for i=1:m,
    % Compute the scores for both categories.
    scores = evaluateRBFN(Centers, betas, Theta, imdb.test(i, :));    
	[~, category] = max(scores);
    confusion(category, imdb.test_labels(i)) = confusion(category, imdb.test_labels(i)) + 1;
    % Validate the result.
    if (category == imdb.test_labels(i))
        numRight = numRight + 1;
    end
end
accuracy = numRight / m * 100;
fprintf('Testing accuracy: %d / %d, %.1f%%\n', numRight, m, accuracy);

fprintf('Confusion matrix for test data:\n');
for i=1:opts.num_labels,
    for j=1:opts.num_labels,
        fprintf('%5d ', confusion(i, j));
    end;
    fprintf('\n');
end;

if exist('OCTAVE_VERSION'), 
    fflush(stdout); 
end;

end

