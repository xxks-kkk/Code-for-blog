function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% a 64x3 matrix: each row represents a model with a specific
% combination of C and sigma. 1st column is C, 2nd column is sigma,
% and 3rd column is prediction error.
results = eye(64,3); 
counter = 0;

for C_hat = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30],
  for sigma_hat = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30],
    counter++;
    model= svmTrain(X, y, C_hat, @(x1, x2) gaussianKernel(x1, x2, sigma_hat));
    predictions = svmPredict(model, Xval);
    error_val = mean(double(predictions ~= yval));
    results(counter, :) = [C_hat, sigma_hat, error_val];
  end
end

[M,I] = min(results, [], 1);  % get the indices of the min value for each column
C = results(I(3),1);
sigma = results(I(3),2);

% =========================================================================

end
