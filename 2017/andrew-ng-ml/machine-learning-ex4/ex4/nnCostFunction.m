function [J grad] = nnCostFunction(nn_params, ...
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
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1)); % Theta1 has size 25 x 401

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1)); % Theta2 has size 10 x 26

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

a1 = [ones(m,1), X];
% Theta1 * a1' is not recommended b/c resulting matrix is 25x5000
% a2 is 5000x25 matrix 
a2 = sigmoid(a1 * Theta1'); 
% a3 is 5000x10 matrix (5000x26 * 26x10)
a3 = sigmoid([ones(size(a2, 1), 1), a2] * Theta2'); 
% yVec is 5000x10 matrix
yVec = zeros(m, num_labels); 

for i = 1:m,
  yVec(i, y(i)) = 1;
end

J = 1/m * sum(sum(-yVec .* log(a3) - (1 - yVec) .* log(1-a3))); % unregularized cost function

theta1_nobias = Theta1(:, 2:end); % Theta1 without bias parameters
theta2_nobias = Theta2(:, 2:end); % Theta2 without bias parameters

J = J + lambda / 2 / m * (sum(sum(theta1_nobias.^2)) + sum(sum(theta2_nobias.^2))); % regularized cost function

for t = 1:m,
  a_1 = [1, X(t,:)]';         % 401x1  matrix  
  z_2 = Theta1 * a_1;         % 25x1   matrix  
  a_2 = [1;sigmoid(z_2)];     % 26x1   matrix
  z_3 = Theta2 * a_2;  
  a_3 = sigmoid(z_3);         % 10x1   matrix  
  delta_3 = a_3 - yVec(t,:)'; % 10x1   matrix
  delta_2 = Theta2(:, 2:end)'*delta_3 .* sigmoidGradient(z_2);
  %printf("delta_2 size: %d, %d\n", size(delta_2,1), size(delta_2,2));
  
  %printf("Theta1_grad size: %d, %d\n", size(Theta1_grad,1), size(Theta1_grad,2));
  %printf("Theta2_grad size: %d, %d\n", size(Theta2_grad,1), size(Theta2_grad,2));
  Theta1_grad += delta_2*a_1';
  Theta2_grad += delta_3*a_2';
end

Theta1_grad = (1/m) * Theta1_grad + (lambda/m) * [zeros(size(Theta1,1),1) Theta1(:, 2:end)];
Theta2_grad = (1/m) * Theta2_grad + (lambda/m) * [zeros(size(Theta2,1),1) Theta2(:, 2:end)];













% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
