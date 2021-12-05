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
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

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

% Part 1 - Cost function and forward propagation
newCol = ones(length(X),1); % create the bias column to be added to X data
X = [newCol X]; % add the bias column to the X data
a1 = X;

z2 = Theta1 * a1';
a2 = sigmoid(z2);
a2 = [ones(length(a2),1) a2']; % add the bias term to the second activation layer
z3 = Theta2 * a2';
a3 = sigmoid(z3');
h = a3;

% update the rows of the new reshaped y_r matrix. The result is a matrix
% 5000 x 10 where the rows are all zeros, but the training data
% (handwritten digit) corresponds to the index position of the non-zero
y_r = eye(num_labels);
y_r = y_r(y,:);

% calculate the cost function J(theta) using vectorized method (spent much time on this, could not figure it out so I resorted to the iterative method below)
% t1 = sum(-y_r * log(h'));
% t2 = sum((1-y_r) * log(1 - h'));
% J = (1/m)*sum((t1 - t2));

sum_term = 0;
% iterative version for calculating the cost function
for i = 1:m % training example index - rows
    for k = 1:num_labels % class index - cols
        sum_term = sum_term + -y_r(i,k)*log(h(i,k)) - (1-y_r(i,k))*log(1-h(i,k));
    end
end

J_unreg = sum_term/m; % the unregularlized term in the regularized cost equation

% implement the regularization term in the calculation of the cost
% function. Note that the regularized terms need to ignore the bias term in
% each of the theta1 and theta2 matrices - that is why the column index
% starts at 2 (excludes the column representing the bias terms)
reg_term = 0;
for j = 1:hidden_layer_size % row index
    for k = 2:input_layer_size + 1 % col index
        reg_term = reg_term + (Theta1(j,k).^2);
    end
end

for j = 1:num_labels % row index
    for k = 2:hidden_layer_size + 1 % col index
        reg_term =  reg_term + (Theta2(j,k).^2);
    end
end

J_reg = reg_term*lambda/(2*m);
J = J_unreg + J_reg; % final cost function calculation

% Part 2 - Back propagatio


% back propagation to find delta terms
d3 = a3 - y_r;
d2 = (d3 * Theta2(:,2:end)) .* (a2(:,2:end) .* (1-a2(:,2:end))); % do NOT include bias values from theta2 and from activation layer 2
% d2 = d2(:,2:end); % remove delta_0 from the matrix (1st column)

Theta1_grad = (Theta1_grad + d2'*a1)/m;
Theta2_grad = (Theta2_grad + d3'*a2)/m;

% Part 3 - implement regularization with cost function and gradients

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m)*Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m)*Theta2(:,2:end);



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
