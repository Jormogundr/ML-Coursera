function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h = theta' * X';
arg = h - y';
J = 1/(2*m)*sum(arg.^2) + lambda/(2*m)*sum(theta(2:end).^2); % the two index is for the special case when j = 0, which is not included in the reg. sum

grad = (1/m)*arg*X + (lambda/m)*theta'; % for j > 0
grad(1) = (1/m)*sum(arg*X(1)); % handle the special case, for j = 0

% =========================================================================

grad = grad(:);

end
