function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

% You need to return the following variables correctly.
Z = zeros(size(X, 1), K);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the projection of the data using only the top K 
%               eigenvectors in U (first K columns). 
%               For the i-th example X(i,:), the projection on to the k-th 
%               eigenvector is given as follows:
%                    x = X(i, :)';
%                    projection_k = x' * U(:, 1:k);;
% Recall: X_approx = Z = U_reduce'*X, where X is the original data and
% U_reduce is the n x n matrix of eigenvectors returned by the svd
% function.

% Return Z, the product of X and the first 'K' columns of U. 
% 
% X is size (m x n), and the portion of U is (n x K). Z is size (m x K).
%

[m n] = size(X);
for i = 1:m
    U_reduce = U(:, 1:K);
    x = X(i, :)';
    Z(i, :) = x' * U_reduce;
end







% =============================================================

end
