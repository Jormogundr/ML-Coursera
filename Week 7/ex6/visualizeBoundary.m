function visualizeBoundary(X, y, model, varargin)
%VISUALIZEBOUNDARY plots a non-linear decision boundary learned by the SVM
%   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a non-linear decision 
%   boundary learned by the SVM and overlays the data on it

% Plot the training data on top of the boundary
plotData(X, y)

% Make classification predictions over a grid of values
x1plot = linspace(min(X(:,1)), max(X(:,1)), 100)'; % 100 element vector for the x1 components of the plot
x2plot = linspace(min(X(:,2)), max(X(:,2)), 100)'; % 100 element vector for the x2 components of the plot
[X1, X2] = meshgrid(x1plot, x2plot); % use the x1 and x2 vector to define a meshgrid object
vals = zeros(size(X1)); % initialize vals vector
for i = 1:size(X1, 2) 
   this_X = [X1(:, i), X2(:, i)];
   vals(:, i) = svmPredict(model, this_X);
end

% Plot the SVM boundary
hold on
contour(X1, X2, vals, [0.5 0.5], 'b');
hold off;

end
