function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
num = size(theta);
n = size(theta);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% J calculation
sum_normal = 0;
sum_thetas = 0;

for i = 1:m
    h = 1/ (1 + exp(- theta' * X(i,:)'));
    sum_normal= sum_normal + (-1* y(i) * log(h) - ((1-y(i))* log(1-h)));
end

for i = 2:n
    sum_thetas = sum_thetas + theta(i)^2;
end

J = sum_normal/m + lambda * sum_thetas/(2*m);

% Calculate derivatives of J with respect to several thetas.
sum = 0;
for i = 1:m
    sum = sum + ( 1/ (1 + exp(- theta' * X(i,:)')) - y(i)) * X(1,1);
end

grad(1) = 1/m * sum;

for j = 2:num
    sum = 0;
    for i = 1:m
    sum = sum + ( 1/ (1 + exp(- theta' * X(i,:)')) - y(i)) * X(i,j);
    end
    grad(j) = 1/m *sum + lambda/m * theta(j); 
% =============================================================

end
