function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% we need to return the following variables correctly 
g = zeros(size(z));

% ======================  CODE HERE ======================
              
for i = 1:numel(z)
  g(i)=1./(1+exp(-1*z(i)));
endfor
% =============================================================

end
