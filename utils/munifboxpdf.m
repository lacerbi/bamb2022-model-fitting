function y = munifboxpdf(x,a,b)
%MUNIFBOX Multivariate uniform box probability density function.

% a < b

nf = prod(b - a,2);
y(:) = 1 ./ nf * ones(size(x,1),1);
idx = any(bsxfun(@lt, x, a),2) | any(bsxfun(@gt, x, b),2);
y(idx) = 0;