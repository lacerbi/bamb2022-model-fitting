function y = mtrapezpdf(x,a,b,c,d)
%MTRAPEZPDF Multivariate trapezoidal probability density function (pdf).
%   Y = MTRAPEZPDF(X,A,B,C,D) returns the pdf of the multivariate trapezoidal
%   distribution with external bounds A and D and internal points B and C,
%   evaluated at the values in X. The multivariate trapezoidal
%   pdf is the product of univariate trapezoidal pdfs in each dimension. 
%
%   For each dimension i, the univariate trapezoidal pdf is defined as:
%
%                 |       __________
%                 |      /|        |\
%         p(X(i)) |     / |        | \ 
%                 |    /  |        |  \
%                 |___/___|________|___\____
%                    A(i) B(i)     C(i) D(i)
%                             X(i)
%                          
%   X can be a matrix, where each row is a separate point and each column
%   is a different dimension. Similarly, A, B, C, and D can also be
%   matrices of the same size as X.

% Luigi Acerbi 2022

[N,D] = size(x);

if D > 1
    if isscalar(a); a = a*ones(1,D); end
    if isscalar(b); b = b*ones(1,D); end
    if isscalar(c); c = c*ones(1,D); end
    if isscalar(d); d = d*ones(1,D); end
end

if size(a,2) ~= D || size(b,2) ~= D || size(c,2) ~= D || size(d,2) ~= D
    error('mtrapezpdf:SizeError', ...
        'A, B, C, D should be scalars or have the same number of columns as X.');
end

if size(a,1) == 1; a = repmat(a,[N,1]); end
if size(b,1) == 1; b = repmat(b,[N,1]); end
if size(c,1) == 1; c = repmat(c,[N,1]); end
if size(d,1) == 1; d = repmat(d,[N,1]); end

y = zeros(size(x));
nf = 0.5 * (d - a + c - b) .* (b - a);

for ii = 1:D
    idx = x(:,ii) >= a(:,ii) & x(:,ii) < b(:,ii);
    y(idx,ii) = (x(idx,ii) - a(idx,ii)) ./ nf(idx,ii);
    
    idx = x(:,ii) >= b(:,ii) & x(:,ii) < c(:,ii);
    y(idx,ii) = (b(idx,ii)-a(idx,ii)) ./ nf(idx,ii);
    
    idx = x(:,ii) >= c(:,ii) & x(:,ii) < d(:,ii);
    y(idx,ii) = (d(idx,ii) - x(idx,ii))./(d(idx,ii) - c(idx,ii)) .*  (b(idx,ii)-a(idx,ii)) ./ nf(idx,ii);
end

y = prod(y,2);