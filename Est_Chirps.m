function X = Est_Chirps (y,Z,Gamma,alpha,beta,rho)

% Written by: Xiangxia Meng
% Paper: "Estimation of chirp signals with time-varying amplitudes"
% Authors: Xiangxia Meng, Andreas Jakobsson, Xiukun Li, Yahui Lei

% Input:
%       y       - Data vector
%       Z       - Predefined dictionary
%       alpha   - Tuning Parameter
%       beta    - Tuning Parameter
%       rho     - The augmented Lagrangian parameter
% Output
%       X       - Estimated weighting matrix

t_start = tic;

QUIET    = 1;
if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
        'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
end

MAX_ITER = 500;
ABSTOL   = 1e-4;
RELTOL   = 1e-2;

[N,P] = size(Z);
[~,R] = size(Gamma);

Xv  = zeros(R*P,1);
Vv2 = Xv;
Uv2 = Xv;

Vv1 = zeros(N,1);
Uv1 = Vv1;

Vv3 = zeros(N*P,1);
Uv3 = Vv3;

Phi = kron(eye(P),Gamma);

H = zeros(R*P,N);
for i = 1:N
    H(:,i) = kron(Z(i,:),Gamma(i,:)).';
end
H = H.';

Hp = [H;Phi];
[m, n] = size(Hp);

[L, U] = factor(H, Gamma, 1, m, n, P);

for k = 1:MAX_ITER
    % Xv
    q = H'*(Vv1-Uv1) + (Vv2-Uv2) + Phi'*(Vv3 - Uv3);    % temporary value
    if( m >= n )    % if skinny
        Xv = U \ (L \ q);
    else            % if fat
        Xv = q - (H'*(U \ ( L \ (H*q) )));
    end
    X = reshape(Xv,R,P);
    
    % Vv
    Vv1old = Vv1;
    Vv2old = Vv2;
    Vv3old = Vv3;
    Vv1 = (y + rho*(Uv1+diag(Gamma*X*Z')))/(1+rho);
    for i = 1:P
        Vv2(R*(i-1)+1:R*i,1) = shrinkage(Xv(R*(i-1)+1:R*i) + Uv2(R*(i-1)+1:R*i)...
            , alpha/rho);
        Vv3(N*(i-1)+1:N*i,1) = shrinkage(Gamma*Xv(R*(i-1)+1:R*i) + Uv3(N*(i-1)+1:N*i)...
            , beta/rho);
    end
    
    % Uv
    Uv1 = Uv1 + diag(Gamma*X*Z') - Vv1;
    Uv2 = Uv2 + (Xv - Vv2);
    Uv3 = Uv3 + (vec(Gamma*X) - Vv3);
    
    % diagnostics, reporting, termination checks
    history.objval(k)  = 0;%objective(A, y, lambda, X, Z,gamma,INDEX);
    
    history.r_norm(k)  = norm(Xv - Vv2);
    history.s_norm(k)  = norm(-rho*(Vv2 - Vv2old));
    
    history.eps_pri(k) = sqrt(N)*ABSTOL + RELTOL*max(norm(Xv), norm(-Vv2));
    history.eps_dual(k)= sqrt(N)*ABSTOL + RELTOL*norm(rho*Uv2);
    
%     if ~QUIET
%         fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
%             history.r_norm(k), history.eps_pri(k), ...
%             history.s_norm(k), history.eps_dual(k), ...
%         history.objval(k));
%     end
    
    if ~QUIET
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
            history.r_norm(k), history.eps_pri(k), ...
            history.s_norm(k), history.eps_dual(k), ...
        norm(diag(Gamma*X*Z') - Vv1),norm(vec(Gamma*X) - Vv3) );
    end
    
    if ( k > 1 && history.r_norm(k) < history.eps_pri(k) && ...
            history.s_norm(k) < history.eps_dual(k))
        break;
    end
            
end

if ~QUIET
    toc(t_start);
end

end    


function [L, U] = factor(A, B, rho, m1, n1, P)
if ( m1 >= n1 )    % if skinny
    A1 = A'*A + kron(speye(P),B'*B);
    L = chol( A1 + rho*speye(n1), 'lower' );
else            % if fat
    L = chol( speye(m1) + 1/rho*(A*A'), 'lower' );
end
% force matlab to recognize the upper / lower triangular structure
L = sparse(L);
U = sparse(L');
end

function z = shrinkage(x,kappa)
temp = norm(x);
if temp<kappa
    z = zeros(size(x));
else
    z = x*(temp-kappa)/temp;
end
end

function v = vec( x )
v = reshape( x, numel( x ), 1 );
end
