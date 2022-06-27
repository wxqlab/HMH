function P = HMH(X,S,Z,bit,alpha,beta,maxItr)

%% Initialization
[d,~] = size(X);
[~,num] = size(S);
XS = X*S;
D = diag(sum(S,2));
XDX = X*D*X';
hatZ = Z'*Z;
G = eye(d,d);
randn('seed',2);
B=sign(randn(num,bit));

%% Start iteration
for ite = 1:maxItr
    % obtain P
    P = (beta*G+XDX)\XS*B;
    % obtain objective function value
    Fval(ite) = - trace(2.*P'*XS*B) - trace(2.*alpha.*B'*Z*B) + trace(alpha.*B'*hatZ*B) + trace(P'*XDX*P) + trace(beta.*P'*G*P);
    % obtain B
    PXS=P'*XS;
    for bb = 1:bit
        for mm = 1:num
            b = - 2.*PXS(bb,mm) + B(:,bb)'*(-2*alpha.*Z(:,mm)-2*alpha.*Z(mm,:)'+alpha.*hatZ(:,mm)+alpha.*hatZ(mm,:)')...
                - B(mm,bb)*(2*alpha.*hatZ(mm,mm)-4*alpha.*Z(mm,mm));
            B(mm,bb) = sign(b);
        end
    end
    % update G
    pi = sqrt(sum(P.*P,2)+eps);
    gi = 0.5./pi;
    G = diag(gi);
end

end