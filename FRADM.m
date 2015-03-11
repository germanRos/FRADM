function [A_hat, E_hat, iter, U, V, B] = FRADM(D, lambda, lrank, tol, maxIter)
% Nov. 2014
% This code implements the Fixed-Rank Alternated Direction Method with Augmented Lagrange Multiplier 
% algorithm for Robust Fixed-Rank Sparse Decompositions.
%
% D - m x n matrix of probably corrupted data
%
% lambda - weight on sparse error term in the cost function
%
% tol - tolerance for stopping criterion.
%
% maxIter - maximum number of iterations
%
% This code has been developed by:
%
%	* German Ros 	(gros@cvc.uab.es)
%	Computer Vision Center and Universitat Autonoma de Barcelona (Spain)
%
%	* Julio Guerrero	(juguerre@um.es)
%	Dept. of Applied Mathematics, Universidad de Murcia (Spain)
%
% To know more details, please check our arXiv submission "Fast and Robust Fixed-Rank Matrix Recovery"
% at http://arxiv.org/abs/1503.03004
%
% 
% This implementation is based on the implementation of the iALM method by Minming Chen & Arvind Ganesh
% and LMAFIT by Yin Zhang. Thank you very much!
%
%%%%%%%%%%%%%%%%
	[m n] = size(D);

	% data re-scaling
	Dscale = norm(D(:), inf);
        D = D / Dscale;

	% initialize
	Y = D;
	norm_two = lansvd(Y, 1, 'L');
	norm_inf = norm(Y(:), inf) / lambda;
	dual_norm = max(norm_two, norm_inf);
	Y = Y / dual_norm;

	A_hat = zeros(m, n);
	E_hat = zeros(m, n);
	mu = 1.25/norm_two;
	mu_bar = mu * 1e7;
	rho = 1.7;
	d_norm = norm(D, 'fro');
	tol = tol*norm(D,'fro');


	iter = 0;
	converged = false;
	stopCriterion = 1;

	U = eye(m, lrank);
	V = eye(n, lrank);
	B = eye(lrank);


	while ~converged       
		iter = iter + 1;
		temp_T = D - A_hat + (1/mu)*Y;
		E_hat = max(temp_T - lambda/mu, 0) + min(temp_T + lambda/mu, 0);

		% Projection onto the FixedRank manifold by using FixedRankOptStep
		[U B V] = FixedRankExactOpt(D - E_hat + (1/mu)*Y, lrank, U, B, V, 1, 0, 0, 0);
		A_hat = U*B*V';
		Z = D - A_hat - E_hat;

		Y = Y + mu*Z;

		% continuation
		mu = mu*rho;

		%% stop Criterion but save the first times!
		if(iter > 20)
			i1 = randi(m*n,5*m,1);
			stopCriterion = norm(Z(i1), 'fro') / norm(D(i1), 'fro');    
			if( (stopCriterion < tol) || (iter >= maxIter) )
				converged = true;
			end  
		end    
	end

	A_hat = A_hat * Dscale;
	E_hat = E_hat * Dscale;
    	U = U * Dscale;
end


function [U, B, V,iters,errorL] = FixedRankExactOpt(M, r, Uo, Bo, Vo, MaxIters,tol,verb,logvalue)
% INPUT:
%   M : Perturbed matrix, of the form M = L + E where L is lowrank  (rank = r) and E is some perturbation
%   r : rank of the low rank matrix L
%   Bo:  initial aproximation to SPD component of L
%   Uo:  initial aproximation to Stiefel U component of L
%   Vo:  initial aproximation to Stiefel V component of L
%   MaxIter: maximum number of iterations to perform
%   tol: relative error of the aproximation obtained (if set to 0 this check is avoided)
%
%
% This file is part of FixedRankOpt project
% Original authors: Julio Guerrero and Germán Ros, Jan.  2014.
% Contributors: 
% Change log: 
	logfilename=0;
	  
	% Size of the original matrix
	[m, n] = size(M);

	U=Uo;
	B=Bo;
	V=Vo;

	if (tol ~= 0)
	   Lo=Uo*Bo*Vo';
	end

	% Begin of for loop
	for i=1:MaxIters
		iters=i;
	     
		[U,B,V] = FixedRankStepExact(M, r,U,B,V); 

	     if (tol ~= 0)
		 L=U*B*V';
		 errorL=norm(L-Lo,'fro')/norm(L,'fro');
		 Lo=L;

		 if (verb > 0)
		    if(mod(i,logvalue) == 0)              
		       str = sprintf('\nFixedRankOpt: End of iteration [%d] ;  Relative error in L = %4.2e \n',i,errorL);
		    fprintf(logfilename, str);                         
		   end
		 end
		if errorL < tol    
		 break
		end
	     end
	    
	% End of for loop
	end

	if (verb > 0)
	str = sprintf('\nTotal no. of FixedRankStep iterations performed = %i \n',iters);
	fprintf(logfilename, str); 
	end

	if (tol==0)
	    errorL=0;
	end

		    	
	% En of function
end


function [U B V] = FixedRankStepExact(M,r,Uo,Bo,Vo)
%
% Exact FixedRank Step with  Alternating linesearch in U, V , B (other orders are also possible, but this seems to work particularly well)
%
%
% INPUT:
%   M : Perturbed matrix, of the form M = L + E where L is lowrank  (rank = r) and E is some perturbation
%   r : rank of the low rank matrix L, which is of th the form L = U*B*V'
%   Bo:  initial aproximation to SPD component of L
%   Uo:  initial aproximation to Stiefel U component of L
%   Vo:  initial aproximation to Stiefel V component of L
%

% OUTPUT:
% 
%   B: new SPD component of the new L (which is not explicitely computed)
%   U: new Stiefel U component of Lnew
%   V: new Stiefel V component of Lnew

% This file is part of FixedRankOpt project
% Original authors: Julio Guerrero and Germán Ros, Jan.  2014.
% Contributors: 
% Change log: 

	% First, actualize U
	%tic;
	U = procrustes(Uo,Bo,Vo,M);

	%toc

	% Second, actualize V. We use the actualized U


	%tic;

	V = procrustes(Vo,Bo,U,M');

	%toc
	   
	% Third, actualize B.  We use the actualized U and V

	%tic;
	B = SPD_min(U,Bo,V,M);
	%toc

	%chol(B_new);

end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5



function B = SPD_min(Uo,Bo,Vo,M)

[m n]=size(M);
         if m>n 
              M_L=(Uo'*M)*Vo;
         else
              M_L=Uo'*(M*Vo);
         end
B=Sym(M_L);

end

function A = procrustes(Ao,Bo,Co,M)
	[m n]=size(M);
	if m>n 
	    M_temp=M*(Co*Bo);
	else
	    M_temp=(M*Co)*Bo;
	end
	[U D V]= svd(M_temp,'econ');  % This can be substituted by a Lanczos algorithm
	
	A=U*V';
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
function S=Sym(A)
         S = (A + A')/2;
end 

function S=Skew(A)
         S = (A - A')/2;
end  

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function  [U, B, V]=RandomFixedRank(m,n,r,sizeB);
     U=randomStiefel(m,r);
     B=sizeB*randomSPD(r);
     V=randomStiefel(n,r);    
end


function Q = randomStiefel(n,r)        
       [Q, R] = qr(randn(n,r), 0);         
end

function B = randomSPD(r)
        D = 0.5*diag(1+rand(r, 1));
        [Q R] = qr(randn(r)); 
        B = Q*D*Q';
end
