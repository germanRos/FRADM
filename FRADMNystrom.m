function [A, E, iters, U] = FRADMNystrom(M, lambda, lrank, tol, maxIter)
% Nov. 2014
% This code implements the Nystrom's variant of the Fixed-Rank Alternated Direction Method with Augmented Lagrange Multiplier 
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
	ll = min(5*lrank, min(size(M)));

	randRows = randsample(size(M, 1), ll);
	randCols = randsample(size(M, 2), ll);

	% recover M_top
	[A_top, E_top, it1] = inexact_alm_rpcaFR(M(randRows, :), lambda, lrank, tol, maxIter);
	%A_top = A_top';
	% recover M_left
	[A_left, E_left, it2] = inexact_alm_rpcaFR(M(:, randCols), lambda, lrank, tol, maxIter);
	% estimate A_top-left
	A_tl = A_top(:, randCols);

	A = A_left * pinv(A_tl) * A_top;
	E = M - A;

	iters = it1 + it2;
end
