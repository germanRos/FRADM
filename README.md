# Nov. 2014
# This code implements the Fixed-Rank Alternated Direction Method with Augmented Lagrange Multiplier 
# algorithm for Robust Fixed-Rank Sparse Decompositions.
#
# D - m x n matrix of probably corrupted data
#
# lambda - weight on sparse error term in the cost function
#
# tol - tolerance for stopping criterion.
#
# maxIter - maximum number of iterations
#
# This code has been developed by:
#
#	* German Ros 	(gros@cvc.uab.es)
#	Computer Vision Center and Universitat Autonoma de Barcelona (Spain)
#
#	* Julio Guerrero	(juguerre@um.es)
#	Dept. of Applied Mathematics, Universidad de Murcia (Spain)
#
# To know more details, please check our arXiv submission "Fast and Robust Fixed-Rank Matrix Recovery"
# at http://arxiv.org/abs/1503.03004
#
# 
# This implementation is based on the implementation of the iALM method by Minming Chen & Arvind Ganesh
# and LMAFIT by Yin Zhang. Thank you very much!
#
################
