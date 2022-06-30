# RelativeShift
Matlab code for fitting relative-shift regression for compositional predictors

The relative-shift regression is a new regression framework for compositional predictors (e.g., microbiome relative abundance). It directly models the relationship between a univariate response and a compositional data vector (and additional non-compositional covariates). The framework also allows feature aggregation through equi-sparsity-inducing regularization. When desired, a tree structure among the compositional predictors (e.g., taxonomic tree among microbes) can be provided to further guide the aggregation process. Check out the paper "Li et al. (2022) It's all relative: Regression analysis with compositional predictors" for more details.



Note: To run the proposed methods, please first compile the SPG code. Specifically, go to the SPG folder and run install_mex.m to mex .c files (I have mexed the .c files under windows for both 32 and 64 bit machines). To get an idea of how to use the methods, run Simulation.m file to get simulation results of the proposed methods in the paper. 

If there is any comment or question, please contact: ligen@umich.edu
