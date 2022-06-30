Smoothing Proximal Gradient Method for General Structured Sparse Learning (uni-response)

Please run install_mex.m to mex .c files (I have mexed the .c files under windows for both 32 and 64 bit machines)

For testing overlapping group lasso, run group_test.m

For testing graph_guided fused lasso, run graph_test.m

The files for main optimization procedure include

(1) SPG.m  Smoothing Proximal Gradient (SPG) with a pre-computed Lipschitz constant, suitable for medium-scale problem

(2) SPG_linesearch.m  Smoothing Proximal Gradient (SPG) with a line-search scheme on Lipschitz constant, suitable for large-scale problem

The first argument for both SPG.m and SPG_linesearch.m is 'prob'. It takes either 
'group' or 'graph', which indicates that the problem to solve is overlapping group lasso or graph-guided fused lasso.

Please cite if you use the code in your research. 

@INPROCEEDINGS{xichen:UAI:11,
  author = {Xi Chen and Qihang Lin and Seyoung Kim and Jaime Carbonell and Eric
    Xing},
  title = {Smoothing Proximal Gradient Method for General Structured Sparse
    Learning},
  booktitle = {Proceedings of Uncertainty in Artificial Intelligence (UAI)},
  year = {2011}  
}

If there is any comment or question, please contact: xichen@cs.cmu.edu




