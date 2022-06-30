#include "mex.h"

/* [w_threshold, nonzero]=soft_threshold(w, lambda)
   w is the dense column vector
*/
void CheckInput(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  /* Check for proper number of arguments. */
  if (nrhs < 1)
     mexErrMsgTxt("soft_threshold(W, lambda).\n");
  
  if(mxIsSparse(prhs[0]) || mxIsComplex(prhs[0]))
    mexErrMsgTxt("Input must be a dense real matrix\n");
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,const mxArray *prhs[])
{
  double lambda;
  double *w, *s, *nz;
  mwSize len, m, n, i;
  
      
  CheckInput(nlhs, plhs, nrhs, prhs);
  
  m=mxGetM(prhs[0]);
  n=mxGetN(prhs[0]);
  w=mxGetPr(prhs[0]);
  if (nrhs>1)
    lambda=mxGetScalar(prhs[1]);
  else
    lambda=1;
  
  plhs[0]=mxCreateDoubleMatrix(m,n,mxREAL);
  s=mxGetPr(plhs[0]);
  len=m*n;
  plhs[1]=mxCreateDoubleMatrix(1,1,mxREAL);
  nz=mxGetPr(plhs[1]);
  *nz=0;
  
  
  for (i=0; i<len; ++i){
      if (w[i]>lambda){
          s[i]=w[i]-lambda;          
          (*nz)++;
      }
      else if (w[i]<-lambda){
          s[i]=w[i]+lambda;          
          (*nz)++;
      }
      else {
          s[i]=0;       
      }
  }  
}
