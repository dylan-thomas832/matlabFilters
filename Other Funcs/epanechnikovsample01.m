function [xarray,ferrorvec,nitervec,iflagtermvec] = ...
                              epanechnikovsample01(n,Ns)
%
%  Copyright (c) 2009 Mark L. Psiaki.  All rights reserved.  
%
%
%  This function draws Ns samples from an n-dimensional
%  Epanechnikov kernel distribution.  It does so by
%  first sampling a Gaussian distribution and then by
%  rescaling the amplitude based on a sample from a uniform
%  distribution and the solution of an equation involving
%  the cumulative magnitude distribution of the
%  Epanechnikov kernel distribution.  This re-scaling
%  of the amplitude exploits the fact that the 
%  Epanechnikov kernel distribution and the zero-mean,
%  identity-covariance Gaussian distribution both
%  produce equi-probable unit direction vectors in n-space.
%
%  Note that this function calls both the randn and the rand
%  random number generators.  Therefore, they must have
%  been appropriately seeded.  This function also makes use
%  of the inverse cumulative distribution function for the
%  Epanechnikov kernel sample magnitude, epanechnikovmagsolve01.m.
%
%  Inputs:
%
%    n                 The dimension of the space in which the
%                      Epanechnikov kernel is defined.
%
%    Ns                The number of samples to be drawn
%                      from the Epanechnikov kernel distribution.
%
%  Outputs:
%
%    xarray            The n-by-Ns array that contains the Ns
%                      samples from the Epanechnikov kernel 
%                      distribution.  xarray(:,jj) is the jjth
%                      sample.
%
%    ferrorvec         The 1-by-Ns vector of error outputs
%                      from the inverse Epanechnikov kernel 
%                      magnitude cumulative distribution
%                      inversion solver function
%                      epanechnikovmagsolve01.m.
%
%    nitervec          The 1-by-Ns vector of iteration count 
%                      outputs from the inverse Epanechnikov  
%                      kernel magnitude cumulative distribution
%                      inversion solver function
%                      epanechnikovmagsolve01.m.
%
%    iflagtermvec      The 1-by-Ns vector of termination status 
%                      flag outputs from the inverse Epanechnikov  
%                      kernel magnitude cumulative distribution
%                      inversion solver function
%                      epanechnikovmagsolve01.m.  These all should
%                      be zero.  If any are not, then a warning
%                      will be sent to the display.
%

%
%  Sample the entries of xarray initially from an n-dimensional
%  Gaussian distribution with zero mean and identity covariance.
%
   xarray = randn(n,Ns);
%
%  Find the number of column entries of xarray that have zero
%  magnitude.  Re-sample any such column entries until all
%  have non-zero magnitude.
%
   magsqvec = (sum(xarray.^2,1))';
   idum = find(magsqvec == 0);
   ndum = size(idum,1);
   while ndum > 0
      xarraydum = randn(n,ndum);
      magsqvecdum = (sum(xarraydum.^2,1))';
      xarray(:,idum) = xarraydum;
      magsqvec(idum,1) = magsqvecdum;
      jdum = find(magsqvecdum == 0);
      ndum = size(jdum,1);
      if ndum > 0
         idum = idum(jdum,1);
      else
         idum = [];
      end
   end
%
%  Prepare a factor that would normalize the Gaussian vectors.
%
   oomagvec = ((sqrt(magsqvec)).^(-1))';
%
%  Determine the magnitudes of the final samples from
%  the Epanechnikov kernel distribution by sampling
%  a flat distribution from 0 to 1 and then using the
%  inverse cdf function for the Epanechnikov kernel 
%  distribution.
%
   probcdfmagvec = rand(1,Ns);
   rhovec = zeros(1,Ns);
   ferrorvec = zeros(1,Ns);
   nitervec = zeros(1,Ns);
   iflagtermvec = zeros(1,Ns);
   for jj = 1:Ns
      [rhojj,ferrorjj,niterjj,iflagtermjj] = ...
               epanechnikovmagsolve01(probcdfmagvec(1,jj),n);
      rhovec(1,jj) = rhojj;
      ferrorvec(1,jj) = ferrorjj;
      nitervec(1,jj) = niterjj;
      iflagtermvec(1,jj) = iflagtermjj;
   end
%
%  Rescale the magnitudes of the columns of xarray in order
%  to transform from a Gaussian distribution to an Epanechnikov 
%  kernel distribution.
   
   xarray = xarray.*(ones(n,1)*(rhovec.*oomagvec));
%
%  Display a warning if any of the magnitude calculations did
%  not terminate properly.
%
   if max(abs(iflagtermvec)) > 0
      disp(' ')
      disp('Warning in epanechnikovsample01.m: One or more')
      disp(' solution failures in the inversions of the')
      disp(' cumulative probability distribution of the')
      disp(' Epanechnikov kernel distribution''s magnitude.')
   end