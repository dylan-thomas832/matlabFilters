function [rho,ferror,niter,iflagterm] = epanechnikovmagsolve01(x,n)
%
%  Copyright (c) 2009 Mark L. Psiaki.  All rights reserved.  
%
%
%  This function solves the equation
%
%             x = 0.5*((n+2)*(rho^n) - n*(rho^(n+2)))
%
%  for an x in the range 0 <= x <= 1 to produce a rho in 
%  this range.  n must be an integer.
%
%  This function is a utility for use in sampling a Epanechnikov
%  distribution.  If x is sampled from a uniform distribution
%  on the interval from 0 to 1, then rho will be the magnitude
%  of a sample from an Epanechnikov kernel distribution in
%  a space of dimension n.
%
%  This function solves for rho by using a guarded secant method
%  of numerical equation solving.
%
%  Inputs:
%
%    x                 The input in the range 0 <= x <= 1 that defines
%                      the cumulative probability of the sample
%                      magnitude at a sample of an Epanechnikov
%                      kernel.  If it exceeds this range, then
%                      it will be clipped, and a warning message
%                      will be sent to the display.
%
%    n                 The dimension of the space in which the
%                      Epanechnikov kernel is defined.
%
%  Outputs:
%
%    rho               The magnitude of the sample of the 
%                      Epanechnikov kernel in n-space.
%
%    ferror            = 0.5*((n+2)*(rho^n) - n*(rho^(n+2))) - x,
%                      the remaining error in the equation to
%                      be solved.
%
%    niter             The number of secant iterations needed in
%                      order to solve for rho.
%
%    iflagterm         A termination status flag that tells
%                      whether (iflagterm == 0) or not
%                      (iflagterm == 1) the algorithm converged
%                      in 60 secant iterations.
%

%
%  Clip x if needed
%
   x_act = x;
   if x_act > 1
      x_act = 1;
      disp(' ');
      disp('Warning in epanechnikovmagsolve01.m: x entered above 1.')
      disp(' It has been changed to equal 1.')
      disp(' ');
   elseif x_act < 0
      x_act = 0;
      disp(' ');
      disp('Warning in epanechnikovmagsolve01.m: x entered below 0.')
      disp(' It has been changed to equal 0.')
      disp(' ');
   end
%
%  Do the obvious solutions first at the endpoints.
%
   if x_act == 1
      rho = 1;
      ferror = 1 - x;
      niter = 1;
      iflagterm = 0;
      return
   elseif x_act == 0
      rho = 0;
      ferror = 0 - x;
      niter = 1;
      iflagterm = 0;
      return
   end
%
%  Set up for the secant method.  Note that f_a and f_b
%  must always have opposite signs.
%
   rho_a = 0;
   drho_ba = 1;
   drhovec = [0;1];
   fvec = [0;1] - x_act;
   f_a = fvec(1,1);
   f_b = fvec(2,1);
   absf_a = abs(f_a);
   absf_b = abs(f_b);
   if absf_a < absf_b
      iflagba_current = 0;
   else
      iflagba_current = 1;
   end
   absfvec = [absf_a;absf_b];
   [absfvec,idum] = sort(absfvec);
   fvec = fvec(idum,1);
   drhovec = drhovec(idum,1);
   niter = 2;
   iflagterm = 0;
%
   np2 = n+2;
%
%  This is the main loop that executes one iteration of
%  the secant calculations for each iteration.  It is a
%  guarded secant method that uses bisection if that
%  would produce a greater decrease in the length
%  of the uncertainty interval from rho_a to rho_b.
%
   while drho_ba > (2*eps)
      if niter >= 60
         iflagterm = 1;
         break
      end
      niter = niter + 1;
      drho_new = drhovec(1,1) + (drhovec(2,1) - drhovec(1,1))*...
                 ((-fvec(1,1))/(fvec(2,1) - fvec(1,1)));
      drho_bao2 = 0.5*drho_ba;
      if (drho_new <= 0) | (drho_ba <= drho_new)
         drho_new = drho_bao2;
      else
         if iflagba_current == 0
            if drho_new > drho_bao2
               drho_new = drho_bao2;
            end
         else
            if drho_new < drho_bao2
               drho_new = drho_bao2;
            end
         end
      end
      rho_new = drho_new + rho_a;
      f_new = 0.5*(np2*(rho_new^n) - n*(rho_new^np2)) - x_act;
      if f_new == 0
         rho_a = rho_new;
         drho_ba = 0;
         break
      end
      absf_new = abs(f_new);
%      
      fvec = [fvec;f_new];
      absfvec = [absfvec;absf_new];
      drhovec = [drhovec;drho_new];
      [absfvec,idum] = sort(absfvec);
      absfvec = absfvec(1:2,1);
      idum = idum(1:2,1);
      fvec = fvec(idum,1);
      drhovec = drhovec(idum,1);
%
      if f_new*f_b > 0
         drho_ba = drho_new;
         f_b = f_new;
         absf_b = absf_new;
         if absf_a < absf_b
            iflagba_current = 0;
         else
            iflagba_current = 1;
         end
      else
         rho_a = rho_new;
         drho_ba = drho_ba - drho_new;
         drho_ba = max([drho_ba;0]);
         drhovec = drhovec - drho_new;
         f_a = f_new;
         absf_a = absf_new;
         if absf_a < absf_b
            iflagba_current = 0;
         else
            iflagba_current = 1;
         end
      end
   end
%
%  Assign the outputs.
%
   rho = rho_a + 0.5*drho_ba;
   ferror = 0.5*((np2)*(rho^n) - n*(rho^(np2))) - x;