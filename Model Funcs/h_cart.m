function [h,dh_dx] = h_cart(x,idervflag)
%
%  Copyright (c) 2002 Mark L. Psiaki.  All rights reserved.  
%
% 
%  This function computes the radar measurements for the tricycle cart EKF
%  problem along with their partial derivatives.
%
%  Inputs:
%
%    x               The 5x1 cart state vector.
%                    x(1,1) is the heading angle in radians, x(2,1) is
%                    the east position in meters, x(3,1) is the north
%                    position in meters, x(4,1) is the steer angle
%                    in radians, and x(5,1) is the speed in meters/sec.
%
%    idervflag       A flag that tells whether (idervflag = 1) or not
%                    (idervflag = 0) the partial derivative Jacobian matrix 
%                    dh_dx must be calculated.  If idervflag = 0, then 
%                    this output will be an empty array.
%  
%  Outputs:
%
%    h               = [rhoa;rhob], the 2x1 radar output vector.  rhoa
%                    is the measured distance from radar a to the cart in
%                    meters.  rhob is the measured distance from radar b to 
%                    the cart in meters.  These distances are to the 
%                    mid point between the cart's two rear wheels.
%
%    dh_dx           The partial derivative of h with respect to
%                    x.  This is a Jacobian matrix.  It is evaluated and
%                    output only if idervflag = 1.  Otherwise, an
%                    empty array is output.
%

%
%  Set up problem constants.
%
   lradara = -1;
   lradarb = 1;
%
%  Determine the cart position relative to the two radar.
%
   dely1a = lradara - x(2,1);
   dely1b = lradarb - x(2,1);
   dely2 = x(3,1);
%
%  Compute the h output.
%
   dely1asq = dely1a^2;
   dely1bsq = dely1b^2;
   dely2sq = dely2^2;
   h = zeros(2,1);
   h(1,1) = sqrt(dely1asq + dely2sq);
   h(2,1) = sqrt(dely1bsq + dely2sq);
%
%  Return if neither first derivatives nor second derivatives
%  need to be calculated.
%
   if idervflag == 0
      dh_dx = [];
      return
   end
%
%  Calculate the first derivatives.  Use analytic formulas.
%
   ddely1adx = -[0,1,0,0,0];
   ddely1bdx = -[0,1,0,0,0];
   ddely2dx = [0,0,1,0,0];
   dh_dx = zeros(2,5);
   one_over_rhoa = 1/h(1,1);
   dh_dx(1,:) = one_over_rhoa*(dely1a*ddely1adx + dely2*ddely2dx);
%
   one_over_rhob = 1/h(2,1);
   dh_dx(2,:) = one_over_rhob*(dely1b*ddely1bdx + dely2*ddely2dx);