function [fscript,dfscript_dx,dfscript_dvtil] = ...
                                fscript_cart(t,x,u,vtil,idervflag)
%
%  Copyright (c) 2002 Mark L. Psiaki.  All rights reserved.  
%
% 
%  This function defines the dynamics model of the cart and its partial
%  derivatives with respect to its inputs.  This is for use in the 
%  EKF problem for the tricycle cart.
%
%  Inputs:
%
%    t               The time at which xdot is to be known, in seconds.
%                    Note: this does not actually get used in the
%                    present function, but it needs to appear in
%                    the argument list so that c2dnonlinear.m can
%                    call it.
%
%    x               The 5x1 cart state vector at time t.
%                    x(1,1) is the heading angle in radians, x(2,1) is
%                    the east position in meters, x(3,1) is the north
%                    position in meters, x(4,1) is the steer angle
%                    in radians, and x(5,1) is the speed in meters/sec.
%
%    u               The 0x1 control vector at time t.
%
%    vtil            The 2x1 process noise disturbance vector at time t.
%                    vtil(1,1) is the white noise that drives the steer
%                    angle Markov process, and vtil(2,1) is the white noise
%                    that drives the speed Markov process.
%
%    idervflag       A flag that tells whether (idervflag = 1) or not
%                    (idervflag = 0) the partial derivatives 
%                    dfscript_dx and dfscript_dvtil must be calculated.
%                    If idervflag = 0, then these outputs will be
%                    empty arrays.
%  
%  Outputs:
%
%    fscript         The time derivative of x at time t as determined
%                    by the cart differential equations and by the
%                    Markov process differential equations.
%
%    dfscript_dx     The partial derivative of fscript with respect to
%                    x.  This is a Jacobian matrix.  It is evaluated and
%                    output only if idervflag = 1.  Otherwise, an
%                    empty array is output.
%
%    dfscript_dvtil  The partial derivative of fscript with respect to
%                    vtil.  This is a Jacobian matrix.  It is evaluated and
%                    output only if idervflag = 1.  Otherwise, an
%                    empty array is output.
%

%
%  Set up the known constants of the system.
%
   b = 0.1;
   tausteer = 0.25;
   tauspeed = 0.60;
   meanspeed = 2.1;
%
%  Evaluate the cart differential equations.
%
   fscript = zeros(5,1);
   tanbeta = tan(x(4,1));
   %%%%
   fscript(1,1) = -(x(5)*tanbeta)/b;
   cospsi = cos(x(1,1));
   sinpsi = sin(x(1,1));
   %%%%
   fscript(2,1) = x(5)*cospsi;
   %%%%
   fscript(3,1) = x(5)*sinpsi;
%
%  Evaluate the Markov differential equations.
%
   %%%%%
   fscript(4,1) = -(1/tausteer)*x(4) + vtil(1);
   %%%%%
   fscript(5,1) = -(1/tauspeed)*(x(5)-meanspeed) + vtil(2);
%
%  Calculate the partial derivative if they are needed.
%
   if idervflag == 1
      dfscript_dx = zeros(5,5);
      dfscript_dx(1,4) = -x(5)*(sec(x(4))^2)/b;
      dfscript_dx(1,5) = -tanbeta/b;
      dfscript_dx(2,1) = -x(5)*sinpsi;
      dfscript_dx(2,5) = cospsi;
      dfscript_dx(3,1) = x(5)*cospsi;
      dfscript_dx(3,5) = sinpsi;
      dfscript_dx(4,4) = -1/tausteer;
      dfscript_dx(5,5) = -1/tauspeed;
      dfscript_dvtil = zeros(5,2);
      dfscript_dvtil(4,1) = 1;
      dfscript_dvtil(5,2) = 1;
   else
      dfscript_dx = [];
      dfscript_dvtil = [];
   end