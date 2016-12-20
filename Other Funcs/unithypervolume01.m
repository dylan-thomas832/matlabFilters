function cn = unithypervolume01(n)
%
%  Copyright (c) 2009 Mark L. Psiaki.  All rights reserved.  
%
%
%  This function computes the hypervolume of the unit hypersphere
%  in n space.  Its formula has been derived based on an
%  integral that involves hyperspherical coordinates.
%
%  Inputs:
%
%    n                 The dimension of the space whose
%                      unit hypersphere's hypervolume is
%                      to be computed.
%
%  Outputs:
%
%    cn                The hypervolume of the unit hypersphere
%                      in n space.  Note: the following
%                      standard values apply:
%
%                          n     cn
%                          1     2
%                          2     pi
%                          3     4*pi/3
%                          4     (pi^2)/2.
%
%                      This is a generalization of the length
%                      of the line segment from -1 to 1 (n = 1),
%                      the area of a circle with unit radius
%                      (n = 2), the volume of a sphere with
%                      unit radius (n = 3), etc.
%

%
%  Deal with the first 2 cases using brute force.
%
   if n == 1
      cn = 2;
      return
   elseif n == 2
      cn = pi;
      return
   end
%
%  Do general case.
%
   mm = floor(n/2);
   if mm >= 2
      mmprodnumvec = 2*[2:mm] - 3;
      mmproddenvec = mmprodnumvec + 1;
      mmprodvec = mmprodnumvec./mmproddenvec;
      mmcumprodvec = cumprod(mmprodvec);
      mmprod = 2*prod(mmcumprodvec);
   else
      mmprod = 2;
   end
   ll = floor((n+1)/2);
   if ll >= 3
      llprodnumvec = 2*[3:ll] - 4;
      llproddenvec = llprodnumvec + 1;
      llprodvec = [2,(llprodnumvec./llproddenvec)];
      llcumprodvec = cumprod(llprodvec);
      llprod = prod(llcumprodvec);
   else
      llprod = 2;
   end
   cn = (1/n)*(pi^mm)*mmprod*llprod;