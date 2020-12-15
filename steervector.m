function [v,tau,phi]=steervector(teta,L,prm,f0,Fdir,tetadir)
%
% [v,tau,phi]=steervector(teta,L,prm,f0,Fdir,tetadir)
%
% teta = DOA
% L = # elements
% prm = [d1,d2,...] = distances among elements
% f0 = frequency
% Fdir,tetadir = directivity
%       
% Angle and array convention: 
%
%                                          teta(-pi/2,+pi/2)
%                                          | /
%                                          |/
%                                  o---------------o
%                                  2       |       1 (reference phase of element 1 = 0) 
%

if (nargin==3)
   f0=0;
   Fdir=[];
   tetadir=[];
end
if (nargin==4)
   Fdir=[];
   tetadir=[];
end
if (nargin==5)
   Fdir=[];
   tetadir=[];
end


c=3e8; %m
if (f0==0)
    f0=c;
end
lambda=c/f0;

v=[];
tau=[];
phi=[];

if (L>1)
   %
   if (length(prm)<(L-1))
      d=prm(1)*ones(1,L-1);
   else
      d=prm;
   end
   for k=1:length(teta)
      %
      phi=2*pi*(cumsum(d)/lambda)*sin(teta(k));
      phi=[0,phi];
      %
      if isempty(Fdir);
         sqdsa=1;
      else
         sqdsa=sqrt(interpcirc(tetadir,Fdir,teta(k)));
      end
      %
      v=[v;sqdsa*exp(1i*phi)];
      tau=[tau;phi/(2*pi*f0)];
      %
   end
else
   for k=1:length(teta)
      %
      if isempty(Fdir);
         sqdsa=1;
      else
         sqdsa=sqrt(interpcirc(tetadir,Fdir,teta(k)));
      end
      v=[v;sqdsa];
      tau=[tau;0];
   end
end

