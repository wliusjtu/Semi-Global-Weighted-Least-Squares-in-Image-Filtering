%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    This is the released code for the following paper:
%
%    "Semi-global weighted least squares in image filtering.", Wei Liu, Xiaogang Chen, 
%     Chuanhua Shen, Zhi Liu, and Jie Yang. In ICCV 2017.
%  
%   The code and the algorithm are for non-comercial use only.

%  ---------------------- Input------------------------
%  img:                  input image to be filtered, can be gray image or RGB color image
%  img_g:              guidance image, can be gray image or RGB color image
%  lambda:            \lambda in Eq.(1), control smoothing strength
%  sigmaR:            range sigma in Eq. (2)
%  sigmaS:            spatial sigma in Eq. (2)
%  r:                      neighborhood radius in Eq. (1)
%  step:                 step size between each SG_WLS described in Sec. 2.3
%  iterNum:            iteration number of the SG_WLS 
%  weightChoice:    choice of the weight in Eq. (2), 0 for exponential weight, 1 for fractional weight

%  ---------------------- Output------------------------
%  result:             smoothed image

function result = SG_WLS(img, img_g, lambda, sigmaR, sigmaS, r, step, iterNum, weightChoice)

img = double(img);  
img_g = double(img_g);  

if max(img(:)) <= 1.0
    error('input image should be in range [0, 255].\n');
end

if max(img_g(:)) <= 1.0
    error('guidance image should be in range [0, 255].\n');
end

% % naive implementation, more time consuming
% result = mexSG_WLS_naive(img, img_g, lambda, sigmaR, sigmaS, r, step, iterNum, weightChoice);

% optimized code, code-level optimization, produce the same results, but
% faster
result = mexSG_WLS_opt(img, img_g, lambda, sigmaR, sigmaS, r, step, iterNum, weightChoice);


