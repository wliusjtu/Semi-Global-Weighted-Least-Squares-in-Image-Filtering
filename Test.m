close all
img = imread('beach.png');
img_g = double(rgb2gray(img));  % must be "double"
img = double(img);  % must be "double"
lambda = 900;
sigmaR = 2;
r = 1;
sigmaS = r;
step = r + 1;
iterNum = 2;
weightChoice = 1;  % 0 for exponential weight and 1 for fractional weight

result = SG_WLS(img, img_g, lambda, sigmaR, sigmaS, r, step, iterNum, weightChoice);

diff = img - result;
imgE = img + 5 * diff;
figure
imshow(uint8(img))
figure
imshow(uint8(result))
figure
imshow(uint8(imgE))