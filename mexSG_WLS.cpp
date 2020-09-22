#include <math.h>
#include <time.h>
#include "mex.h"

// size of input image, guidance image and SG_WLS
int rowNum, colNum, rc;
int chaNumImg, chaNumGuide;
int r, sysLen_row, sysLen_col; 

// memory management
double *** memAllocDouble3(int row, int col, int cha);
double ** memAllocDouble2(int row, int col);
void memFreeDouble3(double ***p);
void memFreeDouble2(double **p);

// functions
void expLUT(double *LUT, double sigma, int chaNum, int len);
void fracLUT(double *LUT, double sigma, int chaNum, int len);
void img2vector_col(double ***img, double ***imgGuide, double **vectorImg, double **vectorGuide, int col);
void img2vector_row(double ***img, double ***imgGuide, double **vectorImg, double **vectorGuide, int row);
void vector2img_col(double ***img, double **vector, double **count, int col);
void vector2img_row(double ***img, double **vector, double **count, int row);
void getLaplacian(double **vectorGuide, double *a, double **b, double **c, double lambda, double *rangeLUT, double *spatialLUT, int sysLen);
void pointDiv(double ***imgInter, double ***imgFiltered, double **count);
void valueSet2(double **input, double value);
void valueSet3(double ***input, double value, int chaNum);

// solvers
void solverForRadius1(double *a, double **b, double **c, double *alpha, double **gamma, double **beta, double **F, double **Y, double **X, int sysLen);
void solverForRadius2(double *a, double **b, double **c, double *alpha, double **gamma, double **beta, double **F, double **Y, double **X, int sysLen);
void solverForRadiusLargerThan2(double *a, double **b, double **c, double *alpha, double **gamma, double **beta, double **F, double **Y, double **X, int sysLen);

// main function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    //////  Input images ////////
    double *img = (double*)mxGetData(prhs[0]), *guidance = (double*)mxGetData(prhs[1]);  // input image to be filtered and the guidance image
    
    if((mxGetDimensions(prhs[0])[0] != mxGetDimensions(prhs[1])[0]) || (mxGetDimensions(prhs[0])[1] != mxGetDimensions(prhs[1])[1]))
        mexErrMsgTxt("The input image and the guidance image should be of the same size.");
    
    // Get row, column and channel number
    rowNum = mxGetDimensions(prhs[0])[0];  // row number 
    colNum = mxGetDimensions(prhs[0])[1];  // column number
    rc = rowNum * colNum;
    
    if(mxGetNumberOfDimensions(prhs[0]) == 2)  // channel number of image to be filtered
        chaNumImg = 1;  // single channel image
    else
        chaNumImg = 3;  // rgb image
    
    if(mxGetNumberOfDimensions(prhs[1]) == 2)  // channel number of guidance image
        chaNumGuide = 1;  // single channel image
    else 
        chaNumGuide = 3;  // rgb image
    
    //////// SG_WLS parameters  ///////////////
    double lambda = mxGetScalar(prhs[2]); // lambda of the SG_WLS
    double sigmaR = mxGetScalar(prhs[3]); // range sigma for the guidance weight
    double sigmaS = mxGetScalar(prhs[4]); // spatial sigma for the guidance weight
    r = (int)mxGetScalar(prhs[5]); // raius of the neighborhood
    int step = (int)mxGetScalar(prhs[6]);  // the step size between each SG_WLS
    int iterNum = (int)mxGetScalar(prhs[7]);  // number of iterations
    int weightChoice = (int)mxGetScalar(prhs[8]);  // 0 for exp weight, 1 for frac weight
    
    // Transfer the input image and the guidance image into 3-dimention arrays
    double ***imgFiltered = memAllocDouble3(rowNum, colNum, chaNumImg);  // initialize the filtered image with the input image
    for(int i=0; i<rowNum; i++)
    {
        for(int j=0; j<colNum; j++)
        {
            for(int k=0; k<chaNumImg; k++)
                imgFiltered[i][j][k] = img[k * rc + j *rowNum + i];
        }
    }
    
    double ***imgGuide = memAllocDouble3(rowNum, colNum, chaNumGuide);
    for(int i=0; i<rowNum; i++)
    {
        for(int j=0; j<colNum; j++)
        {
            for(int k=0; k<chaNumGuide; k++)
                imgGuide[i][j][k] = guidance[k * rc + j * rowNum + i];
        }
    }
    
    // Intermediate output
    double *** imgInter = memAllocDouble3(rowNum, colNum, chaNumImg);
    
    // Output
    plhs[0] = mxDuplicateArray(prhs[0]); // the output has the same size as the input image
    double *imgResult = (double*) mxGetData(plhs[0]);
    
    // SG-WLS related variables
    sysLen_row = colNum * (2 * r + 1);  // the size of the SG-WLS along the row direction
    sysLen_col = rowNum * (2 * r + 1);  // the size of the SG-WLS along the column direction
    
    double *a_row = (double*) mxGetData(mxCreateDoubleMatrix(sysLen_row, 1, mxREAL));
    double **b_row = memAllocDouble2(sysLen_row, r);
    double **c_row = memAllocDouble2(sysLen_row, r);
    
    double *a_col = (double*) mxGetData(mxCreateDoubleMatrix(sysLen_col, 1, mxREAL));
    double **b_col = memAllocDouble2(sysLen_col, r);
    double **c_col = memAllocDouble2(sysLen_col, r);
    
    double *alpha_row = (double*) mxGetData(mxCreateDoubleMatrix(sysLen_row, 1, mxREAL));
    double **gamma_row = memAllocDouble2(sysLen_row, r);
    double **beta_row = memAllocDouble2(sysLen_row, r);
    
    double *alpha_col = (double*) mxGetData(mxCreateDoubleMatrix(sysLen_col, 1, mxREAL));
    double **gamma_col = memAllocDouble2(sysLen_col, r);
    double **beta_col = memAllocDouble2(sysLen_col, r);
    
    double **vectorImg_row = memAllocDouble2(sysLen_row, chaNumImg);
    double **vectorGuide_row = memAllocDouble2(sysLen_row, chaNumGuide + 2);  // RGB pixel values + x/y coordinates for spatial weight computation
    double **vectorInter_row = memAllocDouble2(sysLen_row, chaNumImg);  // intermediate variables 'F' in solving P*Y=F, Q*U=F
    double **vectorFiltered_row = memAllocDouble2(sysLen_row, chaNumImg);  // smoothed output 'U' in solving P*Y=F, Q*U=F
    
    double **vectorImg_col = memAllocDouble2(sysLen_col, chaNumImg);
    double **vectorGuide_col = memAllocDouble2(sysLen_col, chaNumGuide + 2); 
    double **vectorInter_col = memAllocDouble2(sysLen_col, chaNumImg);
    double **vectorFiltered_col = memAllocDouble2(sysLen_col, chaNumImg);
    
    // weight lookup table
    int maxRange = 255 + 10;
    maxRange = chaNumGuide * maxRange * maxRange;
    int maxSpatial = 2 * (2 * r + 1) * (2 * r + 1);
    double *rangeLUT = (double *)mxGetData(mxCreateDoubleMatrix(maxRange, 1, mxREAL));
    double *spatialLUT = (double *)mxGetData(mxCreateDoubleMatrix(maxSpatial, 1, mxREAL));
    
    if(weightChoice == 0)
    {
        expLUT(rangeLUT, sigmaR, chaNumGuide, maxRange);
        expLUT(spatialLUT, sigmaS, 1, maxSpatial);
    }
    else if(weightChoice ==1)
    {
        fracLUT(rangeLUT, sigmaR, chaNumGuide, maxRange);
        fracLUT(spatialLUT, sigmaS, 1, maxSpatial);
    }
    else
        mexErrMsgTxt("Weight choice should be 0 (exponential) or 1 (fractional)\n.");
    
    // accumulate the count of the filtered value at the same location
    double **count = memAllocDouble2(rowNum, colNum);
    
    // maximum row/column number 
    int maxIterColNum = r + 1 + ((colNum - r - r - 1)/step)*step;
    int maxIterRowNum = r + 1 + ((rowNum - r - r - 1)/step)*step;
    int col, row, iter;
    
    // do filtering
    if(r == 1)  // radius is 1
    {
        clock_t tStart = clock(); // time measurement;
        
        for(iter=0; iter<iterNum; iter++)
        {
            /////////////// column direction //////////////
            valueSet2(count, 0.0);
            valueSet3(imgInter, 0.0, chaNumImg);
            
            for(col=r; col<maxIterColNum; col+=step)
            {
                img2vector_col(imgFiltered, imgGuide, vectorImg_col, vectorGuide_col, col);
                getLaplacian(vectorGuide_col, a_col, b_col, c_col, lambda, rangeLUT, spatialLUT, sysLen_col);
                solverForRadius1(a_col, b_col, c_col, alpha_col, gamma_col, beta_col, vectorImg_col, vectorInter_col, vectorFiltered_col, sysLen_col);
                vector2img_col(imgInter, vectorFiltered_col, count, col);
            }
            
            // the last 2 * r + 1 columns
            col = colNum - 1 - r;
            img2vector_col(imgFiltered, imgGuide, vectorImg_col, vectorGuide_col, col);
            getLaplacian(vectorGuide_col, a_col, b_col, c_col, lambda, rangeLUT, spatialLUT, sysLen_col);
            solverForRadius1(a_col, b_col, c_col, alpha_col, gamma_col, beta_col, vectorImg_col, vectorInter_col, vectorFiltered_col, sysLen_col);
            vector2img_col(imgInter, vectorFiltered_col, count, col);
            
            pointDiv(imgInter, imgFiltered, count);
            
            ///////////////////// row direction //////////////////////
            valueSet2(count, 0.0);
            valueSet3(imgInter, 0.0, chaNumImg);
            
            for(row=r; row<maxIterRowNum; row+=step)
            {
                img2vector_row(imgFiltered, imgGuide, vectorImg_row, vectorGuide_row, row);
                getLaplacian(vectorGuide_row, a_row, b_row, c_row, lambda, rangeLUT, spatialLUT, sysLen_row);
                solverForRadius1(a_row, b_row, c_row, alpha_row, gamma_row, beta_row, vectorImg_row, vectorInter_row, vectorFiltered_row, sysLen_row);
                vector2img_row(imgInter, vectorFiltered_row, count, row);
            }
            
            // the last 2 * r + 1 rows
            row = rowNum - 1 - r;
            img2vector_row(imgFiltered, imgGuide, vectorImg_row, vectorGuide_row, row);
            getLaplacian(vectorGuide_row, a_row, b_row, c_row, lambda, rangeLUT, spatialLUT, sysLen_row);
            solverForRadius1(a_row, b_row, c_row, alpha_row, gamma_row, beta_row, vectorImg_row, vectorInter_row, vectorFiltered_row, sysLen_row);
            vector2img_row(imgInter, vectorFiltered_row, count, row);
            
            pointDiv(imgInter, imgFiltered, count);
        }
        
        mexPrintf("Elapsed time is %f seconds.\n", double(clock() - tStart)/CLOCKS_PER_SEC);
    }
    //////////////////////////////
    else if(r == 2)  // radius is 2
    {
        clock_t tStart = clock(); // time measurement;
        
        for(iter=0; iter<iterNum; iter++)
        {
            /////////////// column direction //////////////
            valueSet2(count, 0.0);
            valueSet3(imgInter, 0.0, chaNumImg);
            
            for(col=r; col<maxIterColNum; col+=step)
            {
                img2vector_col(imgFiltered, imgGuide, vectorImg_col, vectorGuide_col, col);
                getLaplacian(vectorGuide_col, a_col, b_col, c_col, lambda, rangeLUT, spatialLUT, sysLen_col);
                solverForRadius2(a_col, b_col, c_col, alpha_col, gamma_col, beta_col, vectorImg_col, vectorInter_col, vectorFiltered_col, sysLen_col);
                vector2img_col(imgInter, vectorFiltered_col, count, col);
            }
            
            // the last 2 * r + 1 columns
            col = colNum - 1 - r;
            img2vector_col(imgFiltered, imgGuide, vectorImg_col, vectorGuide_col, col);
            getLaplacian(vectorGuide_col, a_col, b_col, c_col, lambda, rangeLUT, spatialLUT, sysLen_col);
            solverForRadius2(a_col, b_col, c_col, alpha_col, gamma_col, beta_col, vectorImg_col, vectorInter_col, vectorFiltered_col, sysLen_col);
            vector2img_col(imgInter, vectorFiltered_col, count, col);
            
            pointDiv(imgInter, imgFiltered, count);
            
            ///////////////////// row direction //////////////////////
            valueSet2(count, 0.0);
            valueSet3(imgInter, 0.0, chaNumImg);
            
            for(row=r; row<maxIterRowNum; row+=step)
            {
                img2vector_row(imgFiltered, imgGuide, vectorImg_row, vectorGuide_row, row);
                getLaplacian(vectorGuide_row, a_row, b_row, c_row, lambda, rangeLUT, spatialLUT, sysLen_row);
                solverForRadius2(a_row, b_row, c_row, alpha_row, gamma_row, beta_row, vectorImg_row, vectorInter_row, vectorFiltered_row, sysLen_row);
                vector2img_row(imgInter, vectorFiltered_row, count, row);
            }
            
            // the last 2 * r + 1 rows
            row = rowNum - 1 - r;
            img2vector_row(imgFiltered, imgGuide, vectorImg_row, vectorGuide_row, row);
            getLaplacian(vectorGuide_row, a_row, b_row, c_row, lambda, rangeLUT, spatialLUT, sysLen_row);
            solverForRadius2(a_row, b_row, c_row, alpha_row, gamma_row, beta_row, vectorImg_row, vectorInter_row, vectorFiltered_row, sysLen_row);
            vector2img_row(imgInter, vectorFiltered_row, count, row);
            
            pointDiv(imgInter, imgFiltered, count);  
        }
        
        mexPrintf("Elapsed time is %f seconds.\n", double(clock() - tStart)/CLOCKS_PER_SEC);
        
    }
    ///////////////////////////////
    else  // radius larger than 2
    {
        clock_t tStart = clock(); // time measurement;
        
        for(iter=0; iter<iterNum; iter++)
        {
            /////////////// column direction //////////////
            valueSet2(count, 0.0);
            valueSet3(imgInter, 0.0, chaNumImg);
            
            for(col=r; col<maxIterColNum; col+=step)
            {
                img2vector_col(imgFiltered, imgGuide, vectorImg_col, vectorGuide_col, col);
                getLaplacian(vectorGuide_col, a_col, b_col, c_col, lambda, rangeLUT, spatialLUT, sysLen_col);
                solverForRadiusLargerThan2(a_col, b_col, c_col, alpha_col, gamma_col, beta_col, vectorImg_col, vectorInter_col, vectorFiltered_col, sysLen_col);
                vector2img_col(imgInter, vectorFiltered_col, count, col);
            }
            
            // the last 2 * r + 1 columns
            col = colNum - 1 - r;
            img2vector_col(imgFiltered, imgGuide, vectorImg_col, vectorGuide_col, col);
            getLaplacian(vectorGuide_col, a_col, b_col, c_col, lambda, rangeLUT, spatialLUT, sysLen_col);
            solverForRadiusLargerThan2(a_col, b_col, c_col, alpha_col, gamma_col, beta_col, vectorImg_col, vectorInter_col, vectorFiltered_col, sysLen_col);
            vector2img_col(imgInter, vectorFiltered_col, count, col);
            
            pointDiv(imgInter, imgFiltered, count);
            
            ///////////////////// row direction //////////////////////
            valueSet2(count, 0.0);
            valueSet3(imgInter, 0.0, chaNumImg);
            
            for(row=r; row<maxIterRowNum; row+=step)
            {
                img2vector_row(imgFiltered, imgGuide, vectorImg_row, vectorGuide_row, row);
                getLaplacian(vectorGuide_row, a_row, b_row, c_row, lambda, rangeLUT, spatialLUT, sysLen_row);
                solverForRadiusLargerThan2(a_row, b_row, c_row, alpha_row, gamma_row, beta_row, vectorImg_row, vectorInter_row, vectorFiltered_row, sysLen_row);
                vector2img_row(imgInter, vectorFiltered_row, count, row);
            }
            
            // the last 2 * r + 1 rows
            row = rowNum - 1 - r;
            img2vector_row(imgFiltered, imgGuide, vectorImg_row, vectorGuide_row, row);
            getLaplacian(vectorGuide_row, a_row, b_row, c_row, lambda, rangeLUT, spatialLUT, sysLen_row);
            solverForRadiusLargerThan2(a_row, b_row, c_row, alpha_row, gamma_row, beta_row, vectorImg_row, vectorInter_row, vectorFiltered_row, sysLen_row);
            vector2img_row(imgInter, vectorFiltered_row, count, row);
            
            pointDiv(imgInter, imgFiltered, count);  
        }
        
        mexPrintf("Elapsed time is %f seconds.\n", double(clock() - tStart)/CLOCKS_PER_SEC);
    }
    
    // transfer to the output
    for(int i=0; i<rowNum; i++)
    {
        for(int j=0; j<colNum; j++)
        {
            for(int k=0; k<chaNumImg; k++)
                imgResult[k * rc + j * rowNum + i] = imgFiltered[i][j][k];
        }
    }

}


//============ Functions =============//

double *** memAllocDouble3(int row, int col, int cha)
{   
    // allocate the memory for a 3-dimension array which can be indexed as pp[row][col][cha]
    
	int padding=10;
	double *a, **p, ***pp;
    
	a=(double*) malloc(sizeof(double) * (row * col * cha + padding));
	if(a==NULL) {mexErrMsgTxt("memAllocDouble: Memory allocate failure.\n"); }
	p=(double**) malloc(sizeof(double*) * row * col);
	pp=(double***) malloc(sizeof(double**) * row);
    
    int cc = col * cha;
	int i, j;
    
	for(i=0; i<row; i++) 
    {
		for(j=0; j<col; j++) 
			p[i * col + j] = &a[i * cc + j * cha];      
    }
    
	for(i=0; i<row; i++) 
		pp[i] = &p[i* col];
    
	return(pp);
}


///////////////////////////////
void memFreeDouble3(double ***p)
{
    // free the memory of an allocated 3-dimension array
    
	if(p!=NULL)
	{
		free(p[0][0]);
		free(p[0]);
		free(p);
		p=NULL;
	}
}


////////////////////////////////
double** memAllocDouble2(int row, int col)
{
    // allocate the memory for a 2-dimension array which can be indexed as p[row][col]
    
	int padding=10;
	double *a, **p;
    
	a=(double*) malloc(sizeof(double) * (row * col + padding));
	if(a==NULL) {mexErrMsgTxt("memAllocDouble: Memory allocate failure.\n"); }
	p=(double**) malloc(sizeof(double*) * row);
    
	for(int i=0; i<row; i++) p[i] = &a[i * col];
    
	return(p);
}


//////////////////////////////////
void memFreeDouble2(double **p)
{
    // free the memory of an allocated 2-demision array
    
	if(p!=NULL)
	{
		free(p[0]);
		free(p);
		p=NULL;
	}
}


////////////////////////////////////////////////////
void expLUT(double *LUT, double sigma, int chaNum, int len)
{
    for(int i=0; i<len; i++)   LUT[i] = exp(double(-i) / (double(chaNum) * 2 * sigma * sigma));
    
}


///////////////////////////////////////////////////
void fracLUT(double *LUT, double sigma, int chaNum, int len)
{
    for(int i=0; i<len; i++)   LUT[i] = 1 / (pow(sqrt(double(i) / double(chaNum)), sigma) + 0.00001);
    
}


//////////////////////////////////////////////////
void img2vector_col(double ***img, double ***imgGuide, double **vectorImg, double **vectorGuide, int col)
{
    // transform the block centered along column 'col' with radius 'r' into a vector whose length is 'rowNum * (2 * r + 1)'.
    
    int i, j, k, colSlide;
    int pr=0, dir=1;
    
    for(i=0; i<rowNum; i++)
    {
        for(j=-r; j<=r; j++)
        {
            colSlide = col + (j * dir);   // the column number of the current sliding position
            
            // for the image to be filtered
            for(k=0; k<chaNumImg; k++)
                vectorImg[pr][k] = img[i][colSlide][k];
            
            // for the guidance image
            for(k=0; k<chaNumGuide; k++)
                vectorGuide[pr][k] = imgGuide[i][colSlide][k];
            
            // store the x/y pixel coordinate in the image coordinate system, to compute the spatial weight
            vectorGuide[pr][chaNumGuide] = i;
            vectorGuide[pr][chaNumGuide + 1] = colSlide;
            
            pr++; // move the pointer forward    
        }
        
        dir *= -1;  // change the extraction sliding direction between left-to-right and right-to-left in each row of the image
    }
}


//////////////////////////////////////////////////
void img2vector_row(double ***img, double ***imgGuide, double **vectorImg, double **vectorGuide, int row)
{
    // transform the block centered along row 'row' with radius 'r' into a vector whose length is 'colNum * (2 * r + 1)'.
    
    int i, j, k, rowSlide;
    int pr=0, dir=1;
    
    for(j=0; j<colNum; j++)
    {
        for(i=-r; i<=r; i++)
        {
            rowSlide = row + (i * dir);   // the row number of the current sliding position
            
            // for the image to be filtered
            for(k=0; k<chaNumImg; k++)
                vectorImg[pr][k] = img[rowSlide][j][k];
            
            // for the guidance image
            for(k=0; k<chaNumGuide; k++)
                vectorGuide[pr][k] = imgGuide[rowSlide][j][k];
            
            // store the pixel coordinate in the image coordinate system, to compute the spatial weight
            vectorGuide[pr][chaNumGuide] = rowSlide;
            vectorGuide[pr][chaNumGuide + 1] = j;
            
            pr++; // move the pointer forward    
        }
        
        dir *= -1;  // change the extraction sliding direction between left-to-right and right-to-left in each row of the image
    }
}

//////////////////////////////////////////////////
void vector2img_col(double ***img, double **vector, double **count, int col)
{
    // the inverse transform of 'img2vector_column'
    
    int i, j, k, colSlide;
    int pr=0, dir=1;
    
    for(i=0; i<rowNum; i++)
    {
        for(j=-r; j<=r; j++)
        {
            colSlide = col + (j * dir);   // the column number of the current sliding position
            
            for(k=0; k<chaNumImg; k++)
                img[i][colSlide][k] += vector[pr][k];
            
            // for counter
            count[i][colSlide] += 1.0;
            
            pr++; // move the pointer forward    
        }
        
        dir *= -1;  // change the extraction sliding direction between left-to-right and right-to-left in each row of the image
    }
}


//////////////////////////////////////////////////
void vector2img_row(double ***img, double **vector, double **count, int row)
{
    // inverse transform of the function 'img2vector_row'
    
    int i, j, k, rowSlide;
    int pr=0, dir=1;
    
    for(j=0; j<colNum; j++)
    {
        for(i=-r; i<=r; i++)
        {
            rowSlide = row + (i * dir);   // the row number of the current sliding position
            
            // for the image to be filtered
            for(k=0; k<chaNumImg; k++)
                img[rowSlide][j][k] += vector[pr][k];
            
            // for counter
            count[rowSlide][j] += 1.0;
            
            pr++; // move the pointer forward    
        }
        
        dir *= -1;  // change the extraction sliding direction between left-to-right and right-to-left in each row of the image
    }
}


////////////////////
void getLaplacian(double **vectorGuide, double *a, double **b, double **c, double lambda, double *rangeLUT, double *spatialLUT, int sysLen)
{
    double diffR, temp, diffS, weightR, weightS;
    int i, j, k, n=sysLen;
    
    // compute b and c first
    for(i=0; i<n-r; i++)
    {
        for(j=1; j<=r; j++)
        {
            // range weight
            diffR = 0;
            for(k=0; k<chaNumGuide; k++)
            {
                temp = vectorGuide[i][k]- vectorGuide[i + j][k];
                diffR += temp * temp;
            }
            weightR = rangeLUT[(int)diffR];
            
            // spatial weight
            diffS = 0;
            for(k=chaNumGuide; k<chaNumGuide + 2; k++)
            {
                temp = vectorGuide[i][k] - vectorGuide[i + j][k];
                diffS += temp * temp; 
            }
            weightS = spatialLUT[(int)diffS];
            
            b[i][j - 1] = -lambda * weightR * weightS; 
            c[i][j - 1] = b[i][j - 1];
        }
        
    }
    ///////////////////
    for(i=n - r; i<n - 1; i++)
    {
        for(j=1; j<=n - 1 - i; j++)
        {
            // range weight
            diffR = 0;
            for(k=0; k<chaNumGuide; k++)
            {
                temp = vectorGuide[i][k] - vectorGuide[i + j][k];
                diffR += temp * temp;
            }
            weightR = rangeLUT[(int)diffR];
            
            // spatial weight
            diffS = 0;
            for(k=chaNumGuide; k<chaNumGuide + 2; k++)
            {
                temp = vectorGuide[i][k] - vectorGuide[i + j][k];
                diffS += temp*temp;
            }
            weightS = spatialLUT[(int)diffS];
            
            b[i][j - 1] = -lambda * weightR * weightS; 
            c[i][j - 1] = b[i][j - 1];
        }
      
    }
    
    // compute a with the computed b and c
    temp = 0;
    for(j=1; j<=r; j++)
    {
        temp = temp - b[0][j - 1];
    }
    a[0] = 1 + temp;
    
    ///////////////
    for(i=1; i<r+1; i++)
    {
        temp = 0;
        for(j=1; j<=i; j++)
            temp = temp - c[i - j][j - 1]; 
        
        for(j=1; j<=r; j++)
            temp = temp - b[i][j - 1];
        
        a[i] = 1 + temp;
    }
    
    //////////
    for(i=r + 1; i<n - r; i++)
    {
        temp = 0;
        for(j=1; j<=r; j++)
            temp = temp - b[i][j - 1] - c[i - j][j - 1];
        
        a[i] = 1 + temp;
    }
    
    ////////////////
    for(i=n - r; i<n - 1; i++)
    {
        temp = 0;
        for(j=1; j<=r; j++)
            temp = temp - c[i - j][j - 1]; 
        
        for(j=1; j<=n - 1 - i; j++)
            temp = temp - b[i][j - 1];
        
        a[i] = 1 + temp;   
    }
    
    ////////////////////////
    i = n - 1;
    temp = 0;
    for(j=1; j<=r; j++)
        temp = temp - c[i - j][j - 1]; 
    
    a[n - 1] = 1 + temp;
    
}


/////////////////////////////////////
void solverForRadius1(double *a, double **b, double **c, double *alpha, double **gamma, double **beta, double **F, double **Y, double **X, int sysLen)
{
    int k, cha, n=sysLen;
    
    // LU decomposition
    alpha[0] = a[0];
    gamma[0][0] = c[0][0];
    beta[0][0] = b[0][0] / alpha[0];
    
    for(k=1; k<n - 1; k++)
    {
        gamma[k][0] = c[k][0];
        alpha[k] = a[k] - gamma[k - 1][0] * beta[k - 1][0];
        beta[k][0] = b[k][0] / alpha[k];
    }
    alpha[n - 1] = a[n - 1] - gamma[n - 2][0] * beta[n - 2][0];
    
    for(cha=0; cha<chaNumImg; cha++)
    {
        // L*Y = F 
        Y[0][cha] = F[0][cha] / alpha[0];
        for (k=1; k<n; k++)
            Y[k][cha] = (F[k][cha] - gamma[k - 1][0] * Y[k - 1][cha]) / alpha[k];

        // U*X = Y;
       X[n - 1][cha] = Y[n - 1][cha];
       for(k=n-2; k>=0; k--)
           X[k][cha] = Y[k][cha] - beta[k][0] * X[k + 1][cha];
    }
   
}


//////////////////////////////
void solverForRadius2(double *a, double **b, double **c, double *alpha, double **gamma, double **beta, double **F, double **Y, double **X, int sysLen)
{
    int i, k, t, n=sysLen;
    double tempAlpha, tempGamma, tempBeta;
    
    //////////////////// LU decomposition ///////////////////
    k = 0; 
    alpha[0] = a[0];
    for(i=0; i<r; i++)
    {
        gamma[0][i] = c[0][i];
        beta[0][i] = b[0][i] / alpha[0];
    }
    
    //////////
    k = 1;
    alpha[1] = a[1] - gamma[0][0] * beta[0][0];
    gamma[1][0] = c[1][0] - gamma[0][1] * beta[0][0];
    beta[1][0] = (b[1][0] - gamma[0][0] * beta[0][1]) / alpha[1];
    
    gamma[1][r - 1] = c[1][r - 1];
    beta[1][r - 1] = b[1][r - 1] / alpha[r - 1];
    
    /////////// 
    for(k=2; k<n-r; k++)
    {
        // alpha 
        tempAlpha = 0;
        for(t=1; t<=r; t++)
            tempAlpha += gamma[k - t][t - 1] * beta[k - t][t - 1];
            
        alpha[k] = a[k] - tempAlpha;
        
        // gamma, beta
        for(i=0; i<r-1; i++)
        {
            tempGamma = 0;
            tempBeta = 0;
            
            for(t=1; t<=r-i-1; t++)
            {
                tempGamma += gamma[k - t][i + t] * beta[k - t][t - 1];
                tempBeta += gamma[k - t][t - 1] * beta[k - t][i + t];
            }
            
            gamma[k][i] = c[k][i] - tempGamma;
            beta[k][i] = (b[k][i] - tempBeta) / alpha[k];       
        }
        
        gamma[k][r - 1] = c[k][r - 1];
        beta[k][r - 1] = b[k][r - 1] / alpha[k];
        
    }
    
   ////////
   for(k=n-r; k<n-1; k++)
   {
        //// alpha 
        tempAlpha = 0;
        for(t=1; t<=r; t++)
            tempAlpha += gamma[k - t][t - 1] * beta[k - t][t - 1];
        
        alpha[k] = a[k] - tempAlpha;
        
        ////// gamma, beta
        for(i=0; i<n-k-1; i++)
        {
            tempGamma = 0;
            tempBeta = 0;
            
            for(t=1; t<=r-i-1; t++)
            {
                tempGamma += gamma[k - t][i + t] * beta[k - t][t - 1];
                tempBeta += gamma[k - t][t - 1] * beta[k - t][i + t];
            }
            
            gamma[k][i] = c[k][i] - tempGamma;
            beta[k][i] = (b[k][i] - tempBeta) / alpha[k];       
        }
        
    }
    
    ////////
    k = n - 1;
    tempAlpha = 0;
    for(t=1; t<=r; t++)
        tempAlpha += gamma[k - t][t - 1] * beta[k - t][t - 1];
    
    alpha[k] = a[k] - tempAlpha;
    
    ///////////////////////////// Solve /////////////////////////
    int cha;
    double tempX, tempY;
    
    for(cha=0; cha<chaNumImg; cha++)
    {
        ////////////////// L*Y = F ///////////////////
        // k = 0;
        Y[0][cha] = F[0][cha] / alpha[0];
        
        // k = 1;
        k = 1;
        Y[k][cha] = (F[k][cha] - gamma[0][0] * Y[0][cha]) / alpha[k];

        //////////
        for(k=2; k<n; k++)
        {
            tempY = 0;
            for(t=1; t<=r; t++)
                tempY += gamma[k - t][t - 1] * Y[k - t][cha];
            
            Y[k][cha] = (F[k][cha] - tempY) / alpha[k];
        }

        ////////////// U*X = Y ////////////////////
        // k = n - 1
        X[n - 1][cha] = Y[n - 1][cha];

        // k = n - 2;
        k = n - 2;
        X[k][cha] = Y[k][cha] - beta[k][0] * X[k + 1][cha];

        //////////
        for(k=n-3; k>=0; k--)
        {
            tempX = 0;
            for(t=1; t<=r; t++)
                tempX += beta[k][t - 1] * X[k + t][cha];
            
            X[k][cha] = Y[k][cha] - tempX;
            
        }
        
    }
    
}


//////////////////////////////
void  solverForRadiusLargerThan2(double *a, double **b, double **c, double *alpha, double **gamma, double **beta, double **F, double **Y, double **X, int sysLen)
{
    int i, k, t, n=sysLen;
    double tempAlpha, tempGamma, tempBeta;
    
    //////////////////// Decomposition ///////////////////
    k = 0; 
    alpha[0] = a[0];
    for(i=0; i<r; i++)
    {
        gamma[0][i] = c[0][i];
        beta[0][i] = b[0][i] / alpha[0];
    }
    
    //////////
    for(k=1;k<r;k++)
    {
        //// alpha 
        tempAlpha = 0;
        for(t=1; t<=k; t++)
            tempAlpha += gamma[k - t][t - 1] * beta[k - t][t - 1];
        
        alpha[k] = a[k] - tempAlpha;
        
        ////// gamma, beta
        for(i=0; i<r-k; i++)
        {
            tempGamma = 0;
            tempBeta = 0;
            
            for(t=1; t<=k; t++)
            {
                tempGamma += gamma[k - t][i + t] * beta[k - t][t - 1];
                tempBeta += gamma[k - t][t - 1] * beta[k - t][i + t];
            }
            
            gamma[k][i] = c[k][i] - tempGamma;
            beta[k][i] = (b[k][i] - tempBeta) / alpha[k];       
        }
        
        gamma[k][r - 1] = c[k][r - 1];
        beta[k][r - 1] = b[k][r - 1] / alpha[k]; 
        
    }
    
    //////////
    for(k=2; k<r; k++)
    {
        ////// gamma, beta
        for(i=r-k; i<r-1; i++)
        {
            tempGamma = 0;
            tempBeta = 0;
            
            for(t=1; t<=r-i-1; t++)
            {
                tempGamma += gamma[k - t][i + t] * beta[k - t][t - 1];
                tempBeta += gamma[k - t][t - 1] * beta[k - t][i + t];
            }
            
            gamma[k][i] = c[k][i] - tempGamma;
            beta[k][i] = (b[k][i] - tempBeta) / alpha[k];       
        }
        
    }
    
    /////////////
    for(k=r; k<n-r; k++)
    {
        //// alpha 
        tempAlpha = 0;
        for(t=1; t<=r; t++)
            tempAlpha += gamma[k - t][t - 1] * beta[k - t][t - 1];
        
        alpha[k] = a[k] - tempAlpha;
        
        ////// gamma, beta  
        for(i=0; i<r-1; i++)
        {
            tempGamma = 0;
            tempBeta = 0;
            
            for(t=1; t<=r-i-1; t++)
            {
                tempGamma += gamma[k - t][i + t] * beta[k - t][t - 1];
                tempBeta += gamma[k - t][t - 1] * beta[k - t][i + t];
            }
            
            gamma[k][i] = c[k][i] - tempGamma;
            beta[k][i] = (b[k][i] - tempBeta) / alpha[k];       
        }
        
        gamma[k][r - 1] = c[k][r - 1];
        beta[k][r - 1] = b[k][r - 1] / alpha[k]; 
        
    }
    
   ////////
   for(k=n-r; k<n-1; k++)
   {
        //// alpha 
        tempAlpha = 0;
        for(t=1; t<=r; t++)
            tempAlpha += gamma[k - t][t - 1]*beta[k - t][t - 1];
        
        alpha[k] = a[k] - tempAlpha;
        
        ////// gamma, beta  
        for(i=0; i<n-k-1; i++)
        {
            tempGamma = 0;
            tempBeta = 0;
            
            for(t=1; t<=r-i-1; t++)
            {
                tempGamma += gamma[k - t][i + t] * beta[k - t][t - 1];
                tempBeta += gamma[k - t][t - 1] * beta[k - t][i + t];
            }
            
            gamma[k][i] = c[k][i] - tempGamma;
            beta[k][i] = (b[k][i] - tempBeta) / alpha[k];       
        }
        
    }
    
    ////////
    k = n - 1;
    tempAlpha = 0;
    for(t=1; t<=r; t++)
        tempAlpha += gamma[k - t][t - 1] * beta[k - t][t - 1];
    
    alpha[k] = a[k] - tempAlpha;
    
    ///////////////////////////// Solve /////////////////////////
    int cha;
    double tempX, tempY;
    
    for(cha=0; cha<chaNumImg; cha++)
    {
        ////////////////// L*Y = F ///////////////////
        // k = 0;
        Y[0][cha] = F[0][cha] / alpha[0];
        
        ///////////
        for(k=1; k<r; k++)
        {
            tempY = 0;
            
            for(t=1; t<=k; t++)
                tempY += gamma[k - t][t - 1] * Y[k - t][cha];
            
            Y[k][cha] = (F[k][cha] - tempY) / alpha[k];
        }

        //////////
        for(k=r; k<n; k++)
        {
            tempY = 0;
            for(t=1; t<=r; t++)
                tempY += gamma[k - t][t - 1] * Y[k - t][cha];
            
            Y[k][cha] = (F[k][cha] - tempY) / alpha[k];
        }

        ////////////// U*X = Y ////////////////////
        // k = n - 1
        X[n - 1][cha] = Y[n - 1][cha];

        // k = n - 2;
        for(k=n-2; k>=n-r; k--)
        {
            tempX = 0;

            for(t=1; t<=n-k-1; t++)
                tempX += beta[k][t - 1] * X[k + t][cha];
            
            X[k][cha] = Y[k][cha] - tempX;

        }

        //////////
        for(k=n-r-1; k>=0; k--)
        {
            tempX = 0;
            for(t=1; t<=r; t++)
                tempX += beta[k][t - 1] * X[k + t][cha];
            
            X[k][cha] = Y[k][cha] - tempX;
            
        }
        
    }
    
}

/////////////////////////////////////
void pointDiv(double ***imgInter, double ***imgFiltered, double **count)
{
    int i, j, k;
    
    for(i=0; i<rowNum; i++)
    {
        for(j=0; j<colNum; j++)
        {
            for(k=0; k<chaNumImg; k++)
                imgFiltered[i][j][k] = imgInter[i][j][k] / count[i][j];
        }
    }
    
} 


/////////////////////////////////////////////////////////
void valueSet2(double **input, double value)
{
    for(int i=0; i<rowNum; i++)
        for(int j=0; j<colNum; j++)
                input[i][j] = value;
}


/////////////////////////////////////////////////////////////
void valueSet3(double ***input, double value, int chaNum)
{
    for(int i=0; i<rowNum; i++)
        for(int j=0; j<colNum; j++)
            for(int k=0; k<chaNum; k++)
                input[i][j][k] = value;
}
