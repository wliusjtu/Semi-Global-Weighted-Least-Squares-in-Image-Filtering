#include <math.h>
#include <time.h>
#include <string.h>
#include "mex.h"

#define min(X, Y)  ((X) < (Y) ? (X) : (Y))

// size of input image, guidance image and SG_WLS
int rowNum, colNum, rc, ccImg, ccGuide, ccImgMinusChaNum, ccImgPlusChaNum, ccGuideMinusChaNum, ccGuidePlusChaNum;
int chaNumImg, chaNumGuide, chaNumImgX2, chaNumGuideX2;
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
void vector2img_col(double ***img, double **vector, int col);
void vector2img_row(double ***img, double **vector, int row);
void getCount(double *count, int len, int r, int step);
void getLaplacian(double **vectorGuide, double *a, double **b, double **c, double lambda, double *rangeLUT, double *spatialLUT, int sysLen);
void pointDiv(double ***imgInter, double ***imgFiltered, double *count, int colDir);

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
    
    ccImg = colNum * chaNumImg;
    chaNumImgX2 = 2 * chaNumImg;
    ccImgMinusChaNum = ccImg - chaNumImg;
    ccImgPlusChaNum = ccImg + chaNumImg;
    
    if(mxGetNumberOfDimensions(prhs[1]) == 2)  // channel number of guidance image
        chaNumGuide = 1;  // single channel image
    else 
        chaNumGuide = 3;  // rgb image
    
    ccGuide = colNum * chaNumGuide;
    chaNumGuideX2 = 2 * chaNumGuide;
    ccGuideMinusChaNum = ccGuide - chaNumGuide;
    ccGuidePlusChaNum = ccGuide + chaNumGuide;
    
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
    double *ptr_imgFiltered = &imgFiltered[0][0][0];
    for(int i=0; i<rowNum; i++)
        for(int j=0; j<colNum; j++)
            for(int k=0; k<chaNumImg; k++)
                *ptr_imgFiltered++ = img[k * rc + j *rowNum + i];
    
    double ***imgGuide = memAllocDouble3(rowNum, colNum, chaNumGuide);
    double *ptr_imgGuide = &imgGuide[0][0][0];
    for(int i=0; i<rowNum; i++)
        for(int j=0; j<colNum; j++)
            for(int k=0; k<chaNumGuide; k++)
                *ptr_imgGuide++ = guidance[k * rc + j * rowNum + i];
    
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
    
    // accumulate the count of the filtered value at the same location
    double *count_row = (double *)mxGetData(mxCreateDoubleMatrix(rowNum, 1, mxREAL));
    double *count_col = (double *)mxGetData(mxCreateDoubleMatrix(colNum, 1, mxREAL));
    
    // weight lookup table
    int maxRange = 255 + 10;
    maxRange = chaNumGuide * maxRange * maxRange;
    int maxSpatial = 2 * (2 * r + 1) * (2 * r + 1);
    double *rangeLUT = (double *)mxGetData(mxCreateDoubleMatrix(maxRange, 1, mxREAL));
    double *spatialLUT = (double *)mxGetData(mxCreateDoubleMatrix(maxSpatial, 1, mxREAL));
    
    if(weightChoice == 0){
        expLUT(rangeLUT, sigmaR, chaNumGuide, maxRange);
        expLUT(spatialLUT, sigmaS, 1, maxSpatial);}
    else if(weightChoice ==1){
        fracLUT(rangeLUT, sigmaR, chaNumGuide, maxRange);
        fracLUT(spatialLUT, sigmaS, 1, maxSpatial);}
    
    else mexErrMsgTxt("Weight choice should be 0 (exponential) or 1 (fractional)\n.");
    
    // maximum row/column number 
    int maxIterColNum = r + 1 + ((colNum - r - r - 1)/step)*step;
    int maxIterRowNum = r + 1 + ((rowNum - r - r - 1)/step)*step;
    int col, row, iter;
    
    // do filtering
    clock_t tStart = clock(); // time measurement;

    for(iter=0; iter<iterNum; iter++)
    {
        /////////////// column direction //////////////
        memset(count_col, 0.0, sizeof(double)*colNum);
        memset(imgInter[0][0], 0.0, sizeof(double)*rc*chaNumImg);
        
        getCount(count_col, colNum, r, step);

        for(col=r; col<maxIterColNum; col+=step)
        {
            img2vector_col(imgFiltered, imgGuide, vectorImg_col, vectorGuide_col, col);
            getLaplacian(vectorGuide_col, a_col, b_col, c_col, lambda, rangeLUT, spatialLUT, sysLen_col);
            if(r==1)
                solverForRadius1(a_col, b_col, c_col, alpha_col, gamma_col, beta_col, vectorImg_col, vectorInter_col, vectorFiltered_col, sysLen_col);
            else if(r==2)
                solverForRadius2(a_col, b_col, c_col, alpha_col, gamma_col, beta_col, vectorImg_col, vectorInter_col, vectorFiltered_col, sysLen_col);
            else
                solverForRadiusLargerThan2(a_col, b_col, c_col, alpha_col, gamma_col, beta_col, vectorImg_col, vectorInter_col, vectorFiltered_col, sysLen_col);
            vector2img_col(imgInter, vectorFiltered_col, col);
        }

        // the last 2 * r + 1 columns
        col = colNum - 1 - r;
        img2vector_col(imgFiltered, imgGuide, vectorImg_col, vectorGuide_col, col);
        getLaplacian(vectorGuide_col, a_col, b_col, c_col, lambda, rangeLUT, spatialLUT, sysLen_col);
        if(r==1)
            solverForRadius1(a_col, b_col, c_col, alpha_col, gamma_col, beta_col, vectorImg_col, vectorInter_col, vectorFiltered_col, sysLen_col);
        else if(r==2)
            solverForRadius2(a_col, b_col, c_col, alpha_col, gamma_col, beta_col, vectorImg_col, vectorInter_col, vectorFiltered_col, sysLen_col);
        else
            solverForRadiusLargerThan2(a_col, b_col, c_col, alpha_col, gamma_col, beta_col, vectorImg_col, vectorInter_col, vectorFiltered_col, sysLen_col);
        vector2img_col(imgInter, vectorFiltered_col, col);

        pointDiv(imgInter, imgFiltered, count_col, 1);

        ///////////////////// row direction //////////////////////
        memset(count_row, 0.0, sizeof(double)*rowNum);
        memset(imgInter[0][0], 0.0, sizeof(double)*rc*chaNumImg);
        
        getCount(count_row, rowNum, r, step);

        for(row=r; row<maxIterRowNum; row+=step)
        {
            img2vector_row(imgFiltered, imgGuide, vectorImg_row, vectorGuide_row, row);
            getLaplacian(vectorGuide_row, a_row, b_row, c_row, lambda, rangeLUT, spatialLUT, sysLen_row);
            if(r==1)
                solverForRadius1(a_row, b_row, c_row, alpha_row, gamma_row, beta_row, vectorImg_row, vectorInter_row, vectorFiltered_row, sysLen_row);
            else if(r==2)
                solverForRadius2(a_row, b_row, c_row, alpha_row, gamma_row, beta_row, vectorImg_row, vectorInter_row, vectorFiltered_row, sysLen_row);
            else
                solverForRadiusLargerThan2(a_row, b_row, c_row, alpha_row, gamma_row, beta_row, vectorImg_row, vectorInter_row, vectorFiltered_row, sysLen_row);
            vector2img_row(imgInter, vectorFiltered_row, row);
        }

        // the last 2 * r + 1 rows
        row = rowNum - 1 - r;
        img2vector_row(imgFiltered, imgGuide, vectorImg_row, vectorGuide_row, row);
        getLaplacian(vectorGuide_row, a_row, b_row, c_row, lambda, rangeLUT, spatialLUT, sysLen_row);
        if(r==1)
            solverForRadius1(a_row, b_row, c_row, alpha_row, gamma_row, beta_row, vectorImg_row, vectorInter_row, vectorFiltered_row, sysLen_row);
        else if(r==2)
            solverForRadius2(a_row, b_row, c_row, alpha_row, gamma_row, beta_row, vectorImg_row, vectorInter_row, vectorFiltered_row, sysLen_row);
        else
            solverForRadiusLargerThan2(a_row, b_row, c_row, alpha_row, gamma_row, beta_row, vectorImg_row, vectorInter_row, vectorFiltered_row, sysLen_row);
        vector2img_row(imgInter, vectorFiltered_row, row);

        pointDiv(imgInter, imgFiltered, count_row, 0);
    }

    mexPrintf("Elapsed time is %f seconds.\n", double(clock() - tStart)/CLOCKS_PER_SEC);
    
    // transfer to the output
    ptr_imgFiltered = &(imgFiltered[0][0][0]);
    for(int i=0; i<rowNum; i++)
    {
        for(int j=0; j<colNum; j++)
        {
            for(int k=0; k<chaNumImg; k++)
                imgResult[k * rc + j * rowNum + i] = *ptr_imgFiltered++;
        }
    }
    
    memFreeDouble3(imgFiltered);
    memFreeDouble3(imgGuide);
    memFreeDouble3(imgInter);
    a_row = NULL;
    a_col = NULL;
    memFreeDouble2(b_row);
    memFreeDouble2(b_col);
    memFreeDouble2(c_row);
    memFreeDouble2(c_col);
    alpha_row = NULL;
    alpha_col = NULL;
    memFreeDouble2(beta_row);
    memFreeDouble2(beta_col);
    memFreeDouble2(gamma_row);
    memFreeDouble2(gamma_col);
    memFreeDouble2(vectorImg_row);
    memFreeDouble2(vectorImg_col);
    memFreeDouble2(vectorGuide_row);
    memFreeDouble2(vectorGuide_col);
    memFreeDouble2(vectorFiltered_row);
    memFreeDouble2(vectorFiltered_col);
    memFreeDouble2(vectorInter_row);
    memFreeDouble2(vectorInter_col);

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
		for(j=0; j<col; j++) 
            p[i * col + j] = &a[i * cc + j * cha];
    
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
    double *ptr=&LUT[0];
    for(int i=0; i<len; i++)   *ptr++ = exp(double(-i) / (double(chaNum) * 2 * sigma * sigma));
    
}


///////////////////////////////////////////////////
void fracLUT(double *LUT, double sigma, int chaNum, int len)
{
    double *ptr=&LUT[0];
    for(int i=0; i<len; i++)  *ptr++ = 1 / (pow(sqrt(double(i) / double(chaNum)), sigma) + 0.00001);
    
}


////////////////////////////////////////////////////
void img2vector_col(double ***img, double ***imgGuide, double **vectorImg, double **vectorGuide, int col)
{
    // transform the block centered along column 'col' with radius 'r' into a vector whose length is 'rowNum * (2 * r + 1)'.
    
    double  *ptr_img, *ptr_guide, *ptr_vecImg, *ptr_vecGuide;
    int i, j, k, colSlide;
    bool forward=true;
    
    colSlide = col - r;
    ptr_vecImg = &vectorImg[0][0];
    ptr_vecGuide = &vectorGuide[0][0];
    ptr_img = &img[0][colSlide][0];
    ptr_guide = &imgGuide[0][colSlide][0];
    
    for(i=0; i<rowNum; i++)
    {
        for(j=-r; j<=r; j++)
        {
            if(forward){  // left-to-right extraction sliding direction
                // for the image to be filtered
                for(k=0; k<chaNumImg; k++)
                    *ptr_vecImg++ = *ptr_img++;

                // for the guidance image
                for(k=0; k<chaNumGuide; k++)
                    *ptr_vecGuide++ = *ptr_guide++;

                // store the x/y pixel coordinate in the image coordinate system, to compute the spatial weight
                *ptr_vecGuide++ = i;
                *ptr_vecGuide++ = colSlide++;  
            }
            else {   // right-to-left extraction sliding direction
                // for the image to be filtered
                ptr_img -= chaNumImgX2;
                for(k=0; k<chaNumImg; k++)
                    *ptr_vecImg++ = *ptr_img++;

                // for the guidance image
                ptr_guide -= chaNumGuideX2;
                for(k=0; k<chaNumGuide; k++)
                    *ptr_vecGuide++ = *ptr_guide++;

                // store the x/y pixel coordinate in the image coordinate system, to compute the spatial weight
                *ptr_vecGuide++ = i;
                *ptr_vecGuide++ = colSlide--; 
            }
        }
        
        if(forward){
            // move to the next row
            ptr_img += ccImgPlusChaNum;
            ptr_guide += ccGuidePlusChaNum;
            colSlide--;
        }
        else{
            // move to the next row
            ptr_img += ccImgMinusChaNum;
            ptr_guide += ccGuideMinusChaNum;
            colSlide++;
        }
        
         forward = !forward;   // change sliding direction between left-to-right and right-to-left
    }
}


//////////////////////////////////////////////////
void img2vector_row(double ***img, double ***imgGuide, double **vectorImg, double **vectorGuide, int row)
{
    // transform the block centered along row 'row' with radius 'r' into a vector whose length is 'colNum * (2 * r + 1)'.
    double *ptr_img, *ptr_guide, *ptr_vecImg, *ptr_vecGuide;
    int i, j, k, rowSlide;
    bool forward=true;
    
    rowSlide = row - r;
    ptr_vecImg = &vectorImg[0][0];
    ptr_vecGuide = &vectorGuide[0][0];
    ptr_img = &img[rowSlide][0][0];
    ptr_guide = &imgGuide[rowSlide][0][0];
   
    for(j=0; j<colNum; j++)
    {
        for(i=-r; i<=r; i++)
        {
            if(forward){  // up-bottom extraction sliding direction
                // for the image to be filtered
                for(k=0; k<chaNumImg; k++) 
                    *ptr_vecImg++ = *ptr_img++;
                ptr_img += ccImgMinusChaNum;        
                
                // for the guidance image
                for(k=0; k<chaNumGuide; k++)
                    *ptr_vecGuide++ = *ptr_guide++;
                ptr_guide += ccGuideMinusChaNum;
                
                // store the x/y pixel coordinate in the image coordinate system, to compute the spatial weight
                *ptr_vecGuide++ = rowSlide++;
                *ptr_vecGuide++ = j;    
            }
            else{  // bottom-up extraction sliding direction
                 // for the image to be filtered
                for(k=0; k<chaNumImg; k++) 
                    *ptr_vecImg++ = *ptr_img++;
                ptr_img -= ccImgPlusChaNum;        
                
                // for the guidance image
                for(k=0; k<chaNumGuide; k++)
                    *ptr_vecGuide++ = *ptr_guide++;
                ptr_guide -= ccGuidePlusChaNum;
                
                // store the x/y pixel coordinate in the image coordinate system, to compute the spatial weight
                *ptr_vecGuide++ = rowSlide--;
                *ptr_vecGuide++ = j;    
            }      
        }
       
        if(forward){
            // move to the next column
            ptr_img -=  ccImgMinusChaNum;
            ptr_guide -= ccGuideMinusChaNum;
            rowSlide--;
        }
        else{
            // move to the next column
            ptr_img +=  ccImgPlusChaNum;
            ptr_guide += ccGuidePlusChaNum;
            rowSlide++;
        }
        
        forward = !forward;  // change the extraction sliding direction between bottom-up and up-bottom in each column of the image
        
    }
}


///////////////////////////////////////////////////
void vector2img_col(double ***img, double **vector, int col)
{
    // the inverse transform of 'img2vector_column'

    double *ptr_img, *ptr_vec;
    int i, j, k, colSlide;
    bool forward=true;
    
    colSlide = col - r;
    ptr_vec = &vector[0][0];
    ptr_img = &img[0][colSlide][0];
    
    for(i=0; i<rowNum; i++)
    {
        for(j=-r; j<=r; j++)
        {
            if(forward){
                // for image
                for(k=0; k<chaNumImg; k++)
                    *ptr_img++ += *ptr_vec++;}
            else {
                // for image
                ptr_img -= chaNumImgX2;
                for(k=0; k<chaNumImg; k++)
                    *ptr_img++ += *ptr_vec++;}
        }
        
        if(forward)
            // move to the next row
            ptr_img += ccImgPlusChaNum;
        else
            // move to the next row
            ptr_img += ccImgMinusChaNum;
        
        forward = !forward; // change the extraction sliding direction between left-to-right and right-to-left in each row of the image
        
    }
}


///////////////////////////////////////////////////
void vector2img_row(double ***img, double **vector, int row)
{
    // the inverse transform of 'img2vector_column'
    
    double *ptr_img, *ptr_vec;
    int i, j, k, rowSlide;
    bool forward=true;
    
    rowSlide = row - r;
    ptr_img = &img[rowSlide][0][0];
    ptr_vec = &vector[0][0];
    
    for(j=0; j<colNum; j++)
    {
        for(i=-r; i<=r; i++)
        {
            if(forward){  // up-bottom extraction sliding direction
                // for image to be filtered
                for(k=0; k<chaNumImg; k++)
                    *ptr_img++ += *ptr_vec++;
                ptr_img += ccImgMinusChaNum;}
            else{  // bottom-up extraction sliding direction
                // for image to be filtered
                for(k=0; k<chaNumImg; k++)
                    *ptr_img++ += *ptr_vec++;
                ptr_img -= ccImgPlusChaNum;}   
        }
        
        if(forward)
            // move to the next column
            ptr_img -= ccImgMinusChaNum;
        else
            // move to the next column
            ptr_img += ccImgPlusChaNum;
        
        forward = !forward;  // change the extraction sliding direction between bottom-up and up-bottom in each column of the image
        
    } 
}


////////////////////
void getLaplacian(double **vectorGuide, double *a, double **b, double **c, double lambda, double *rangeLUT, double *spatialLUT, int sysLen)
{
    double *ptr_guide_cur, *ptr_guide_nei, *ptr_a, *ptr_b, *ptr_c;
    double diffR, temp, diffS, weightR, weightS, weight;
    int i, j, k, n=sysLen, colNumGuideVector = chaNumGuide + 2;
    
    ptr_guide_cur = &vectorGuide[0][0];
    ptr_guide_nei = &vectorGuide[1][0];
    ptr_a = &a[0];
    ptr_b = &b[0][0];
    ptr_c = &c[0][0];
    
    // compute b and c first
    for(i=0; i<n-1; i++)
    {
        for(j=1; j<=min(r, n - 1 - i); j++)
        {
            // range weight
            diffR = 0;
            for(k=0; k<chaNumGuide; k++){
                temp = *ptr_guide_cur++ - *ptr_guide_nei++;
                diffR += temp * temp;}
            weightR = rangeLUT[(int)diffR];
            
            // spatial weight
            diffS = 0;
            for(k=chaNumGuide; k<colNumGuideVector; k++){
                temp =  *ptr_guide_cur++ - *ptr_guide_nei++;
                diffS += temp * temp; }
            weightS = spatialLUT[(int)diffS];
            
            weight = -lambda * weightR * weightS;
            *ptr_b = weight;
            *ptr_c = weight;
            
            ptr_b++; ptr_c++;
            ptr_guide_cur -= colNumGuideVector;
        }
        
        ptr_b += r - j + 1;
        ptr_c += r - j + 1;
        
        ptr_guide_cur += colNumGuideVector;
        ptr_guide_nei = ptr_guide_cur + colNumGuideVector;
    }
    
    // compute a with the computed b and c
    double *ptr_c_anchor;
    
    ptr_a = &a[0];
    ptr_b = &b[0][0];
    ptr_c_anchor = &c[0][0];
    
    temp = 0;
    for(j=1; j<=r; j++)
        temp -= *ptr_b++;
    *ptr_a++ = 1 + temp;
    
    ///////////////
    for(i=1; i<r+1; i++)
    {
        ptr_c = ptr_c_anchor++;
        temp = 0;
        for(j=1; j<=i; j++){
            temp -= *ptr_c;
            ptr_c += r - 1;}
        
        for(j=1; j<=r; j++)
            temp -= *ptr_b++;
        
        *ptr_a++ = 1 + temp;
    }
    
    //////////
    ptr_c_anchor--;
    for(i=r + 1; i<n - r; i++)
    {
        ptr_c_anchor += r;
        ptr_c = ptr_c_anchor;
        temp = 0;
        for(j=1; j<=r; j++){
            temp -= *ptr_c + *ptr_b++;
            ptr_c += r - 1;}
        
        *ptr_a++ = 1 + temp;
    }
    
    ////////////////
    for(i=n - r; i<n - 1; i++)
    {
        ptr_c_anchor += r;
        ptr_c = ptr_c_anchor;
        temp = 0;
        for(j=1; j<=r; j++){
            temp -= *ptr_c;
            ptr_c += r - 1;}
        
        for(j=1; j<=n - 1 - i; j++)
            temp -= *ptr_b++;
        ptr_b += r - j + 1;
        
        *ptr_a++ = 1 + temp;
    }
    
    ////////////////////////
    i = n - 1;
    ptr_c_anchor += r;
    ptr_c = ptr_c_anchor;
    temp = 0;
    for(j=1; j<=r; j++){
        temp -= *ptr_c;
        ptr_c += r - 1;}
    
    *ptr_a = 1 + temp;
    
}

/////////////////////////////////////
void solverForRadius1(double *a, double **b, double **c, double *alpha, double **gamma, double **beta, double **F, double **Y, double **X, int sysLen)
{
    int k, cha, n=sysLen;
    double *ptr_a, *ptr_b, *ptr_c, *ptr_alpha, *ptr_gamma, *ptr_beta;
    double tempAlpha, tempGamma, tempBeta;
    
    ptr_a = &a[0];  ptr_b = &b[0][0];  ptr_c = &c[0][0];
    ptr_alpha = &alpha[0];  ptr_beta = &beta[0][0];  
    
    // LU decomposition
    tempAlpha = *ptr_a++;
    tempGamma = *ptr_c++;
    tempBeta = tempGamma / tempAlpha;
//     tempBeta = *ptr_b++ / tempAlpha;
    *ptr_alpha++ = tempAlpha;
    *ptr_beta++ = tempBeta;
    
    for(k=1; k<n - 1; k++)
    {
        tempAlpha = *ptr_a++ - tempGamma * tempBeta;
        *ptr_alpha++ = tempAlpha;
        tempGamma = *ptr_c++;
        tempBeta = tempGamma / tempAlpha;
//         tempBeta = *ptr_b++ / tempAlpha;
        *ptr_beta++ = tempBeta;
    }
    *ptr_alpha = *ptr_a - tempGamma * tempBeta;
    
    ///////  Solve ///////////
    double *ptr_F, *ptr_Y, *ptr_X, *ptr_Y_pre;
    
    ptr_alpha = &alpha[0];
    ptr_gamma = &c[0][0];  // ptr_gamma = &gamma[0][0];  we always have \gamma_k,i = c_k,i when r=1
    ptr_Y_pre=&Y[0][0];
    ptr_Y = &Y[0][0];
    ptr_F = &F[0][0];
    
    // L * Y = F
    tempAlpha = *ptr_alpha++;
    for(cha=0; cha<chaNumImg; cha++)
        *ptr_Y++ = *ptr_F++ / tempAlpha;
    
    for(k=1; k<n; k++)
    {
        tempAlpha = *ptr_alpha++;
        tempGamma = *ptr_gamma++;
        for(cha=0; cha<chaNumImg; cha++)
            *ptr_Y++ = (*ptr_F++ - tempGamma * (*ptr_Y_pre++)) / tempAlpha;
    }
    
    // U*X = Y;
    double *ptr_X_pre=&X[n-1][chaNumImg - 1];
    ptr_X = ptr_X_pre;
    ptr_Y = &Y[n-1][chaNumImg - 1];
    
    for(cha=0; cha<chaNumImg; cha++)
        *ptr_X-- = *ptr_Y--;
    
    ptr_beta = &beta[n-2][0];
    for(k=n-2; k>=0; k--)
    {
        tempBeta = *ptr_beta--;
        for(cha=0; cha<chaNumImg; cha++)
            *ptr_X-- = *ptr_Y-- - tempBeta * (*ptr_X_pre--);
    }
}


//////////////////////////////
void solverForRadius2(double *a, double **b, double **c, double *alpha, double **gamma, double **beta, double **F, double **Y, double **X, int sysLen)
{
    int i, k, t, n=sysLen;
    double *ptr_a, *ptr_b, *ptr_c,*ptr_alpha, *ptr_gamma, *ptr_beta, *ptr_gamma_temp, *ptr_beta_temp;
    double tempAlpha, tempGamma, tempBeta;
    
    ptr_a = &a[0];  ptr_b = &b[0][0];  ptr_c = &c[0][0];
    ptr_alpha = &alpha[0];  ptr_gamma = &gamma[0][0];  ptr_beta = &beta[0][0];
    ptr_gamma_temp = ptr_gamma;  ptr_beta_temp = ptr_beta;
    
    //////////////////// LU decomposition ///////////////////
    // k = 0; 
    tempAlpha = *ptr_a++;
    *ptr_alpha++ = tempAlpha;
    for(i=0; i<r; i++){
        *ptr_gamma++ = *ptr_c++;
        *ptr_beta++ = *ptr_b++ / tempAlpha;
    }
    
    //////////
    // k = 1;
    tempAlpha = *ptr_a++ - *ptr_gamma_temp * (*ptr_beta_temp); 
    *ptr_alpha++ = tempAlpha;
    *ptr_gamma++ = *ptr_c++ - *(ptr_gamma_temp + 1) * *ptr_beta_temp; 
    *ptr_beta++ = (*ptr_b++ - *ptr_gamma_temp * (*(ptr_beta_temp + 1))) / tempAlpha;
    
    *ptr_gamma++ = *ptr_c++;
    *ptr_beta++ = *ptr_b++ / tempAlpha;
    
    ptr_gamma_temp++; 
    ptr_beta_temp++; 
    
    /////////// 
    for(k=2; k<n-r; k++)
    {
        // alpha 
        tempAlpha = 0;
        for(t=1; t<=r; t++)
            tempAlpha += *ptr_gamma_temp++ * (*ptr_beta_temp++);
        
        tempAlpha  = *ptr_a++ - tempAlpha;
        *ptr_alpha++ = tempAlpha;
        
        //gamma, beta
        *ptr_gamma++ = *ptr_c++ - *ptr_gamma_temp * (*(ptr_beta_temp - 1));
        *ptr_beta++ = (*ptr_b++ - *(ptr_gamma_temp - 1) * (*ptr_beta_temp)) / tempAlpha;
         
        *ptr_gamma++ = *ptr_c++;
        *ptr_beta++ = *ptr_b++ / tempAlpha;
    }
    
    ////////
    // k=n-2
    //// alpha 
    tempAlpha = 0;
    for(t=1; t<=r; t++)
        tempAlpha += *ptr_gamma_temp++ * (*ptr_beta_temp++);

    tempAlpha  = *ptr_a++ - tempAlpha;
    *ptr_alpha++ = tempAlpha;

    //gamma, beta
    *ptr_gamma++ = *ptr_c++ - *ptr_gamma_temp * (*(ptr_beta_temp - 1));
    *ptr_beta++ = (*ptr_b++ - *(ptr_gamma_temp - 1) * (*ptr_beta_temp)) / tempAlpha;

    ////////
    // k = n - 1;
    tempAlpha = 0;
    for(t=1; t<=r; t++)
        tempAlpha += *ptr_gamma_temp++ * (*ptr_beta_temp++);
    
    *ptr_alpha = *ptr_a - tempAlpha;
    
    ///////////////////////////// Solve /////////////////////////
    int cha;
    double tempX, tempY;//, tempGamma, tempBeta;
    double *ptr_X, *ptr_X_pre, *ptr_F, *ptr_Y, *ptr_Y_pre;
    
    ptr_Y = &Y[0][0];  ptr_Y_pre = ptr_Y;  ptr_F = &F[0][0];
    ptr_alpha = &alpha[0];  ptr_gamma = &gamma[0][0];
    
    ////////////////// L*Y = F ///////////////////
    // k = 0;
    tempAlpha = *ptr_alpha++;
    for(cha=0; cha<chaNumImg; cha++)
        *ptr_Y++ = *ptr_F++ / tempAlpha;
    
    // k = 1;
    tempGamma = *ptr_gamma++;
    tempAlpha = *ptr_alpha++;
    for(cha=0; cha<chaNumImg; cha++)
        *ptr_Y++ = (*ptr_F++ - tempGamma * (*ptr_Y_pre++)) / tempAlpha;
    
    //////////
    ptr_Y_pre -= chaNumImg;
    
    for(k=2; k<n; k++)
    {
        tempAlpha = *ptr_alpha++;
        for(cha=0; cha<chaNumImg; cha++){
            tempY = 0;
            for(t=1; t<=r; t++) {
                tempY += *ptr_gamma++ * (*ptr_Y_pre);
                ptr_Y_pre += chaNumImg;}
            
            *ptr_Y++ = (*ptr_F++ - tempY) / tempAlpha;
            ptr_gamma -= r;  ptr_Y_pre -= r * chaNumImg - 1;
        }
        
        ptr_gamma += r;  
    }
    
    ////////////// U*X = Y ////////////////////
    ptr_X = &X[n - 1][chaNumImg - 1];  ptr_X_pre = ptr_X;  ptr_Y = &Y[n - 1][chaNumImg - 1]; 
    
    // k = n - 1
    for(cha=0; cha<chaNumImg; cha++)
        *ptr_X-- = *ptr_Y--;
    
    // k = n - 2;
    ptr_beta = &beta[n - 2][0];
    tempBeta = *ptr_beta--;
    for(cha=0; cha<chaNumImg; cha++)
        *ptr_X-- = *ptr_Y-- - tempBeta * (*ptr_X_pre--); 
    
    //////////
    ptr_X_pre += chaNumImg;
    
    for(k=n-3; k>=0; k--)
    {
        for(cha=0; cha<chaNumImg; cha++){
            tempX = 0;
            for(t=1; t<=r; t++){
                tempX += *ptr_beta-- * (*ptr_X_pre);
                ptr_X_pre -= chaNumImg;}
            
            *ptr_X-- = *ptr_Y-- - tempX;
            ptr_beta += r;  ptr_X_pre += r * chaNumImg - 1;
        }
        
        ptr_beta -= r;     
    }
    
}


//////////////////////////////
void  solverForRadiusLargerThan2(double *a, double **b, double **c, double *alpha, double **gamma, double **beta, double **F, double **Y, double **X, int sysLen)
{
    int i, maxt, k, t, n=sysLen, rMinusOne = r - 1;
    double *ptr_a, *ptr_b, *ptr_c, *ptr_alpha, *ptr_gamma, *ptr_beta, *ptr_gamma_temp, *ptr_beta_temp;
    double tempAlpha, tempGamma, tempBeta;
    
    ptr_a = &a[0];  ptr_b = &b[0][0];  ptr_c = &c[0][0];
    ptr_alpha = &alpha[0];  ptr_gamma = &gamma[0][0];  ptr_beta = &beta[0][0];
    ptr_gamma_temp = ptr_gamma;  ptr_beta_temp = ptr_beta;
    
    //////////////////// Decomposition ///////////////////
    // k = 0; 
    tempAlpha = *ptr_a++;
    *ptr_alpha++ = tempAlpha;
    for(i=0; i<r; i++){
        *ptr_gamma++ = *ptr_c++;
        *ptr_beta++ = *ptr_b++ / tempAlpha;}
    
    //////////
    // k = 1;
    //// alpha 
    tempAlpha = *ptr_a++ - *ptr_gamma_temp * (*ptr_beta_temp);
    *ptr_alpha++ = tempAlpha;
    
    ////// gamma, beta
    for(i=0; i<r-1; i++){
        *ptr_gamma++ = *ptr_c++ - *(ptr_gamma_temp + i + 1) * (*ptr_beta_temp);
        *ptr_beta++ =(*ptr_b++ - *ptr_gamma_temp * (*(ptr_beta_temp + i + 1))) / tempAlpha;}
    
    *ptr_gamma++ = *ptr_c++;
    *ptr_beta++ = *ptr_b++ / tempAlpha;
    
    //////////
    for(k=2; k<r; k++)
    {
        ptr_gamma_temp += r;
        ptr_beta_temp += r;
        
        //// alpha 
        tempAlpha = 0;
        for(t=1; t<=k; t++){
            tempAlpha += *ptr_gamma_temp * (*ptr_beta_temp);
            ptr_gamma_temp -= rMinusOne;
            ptr_beta_temp -= rMinusOne;}
        
        tempAlpha = *ptr_a++ - tempAlpha;
        *ptr_alpha++ = tempAlpha;
        
        ptr_gamma_temp += k * rMinusOne;
        ptr_beta_temp += k * rMinusOne;
        
        ////// gamma, beta
        for(i=0; i<r-1; i++)  
        {
            maxt = min(k, r - i - 1);
            tempGamma = 0;
            tempBeta = 0;
            for(t=1; t<=maxt; t++){
                tempGamma += *(ptr_gamma_temp + i + 1) * (*ptr_beta_temp);
                tempBeta += *ptr_gamma_temp * (*(ptr_beta_temp + i + 1));
                ptr_gamma_temp -= rMinusOne;
                ptr_beta_temp -= rMinusOne;}
            
            *ptr_gamma++ = *ptr_c++ - tempGamma;
            *ptr_beta++ = (*ptr_b++ - tempBeta) / tempAlpha;     
            
            ptr_gamma_temp += maxt * rMinusOne;
            ptr_beta_temp += maxt * rMinusOne;
        }
        
        *ptr_gamma++ = *ptr_c++;
        *ptr_beta++ = *ptr_b++ / tempAlpha;      
    }
    
    /////////////
    for(k=r; k<n-r; k++)
    {
        ptr_gamma_temp += r;
        ptr_beta_temp += r;
        
        //// alpha 
        tempAlpha = 0;
        for(t=1; t<=r; t++){
            tempAlpha += *ptr_gamma_temp * (*ptr_beta_temp);
            ptr_gamma_temp -= rMinusOne;
            ptr_beta_temp -= rMinusOne;}
        
        tempAlpha = *ptr_a++ - tempAlpha;
        *ptr_alpha++ = tempAlpha;
        
        ptr_gamma_temp += r * rMinusOne;
        ptr_beta_temp += r * rMinusOne;
        
        ////// gamma, beta  
        for(i=0; i<r-1; i++)
        {
            maxt = r - i - 1;
            tempGamma = 0;
            tempBeta = 0;
            for(t=1; t<=maxt; t++){
                tempGamma += *(ptr_gamma_temp + i + 1) * (*ptr_beta_temp);
                tempBeta += *ptr_gamma_temp * (*(ptr_beta_temp + i + 1));
                ptr_gamma_temp -= rMinusOne;
                ptr_beta_temp -= rMinusOne;}
            
            *ptr_gamma++ = *ptr_c++ - tempGamma;
            *ptr_beta++ = (*ptr_b++ - tempBeta) / tempAlpha;
            
            ptr_gamma_temp += maxt * rMinusOne;
            ptr_beta_temp += maxt * rMinusOne;
        }

        *ptr_gamma++ = *ptr_c++;
        *ptr_beta++ = *ptr_b++ / tempAlpha;   
    }
    
   ////////
   for(k=n-r; k<n-1; k++)
   {
       ptr_gamma_temp += r;
       ptr_beta_temp += r;
        
       //// alpha 
       tempAlpha = 0;
       for(t=1; t<=r; t++){
           tempAlpha += *ptr_gamma_temp * (*ptr_beta_temp);
           ptr_gamma_temp -= rMinusOne;
           ptr_beta_temp -= rMinusOne;}
        
        tempAlpha = *ptr_a++ - tempAlpha;
        *ptr_alpha++ = tempAlpha;
        
        ptr_gamma_temp += r * rMinusOne;
        ptr_beta_temp += r * rMinusOne;
        
        ////// gamma, beta  
        for(i=0; i<n-k-1; i++)
        {
            maxt = r - i - 1;
            tempGamma = 0;
            tempBeta = 0;
            for(t=1; t<=maxt; t++){
                tempGamma += *(ptr_gamma_temp + i + 1) * (*ptr_beta_temp);
                tempBeta += *ptr_gamma_temp * (*(ptr_beta_temp + i + 1));
                ptr_gamma_temp -= rMinusOne;
                ptr_beta_temp -= rMinusOne;}
            
            *ptr_gamma++ = *ptr_c++ - tempGamma;
            *ptr_beta++ = (*ptr_b++ - tempBeta) / tempAlpha; 
            
            ptr_gamma_temp += maxt * rMinusOne;
            ptr_beta_temp += maxt * rMinusOne;
        }
        
        ptr_gamma += r + k - n + 1;  ptr_beta += r + k - n + 1;
        ptr_b += r + k - n + 1;  ptr_c += r + k - n + 1;
   }
    
   ////////
   //k = n - 1;
   ptr_gamma_temp += r;
   ptr_beta_temp += r;
   
   //// alpha 
   tempAlpha = 0;
   for(t=1; t<=r; t++){
       tempAlpha += *ptr_gamma_temp * (*ptr_beta_temp);
       ptr_gamma_temp -= rMinusOne;
       ptr_beta_temp -= rMinusOne;}
 
   *ptr_alpha = *ptr_a - tempAlpha;
    
    ///////////////////////////// Solve /////////////////////////
    int cha;
    double *ptr_F, *ptr_X, *ptr_X_pre, *ptr_Y, *ptr_Y_pre;
    double tempX, tempY;
    
    ptr_F = &F[0][0];  ptr_Y = &Y[0][0];  ptr_Y_pre = ptr_Y;
    ptr_alpha = &alpha[0];  ptr_gamma = &gamma[0][0];
    
    ////////////////// L*Y = F ///////////////////
    // k = 0;
    tempAlpha = *ptr_alpha++;
    for(cha=0; cha<chaNumImg; cha++)
        *ptr_Y++ = *ptr_F++ / tempAlpha;
    
    for(k=1; k<r; k++)
    {
        tempAlpha = *ptr_alpha++;
        for(cha=0; cha<chaNumImg; cha++){
            maxt = k;
            tempY = 0;
            for(t=1; t<=maxt; t++){
                tempY += *ptr_gamma * (*ptr_Y_pre);
                ptr_gamma -= rMinusOne;
                ptr_Y_pre -= chaNumImg;}
            
            *ptr_Y++ = (*ptr_F++ - tempY) / tempAlpha;
            ptr_gamma += maxt * rMinusOne;
            ptr_Y_pre += maxt * chaNumImg + 1;}
        
        ptr_gamma += r;
    }
    
    for(k=r; k<n; k++)
    {
        tempAlpha = *ptr_alpha++;
        for(cha=0; cha<chaNumImg; cha++){
            maxt = r;
            tempY = 0;
            for(t=1; t<=maxt; t++){
                tempY += *ptr_gamma * (*ptr_Y_pre);
                ptr_gamma -= rMinusOne;
                ptr_Y_pre -= chaNumImg;}
            
            *ptr_Y++ = (*ptr_F++ - tempY) / tempAlpha;
            ptr_gamma += maxt * rMinusOne;
            ptr_Y_pre += maxt * chaNumImg + 1;}
        
        ptr_gamma += r;
    }
    
    ////////////// U*X = Y ////////////////////
    ptr_X = &X[n - 1][chaNumImg - 1];  ptr_X_pre = ptr_X;  ptr_Y = &Y[n - 1][chaNumImg - 1];
    ptr_beta = &beta[n - 2][0];
    
    // k = n - 1
    for(cha=0; cha<chaNumImg; cha++)
        *ptr_X -- = *ptr_Y--;
    
    for(k=n-2; k>=n-r; k--)
    {
        for(cha=0; cha<chaNumImg; cha++){
            maxt = n - k - 1;
            tempX = 0;
            for(t=1; t<=maxt; t++){
                tempX += *ptr_beta++ * (*ptr_X_pre);
                ptr_X_pre += chaNumImg;}
            
            *ptr_X-- = *ptr_Y-- - tempX;
            
            ptr_beta -= maxt;
            ptr_X_pre -= maxt * chaNumImg + 1;}
        
        ptr_beta -= r;
    }
    
    for(k=n-r-1; k>=0; k--)
    {
        for(cha=0; cha<chaNumImg; cha++){
            maxt = r;
            tempX = 0;
            for(t=1; t<=maxt; t++){
                tempX += *ptr_beta++ * (*ptr_X_pre);
                ptr_X_pre += chaNumImg;}
            
            *ptr_X-- = *ptr_Y-- - tempX;
            
            ptr_beta -= maxt;
            ptr_X_pre -= maxt * chaNumImg + 1;}
        
        ptr_beta -= r;
    }
    
}


/////////////////////////////////////
void pointDiv(double ***imgInter, double ***imgFiltered, double *count, int colDir)
{
    double *ptr_inter, *ptr_filtered, *ptr_count, value;
    ptr_inter = &imgInter[0][0][0];
    ptr_filtered = &imgFiltered[0][0][0];
    ptr_count = &count[0];
    
    // column direction
    if(colDir) {
        for(int i=0; i<rowNum; i++){
            for(int j=0; j<colNum; j++){
                value = *ptr_count++;       
                for(int k=0; k<chaNumImg; k++)
                    *ptr_filtered++ = *ptr_inter++ / value;}
        
        ptr_count -= colNum;
        }
    }
    // row direction
    else{
        for(int i=0; i<rowNum; i++){
            value = *ptr_count++;
            for(int j=0; j<colNum; j++){     
                for(int k=0; k<chaNumImg; k++)
                    *ptr_filtered++ = *ptr_inter++ / value;}
        }   
    } 
} 

void getCount(double *count, int len, int r, int step)
{
    int i, j, step_new, maxIterNum;
    double *ptr;
    
    step_new = 2 * r + 1 - step;
    maxIterNum = r + 1 + ((len - r - r - 1)/step)*step;
    ptr = &count[0];
    
    for(i = r; i < maxIterNum; i += step){
        for(j = -r; j <= r; j++)
            *ptr++ += 1.0;
        ptr -= step_new;}
    
    ptr = &count[len - 2* r - 1];
    for(j = -r; j <= r; j++)
        *ptr++ += 1.0;
}
