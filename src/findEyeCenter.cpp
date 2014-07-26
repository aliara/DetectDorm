#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

//#include <mgl2/mgl.h>

#include <iostream>
#include <queue>
#include <stdio.h>

#include "constants.h"
#include "utils.h"

// Pre-declarations
cv::Mat floodKillEdges(cv::Mat &mat);



cv::Point unscalePoint(cv::Point p, cv::Rect origSize) {
  float ratio = (((float)kFastEyeWidth)/origSize.width);
  int x = round(p.x / ratio);
  int y = round(p.y / ratio);
  return cv::Point(x,y);
}

void scaleToFastSize(const cv::Mat &src,cv::Mat &dst) {
  cv::resize(src, dst, cv::Size(kFastEyeWidth,(((float)kFastEyeWidth)/src.cols) * src.rows));
}

cv::Mat computeMatXGradient(const cv::Mat &mat) {
  cv::Mat out(mat.rows,mat.cols,CV_64F);
  
  for (int y = 0; y < mat.rows; ++y) {
    const uchar *Mr = mat.ptr<uchar>(y);
    double *Or = out.ptr<double>(y);
    
    Or[0] = Mr[1] - Mr[0];
    for (int x = 1; x < mat.cols - 1; ++x) {
      Or[x] = (Mr[x+1] - Mr[x-1])/2.0;
    }
    Or[mat.cols-1] = Mr[mat.cols-1] - Mr[mat.cols-2];
  }
  
  return out;
}

//#pragma mark Main Algorithm

void testPossibleCentersFormula(int x, int y, unsigned char weight,double gx, double gy, cv::Mat &out)
{
	// for all possible centers
	for (int cy = 0; cy < out.rows; ++cy)
	{
		double *Or = out.ptr<double>(cy);
		for (int cx = 0; cx < out.cols; ++cx)
		{
			if (x == cx && y == cy)
			{
				continue;
			}
			// create a vector from the possible center to the gradient origin
			double dx = x - cx;
			double dy = y - cy;
			// normalize d
			double magnitude = sqrt((dx * dx) + (dy * dy));
			dx = dx / magnitude;
			dy = dy / magnitude;
			double dotProduct = dx*gx + dy*gy;
			dotProduct = std::max(0.0,dotProduct);
			// square and multiply by the weight
			if (kEnableWeight)
			{
				Or[cx] += dotProduct * dotProduct * (weight/kWeightDivisor);
			}
			else
			{
				Or[cx] += dotProduct * dotProduct;
			}
		}
  }
}

cv::Point findEyeCenter(cv::Mat face, cv::Rect eye, std::string debugWindow)
{
	cv::Mat eyeROIUnscaled = face(eye);
	cv::Mat eyeROI;
	if(GBStatus)
	{
		cv::GaussianBlur( face, face, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );
	}
	if(debugL1 && GBStatus && (debugWindow == costado))
	{
		std::string gaussianBlur_window = "GaussianBlur";
		cv::namedWindow(gaussianBlur_window,CV_WINDOW_NORMAL);
		cv::moveWindow(gaussianBlur_window, gaussBlurx, gaussBlury);
		imshow(gaussianBlur_window,face);
	}
	scaleToFastSize(eyeROIUnscaled, eyeROI);
	// draw eye region
	rectangle(face,eye,1234);
	//-- Find the gradient
	cv::Mat gradientX = computeMatXGradient(eyeROI);
	///////////////cv::Mat gradientX;
	///////////////cv::Sobel( eyeROI, gradientX, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );
	///////////////cv::Mat gradientY;
	///////////////cv::Sobel( (eyeROI.t()).t(), gradientY, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );
	cv::Mat gradientY = computeMatXGradient(eyeROI.t()).t();
	cv::Mat abs_grad_x, abs_grad_y;

	cv::convertScaleAbs( gradientX, abs_grad_x );
	cv::convertScaleAbs( gradientY, abs_grad_y );

	//	cv::Mat gradientY = computeMatXGradient(eyeROI);
	//-- Normalize and threshold the gradient
	// compute all the magnitudes
	cv::Mat mags = matrixMagnitude(gradientX, gradientY);
//	cv::Mat mags;
//	addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, mags );

	//compute the threshold
	double gradientThresh = computeDynamicThreshold(mags, kGradientThreshold);
	//double gradientThresh = kGradientThreshold;
	//double gradientThresh = 0;
	//normalize
	if(debugL1 && (debugWindow == costado))
	{
		std::string eyeROI_window = debugWindow;
		cv::namedWindow(eyeROI_window,CV_WINDOW_NORMAL);
		cv::moveWindow(eyeROI_window, 640, 350);
		imshow(eyeROI_window,eyeROI);
		std::string Sobel_window = "Salida del Filtro";
		cv::namedWindow(Sobel_window,CV_WINDOW_NORMAL);
		cv::moveWindow(Sobel_window, sobelx, sobely);
		imshow(Sobel_window,mags);
	}

	for (int y = 0; y < eyeROI.rows; ++y)
	{
		double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
		const double *Mr = mags.ptr<double>(y);
		for (int x = 0; x < eyeROI.cols; ++x)
		{
			double gX = Xr[x], gY = Yr[x];
			double magnitude = Mr[x];
			if (magnitude > gradientThresh)
			{
				Xr[x] = gX/magnitude;
				Yr[x] = gY/magnitude;
			}
			else
			{
				Xr[x] = 0.0;
				Yr[x] = 0.0;
			}
		}
	}

	//imshow(debugWindow,gradientX);
	//-- Create a blurred and inverted image for weighting
	cv::Mat weight;
	GaussianBlur( eyeROI, weight, cv::Size( kWeightBlurSize, kWeightBlurSize ), 0, 0 );
	for (int y = 0; y < weight.rows; ++y)
	{
		unsigned char *row = weight.ptr<unsigned char>(y);
		for (int x = 0; x < weight.cols; ++x)
		{
			row[x] = (255 - row[x]);
		}
	}

	//imshow(debugWindow,weight);
	//-- Run the algorithm!
	cv::Mat outSum = cv::Mat::zeros(eyeROI.rows,eyeROI.cols,CV_64F);
	// for each possible center
//	printf("Eye Size: %ix%i\n",outSum.cols,outSum.rows);
//	imshow("outSum",outSum);

	for (int y = 0; y < weight.rows; ++y)
	{
		const unsigned char *Wr = weight.ptr<unsigned char>(y);
		const double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
		for (int x = 0; x < weight.cols; ++x)
		{
			double gX = Xr[x], gY = Yr[x];
			if (gX == 0.0 && gY == 0.0)
			{
				continue;
			}
			testPossibleCentersFormula(x, y, Wr[x], gX, gY, outSum);
		}
	}

	// scale all the values down, basically averaging them
	double numGradients = (weight.rows*weight.cols);

	cv::Mat out;

	outSum.convertTo(out, CV_32F,1.0/numGradients);

	//imshow(debugWindow,out);
	//-- Find the maximum point
	cv::Point maxP;
	double maxVal;
	cv::minMaxLoc(out, NULL,&maxVal,NULL,&maxP);
	//-- Flood fill the edges

	if(kEnablePostProcess)
	{
		cv::Mat floodClone;
		//double floodThresh = computeDynamicThreshold(out, 1.5);
		double floodThresh = maxVal * kPostProcessThreshold;
		cv::threshold(out, floodClone, floodThresh, 0.0f, cv::THRESH_TOZERO);
		if(kPlotVectorField)
		{
			//plotVecField(gradientX, gradientY, floodClone);
			imwrite("eyeFrame.png",eyeROIUnscaled);
		}
		cv::Mat mask = floodKillEdges(floodClone);
		//imshow(debugWindow + " Mask",mask);
		//imshow(debugWindow,out);
		// redo max
		cv::minMaxLoc(out, NULL,&maxVal,NULL,&maxP,mask);
	}
	return unscalePoint(maxP,eye);
}

//#pragma mark Postprocessing

bool floodShouldPushPoint(const cv::Point &np, const cv::Mat &mat)
{
	return inMat(np, mat.rows, mat.cols);
}

// returns a mask
cv::Mat floodKillEdges(cv::Mat &mat)
{
	rectangle(mat,cv::Rect(0,0,mat.cols,mat.rows),255);
	cv::Mat mask(mat.rows, mat.cols, CV_8U, 255);
	std::queue<cv::Point> toDo;
	toDo.push(cv::Point(0,0));
	while (!toDo.empty())
	{
		cv::Point p = toDo.front();
		toDo.pop();
		if (mat.at<float>(p) == 0.0f)
		{
			continue;
		}
		// add in every direction
		cv::Point np(p.x + 1, p.y); // right
		if (floodShouldPushPoint(np, mat)) toDo.push(np);
		np.x = p.x - 1; np.y = p.y; // left
		if (floodShouldPushPoint(np, mat)) toDo.push(np);
		np.x = p.x; np.y = p.y + 1; // down
		if (floodShouldPushPoint(np, mat)) toDo.push(np);
		np.x = p.x; np.y = p.y - 1; // up
		if (floodShouldPushPoint(np, mat)) toDo.push(np);
		// kill it
		mat.at<float>(p) = 0.0f;
		mask.at<uchar>(p) = 0;
	}
	return mask;
}
