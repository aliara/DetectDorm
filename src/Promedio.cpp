#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <queue>
#include <stdio.h>
#include <math.h>
#include <windows.h>
#include <cstdlib>
#include <conio.h>
#include <iterator>

using namespace cv;
using namespace std;

int promedio(Mat imagen)
{
	int cont = 0;
//	cout<<"Nuevo Frame"<<"    "<<cont<<endl;
	for(int i=0; i<imagen.rows;i++)
	{
		for(int j=0; j<imagen.cols;j++)
	    {
//			gray.at<CvType<type>::type_t>(1, 1);
			cont = cont +(int)imagen.at<unsigned char>(i, j);
//        	cout<<(int)imagen.at<unsigned char>(i, j)<<endl;
	    }
	}
//	cout <<cont<<"    "<< (imagen.rows*imagen.cols)<<"    "<<cont/(imagen.rows*imagen.cols)<<endl;
	return cont/(imagen.rows*imagen.cols);
}



