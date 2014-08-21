#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <fstream>

using namespace std;


void calculos(std::string nombre, cv::Mat src)
{
	static bool cabecera = true;
	ofstream myfile ("cerrados.csv",ios::app);
	if (myfile.is_open())
	{
		if(cabecera)
		{
			myfile<<"nombre,mean,countNonZero,dft mean,"<<endl;
			cabecera=false;
		}
		myfile<<nombre<<",";
		cv::Mat fflo, ffou,covar,meanCovar;
		myfile<<cv::mean(src)<<",";
		myfile<<cv::countNonZero(src)<<",";
		src.convertTo(fflo, CV_32FC1); // or CV_32F works (too)
		dft(fflo,ffou);
		myfile<<cv::mean(ffou)<<endl;
		ffou.release();
		fflo.release();
		covar.release();
		meanCovar.release();
		myfile.close();
	}
	else cout << "Unable to open file";



}



