#ifndef CONSTANTS_H
#define CONSTANTS_H

// Debugging
const bool kPlotVectorField = false;

// Size constants
const int kEyePercentTop = 25;
const int kEyePercentSide = 13;
const int kEyePercentHeight = 30;
const int kEyePercentWidth = 35;

// Preprocessing
const bool kSmoothFaceImage = false;
const float kSmoothFaceFactor = 0.005;

// Algorithm Parameters
const int kFastEyeWidth = 50;
const int kWeightBlurSize = 5;
const bool kEnableWeight = false;
const float kWeightDivisor = 150.0;
const double kGradientThreshold = 50.0;

// Postprocessing
const bool kEnablePostProcess = false;
const float kPostProcessThreshold = 0.97;

// Eye Corner
const bool kEnableEyeCorner = false;

//Config
const bool GBStatus = true;


//Debug
const bool debugL0=true;
const bool debugL1=false;
const bool debugL2=false;
const bool debugL3=false;

//Umbrales
const int persistCuadros = 5;
const int alturaOjos = 15;
const int pupilDetUmb = 10;
const char costado[] = "Left Eye";

//Sobel
const int scale = 1;
const int delta = 0;
const int ddepth = CV_16S;

//Posicion de las pantallas
const int gaussBlurx = 0;
const int gaussBlury = 350;
const int sobelx = 320;
const int sobely = 350;


#endif
