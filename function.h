#pragma once
#include <OpenNI.h>
#include <stdio.h>
#include <cstdio>
#include <iostream>
#include <conio.h>
#include <tgmath.h>
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace openni;
using namespace cv;

void encode(Mat input, DepthPixel* ref, int* ROI, int maxz, int minz, double* Enstr);
void decode(Mat compimg, Mat output, int range, int* ROI, double* Enstr, Mat vismat);
void show_Enstr(Mat vismat, double *Enstr);
void normalize_map(Mat vismat);
void show_error(Mat vismat, Mat depthmat, Mat reconmat, double *maxError);
