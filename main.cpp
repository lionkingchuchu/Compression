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

const double pi_ = 3.14159265358979;
int height, width;
int ROI[2] = {240,320}; //y,x
double Enstr[480 * 640];



int wasKeyboardHit()
{
    return (int)_kbhit();
}

void encode(Mat input, DepthPixel* ref, int ROI_h, int ROI_w, int maxz, int minz, double* Enstr) {
    int NMIN = 1, NMAX = 18, SIGMA = 50;
    double ALPHA = 1.05, BETA = 5, minE = 99999, maxEE = -99999, maxE = 0, minN = 99999999, maxN = 0, range = (double)maxz - (double)minz;;

    for (int i = 0; i < height; i++) { // Calculate E
        for (int j = 0; j < width; j++) {
            Enstr[i * width + j] = sqrt(pow(ROI_h - i, 2) + pow(ROI_w - j, 2));
            maxE = max(maxE, Enstr[i * width + j]);
        }
    }

    for (int i = 0; i < height; i++) { // Calculate Nstr
        for (int j = 0; j < width; j++) {
            Enstr[i * width + j] = min(1., pow((ALPHA - (Enstr[i * width + j] / maxE)), BETA)) * (NMAX - NMIN) + NMIN;
        }
    }

    for (int i = 0; i < height; i++) { // Encode
        uchar* pi = input.ptr<uchar>(i);
        for (int j = 0; j < width; j++) {
            *pi++ = (int)round(255. * double(0.5 * sin(ref[i * width + j] * Enstr[i * width + j] * 2 * pi_ / range) + 0.5));
            *pi++ = (int)round(255. * double(0.5 * cos(ref[i * width + j] * Enstr[i * width + j] * 2 * pi_ / range) + 0.5));
            *pi++ = (int)round(255. * (double)((ref[i * width + j]) - minz) / range);
        }
    }
}

void decode(Mat compimg, Mat output, int range, int ROI_h, int ROI_w, double* Enstr, Mat vismat) {
    double mink = 99999, maxk = -99999;
    double phHF, phLF, uph, k;
    double i1, i2, i3;
    int z;

    for (int i = 0; i < height; i++) {
        uchar* pc = compimg.ptr<uchar>(i);
        double* pv = vismat.ptr<double>(i);
        ushort* po = output.ptr<ushort>(i);

        for (int j = 0; j < width; j++) { // Decode from pre-calculated Nstr (Enstr)
            i1 = (*pc++)/255., i2 = (*pc++)/255., i3 = (*pc++)/255.;
            phHF = std::atan2(i1-0.5, i2-0.5);
            phLF = i3 * 2 * pi_;
            k = round(((phLF * Enstr[i * width + j] - phHF)) / (2 * pi_));
            uph = phHF + 2 * pi_ * (k);
            z = int(uph * range / (2 * pi_ * Enstr[i * width + j]));
            *po++ = z;
            //*pv++ =  // visualize internal data
        }
    }
}

void show_Enstr(Mat vismat, double *Enstr) {
    double maxEnstr = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) maxEnstr = max(maxEnstr, Enstr[i * width + j]);
    }
    for (int i = 0; i < height; i++) { //Enstr Normalize
        double* pe = vismat.ptr<double>(i);
        for (int j = 0; j < width; j++) {
            *pe++ = Enstr[i * width + j] / maxEnstr;
        }
    }
}

void normalize_map(Mat vismat) {
    double maxEnstr = 0;
    for (int i = 0; i < height; i++) {
        double* pe = vismat.ptr<double>(i);
        for (int j = 0; j < width; j++) maxEnstr = max(maxEnstr, *pe++);
    }
    for (int i = 0; i < height; i++) { //Enstr Normalize
        double* pe = vismat.ptr<double>(i);
        for (int j = 0; j < width; j++) {
            *pe++ /= maxEnstr;
        }
    }
    
}

void show_error(Mat vismat, Mat depthmat, Mat reconmat, double *maxError) {
    for (int i = 0; i < height; i++) { //Enstr Normalize
        double* pv = vismat.ptr<double>(i);
        ushort* pd = depthmat.ptr<ushort>(i);
        ushort* pr = reconmat.ptr<ushort>(i);

        for (int j = 0; j < width; j++) {
            *pv = (double)(*pd++ - *pr++);
            *maxError = max(*maxError, *pv++);
        }
    }
    for (int i = 0; i < height; i++) { //Enstr Normalize
        double* pv = vismat.ptr<double>(i);
        for (int j = 0; j < width; j++) {
            *pv++ /= *maxError;
        }
    }
}

void on_mouse(int event, int x, int y, int flags, void*) {
    if (event == EVENT_LBUTTONDOWN) {
        ROI[0] = y;
        ROI[1] = x;
    }
}

int main(int argc, char** argv)
{
    Status rc = OpenNI::initialize();
    if (rc != STATUS_OK)
    {
        printf("Initialize failed\n%s\n", OpenNI::getExtendedError());
        return 1;
    }

    Device device;
    rc = device.open(ANY_DEVICE);
    if (rc != STATUS_OK)
    {
        printf("Couldn't open device\n%s\n", OpenNI::getExtendedError());
        return 2;
    }

    VideoStream depth;
    if (device.getSensorInfo(SENSOR_DEPTH) != NULL)
    {
        rc = depth.create(device, SENSOR_DEPTH);
        if (rc != STATUS_OK)
        {
            printf("Couldn't create depth stream\n%s\n", OpenNI::getExtendedError());
            return 3;
        }
    }

    rc = depth.start();
    if (rc != STATUS_OK)
    {
        printf("Couldn't start the depth stream\n%s\n", OpenNI::getExtendedError());
        return 4;
    }
    
    VideoFrameRef frame;


    Mat rawdepth(480, 640, CV_16UC1);
    Mat depthimg(480, 640, CV_8UC1);
    Mat compimg(480, 640, CV_8UC3);
    Mat recondepth(480, 640, CV_16UC1);
    Mat errormap(480, 640, CV_64FC1);


    while (!wasKeyboardHit())
    {
        int changedStreamDummy;
        VideoStream* pStream = &depth;

        //wait a new frame
        rc = OpenNI::waitForAnyStream(&pStream, 1, &changedStreamDummy, 10000);
        if (rc != STATUS_OK)
        {
            printf("Wait failed! (timeout is %d ms)\n%s\n", 10, OpenNI::getExtendedError());
            continue;
        }

        //get depth frame
        rc = depth.readFrame(&frame);
        if (rc != STATUS_OK)
        {
            printf("Read failed!\n%s\n", OpenNI::getExtendedError());
            continue;
        }

        //check if the frame format is depth frame format
        if (frame.getVideoMode().getPixelFormat() != PIXEL_FORMAT_DEPTH_1_MM && frame.getVideoMode().getPixelFormat() != PIXEL_FORMAT_DEPTH_100_UM)
        {
            printf("Unexpected frame format\n");
            continue;
        }

        DepthPixel* pDepth = (DepthPixel*)frame.getData();

        unsigned short maxdepth = 0;
        unsigned short mindepth = 65535;
        height = frame.getHeight();
        width = frame.getWidth();

        for (int i = 0; i < height * width; i++) { // find max, min depth
            if (pDepth[i] > maxdepth) maxdepth = pDepth[i];
            if (pDepth[i] < mindepth) mindepth = pDepth[i];
        }


        for (int i = 0; i < height; i++) { // Depth -> Grayscale Img in depthimg
            ushort* p = rawdepth.ptr<ushort>(i);
            uchar* p_ = depthimg.ptr<uchar>(i);
            for (int j = 0; j < width; j++) {
                *p++ = pDepth[i * width + j];
                *p_++ = (int)(255. * ((pDepth[i * width + j]-mindepth) / double(maxdepth-mindepth)));
            }
        }

        encode(compimg, pDepth, ROI[0], ROI[1], maxdepth, mindepth, Enstr);
        decode(compimg, recondepth, maxdepth - mindepth, ROI[0], ROI[1], Enstr, errormap);

        // VISUALIZATION //
        double maxerror = 0;
        show_error(errormap, rawdepth, recondepth, &maxerror);
        //show_Enstr(errormap, Enstr);
        //normalize_map(errormap); // visualize internal data
        ///////////////////
        

        String s = "maxerror: ";
        s.append(std::to_string(maxerror));

        putText(errormap, s, Point(10,20), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);

        imshow("Window", depthimg);
        imshow("comp", compimg);
        imshow("error", errormap);
        setMouseCallback("comp", on_mouse);
        char ch = waitKey(1);
    }

    depth.stop();
    depth.destroy();
    device.close();
    OpenNI::shutdown();

    std::cout << "hit enter to exit program" << std::endl;
    system("pause");

    return 0;
}
