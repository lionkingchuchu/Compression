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
#include "function.h"

using namespace openni;
using namespace cv;

double pi_ = 3.14159265358979;
int height, width;
int ROI[2] = {240,320}; //y,x
double Enstr[480 * 640];

int wasKeyboardHit()
{
    return (int)_kbhit();
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

        encode(compimg, pDepth, ROI, maxdepth, mindepth, Enstr);
        decode(compimg, recondepth, maxdepth - mindepth, ROI, Enstr, errormap);

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
