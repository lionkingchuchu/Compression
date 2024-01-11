#include "function.h"

extern int height, width;
extern double pi_;

void encode(Mat input, DepthPixel* ref, int* ROI, int maxz, int minz, double* Enstr) {
    int NMIN = 1, NMAX = 18, SIGMA = 50, ROI_x = ROI[0], ROI_y = ROI[1];
    double ALPHA = 1.05, BETA = 5, maxE = 0, range = (double)maxz - (double)minz;;

    for (int i = 0; i < height; i++) { // Calculate E
        for (int j = 0; j < width; j++) {
            Enstr[i * width + j] = sqrt(pow(ROI_x - i, 2) + pow(ROI_y - j, 2));
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

void decode_fast(Mat compimg, Mat output, int range, int* ROI, double* Enstr, Mat vismat) {
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

void decode(Mat compimg, Mat output, int range, int* ROI, double* Enstr, Mat vismat) {
    int NMIN = 1, NMAX = 18, SIGMA = 50, ROI_x = ROI[0], ROI_y = ROI[1];
    double phHF, phLF, uph, k;
    double ALPHA = 1.05, BETA = 5, maxE = 0;
    double i1, i2, i3;
    int z;

    for (int i = 0; i < height; i++) { // Calculate E
        for (int j = 0; j < width; j++) {
            Enstr[i * width + j] = sqrt(pow(ROI_x - i, 2) + pow(ROI_y - j, 2));
            maxE = max(maxE, Enstr[i * width + j]);
        }
    }

    for (int i = 0; i < height; i++) { // Calculate Nstr
        for (int j = 0; j < width; j++) {
            Enstr[i * width + j] = min(1., pow((ALPHA - (Enstr[i * width + j] / maxE)), BETA)) * (NMAX - NMIN) + NMIN;
        }
    }

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
