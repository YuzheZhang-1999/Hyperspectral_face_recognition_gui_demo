#ifndef IMAGEPROCESSING_H
#define IMAGEPROCESSING_H

#include <PvSampleUtils.h>
#include <PvDevice.h>
#include <PvBuffer.h>
#include <PvStream.h>
#include <PvStreamU3V.h>
#include <PvPipeline.h>
#include <PvPixelType.h>
#include <ImagingBuffer.h>
#include <ImagingContrastFilter.h>
#include <PvBufferWriter.h>
#include <PvBufferConverter.h>
#include <PvBufferConverterRGBFilter.h>


#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <pthread.h>
#include <thread>

#include <boost/function.hpp>
#include <boost/thread.hpp>

#include <cstring>
#include <opencv2/opencv.hpp>

// Desired buffer count
#define BUFFER_COUNT ( 16 )


class Jai_Camera_Class
{
    // FIXME it is necessary to specialize the exceptions...
    PvDevice * device;
    PvStream * stream;
    PvPipeline * pipeline;
    PvString camera_id;
    PvResult Result;
    

    PvGenParameterArray * device_parameters;
    PvGenParameterArray * stream_parameters;

    boost::shared_ptr<boost::thread> image_thread;

public:
    boost::function<void(const cv::Mat &image)> callback;


    Jai_Camera_Class();
   ~Jai_Camera_Class();

    void start();
    void stop();
    void my_acquireImages(uchar* data);
    void acquireImages(uchar* data);
    unsigned char *image_data_buffer;

private:
    void open();
    void close();
    bool Select_My_Device(PvString *aConnectionID);

};

void CreateBuffers(PvDevice* aDevice);
void ReleaseBuffers();


#endif
