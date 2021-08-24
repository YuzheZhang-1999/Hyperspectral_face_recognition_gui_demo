/*
 auther:    zyz
 date:      2021.0123
 function:  acquire img from go5000 camera
 version:   2.0
*/
#include "ImageProcessing.h"

PV_INIT_SIGNAL_HANDLER();

PvBuffer* gPvBuffers;
SimpleImagingLib::ImagingBuffer* gImagingBuffers;
uint32_t gBufferCount;

void CreateBuffers(PvDevice* aDevice, PvStream* aStream)
{
	gPvBuffers = NULL;
	PvGenParameterArray *lDeviceParams = aDevice->GetParameters();

	// Set device in RGB8 to match what our imaging library expects
        lDeviceParams->SetEnumValue("PixelFormat", PvPixelRGB8);
	// Get width, height from device
	int64_t lWidth = 0, lHeight = 0;
	lDeviceParams->GetIntegerValue("Width", lWidth);
	lDeviceParams->GetIntegerValue("Height", lHeight);

	// Use min of BUFFER_COUNT and how many buffers can be queued in PvStream.
	gBufferCount = (aStream->GetQueuedBufferMaximum() < BUFFER_COUNT) ?
		aStream->GetQueuedBufferMaximum() :
		BUFFER_COUNT;

	// Create our image buffers which are holding the real memory buffers
	gImagingBuffers = new SimpleImagingLib::ImagingBuffer[gBufferCount];
	for (uint32_t i = 0; i < gBufferCount; i++)
	{
		gImagingBuffers[i].AllocateImage(static_cast<uint32_t>(lWidth), static_cast<uint32_t>(lHeight), 3);
	}

	// Creates, eBUS SDK buffers, attach out image buffer memory
	gPvBuffers = new PvBuffer[gBufferCount];
	for (uint32_t i = 0; i < gBufferCount; i++)
	{
		// Attach the memory of our imaging buffer to a PvBuffer. The PvBuffer is used as a shell
		// that allows directly acquiring image data into the memory owned by our imaging buffer
		gPvBuffers[i].GetImage()->Attach(gImagingBuffers[i].GetTopPtr(),
			static_cast<uint32_t>(lWidth), static_cast<uint32_t>(lHeight), PvPixelRGB8);

		// Set eBUS SDK buffer ID to the buffer/image index
		gPvBuffers[i].SetID(i);
	}

	// Queue all buffers in the stream
	for (uint32_t i = 0; i < gBufferCount; i++)
	{
		aStream->QueueBuffer(gPvBuffers + i);
	}
}

void ReleaseBuffers()
{
	cout << "Releasing buffers" << endl;

	if (gPvBuffers != NULL)
	{
		delete[] gPvBuffers;
		gPvBuffers = NULL;
	}

	if (gImagingBuffers != NULL)
	{
		delete[]gImagingBuffers;
		gImagingBuffers = NULL;
	}
}


Jai_Camera_Class::Jai_Camera_Class()
{

}


Jai_Camera_Class::~Jai_Camera_Class()
{

}


void Jai_Camera_Class::open()
{
    // Find the camera and connect to it
    if(Select_My_Device(&camera_id)) // The device is reachable and no one is connected to it
    {
        PvResult result;
        //PvStream *lStream;
        std::cout << "Opening stream to device." << std::endl;

        device = static_cast<PvDevice *>(PvDevice::CreateAndConnect(camera_id,&result));
        if(device == NULL || !result.IsOK()) // for some reason the device is not reachable anymore...
                {
                        PvDevice::Free(device);
                        throw std::runtime_error("Device found, but can't connect"); // TODO custom exeception
                }
    }
    else // the device is reachable, but someone is connected to it
        throw std::runtime_error("Device not found"); // TODO custom exeception




    // Open a stream with the device
    PvResult result;
    stream =static_cast<PvStream *>(PvStream::CreateAndOpen(camera_id,&result));
    if(stream == NULL || !result.IsOK())
    {
		throw std::runtime_error("Unable to stream from thie device!");
		PvStream::Free(stream);
		PvDevice::Free(device);
     }

    CreateBuffers(device, stream);

    device_parameters = device->GetParameters();
    device_parameters->SetEnumValue("BalanceWhiteAuto","Continuous");
    device_parameters->SetEnumValue("GainAuto","Continuous");
    device_parameters->SetEnumValue("ExposureAuto","Continuous");
    device_parameters->SetFloatValue("AcquisitionFrameRate",50);

    PvGenParameter * device_parameter = device_parameters->Get("PixelFormat");
    //std::cout<<"Pixel Format is "<< device_parameter<<std::endl;
    PvGenEnum * myPixelFormatParameter = dynamic_cast<PvGenEnum *>(device_parameter);
    myPixelFormatParameter->SetValue("RGB8");
    stream_parameters = stream->GetParameters(); // get stream parameters (for future usages)
    device_parameters->SetEnumValue("PixelFormat", PvPixelRGB8);


    std::cout << "Enable streaming on the controller." << std::endl;

    device->StreamEnable();


}

void Jai_Camera_Class::close()
{
    image_thread->interrupt();
    image_thread->join(); // TODO is it necessary?
    image_thread.reset();

    device->StreamDisable();
    std::cout <<"stream disable!"<< std::endl;
    stream->AbortQueuedBuffers();
    while (stream->GetQueuedBufferCount() > 0)
    {
        PvBuffer *Buffer = NULL;
        PvResult lOperationResult;
        stream->RetrieveBuffer(&Buffer, &lOperationResult);
    }
    ReleaseBuffers();

    stream->Close();
    PvStream::Free(stream);

    device->Disconnect();
    PvDevice::Free(device);
    std::cout <<"Device close!"<< std::endl;

}

/*
 		    PvBuffer BufferDisplay;
                    PvImage *ImageDiaplay = BufferDisplay.GetImage();
                    ImageDiaplay->Alloc(image->GetWidth(),image->GetHeight(), PvPixelRGB8);
                    myConverter.Convert(buffer, &BufferDisplay);
		    aFilterRGB->WhiteBalance(&BufferDisplay);
                    buffer->GetImage()->Attach(data,2048,2560,PvPixelRGB8);
*/

void Jai_Camera_Class::my_acquireImages(uchar * data)
{

    static PvBuffer *buffer = NULL;
    //static PvImage *image = NULL;
    PvResult buffer_result, operation_result;
    int count=0;
    while(1)
    {
        operation_result = stream->RetrieveBuffer(&buffer, &buffer_result, 1000);

        //std::cout <<"Ready output image!"<< std::endl;
        if(operation_result.IsOK()) // operation results says about the retrieving from the pipeline
        {
            if(buffer_result.IsOK()) // buffer results says about the retrieved buffer status
            {

                if(buffer->GetPayloadType() == PvPayloadTypeImage)
                {
                    buffer->GetImage()->Attach(data,2056,2464,PvPixelRGB8);
                    //std::cout<<count<<std::endl;
                    count++;
                    if(count>=10000)
                        count=0;
                }
                else
                    std::cout << " (buffer does not contain image)";

            }
            else
            {
                std::cout <<"error 1"<< "\r";
            }
            stream->QueueBuffer(buffer);

        }
        else
        {
            std::cout <<"error 2"<< "\r";
        }
        boost::this_thread::interruption_point();
    }

}


void Jai_Camera_Class::acquireImages(uchar* data)
{
    //pthread_t tid;
    //pthread_create(&tid,NULL,Jai_Camera_Class::my_acquireImages,(void*)data);
    //std::thread t(&Jai_Camera_Class::my_acquireImages,this,data);
     image_thread.reset(new boost::thread(boost::bind(&Jai_Camera_Class::my_acquireImages, this, data)));
}



bool Jai_Camera_Class::Select_My_Device(PvString *aConnectionID)
{
	const PvDeviceInfo *lSelectedDI = NULL;
	PvSystem lSystem;

	std::cout << std::endl << "Detecting devices." << std::endl;

	for (;; )
	{
		if (gStop)
		{
			return false;
		}
		lSystem.Find();

		// Detect, select device.
		vector<const PvDeviceInfo *> lDIVector;
		for (uint32_t i = 0; i < lSystem.GetInterfaceCount(); i++)
		{
			const PvInterface *lInterface = dynamic_cast<const PvInterface *>(lSystem.GetInterface(i));
			if (lInterface != NULL)
			{
				std::cout << "   " << lInterface->GetDisplayID().GetAscii() << std::endl;
				for (uint32_t j = 0; j < lInterface->GetDeviceCount(); j++)
				{
					const PvDeviceInfo *lDI = dynamic_cast<const PvDeviceInfo *>(lInterface->GetDeviceInfo(j));
					if (lDI != NULL)
					{
						lDIVector.push_back(lDI);
						std::cout << "[" << (lDIVector.size() - 1) << "]" << "\t" << lDI->GetDisplayID().GetAscii() << std::endl;
					}
				}
			}
		}

		if (lDIVector.size() == 0)
		{
			std::cout << "No device found!" << std::endl;
		}

		std::cout << "[" << lDIVector.size() << "] to abort" << std::endl;
		std::cout << "[" << (lDIVector.size() + 1) << "] to search again" << std::endl << std::endl;

		std::cout << "Enter your action or device selection?" << std::endl;
		std::cout << ">";

		// Read device selection, optional new IP address.
		uint32_t lIndex = 1;
		PV_DISABLE_SIGNAL_HANDLER();
		//cin >> lIndex;
		PV_ENABLE_SIGNAL_HANDLER();
		if (lIndex == lDIVector.size())
		{
			// We abort the selection process.
			return false;
		}
		else if (lIndex < lDIVector.size())
		{
			// The device is selected
			lSelectedDI = lDIVector[lIndex];
			break;
		}
	}

	// If the IP Address valid?
	if (lSelectedDI->IsConfigurationValid())
	{
		std::cout << std::endl;
		*aConnectionID = lSelectedDI->GetConnectionID();
		return true;
	}

	if ((lSelectedDI->GetType() == PvDeviceInfoTypeUSB) || (lSelectedDI->GetType() == PvDeviceInfoTypeU3V))
	{
		std::cout << "This device must be connected to a USB 3.0 (SuperSpeed) port." << std::endl;
		return false;
	}

	std::cout << std::endl;
	return false;
}

void Jai_Camera_Class::start()
{
    open();

    // All is set and ready, now say to the camera to start sending images
    std::cout<<"Sending StartAcquisition command to device"<<std::endl;
    device_parameters->ExecuteCommand("AcquisitionStart");
    std::cout <<"stream open!"<< std::endl;
    // Start the thread which polls images from the camera buffer

 

}

void Jai_Camera_Class::stop()
{
    // Tell the camera to stop sending images
    device_parameters->ExecuteCommand( "AcquisitionStop" );
    close();
}



extern "C"
{
    Jai_Camera_Class My_Jai_Camera_Class;

    void start()
    {
        return My_Jai_Camera_Class.start();
    }

    void acquireImages(uchar* data)
    {
        return My_Jai_Camera_Class.acquireImages(data);
    }
    void stop()
    {
        return My_Jai_Camera_Class.stop();
    }


}


int main()
{

    return 0;
}


