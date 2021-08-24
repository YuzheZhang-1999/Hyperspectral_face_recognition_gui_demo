// *****************************************************************************
//
//     Copyright (c) 2012, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVDEVICEEVENTSINK_H__
#define __PVDEVICEEVENTSINK_H__

#include <PvDevice.h>
#include <PvGenParameterList.h>


class PvDevice;


class PV_DEVICE_API PvDeviceEventSink
{
public:

    PvDeviceEventSink();
    virtual ~PvDeviceEventSink();

    // Link disconnected notification
    virtual void OnLinkDisconnected( PvDevice *aDevice );

    // Link reconnected notification: deprecated, no longer in use
    virtual void OnLinkReconnected( PvDevice *aDevice );

#ifdef EBUS_PYTHON_DOXYGEN
    virtual void OnEvent( PvDevice aDevice, int aEventID, int aChannel, int aBlockID, int aTimestamp, ndarray aData );
#else
    // Messaging channel events (raw)
    virtual void OnEvent( PvDevice *aDevice, 
        uint16_t aEventID, uint16_t aChannel, uint64_t aBlockID, uint64_t aTimestamp, 
        const void *aData, uint32_t aDataLength );
#endif

    // Messaging channel events (GenICam)
    virtual void OnEventGenICam( PvDevice *aDevice,
        uint16_t aEventID, uint16_t aChannel, uint64_t aBlockID, uint64_t aTimestamp,
        PvGenParameterList *aData );

#ifdef EBUS_PYTHON_DOXYGEN
	virtual void OnCmdLinkRead(tuple aBuffer);
	virtual void OnCmdLinkWrite(tuple aBuffer);
#else
	// GigE Vision command link GenApi::IPort monitoring hooks
	virtual void OnCmdLinkRead( const void *aBuffer, int64_t aAddress, int64_t aLength );
	virtual void OnCmdLinkWrite( const void *aBuffer, int64_t aAddress, int64_t aLength );
#endif

};

#endif
