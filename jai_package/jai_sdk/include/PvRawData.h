// *****************************************************************************
//
//     Copyright (c) 2010, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVRAWDATA_H__
#define __PVRAWDATA_H__

#include <PvBufferLib.h>
#include <PvPixelType.h>


class PV_BUFFER_API PvRawData
{
public:

    virtual ~PvRawData() {}

    virtual uint64_t GetPayloadLength() const = 0;

    virtual PvResult Alloc( uint64_t aPayloadLength, uint32_t aMaximumChunkLength = 0 ) = 0;
    virtual void Free() = 0;

    virtual PvResult Attach( void * aRawBuffer, uint64_t aPayloadLength, uint32_t aMaximumChunkLength = 0 ) = 0;
    virtual uint8_t *Detach() = 0;

};


#endif
