// *****************************************************************************
//
//     Copyright (c) 2018, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVIMAGEJPEG_H__
#define __PVIMAGEJPEG_H__

#include <PvBufferLib.h>


class PV_BUFFER_API IPvImageJPEG
{
public:

    virtual ~IPvImageJPEG() {}

    virtual const uint8_t *GetDataPointer() const = 0;
    virtual uint8_t *GetDataPointer() = 0;

    virtual uint8_t GetFlag() const = 0;
    virtual uint64_t GetTimestampTickFrequency() const = 0;
    virtual uint32_t GetDataFormat() const = 0;

};


#endif // __PVIMAGEJPEG_H__
