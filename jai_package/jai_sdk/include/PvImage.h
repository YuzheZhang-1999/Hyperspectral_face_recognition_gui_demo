// *****************************************************************************
//
//     Copyright (c) 2010, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVIMAGE_H__
#define __PVIMAGE_H__

#include <PvPixelType.h>


class PvBuffer;
class PvTruesenseConverter;


class PV_BUFFER_API IPvImage
{
public:

    virtual ~IPvImage() {}

#ifdef EBUS_PYTHON_DOXYGEN
    virtual ndarray GetDataPointer() = 0;
#else
    virtual const uint8_t *GetDataPointer() const = 0;
    virtual uint8_t *GetDataPointer() = 0;
#endif
    virtual uint32_t GetImageSize() const = 0;

    virtual uint32_t GetWidth() const = 0;
    virtual uint32_t GetHeight() const = 0;
    virtual PvPixelType GetPixelType() const = 0;
    virtual uint32_t GetBitsPerPixel() const = 0;

    virtual uint32_t GetOffsetX() const = 0;
    virtual uint32_t GetOffsetY() const = 0;
    virtual uint16_t GetPaddingX() const = 0;
    virtual uint16_t GetPaddingY() const = 0;

    virtual uint64_t GetBlockID() const = 0;
    virtual uint64_t GetTimestamp() const = 0;
    virtual uint64_t GetReceptionTime() const = 0;

};


class PV_BUFFER_API PvImage
    : public IPvImage
{
public:

    virtual uint32_t GetMaximumChunkLength() const = 0;

    static uint32_t GetPixelSize( PvPixelType aPixelType );
    static bool IsPixelColor( PvPixelType aPixelType );
    static bool IsPixelHighRes( PvPixelType aPixelType );
    static uint32_t GetBitsPerComponent( PvPixelType aPixelType );
    static PvString PixelTypeToString( PvPixelType aPixelType );

    virtual uint32_t GetRequiredSize() const = 0;
    virtual uint32_t GetEffectiveImageSize() const = 0;

    virtual void SetOffsetX( uint32_t aValue ) = 0;
    virtual void SetOffsetY( uint32_t aValue ) = 0;

    virtual bool IsAllocated() const = 0;
    virtual bool IsAttached()  const = 0;
    virtual PvResult Alloc( uint32_t aSizeX, uint32_t aSizeY, PvPixelType aPixelType, uint16_t aPaddingX = 0, uint16_t aPaddingY = 0, uint32_t aMaximumChunkLength = 0 ) = 0;
    virtual void Free() = 0;
#ifdef EBUS_PYTHON_DOXYGEN
    virtual PvResult Attach( ndarray aRawBuffer, PvPixelType aPixelType, int aPaddingX, int aPaddingY, int aMaximumChunkLength ) = 0;
    virtual ndarray Detach() = 0;
#else
    virtual PvResult Attach( void * aRawBuffer, uint32_t aSizeX, uint32_t aSizeY, PvPixelType aPixelType, uint16_t aPaddingX = 0, uint16_t aPaddingY = 0, uint32_t aMaximumChunkLength = 0 ) = 0;
    virtual uint8_t *Detach() = 0;
#endif

    virtual bool IsPartialLineMissing() const = 0;
    virtual bool IsFullLineMissing() const = 0;
    virtual void SetEOFByLineCount( bool aValue = true ) = 0;
    virtual bool IsEOFByLineCount() const = 0;
    virtual bool IsInterlacedEven() const = 0;
    virtual bool IsInterlacedOdd() const = 0;
    virtual bool IsImageDropped() const = 0;
    virtual bool IsDataOverrun() const = 0;

    virtual PvBuffer *GetBuffer() = 0;

private:

	friend class PvTruesenseConverter;

};


#endif
