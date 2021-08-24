// *****************************************************************************
//
//     Copyright (c) 2018, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVMULTIPARTCONTAINER_H__
#define __PVMULTIPARTCONTAINER_H__

#include <PvMultiPartSection.h>


class PV_BUFFER_API IPvMultiPartContainerReader
{
public:

    virtual ~IPvMultiPartContainerReader() {}

    virtual uint32_t GetPartCount() = 0;

    virtual const IPvMultiPartSection *GetPart( uint32_t aIndex ) const = 0;
    virtual IPvMultiPartSection *GetPart( uint32_t aIndex ) = 0;

    virtual const IPvMultiPartSection *operator[]( uint32_t aIndex ) const = 0;
    virtual IPvMultiPartSection *operator[]( uint32_t aIndex ) = 0;

};

class PV_BUFFER_API IPvMultiPartContainerWriter
    : public IPvMultiPartContainerReader
{
public:

    virtual ~IPvMultiPartContainerWriter() {}

    virtual void Reset() = 0;
    virtual PvResult AddImagePart( PvMultiPartDataType aDataType, uint32_t aWidth, uint32_t aMaxHeight, PvPixelType aPixelType, uint32_t aOffsetX = 0, uint32_t aOffsetY = 0, uint16_t aPaddingX = 0 ) = 0;
    virtual PvResult AddJPEGPart( PvMultiPartDataType aDataType, uint32_t aMaxLength, uint8_t aFlag, uint64_t aTimestampTickFrequency, uint32_t aDataFormat ) = 0;
    virtual PvResult AddChunkPart( uint32_t aMaxLength, uint32_t aChunkLayoutID ) = 0;

    virtual PvResult SetPartIDs( uint32_t aIndex, uint32_t aSourceID, uint32_t aDataPurposeID, uint32_t aRegionID ) = 0;
    virtual PvResult SetPartZoneInfo( uint32_t aIndex, uint8_t aAdditionalZones, uint32_t aZoneDirectionMask ) = 0;

    virtual PvResult AllocAllParts() = 0;
    virtual PvResult AllocPart( uint32_t aIndex ) = 0;
    virtual PvResult AttachPart( uint32_t aIndex, uint8_t *aBuffer, uint64_t aLength ) = 0;

    virtual PvResult SetPartFinalLength( uint32_t aIndex, uint32_t aLength ) = 0;
    virtual PvResult SetPartFinalImageHeight( uint32_t aIndex, uint32_t aHeight ) = 0;

    virtual PvResult Validate() = 0;

};

class PV_BUFFER_API IPvMultiPartContainer
    : public IPvMultiPartContainerWriter
{
public:

    virtual ~IPvMultiPartContainer() {}

};


#endif
