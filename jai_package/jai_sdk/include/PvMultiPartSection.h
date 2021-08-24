// *****************************************************************************
//
//     Copyright (c) 2018, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVMULTIPARTSECTION_H__
#define __PVMULTIPARTSECTION_H__

#include <PvBufferLib.h>
#include <PvImage.h>
#include <PvChunkData.h>
#include <PvImageJPEG.h>


typedef enum
{
    PvMultiPartInvalid = 0,
    PvMultiPart2DImage = 1,
    PvMultiPart2DPlaneOfBiPlanar = 2,
    PvMultiPart2DPlaneOfTriPlanar = 3,
    PvMultiPart2DPlaneOfQuadPlanar = 4,
    PvMultiPart3DImage = 5,
    PvMultiPart3DPlaneOfBiPlanar = 6,
    PvMultiPart3DPlaneOfTriPlanar = 7,
    PvMultiPart3DPlaneOfQuadPlanar = 8,
    PvMultiPartConfidenceMap = 9,
    PvMultiPartChunkData = 10,
    PvMultiPartJPEGImage = 11,
    PvMultiPartJPEG2000Image = 12

} PvMultiPartDataType;


class PV_BUFFER_API IPvMultiPartSection
{
public:

    virtual ~IPvMultiPartSection() {}

    virtual const uint8_t *GetDataPointer() const = 0;
    virtual uint8_t *GetDataPointer() = 0;

    virtual uint32_t GetSize() const = 0;
    virtual uint32_t GetEffectiveSize() const = 0;

    virtual PvMultiPartDataType GetDataType() const = 0;
    virtual operator IPvImage *() = 0;
    virtual operator IPvImageJPEG *() = 0;
    virtual operator IPvChunkData *() = 0;
    virtual IPvImage *GetImage() = 0;
    virtual IPvImageJPEG *GetJPEG() = 0;
    virtual IPvChunkData *GetChunkData() = 0;

    virtual uint32_t GetSourceID() const = 0;
    virtual uint32_t GetDataPurposeID() const = 0;
    virtual uint32_t GetRegionID() const = 0;

    virtual uint32_t GetAdditionalZones() const = 0;
    virtual uint32_t GetZoneDirectionMask() const = 0;

};


#endif
