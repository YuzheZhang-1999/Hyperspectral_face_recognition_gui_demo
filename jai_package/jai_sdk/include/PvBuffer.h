// *****************************************************************************
//
//     Copyright (c) 2007, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVBUFFER_H__
#define __PVBUFFER_H__

#include <PvBufferLib.h>
#include <PvPayloadType.h>
#include <PvImage.h>
#include <PvImage.h>
#include <PvMultiPartContainer.h>
#include <PvRawData.h>
#include <PvH264AccessUnit.h>


namespace PvBufferLib
{
    class Buffer;
}

namespace PvStreamLib
{
	class Pipeline;
}

namespace PvVirtualDeviceLib
{
    class PacketizerGVSP;
    class PacketizerGVSPMultiPart;
    class PacketizerRTPUncompressed;
    class PacketizerRTPH264;
    class Transmitter;
}

class PvPipeline;
class PvStream;
class PvBufferConverter;
class PvBufferConverterRGBFilter;
class PvDeInterlacer;
class PvImage;
class PvTransmitterGEV;


class PV_BUFFER_API PvBuffer
    : public IPvChunkData
{
public:

    PvBuffer( PvPayloadType aPayloadType = PvPayloadTypeImage );
    virtual ~PvBuffer();

    PvPayloadType GetPayloadType() const;

    PvImage *GetImage();
    const PvImage *GetImage() const;
    PvRawData *GetRawData();
    const PvRawData *GetRawData() const;
    PvChunkData *GetChunkData();
    const PvChunkData *GetChunkData() const;
    IPvMultiPartContainer *GetMultiPartContainer();
    const IPvMultiPartContainer *GetMultiPartContainer() const;
    IPvH264AccessUnit *GetH264AccessUnit();
    const IPvH264AccessUnit *GetH264AccessUnit() const;
    
#ifdef EBUS_PYTHON_DOXYGEN
    ndarray GetDataPointer();
#else
    const uint8_t *GetDataPointer() const;
    uint8_t *GetDataPointer();
#endif

    uint64_t GetID() const;
    void SetID( uint64_t aValue );

    bool IsExtendedID() const;
    bool IsAttached() const;
    bool IsAllocated() const;

    uint32_t GetAcquiredSize() const;
    uint32_t GetRequiredSize() const;
    uint32_t GetSize() const;

    PvResult Reset( PvPayloadType aPayloadType = PvPayloadTypeImage );

    PvResult Alloc( uint32_t aSize );
    PvResult AllocChunk( uint32_t aSize );
    void Free();
    void FreeChunk();

#ifdef EBUS_PYTHON_DOXYGEN
    PvResult Attach( ndarray aBuffer );
    ndarray Detach();
#else
    PvResult Attach( void * aBuffer, uint32_t aSize );
    uint8_t *Detach();
#endif

    uint64_t GetBlockID() const;
    PvResult GetOperationResult() const;
    uint64_t GetTimestamp() const;
    uint64_t GetReceptionTime() const;
    uint64_t GetDuration() const;

    PvResult SetTimestamp( uint64_t aTimestamp );
    PvResult SetDuration( uint64_t aDuration );
    PvResult SetBlockID( uint64_t aBlockID );
    PvResult SetReceptionTime( uint64_t aReceptionTime );

    uint32_t GetPacketsRecoveredCount() const;
    uint32_t GetPacketsRecoveredSingleResendCount() const;
    uint32_t GetResendGroupRequestedCount() const;
    uint32_t GetResendPacketRequestedCount() const;
    uint32_t GetLostPacketCount() const;
    uint32_t GetIgnoredPacketCount() const;
    uint32_t GetRedundantPacketCount() const;
    uint32_t GetPacketOutOfOrderCount() const;

    PvResult GetMissingPacketIdsCount( uint32_t& aCount );
    PvResult GetMissingPacketIds( uint32_t aIndex, uint32_t& aPacketIdLow, uint32_t& aPacketIdHigh );

    void ResetChunks();
#ifdef EBUS_PYTHON_DOXYGEN
    PvResult AddChunk( uint32_t aID, ndarray aData );
#else
    PvResult AddChunk( uint32_t aID, const uint8_t *aData, uint32_t aLength );
#endif
    void SetChunkLayoutID( uint32_t aChunkLayoutID );

    bool HasChunks() const;
    uint32_t GetChunkCount();
    PvResult GetChunkIDByIndex( uint32_t aIndex, uint32_t &aID );
    uint32_t GetChunkSizeByIndex( uint32_t aIndex );
    uint32_t GetChunkSizeByID( uint32_t aID );
#ifdef EBUS_PYTHON_DOXYGEN
    ndarray GetChunkRawDataByIndex( int aIndex );
    ndarray GetChunkRawDataByID( int aID );
#else
    const uint8_t *GetChunkRawDataByIndex( uint32_t aIndex );
    const uint8_t *GetChunkRawDataByID( uint32_t aID );
#endif
    uint32_t GetPayloadSize() const;
    uint32_t GetChunkLayoutID();

    uint32_t GetChunkDataSize() const;
    uint32_t GetChunkDataCapacity() const;

    bool IsHeaderValid() const;
    bool IsTrailerValid() const;

private:

    // Not implemented
    PvBuffer( const PvBuffer & );
    const PvBuffer &operator=( const PvBuffer & );

    friend class PvStreamLib::Pipeline;
    friend class PvPipeline;
    friend class PvStream;
    friend class PvBufferConverter;
    friend class PvBufferConverterRGBFilter;
    friend class PvDeInterlacer;
    friend class PvTransmitterGEV;
    friend class PvVirtualDeviceLib::PacketizerGVSP;
    friend class PvVirtualDeviceLib::PacketizerGVSPMultiPart;
    friend class PvVirtualDeviceLib::PacketizerRTPUncompressed;
    friend class PvVirtualDeviceLib::PacketizerRTPH264;
    friend class PvVirtualDeviceLib::Transmitter;

    PvBufferLib::Buffer * mThis;
};


#endif
