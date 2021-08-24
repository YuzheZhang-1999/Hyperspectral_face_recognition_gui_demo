// *****************************************************************************
//
// Copyright (c) 2018, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVSTREAMINGCHANNELSOURCEDEFAULT_H__
#define __PVSTREAMINGCHANNELSOURCEDEFAULT_H__

#include <PvSoftDeviceGEVInterfaces.h>


class PV_VIRTUAL_DEVICE_API PvStreamingChannelSourceDefault
    : public IPvStreamingChannelSource
{
public:

    PvStreamingChannelSourceDefault( uint32_t aWidth = 640, uint32_t aHeight = 480, PvPixelType aPixelType = PvPixelMono8, size_t aBufferCount = 16 );
    virtual ~PvStreamingChannelSourceDefault();

    virtual uint32_t GetWidth() const;
    virtual uint32_t GetHeight() const;
    virtual uint32_t GetOffsetX() const;
    virtual uint32_t GetOffsetY() const;
    virtual PvPixelType GetPixelType() const;
    virtual void GetWidthInfo( uint32_t &aMin, uint32_t &aMax, uint32_t &aInc ) const;
    virtual void GetHeightInfo( uint32_t &aMin, uint32_t &aMax, uint32_t &aInc ) const;
    virtual uint32_t GetChunksSize() const;
    virtual uint32_t GetPayloadSize() const;
    virtual PvScanType GetScanType() const;
    virtual bool GetChunkModeActive() const;
    virtual bool GetChunkEnable( uint32_t aChunkID ) const;
    virtual PvResult GetSupportedPixelType( int aIndex, PvPixelType &aPixelType ) const;
    virtual PvResult GetSupportedChunk( int aIndex, uint32_t &aID, PvString &aName ) const;

    virtual PvResult SetWidth( uint32_t aWidth );
    virtual PvResult SetHeight( uint32_t aHeight );
    virtual PvResult SetOffsetX( uint32_t aOffsetX );
    virtual PvResult SetOffsetY( uint32_t aOffsetY );
    virtual PvResult SetPixelType( PvPixelType aPixelType );
    virtual PvResult SetChunkModeActive( bool aEnabled );
    virtual PvResult SetChunkEnable( uint32_t aChunkID, bool aEnabled );

    virtual void OnOpen( const PvString &aDestIP, uint16_t aDestPort );
    virtual void OnClose();

    virtual void OnStreamingStart();
    virtual void OnStreamingStop();

    virtual PvBuffer *AllocBuffer();
    virtual void FreeBuffer( PvBuffer *aBuffer );

    virtual void AbortQueuedBuffers();

    virtual void CreateRegisters( IPvRegisterMap *aRegisterMap, IPvRegisterFactory *aFactory );
    virtual void CreateGenApiFeatures( IPvRegisterMap *aRegisterMap, IPvGenApiFactory *aFactory );

    virtual bool IsPayloadTypeSupported( PvPayloadType aPayloadType );
    virtual void SetMultiPartAllowed( bool aAllowed );
    virtual PvResult SetTestPayloadFormatMode( PvPayloadType aPayloadType );

private:

    size_t mBufferCount;
    size_t mBufferAllocated;

    uint32_t mStaticWidth;
    uint32_t mStaticHeight;
    PvPixelType mStaticPixelType;

};


#endif // __PVSTREAMINGCHANNELSOURCEDEFAULT_H__
