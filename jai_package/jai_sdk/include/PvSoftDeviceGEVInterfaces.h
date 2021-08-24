// *****************************************************************************
//
//     Copyright (c) 2011, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVSOFTDEVICEGEVINTERFACES_H__
#define __PVSOFTDEVICEGEVINTERFACES_H__

#include <PvVirtualDeviceLib.h>
#include <PvDeviceEnums.h>
#include <PvBuffer.h>
#include <PvGenTypes.h>


class IPvRegisterEventSink;
class IPvSoftDeviceGEV;
class IPvGenApiFactory;
class IPvRegisterFactory;
class IPvRegisterMap;


namespace PvVirtualDeviceLib
{
    class IActionHandler;
}

typedef enum
{
    PvScanTypeInvalid = -1,
    PvScanTypeArea = 0,
    PvScanTypeLine = 1,
    PvScanTypeArea3D = 2,
    PvScanTypeLine3D = 3

} PvScanType;


class PV_VIRTUAL_DEVICE_API IPvSoftDeviceGEVInfo
{
public:

    virtual ~IPvSoftDeviceGEVInfo() {}

    virtual const PvString &GetManufacturerName() = 0;
    virtual const PvString &GetModelName() = 0;
    virtual const PvString &GetDeviceVersion() = 0;
    virtual const PvString &GetManufacturerInformation() = 0;
    virtual const PvString &GetSerialNumber() = 0;

    virtual PvResult SetManufacturerName( const PvString &aValue ) = 0;
    virtual PvResult SetModelName( const PvString &aValue ) = 0;
    virtual PvResult SetDeviceVersion( const PvString &aValue ) = 0;
    virtual PvResult SetDeviceFirmwareVersion( const PvString &aValue ) = 0;
    virtual PvResult SetManufacturerInformation( const PvString &aValue ) = 0;
    virtual PvResult SetSerialNumber( const PvString &aValue ) = 0;

    virtual uint16_t GetGenICamXMLVersionMajor() const = 0;
    virtual uint16_t GetGenICamXMLVersionMinor() const = 0;
    virtual uint16_t GetGenICamXMLVersionSubMinor() const = 0;
    virtual const PvString &GetGenICamXMLProductGUID() = 0;
    virtual const PvString &GetGenICamXMLVersionGUID() = 0;

    virtual PvResult SetGenICamXMLVersion( uint16_t aMajor, uint16_t aMinor, uint16_t aSubMinor ) = 0;
    virtual PvResult SetGenICamXMLGUIDs( const PvString &aProduct, const PvString &aVersion ) = 0;
};

class PV_VIRTUAL_DEVICE_API IPvSoftDeviceGEVStatistics
{
public:

    virtual ~IPvSoftDeviceGEVStatistics() {}
    
    virtual void Reset() = 0;

    virtual uint64_t GetReadMemsReceived() const = 0;
    virtual uint64_t GetWriteMemsReceived() const = 0;
    virtual uint64_t GetReadRegsReceived() const = 0;
    virtual uint64_t GetWriteRegsReceived() const = 0;
    virtual uint64_t GetDiscoveriesReceived() const = 0;
    virtual uint64_t GetActionCommandsReceived() const = 0;
    virtual uint64_t GetForceIpsReceived() const = 0;
    virtual uint64_t GetPacketResendRequestsReceived() const = 0;
    virtual uint64_t GetPendingAcksIssued() const = 0;
    virtual uint64_t GetRetriesReceived() const = 0;
    virtual uint64_t GetRetriesReceivedMax() const = 0;

    virtual uint64_t GetEventsIssued() const = 0;
    virtual uint64_t GetEventDatasIssued() const = 0;
    virtual uint64_t GetEventRetriesIssued() const = 0;
    virtual uint64_t GetEventRetriesIssuedMax() const = 0;

    virtual uint64_t GetSessionsOpened() const = 0;
    virtual uint64_t GetSessionsClosed() const = 0;
    virtual uint64_t GetHeartbeatFailedCount() const = 0;

    virtual uint64_t GetGVSPTestPacketsIssued() const = 0;

};

class PV_VIRTUAL_DEVICE_API IPvMessageChannel
{
public:

    virtual ~IPvMessageChannel() {}

#ifdef EBUS_PYTHON_DOXYGEN
    virtual PvResult FireEvent( uint16_t aEventID, bytes *aData, uint64_t aBlockID = 0, uint16_t aChannelIndex = 0 ) = 0;
#else
    virtual PvResult FireEvent( uint16_t aEventID, uint64_t aBlockID = 0, uint16_t aChannelIndex = 0 ) = 0;
    virtual PvResult FireEvent( uint16_t aEventID, uint8_t *aData, uint32_t aDataLength, uint64_t aBlockID = 0, uint16_t aChannelIndex = 0 ) = 0;
#endif
    virtual bool IsOpened() = 0;

};

class PV_VIRTUAL_DEVICE_API IPvSoftDeviceGEVEventSink
{
public:

    virtual ~IPvSoftDeviceGEVEventSink() {}

    virtual void OnApplicationConnect( IPvSoftDeviceGEV *aDevice, const PvString &aIPAddress, uint16_t aPort, PvAccessType aAccessType ) 
    {
        PVUNREFPARAM( aDevice );
        PVUNREFPARAM( aIPAddress );
        PVUNREFPARAM( aPort );
        PVUNREFPARAM( aAccessType );
    }

    virtual void OnControlChannelStart( IPvSoftDeviceGEV *aDevice, const PvString &aMACAddress, const PvString &aIPAddress, const PvString &aMask, const PvString &aGateway, uint16_t aPort )
    {
        PVUNREFPARAM( aDevice ); 
        PVUNREFPARAM( aMACAddress );
        PVUNREFPARAM( aIPAddress );
        PVUNREFPARAM( aMask );
        PVUNREFPARAM( aGateway );
        PVUNREFPARAM( aPort );
    }

    virtual void OnApplicationDisconnect( IPvSoftDeviceGEV *aDevice ) { PVUNREFPARAM( aDevice ); }
    virtual void OnControlChannelStop( IPvSoftDeviceGEV *aDevice ) { PVUNREFPARAM( aDevice ); }
    virtual void OnDeviceResetFull( IPvSoftDeviceGEV *aDevice ) { PVUNREFPARAM( aDevice ); }
    virtual void OnDeviceResetNetwork( IPvSoftDeviceGEV *aDevice ) { PVUNREFPARAM( aDevice ); }

    virtual void OnCreateCustomRegisters( IPvSoftDeviceGEV *aDevice, IPvRegisterFactory *aFactory ) { PVUNREFPARAM( aDevice ); PVUNREFPARAM( aFactory ); }
    virtual void OnCreateCustomGenApiFeatures( IPvSoftDeviceGEV *aDevice, IPvGenApiFactory *aFactory ) { PVUNREFPARAM( aDevice ); PVUNREFPARAM( aFactory ); }
};

class PV_VIRTUAL_DEVICE_API IPvStreamingChannelSource
{
public:

    virtual ~IPvStreamingChannelSource() {}

    virtual uint32_t GetWidth() const = 0;
    virtual uint32_t GetHeight() const = 0;
    virtual uint32_t GetOffsetX() const = 0;
    virtual uint32_t GetOffsetY() const = 0;
    virtual PvPixelType GetPixelType() const = 0;
#ifdef EBUS_PYTHON_DOXYGEN
    virtual tuple GetWidthInfo() const = 0;
    virtual tuple GetHeightInfo() const = 0;
#else
    virtual void GetWidthInfo( uint32_t &aMin, uint32_t &aMax, uint32_t &aInc ) const = 0;
    virtual void GetHeightInfo( uint32_t &aMin, uint32_t &aMax, uint32_t &aInc ) const = 0;
#endif
    virtual uint32_t GetChunksSize() const = 0;
    virtual uint32_t GetPayloadSize() const = 0;
    virtual PvScanType GetScanType() const = 0;
    virtual bool GetChunkModeActive() const = 0;
    virtual bool GetChunkEnable( uint32_t aChunkID ) const = 0;
#ifdef EBUS_PYTHON_DOXYGEN
    virtual tuple GetSupportedPixelType( int aIndex ) const = 0;
    virtual tuple GetSupportedChunk( int aIndex ) const = 0;
#else
    virtual PvResult GetSupportedPixelType( int aIndex, PvPixelType &aPixelType ) const = 0;
    virtual PvResult GetSupportedChunk( int aIndex, uint32_t &aID, PvString &aName ) const = 0;
#endif

    virtual PvResult SetWidth( uint32_t aWidth ) = 0;
    virtual PvResult SetHeight( uint32_t aHeight ) = 0;
    virtual PvResult SetOffsetX( uint32_t aOffsetX ) = 0;
    virtual PvResult SetOffsetY( uint32_t aOffsetY ) = 0;
    virtual PvResult SetPixelType( PvPixelType aPixelType ) = 0;
    virtual PvResult SetChunkModeActive( bool aEnabled ) = 0;
    virtual PvResult SetChunkEnable( uint32_t aChunkID, bool aEnabled ) = 0;

    virtual void OnOpen( const PvString &aDestIP, uint16_t aDestPort ) = 0;
    virtual void OnClose() = 0;

    virtual void OnStreamingStart() = 0;
    virtual void OnStreamingStop() = 0;

    virtual PvBuffer *AllocBuffer() = 0;
    virtual void FreeBuffer( PvBuffer *aBuffer ) = 0;

    virtual PvResult QueueBuffer( PvBuffer *aBuffer ) = 0;
#ifdef EBUS_PYTHON_DOXYGEN
    virtual tuple RetrieveBuffer( void *unused ) = 0;
#else
    virtual PvResult RetrieveBuffer( PvBuffer **aBuffer ) = 0;
#endif
    virtual void AbortQueuedBuffers() = 0;

    virtual void CreateRegisters( IPvRegisterMap *aRegisterMap, IPvRegisterFactory *aFactory )
    {
        PVUNREFPARAM( aRegisterMap );
        PVUNREFPARAM( aFactory );
    }

    virtual void CreateGenApiFeatures( IPvRegisterMap *aRegisterMap, IPvGenApiFactory *aFactory ) 
    {
        PVUNREFPARAM( aRegisterMap );
        PVUNREFPARAM( aFactory );
    }

    virtual bool IsPayloadTypeSupported( PvPayloadType aPayloadType ) { PVUNREFPARAM( aPayloadType ); return false; }
    virtual void SetMultiPartAllowed( bool aAllowed ) { PVUNREFPARAM( aAllowed ); }
    virtual PvResult SetTestPayloadFormatMode( PvPayloadType aPayloadType ) { PVUNREFPARAM( aPayloadType ); return PvResult::Code::NOT_SUPPORTED; }
};

class PV_VIRTUAL_DEVICE_API IPvRegisterInfo
{
public:

    virtual ~IPvRegisterInfo() {}

    virtual const PvString &GetName() = 0;
    virtual uint32_t GetAddress() const = 0;
    virtual size_t GetLength() const = 0;
    virtual const void *GetContext() const = 0;
    virtual void *GetContext() = 0;

    virtual bool IsWritable() const = 0;
    virtual bool IsReadable() const = 0;

};

class PV_VIRTUAL_DEVICE_API IPvRegister
    : public IPvRegisterInfo
{
public:

    virtual ~IPvRegister() {}

#ifndef EBUS_PYTHON_DOXYGEN
    virtual PvResult Read( uint8_t *aData, uint32_t aByteCount, uint32_t aOffset = 0 ) = 0;
    virtual PvResult Write( const uint8_t *aData, uint32_t aByteCount, uint32_t aOffset = 0 ) = 0;

    virtual PvResult Read( uint32_t &aValue, uint32_t aOffset = 0 ) = 0;
    virtual PvResult Write( uint32_t aValue, uint32_t aOffset = 0 ) = 0;
    virtual PvResult Read( PvString &aValue ) = 0;
    virtual PvResult Write( const PvString &aValue ) = 0;
    virtual PvResult ReadFloat( float &aValue ) = 0;
    virtual PvResult WriteFloat( float aValue ) = 0;
    virtual PvResult ReadDouble( double &aValue ) = 0;
    virtual PvResult WriteDouble( double aValue ) = 0;
#else
    virtual bytes Read( uint32_t aByteCount, uint32_t aOffset = 0 ) = 0;
    virtual PvResult Write( bytes aData, uint32_t aOffset = 0 ) = 0;
    virtual uint32_t ReadInt( uint32_t aOffset = 0 ) = 0;
    virtual PvResult Write( uint32_t aValue, uint32_t aOffset = 0 ) = 0;
    virtual string ReadString() = 0;
    virtual PvResult Write( const PvString &aValue ) = 0;
    virtual float ReadFloat() = 0;
    virtual PvResult WriteFloat( float aValue ) = 0;
    virtual double ReadDouble() = 0;
    virtual PvResult WriteDouble( double aValue ) = 0;
#endif
    virtual PvResult AddEventSink( IPvRegisterEventSink *aEventSink ) = 0;
    virtual PvResult RemoveEventSink( IPvRegisterEventSink *aEventSink ) = 0;
};

class PV_VIRTUAL_DEVICE_API IPvRegisterStore
{
public:

    virtual ~IPvRegisterStore() {}

    virtual PvResult Persist( IPvRegister *aRegister, const PvString &aNameSuffix ) = 0;
};

class PV_VIRTUAL_DEVICE_API IPvRegisterEventSink
{
public:

    virtual ~IPvRegisterEventSink() {}

    virtual PvResult PreRead( IPvRegister *aRegister ) { PVUNREFPARAM( aRegister ); return PvResult::Code::OK; }
    virtual void PostRead( IPvRegister *aRegister ) { PVUNREFPARAM( aRegister ); }

    virtual PvResult PreWrite( IPvRegister *aRegister ) { PVUNREFPARAM( aRegister ); return PvResult::Code::OK; }
    virtual void PostWrite( IPvRegister *aRegister ) { PVUNREFPARAM( aRegister ); }

    virtual PvResult Persist( IPvRegister *aRegister, IPvRegisterStore *aStore ) { PVUNREFPARAM( aRegister ); PVUNREFPARAM( aStore ); return PvResult::Code::NOT_IMPLEMENTED; }
};

class PV_VIRTUAL_DEVICE_API IPvRegisterFactory
{
public:

    virtual ~IPvRegisterFactory() {}

    virtual PvResult AddRegister( const PvString &aName, uint32_t aAddress, uint32_t aLength, PvGenAccessMode aAccessMode,
        IPvRegisterEventSink *aRegisterEventSink = NULL, void *aContext = NULL ) = 0;
};

class PV_VIRTUAL_DEVICE_API IPvRegisterMap
{
public:

    virtual ~IPvRegisterMap() {}

    virtual size_t GetRegisterCount() = 0;
    virtual IPvRegister *GetRegisterByIndex( size_t aIndex ) = 0;
    virtual IPvRegister *GetRegisterByAddress( uint32_t aAddress ) = 0;

    virtual PvResult Lock() = 0;
    virtual PvResult Lock( uint32_t aTimeout ) = 0;
    virtual void Release() = 0;
};

class PV_VIRTUAL_DEVICE_API IPvGenApiFactory
{
public:

    virtual ~IPvGenApiFactory() {}
    
    virtual void SetName( const PvString &aName ) = 0;
    virtual void SetDisplayName( const PvString &aDisplayName ) = 0;
    virtual void SetCategory( const PvString &aCategory ) = 0;
    virtual void SetDescription( const PvString &aDescription ) = 0;
    virtual void SetToolTip( const PvString &aToolTip ) = 0;
    virtual void SetAccessMode( PvGenAccessMode aAccessMode ) = 0;
    virtual void SetRepresentation( PvGenRepresentation aRepresentation ) = 0;
    virtual void SetVisibility( PvGenVisibility aVisibility ) = 0;
    virtual void SetCachable( PvGenCache aCache ) = 0;
    virtual void SetPollingTime( uint32_t aPollingTime ) = 0;
    virtual void SetNameSpace( PvGenNameSpace aNameSpace ) = 0;
    virtual void SetTLLocked( bool aLocked ) = 0;
    virtual void SetStreamable( bool aStreamable ) = 0;
    virtual void SetUnit( const PvString &aUnit ) = 0;
    virtual void SetPValue( const PvString &aFeatureName ) = 0;
    virtual void SetPIsAvailable( const PvString &aFeatureName ) = 0;
    virtual void SetPIsLocked( const PvString &aFeatureName ) = 0;

    virtual void MapChunk( uint32_t aChunkID, uint32_t aAddress, size_t aLength, PvGenEndianness aEndianness = PvGenEndiannessLittle ) = 0;
    virtual void MapEvent( uint32_t aEventID, uint32_t aAddress, size_t aLength, PvGenEndianness aEndianness = PvGenEndiannessLittle, bool aAdjustAddress = true ) = 0;

    virtual void AddSelected( const PvString &aFeatureName ) = 0;
    virtual void AddInvalidator( const PvString &aFeatureName ) = 0;
    virtual void AddEnumEntry( const PvString &aName, uint32_t aValue ) = 0;
    virtual void AddEnumEntry( const PvString &aName, uint32_t aValue, const PvString &aDisplayName, PvGenNameSpace aNameSpace = PvGenNameSpaceCustom ) = 0;
    virtual void AddVariable( const PvString &aFeatureName ) = 0;

    virtual PvResult CreateInteger( IPvRegister *aRegister, int64_t aMin, int64_t aMax, int64_t aInc = 1 ) = 0;
    virtual PvResult CreateInteger( IPvRegister *aRegister = NULL ) = 0;
    virtual PvResult CreateFloat( IPvRegister *aRegister, double aMin, double aMax ) = 0;
    virtual PvResult CreateFloat( IPvRegister *aRegister = NULL ) = 0;
    virtual PvResult CreateString( IPvRegister *aRegister = NULL ) = 0;
    virtual PvResult CreateRegister( IPvRegister *aRegister = NULL ) = 0;
    virtual PvResult CreateBoolean( IPvRegister *aRegister = NULL ) = 0;
    virtual PvResult CreateCommand( IPvRegister *aRegister = NULL ) = 0;
    virtual PvResult CreateEnum( IPvRegister *aRegister = NULL ) = 0;
    virtual PvResult CreateIntSwissKnife( const PvString &aFormula ) = 0;
    virtual PvResult CreateFloatSwissKnife( const PvString &aFormula ) = 0;
    virtual PvResult CreateIntConverter( const PvString &aValueFeatureName, const PvString &aFromFormula, const PvString &aToFormula ) = 0;
    virtual PvResult CreateFloatConverter( const PvString &aValueFeatureName, const PvString &aFromFormula, const PvString &aToFormula ) = 0;

    virtual PvResult AddInvalidatorTo( const PvString &aStandardFeatureName, const PvString &aInvalidatorFeatureName ) = 0;
    virtual PvResult SetPIsAvailableFor( const PvString &aStandardFeatureName, const PvString &aPIsAvailableFeatureName ) = 0;
    virtual PvResult SetPIsAvailableForEnumEntry( const PvString &aStandardFeatureName, const PvString &aEnumEntryName, const PvString &aPIsAvailableFeatureName ) = 0;
    virtual PvResult SetPIsLockedFor( const PvString &aStandardFeatureName, const PvString &aPIsLockedFeatureName ) = 0;
    virtual PvResult SetPValueFor( const PvString &aStandardFeatureName, const PvString &aPValueFeatureName ) = 0;
    virtual PvResult SetPMinFor( const PvString &aStandardFeatureName, const PvString &aPMinFeatureName ) = 0;
    virtual PvResult SetPMaxFor( const PvString &aStandardFeatureName, const PvString &aPMaxFeatureName ) = 0;
    virtual PvResult SetPIncFor( const PvString &aStandardFeatureName, const PvString &aPIncFeatureName ) = 0;

    virtual void SetPMin( const PvString &aFeatureName ) = 0;
    virtual void SetPMax( const PvString &aFeatureName ) = 0;
    virtual void SetPInc( const PvString &aFeatureName ) = 0;
};

class PV_VIRTUAL_DEVICE_API IPvSoftDeviceGEV
{
public:

    virtual ~IPvSoftDeviceGEV() {}

    virtual PvResult AddStream( IPvStreamingChannelSource * aSource ) = 0;
    virtual PvResult SetUserSetCount( uint32_t aCount ) = 0;
    virtual PvResult SetTCPTransportEnabled( bool aEnabled ) = 0;
    virtual PvResult SetRTPProtocolEnabled( bool aEnabled ) = 0;
    virtual PvResult SetActionHandler( PvVirtualDeviceLib::IActionHandler *aActionHandler ) = 0;

    virtual PvResult RegisterEventSink( IPvSoftDeviceGEVEventSink *aEventSink ) = 0;
    virtual PvResult UnregisterEventSink( IPvSoftDeviceGEVEventSink *aEventSink ) = 0;

    virtual PvResult Start( const PvString &aIpAddress ) = 0;
    virtual PvResult Stop() = 0;

    virtual IPvSoftDeviceGEVInfo *GetInfo() = 0;
    virtual IPvRegisterMap *GetRegisterMap() = 0;
    virtual IPvSoftDeviceGEVStatistics *GetStatistics() = 0;
    virtual IPvMessageChannel *GetMessagingChannel() = 0;

    virtual PvResult GetGenICamXMLFile( PvString &aString ) const = 0;
};


#endif
