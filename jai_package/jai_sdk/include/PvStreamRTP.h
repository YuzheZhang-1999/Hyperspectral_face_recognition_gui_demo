// *****************************************************************************
//
//     Copyright (c) 2013, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVSTREAMRTP_H__
#define __PVSTREAMRTP_H__

#include <PvStream.h>

class PvSessionInfoSDP;

class PV_STREAM_API PvStreamRTP 
    : public PvStream
{
public:
	
	PvStreamRTP();
	virtual ~PvStreamRTP();

    // Not supported with RTP
    PvResult Open( const PvDeviceInfo *aDeviceInfo );
    PvResult Open( const PvString &aInfo );

    // Explicitly set payload type
    PvResult PrepareFor( uint32_t aWidth, uint32_t aHeight, PvPixelType aPixelType, uint16_t aPaddingX = 0 );
    PvResult PrepareForH264();
    PvResult ResetPayloadType();

    // Open methods
    PvResult Open( const PvSessionInfoSDP *aSession, const PvString &aLocalIPAddress, uint16_t aLocalPort = 0 ); // From SDP
    PvResult Open( const PvString &aLocalIpAddress, uint16_t aLocalPort ); // Unicast
    PvResult Open( const PvString &aMulticastAddress, uint16_t aDataPort, const PvString &aLocalIpAddress ); // Multicast
    PvResult OpenTCP( const PvString &aServerIpAddress, uint16_t aServerPort ); // TCP

    PvStreamType GetType() const;

    uint16_t GetLocalPort() const;
    PvString GetLocalIPAddress() const;
    PvString GetMulticastIPAddress() const;
    PvString GetTCPServerIPAddress() const;
    uint16_t GetTCPServerPort() const;

    uint32_t GetThreadPriority() const;
    PvResult SetThreadPriority( uint32_t aPriority );

private:

private:

	 // Not implemented
	PvStreamRTP( const PvStreamRTP & );
    const PvStreamRTP &operator=( const PvStreamRTP & );
};


#endif
