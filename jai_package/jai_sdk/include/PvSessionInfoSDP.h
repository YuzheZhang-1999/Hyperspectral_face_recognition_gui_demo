// *****************************************************************************
//
//     Copyright (c) 2007, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PV_SESSIONINFOSDP_H__
#define __PV_SESSIONINFOSDP_H__

#include <PvSystemLib.h>
#include <PvSystemEnums.h>


class PvInterface;


class PV_SYSTEM_API PvSessionInfoSDP
{
public:

	virtual ~PvSessionInfoSDP();
    PvSessionInfoSDP *Copy() const;

    uint32_t GetSessionID() const;
    uint32_t GetSessionVersion() const;
    std::string GetSessionName() const;
    std::string GetSessionInfo() const;

    PvString GetConnectionIP() const;
    PvString GetMediaType() const;
    uint16_t GetMediaTransportPort() const;
    PvString GetMediaSampling() const;
    uint16_t GetMediaDepth() const;
    PvString GetMediaColorimetry() const;
    uint32_t GetMediaWidth() const;
    uint32_t GetMediaHeight() const;
    uint32_t GetMediaFramerate() const;
    bool GetMediaInterlaced() const;
    PvString GetContent() const;

    PvString GetLicenseMessage() const;
    PvString GetDisplayID() const;
    PvString GetUniqueID() const;
    PvString GetConnectionID() const;
    bool IsConfigurationValid() const;
    bool IsLicenseValid() const;

    const PvInterface *GetInterface() const;

protected:

#ifndef PV_GENERATING_DOXYGEN_DOC

	// PvSessionInfoSDP( PvSessionInfoSDPType, PvInterface *aInterface );
	const PvSessionInfoSDP &operator=( const PvSessionInfoSDP &aFrom );

    void Init();

    void SetLicenseValid( bool aValue ) { mLicenseValid = aValue; }
    void SetConnectionID( const std::string &aValue ) { *mConnectionID = aValue; }
    void SetDisplayID( const std::string &aValue ) { *mDisplayID = aValue; }
    void SetUniqueID( const std::string &aValue ) { *mUniqueID = aValue; }
    void SetCompareID( const std::string &aValue ) { *mCompareID = aValue; }
    void SetLicenseMessage( const std::string &aValue ) { *mLicenseMessage = aValue; }

    std::string *GetCompareID() { return mCompareID; }

#endif // PV_GENERATING_DOXYGEN_DOC

private:

    bool mLicenseValid;
    std::string *mConnectionID;
    std::string *mDisplayID;
    std::string *mUniqueID;
    std::string *mCompareID;
    std::string *mLicenseMessage;
    std::string *mContent;

    const PvInterface *mInterface;

	 // Not implemented
    PvSessionInfoSDP();
    PvSessionInfoSDP( const PvSessionInfoSDP & );

};


#endif // __PV_SESSIONINFOSDP_H__

