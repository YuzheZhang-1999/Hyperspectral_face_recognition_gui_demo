// *****************************************************************************
//
//     Copyright (c) 2019, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVH264DECODER_H__
#define __PVH264DECODER_H__

#include <PvAppUtilsLib.h>
#include <PvBuffer.h>


#ifdef WIN32
struct IMFSample;
#endif

namespace PvAppUtilsLib
{
    class IH264Decoder;
}


class PV_APPUTILS_API PvH264Decoder
{
public:

    PvH264Decoder();
    ~PvH264Decoder();

    bool IsAvailable() const;

    PvResult Reset();
    PvResult Process( const IPvH264AccessUnit *aAccessUnit );
    PvResult Retrieve( PvImage *aImage );
#ifdef WIN32
    PvResult Retrieve( IMFSample *aSample );
#endif

    void GetLastError( PvString &aString ) const;
    void ResetLastError();

private:

    PvAppUtilsLib::IH264Decoder *mThis;

    // Not implemented
    PvH264Decoder( const PvH264Decoder & );
    const PvH264Decoder &operator=( const PvH264Decoder & );

};


#endif
