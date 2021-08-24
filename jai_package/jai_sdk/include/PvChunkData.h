// *****************************************************************************
//
//     Copyright (c) 2017, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVCHUNKDATA_H__
#define __PVCHUNKDATA_H__

#include <PvBufferLib.h>
#include <PvPixelType.h>


class PV_BUFFER_API IPvChunkData
{
public:

    virtual ~IPvChunkData() {}

    virtual void ResetChunks() = 0;
#ifdef EBUS_PYTHON_DOXYGEN
    virtual PvResult AddChunk( uint32_t aID, bytes aData ) = 0;
#else
    virtual PvResult AddChunk( uint32_t aID, const uint8_t *aData, uint32_t aLength ) = 0;
#endif
    virtual void SetChunkLayoutID( uint32_t aChunkLayoutID ) = 0;

    virtual bool HasChunks() const = 0;
    virtual uint32_t GetChunkCount() = 0;
    virtual PvResult GetChunkIDByIndex( uint32_t aIndex, uint32_t &aID ) = 0;
    virtual uint32_t GetChunkSizeByIndex( uint32_t aIndex ) = 0;
    virtual uint32_t GetChunkSizeByID( uint32_t aID ) = 0;
#ifdef EBUS_PYTHON_DOXYGEN
    virtual ndarray GetChunkRawDataByIndex( int aIndex ) = 0;
    virtual ndarray GetChunkRawDataByID( int aID ) = 0;
#else
    virtual const uint8_t *GetChunkRawDataByIndex( uint32_t aIndex ) = 0;
    virtual const uint8_t *GetChunkRawDataByID( uint32_t aID ) = 0;
#endif
    virtual uint32_t GetChunkLayoutID() = 0;

    virtual uint32_t GetChunkDataSize() const = 0;
    virtual uint32_t GetChunkDataCapacity() const = 0;

};


class PV_BUFFER_API PvChunkData
    : public IPvChunkData
{
public:

    virtual ~PvChunkData() {}

    virtual uint64_t GetChunkDataPayloadLength() const = 0;
    virtual PvResult Alloc( uint32_t aMaximumChunkLength ) = 0;
    virtual void Free() = 0;
#ifdef EBUS_PYTHON_DOXYGEN
    virtual PvResult Attach( ndarray aRawBuffer ) = 0;
    virtual ndarray Detach() = 0;
#else
    virtual PvResult Attach( void * aRawBuffer, uint32_t aMaximumChunkLength ) = 0;
    virtual uint8_t *Detach() = 0;
#endif

};


#endif
