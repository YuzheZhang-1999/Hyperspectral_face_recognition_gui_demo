// *****************************************************************************
//
//     Copyright (c) 2007, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVGENSTRING_H__
#define __PVGENSTRING_H__

#include <PvGenParameter.h>


class PvGenString : public PvGenParameter
{
public:

	PV_GENICAM_API PvResult SetValue( const PvString &aValue );
#ifndef EBUS_PYTHON_DOXYGEN
	PV_GENICAM_API PvResult GetValue( PvString &aValue ) const;
    PV_GENICAM_API PvResult GetMaxLength( int64_t &aMaxLength ) const;
#else
	PvString GetValue() const;
    int64_t GetMaxLength() const;
#endif

protected:

	PvGenString();
	virtual ~PvGenString();
 
private:

    // Not implemented
    PvGenString( const PvGenString & );
	const PvGenString &operator=( const PvGenString & );
};

#endif
