// *****************************************************************************
//
//     Copyright (c) 2007, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVGENINTEGER_H__
#define __PVGENINTEGER_H__

#include <PvGenParameter.h>


class PvGenInteger : public PvGenParameter
{
public:

	PV_GENICAM_API PvResult SetValue( int64_t aValue );
#ifdef EBUS_PYTHON_DOXYGEN
	int64_t GetValue() const;

	int64_t GetMin() const;
	int64_t GetMax() const;
	int64_t GetIncrement() const;

	PvGenRepresentation GetRepresentation() const;
    PvString GetUnit() const;
#else
	PV_GENICAM_API PvResult GetValue( int64_t &aValue ) const;

	PV_GENICAM_API PvResult GetMin( int64_t &aMin ) const;
	PV_GENICAM_API PvResult GetMax( int64_t &aMax ) const;
	PV_GENICAM_API PvResult GetIncrement( int64_t &aIncrement ) const;

	PV_GENICAM_API PvResult GetRepresentation( PvGenRepresentation &aRepresentation ) const;
    PV_GENICAM_API PvResult GetUnit( PvString &aUnit ) const;
#endif
protected:

	PvGenInteger();
	virtual ~PvGenInteger();

private:

    // Not implemented
	PvGenInteger( const PvGenInteger & );
	const PvGenInteger &operator=( const PvGenInteger & );

};

#endif
