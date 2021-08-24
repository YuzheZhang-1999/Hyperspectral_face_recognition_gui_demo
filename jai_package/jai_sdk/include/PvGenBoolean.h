// *****************************************************************************
//
//     Copyright (c) 2007, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVGENBOOLEAN_H__
#define __PVGENBOOLEAN_H__

#include <PvGenParameter.h>


class PvGenBoolean : public PvGenParameter
{
public:

	PV_GENICAM_API PvResult SetValue( bool aValue );
#ifdef EBUS_PYTHON_DOXYGEN
    bool GetValue();
#else
	PV_GENICAM_API PvResult GetValue( bool &aValue ) const;
#endif

protected:

	PvGenBoolean();
	virtual ~PvGenBoolean();

private:

    // Not implemented
	PvGenBoolean( const PvGenBoolean & );
	const PvGenBoolean &operator=( const PvGenBoolean & );
};

#endif
