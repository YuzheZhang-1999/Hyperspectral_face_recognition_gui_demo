// *****************************************************************************
//
//     Copyright (c) 2007, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVGENCOMMAND_H__
#define __PVGENCOMMAND_H__

#include <PvGenParameter.h>


class PvGenCommand : public PvGenParameter
{
public:

	PV_GENICAM_API PvResult Execute();
#ifdef EBUS_PYTHON_DOXYGEN
	bool IsDone();
#else
	PV_GENICAM_API PvResult IsDone( bool &aDone );
#endif

protected:

	PvGenCommand();
	virtual ~PvGenCommand();

private:

    // Not implemented
	PvGenCommand( const PvGenCommand & );
	const PvGenCommand&operator=( const PvGenCommand & );

};

#endif
