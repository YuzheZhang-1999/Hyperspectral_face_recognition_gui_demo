// *****************************************************************************
//
//     Copyright (c) 2007, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVREGISTERINTERFACEWND_H__
#define __PVREGISTERINTERFACEWND_H__

#include <PvGUILib.h>
#include <PvWnd.h>
#include <PvDeviceGEV.h>


class PV_GUI_API PvRegisterInterfaceWnd : public PvWnd
{
public:

	PvRegisterInterfaceWnd();
	virtual ~PvRegisterInterfaceWnd();

    PvResult SetDevice( PvDeviceGEV *aDevice );

protected:

private:

    // Not implemented
	PvRegisterInterfaceWnd( const PvRegisterInterfaceWnd & );
	const PvRegisterInterfaceWnd &operator=( const PvRegisterInterfaceWnd & );

};

#endif
