// *****************************************************************************
//
//     Copyright (c) 2008, Pleora Technologies Inc., All rights reserved.
//
// *****************************************************************************

#ifndef __PVWND_H__
#define __PVWND_H__

#include <PvGUILib.h>


class Wnd;
#ifdef QT_GUI_LIB
class QWidget;
#endif


class PV_GUI_API PvWnd
{
public:

	void SetPosition( int32_t  aPosX, int32_t  aPosY, int32_t  aSizeX, int32_t  aSizeY );
	void GetPosition( int32_t &aPosX, int32_t &aPosY, int32_t &aSizeX, int32_t &aSizeY );

#ifdef _AFXDLL
	PvResult ShowModal( PvWindowHandle aParentHwnd = 0 );
	PvResult ShowModeless( PvWindowHandle aParentHwnd = 0 );
	PvResult Create( PvWindowHandle aHwnd, uint32_t aID );
#endif

#ifdef QT_GUI_LIB
    PvResult ShowModal();
    PvResult ShowModal( QWidget* aParentHwnd );

    PvResult ShowModeless();
    PvResult ShowModeless( QWidget* aParentHwnd );

    PvResult Create( QWidget* aHwnd );
    QWidget* GetQWidget();
#endif

	PvString GetTitle() const;
	void SetTitle( const PvString &aTitle );

	PvResult Close();

#ifdef _AFXDLL
	PvWindowHandle GetHandle();
    PvResult DoEvents();
#endif

#ifdef QT_GUI_LIB
    static void DoEvents();
#endif

protected:

    PvWnd();
	virtual ~PvWnd();

    Wnd *mThis;

private:

    // Not implemented
	PvWnd( const PvWnd & );
	const PvWnd &operator=( const PvWnd & );
};


#endif
