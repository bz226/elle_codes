
//-----------------------------------------------------------------------------
// Begin single inclusion of this .h file condition
//-----------------------------------------------------------------------------

#ifndef _WIZARDEXT_H_
#define _WIZARDEXT_H_

//-----------------------------------------------------------------------------
// GCC interface
//-----------------------------------------------------------------------------

#if defined(__GNUG__) && !defined(__APPLE__)
#pragma interface "externalprocess.h"
#endif

//-----------------------------------------------------------------------------
// Headers
//-----------------------------------------------------------------------------

#include "wx/dialog.h"
#include "userdat.h"

#include "wx/frame.h"
#include "wx/string.h"

#include "wx/process.h"

#include "wx/numdlg.h"
#include "wx/ffile.h"
#include "wx/utils.h"

#include "wx/txtstrm.h"
#include "wx/wfstream.h"
#include "wx/zipstrm.h"
#include "wx/sstream.h"

//#include "myframe.h"

enum process_external_process
{
	EDITEXTPROC,
	EDITGG,
	EDITMELT
};

//-----------------------------------------------------------------------------
// Class definition: WizardDialog
//-----------------------------------------------------------------------------

class WizardDialogExternalProcess : public wxDialog
{

	public:

		WizardDialogExternalProcess( wxWindow* parent, process_external_process prc = EDITEXTPROC, usdat* userdata = new(usdat) );

    // Destructor.
		~WizardDialogExternalProcess();

		usdat* userdat;
		int proc;
    
	private:

		wxWindow* master;
		
		void SetEditExtProc();
		void CloseEditExtProc();
    
	private:

		void Cancel( wxCommandEvent& event );
		void OnOk( wxCommandEvent& event );
    
    // Any class wishing to process wxWidgets events must use this macro
		DECLARE_EVENT_TABLE()

};

//-----------------------------------------------------------------------------
// End single inclusion of this .h file condition
//-----------------------------------------------------------------------------

#endif

