
//-----------------------------------------------------------------------------
// Begin single inclusion of this .h file condition
//-----------------------------------------------------------------------------

#ifndef _LATTICEGAS_H_
#define _LATTICEGAS_H_

//-----------------------------------------------------------------------------
// GCC interface
//-----------------------------------------------------------------------------

#if defined(__GNUG__) && !defined(__APPLE__)
#pragma interface "latticegas.h"
#endif

//-----------------------------------------------------------------------------
// Headers
//-----------------------------------------------------------------------------

#include "wx/dialog.h"
#include "userdat.h"

#include "editfile.h"

#include "wx/process.h"

#include "wx/numdlg.h"
#include "wx/ffile.h"
#include "wx/utils.h"

//#include "myframe.h"

enum process_latticegas
{
	EDITLATTICEGAS
};

//-----------------------------------------------------------------------------
// Class definition: WizardDialog
//-----------------------------------------------------------------------------

class LatticeGas : public wxDialog
{

	public:

		LatticeGas( wxWindow* parent, process_latticegas prc = EDITLATTICEGAS, usdat* userdata = new(usdat), bool CalledFromNewFile = false );

    // Destructor.
		~LatticeGas();

		usdat* userdat;
		int proc;
    
	private:

		wxWindow* master;

		void CloseLattice();
		void SetLatticeGas();

		bool CalledByNewFile;
    
	private:

		void Cancel( wxCommandEvent& event );
		void OnOk( wxCommandEvent& event );
		void UnCheckOne(wxCommandEvent& event);
		void UnCheckTwo(wxCommandEvent& event);
    
    // Any class wishing to process wxWidgets events must use this macro
		DECLARE_EVENT_TABLE()

};

//-----------------------------------------------------------------------------
// End single inclusion of this .h file condition
//-----------------------------------------------------------------------------

#endif
