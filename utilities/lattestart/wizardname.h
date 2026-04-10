
//-----------------------------------------------------------------------------
// Begin single inclusion of this .h file condition
//-----------------------------------------------------------------------------

#ifndef _WIZARDNAME_H_
#define _WIZARDNAME_H_

//-----------------------------------------------------------------------------
// GCC interface
//-----------------------------------------------------------------------------

#if defined(__GNUG__) && !defined(__APPLE__)
#pragma interface "wizardname.h"
#endif

//-----------------------------------------------------------------------------
// Headers
//-----------------------------------------------------------------------------

#include "wx/dialog.h"
#include "userdat.h"

//#include "myframe.h"

enum process_name
{
    EDITNAME
};

//-----------------------------------------------------------------------------
// Class definition: WizardDialog
//-----------------------------------------------------------------------------

class WizardDialogName : public wxDialog
{

public:

   WizardDialogName( wxWindow* parent, process_name prc = EDITNAME, usdat* userdata = new(usdat) );

    // Destructor.
	~WizardDialogName();

    usdat* userdat;
    int proc;
    
private:

   wxWindow* master;

   void CloseEditName();
	void SetEditName();
    
private:

    void Cancel( wxCommandEvent& event );
    void OnOk( wxCommandEvent& event );
    
    //void OnUpdateCheckboxGauss(  wxUpdateUIEvent &event );
    //void OnUpdateCheckboxLin(  wxUpdateUIEvent &event );
    
    // Any class wishing to process wxWidgets events must use this macro
    DECLARE_EVENT_TABLE()

};

//-----------------------------------------------------------------------------
// End single inclusion of this .h file condition
//-----------------------------------------------------------------------------

#endif 
