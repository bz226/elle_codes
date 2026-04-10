
//-----------------------------------------------------------------------------
// Begin single inclusion of this .h file condition
//-----------------------------------------------------------------------------

#ifndef _WIZARDRUNFUNC_H_
#define _WIZARDRUNFUNC_H_

//-----------------------------------------------------------------------------
// GCC interface
//-----------------------------------------------------------------------------

#if defined(__GNUG__) && !defined(__APPLE__)
#pragma interface "wizardrunfunc.h"
#endif

//-----------------------------------------------------------------------------
// Headers
//-----------------------------------------------------------------------------

#include "wx/dialog.h"
#include "userdat.h"

#include "wx/variant.h"
#include "wx/spinbutt.h"
#include "wx/spinctrl.h"

enum process_run
{
  EDIT_MIKE_RUNFUNCTIONS,
  EDIT_COMPACTION,
  EDIT_PURESHEAR,
  EDIT_BILATERALCOMPACTION,
  EDIT_EXTRADEFORMATION,
  EDIT_STEPSIZE,
  EDIT_VISCOSITY,
  EDIT_DISSOLUTIONSTRAIN,
  EDIT_STYLOLITHEDISSOLUTION,
  EDIT_SHRINKGRAIN,
  EDIT_SHRINKBOX
};

//-----------------------------------------------------------------------------
// Class definition: WizardDialog
//-----------------------------------------------------------------------------

class WizardDialogRunFunc : public wxDialog
  {

  public:

    WizardDialogRunFunc( wxWindow* parent, process_run prc = EDIT_MIKE_RUNFUNCTIONS, usdat* userdata = new(usdat) );

    // Destructor.
    ~WizardDialogRunFunc();

    usdat* userdat;
    int proc;

  private:
    void CloseExtraDeform();
    void SetRunFunc();
    void CloseRunFunc();

  private:
    void Cancel( wxCommandEvent& event );
    void OnOk( wxCommandEvent& event );
    void ExtraDeform( wxCommandEvent& event );
    void SetStepsizeDef (wxCommandEvent& event);
    void SetViscDef (wxCommandEvent& event);
    void SetStyStressDef (wxCommandEvent& event);
    void SetStyShrinkDef (wxCommandEvent& event);
    void SetShrinkNbDef (wxCommandEvent& WXUNUSED(event));
    void SetShrinkShrinkDef (wxCommandEvent& WXUNUSED(event));
    void SetShrinkBoxDef (wxCommandEvent& WXUNUSED(event));
    void CallSetRunFunc( wxCommandEvent& WXUNUSED(event));

    // Any class wishing to process wxWidgets events must use this macro
    DECLARE_EVENT_TABLE()

  };

//-----------------------------------------------------------------------------
// End single inclusion of this .h file condition
//-----------------------------------------------------------------------------

#endif
