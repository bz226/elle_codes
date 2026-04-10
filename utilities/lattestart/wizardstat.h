
//-----------------------------------------------------------------------------
// Begin single inclusion of this .h file condition
//-----------------------------------------------------------------------------

#ifndef _WIZARDSTAT_H_
#define _WIZARDSTAT_H_

//-----------------------------------------------------------------------------
// GCC interface
//-----------------------------------------------------------------------------

#if defined(__GNUG__) && !defined(__APPLE__)
    #pragma interface "wizardstat.h"
#endif

//-----------------------------------------------------------------------------
// Headers
//-----------------------------------------------------------------------------

#include "wx/dialog.h"
#include "userdat.h"
#include "wx/variant.h"
#include "wx/spinbutt.h"
#include "wx/spinctrl.h"

enum process_stat
{
  EDITSTATISTICS,
  STRESS_BOX,
  STRESS_GRAIN,
  STRESS_TWO_GRAINS,
  SURFACE,
  SURFACE_TIME,
  XSTRESS,
  YSTRESS
};

//-----------------------------------------------------------------------------
// Class definition: WizardDialog
//-----------------------------------------------------------------------------

class WizardDialogStat : public wxDialog
  {

  public:

    WizardDialogStat( wxWindow* parent, process_stat prc = EDITSTATISTICS, usdat* userdata = new(usdat) );

    // Destructor.
    ~WizardDialogStat();

    usdat* userdat;
    int proc;

  private:
    void SetStatistics();
    void CloseStatistics();
    void CloseStressBox();
    void CloseYStress();
    void CloseXStress();
    
  private:

    void Cancel( wxCommandEvent& event );
    void OnOk( wxCommandEvent& event );
    void SetStressBox(wxCommandEvent& event);

    void SetXStress( wxCommandEvent& event );

    //void OnUpdateCheckboxGauss(  wxUpdateUIEvent &event );
    //void OnUpdateCheckboxLin(  wxUpdateUIEvent &event );

    // Any class wishing to process wxWidgets events must use this macro
    DECLARE_EVENT_TABLE()

  };

//-----------------------------------------------------------------------------
// End single inclusion of this .h file condition
//-----------------------------------------------------------------------------

#endif
