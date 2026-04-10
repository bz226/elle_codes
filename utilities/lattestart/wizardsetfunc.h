
//-----------------------------------------------------------------------------
// Begin single inclusion of this .h file condition
//-----------------------------------------------------------------------------

#ifndef _WIZARDSETFUNC_H_
#define _WIZARDSETFUNC_H_

//-----------------------------------------------------------------------------
// GCC interface
//-----------------------------------------------------------------------------

#if defined(__GNUG__) && !defined(__APPLE__)
#pragma interface "wizardsetfunc.h"
#endif

//-----------------------------------------------------------------------------
// Headers
//-----------------------------------------------------------------------------

#include "wx/dialog.h"
#include "wx/spinctrl.h"
#include "userdat.h"

enum process_set
{
  EDIT_MIKE_SETTINGS,
  EDITGAUSSPARAMS,
  WALL_EDIT,
  WEAKEN_EDIT,
  WEAKEN_ALL_EDIT,
  EDIT_GRAIN_BOUNDARIES,
  EDITSPRINGGAUSS,
  EDITRATEGAUSS,
  EDITYOUNGGAUSS,
  EDIT_2ND_HOR_LAYER,
  EDITLINEARPARAMS,
  WEAKEN_GRAIN
};

//-----------------------------------------------------------------------------
// Class definition: WizardDialog
//-----------------------------------------------------------------------------

class WizardDialogSetFunc : public wxDialog
  {

  public:

    WizardDialogSetFunc( wxWindow* parent, process_set prc = EDIT_MIKE_SETTINGS, usdat* userdata = new(usdat) );

    // Destructor.
    ~WizardDialogSetFunc();

    usdat* userdat;
    wxWindow *master;
    int proc;

  private:

    void CloseRunFunc();
    void CloseNameEdit();
    void CloseStatEdit();
    void CloseGaussParams();
    void CloseWallEdit();
    void CloseWeakenEdit();
    void CloseWeakenAllEdit();
    void CloseGrainBoundaryEdit();
    void CloseEditMikeSettings();
    void CloseRateGauss();
    void CloseSpringGauss();
    void CloseAddLayer();
    void SetSettings();
    void CloseYoungGauss();

  private:

    void Cancel( wxCommandEvent& event );
    void EditGaussStrengthParas( wxCommandEvent& event );
    void EditGaussRateParas( wxCommandEvent& event );
    void EditGaussSpringParas( wxCommandEvent& event );
    void EditWalls( wxCommandEvent& event );
    void WeakenHoriz( wxCommandEvent& event );
    void WeakenEdit( wxCommandEvent& event );
    void EditGrainBoundaries( wxCommandEvent& event );
    void OnOk( wxCommandEvent& event );
    void ExtraHorLayer(wxCommandEvent& event);
    void EditGaussYoungParas( wxCommandEvent& WXUNUSED(event) );

    void SetDefLinearMean1 ( wxCommandEvent& event);
    void SetDefLinearSigma1 ( wxCommandEvent& event);
    void SetDefLinearMean2 ( wxCommandEvent& event);
    void SetDefLinearSigma2 ( wxCommandEvent& event);
    void SetDefAdLayerYMin( wxCommandEvent& event);
    void SetDefAdLayerYMax( wxCommandEvent& event);
    void SetDefAdLayerConst ( wxCommandEvent& event);
    void SetDefAdLayerVisc ( wxCommandEvent& event);
    void SetDefAdLayerBreak ( wxCommandEvent& event);
    void SetDefHorLayerYMin (wxCommandEvent& event);
    void SetDefHorLayerYMax (wxCommandEvent& event);
    void SetDefHorLayerConst (wxCommandEvent& event);
    void SetDefHorLayerVisc (wxCommandEvent& event);
    void SetDefHorLayerBreak (wxCommandEvent& event);
    void SetDefSetWallBoth ( wxCommandEvent& event);
    void SetDefSetWallConst ( wxCommandEvent& event);
    void SetDefGaussMean ( wxCommandEvent& event);
    void SetDefGaussSigma ( wxCommandEvent& event);
    void SetDefWAConst ( wxCommandEvent& event);
    void SetDefWAVisc ( wxCommandEvent& event);
    void SetDefWABreak ( wxCommandEvent& event);
    void SetDefMGConst (wxCommandEvent& event);
    void SetDefMGBreak (wxCommandEvent& event);
    void SetDefTimeStep (wxCommandEvent& event);
    void SetDefMultRelax (wxCommandEvent& event);
    void SetWeakenGrain(wxCommandEvent& event);
    void CallSetSettings( wxCommandEvent& WXUNUSED(event) );
    void CloseWeakenGrain();

    //void OnUpdateCheckboxGauss(  wxUpdateUIEvent &event );
    //void OnUpdateCheckboxLin(  wxUpdateUIEvent &event );

    // Any class wishing to process wxWidgets events must use this macro
    DECLARE_EVENT_TABLE()

  };

//-----------------------------------------------------------------------------
// End single inclusion of this .h file condition
//-----------------------------------------------------------------------------

#endif
