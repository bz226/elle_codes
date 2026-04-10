
//-----------------------------------------------------------------------------
// GCC implementation
//-----------------------------------------------------------------------------

#ifdef __GNUG__
#pragma implementation "wizardsetfunc.h"
#endif

//-----------------------------------------------------------------------------
// Standard wxWidgets headers
//-----------------------------------------------------------------------------

// For compilers that support precompilation, includes "wx/wx.h".
#include "wx/wxprec.h"
#include "wx/variant.h"

#ifdef __BORLANDC__
    #pragma hdrstop
#endif

// For all others, include the necessary headers (this file is usually all you
// need because it includes almost all "standard" wxWidgets headers)
#ifndef WX_PRECOMP
    #include "wx/wx.h"
#endif

#include "wizardsetfunc.h"

#include "wx/xrc/xmlres.h"              // XRC XML resouces

BEGIN_EVENT_TABLE(WizardDialogSetFunc, wxDialog)
EVT_BUTTON( XRCID( "wxID_CANCEL" ),WizardDialogSetFunc::Cancel )
EVT_BUTTON( XRCID( "wxID_OK" ),WizardDialogSetFunc::OnOk )

EVT_BUTTON( XRCID( "GSP" ),WizardDialogSetFunc::EditGaussStrengthParas )
EVT_BUTTON( XRCID( "GRP" ),WizardDialogSetFunc::EditGaussRateParas )
EVT_BUTTON( XRCID( "GSpP" ),WizardDialogSetFunc::EditGaussSpringParas )

EVT_BUTTON( XRCID( "EDIT_GAUSSIAN_YOUNG" ),WizardDialogSetFunc::EditGaussYoungParas)

EVT_BUTTON( XRCID( "EDIT_WALLS" ),WizardDialogSetFunc::EditWalls )
EVT_BUTTON( XRCID( "WEAKEN_HORIZ" ),WizardDialogSetFunc::WeakenHoriz )
EVT_BUTTON( XRCID( "WEAKEN_ALL" ),WizardDialogSetFunc::WeakenEdit )
EVT_BUTTON( XRCID( "EDIT_GB" ),WizardDialogSetFunc::EditGrainBoundaries )
EVT_BUTTON( XRCID( "ID_BUTTON_2ND_LAYER"),WizardDialogSetFunc::ExtraHorLayer)

EVT_BUTTON( XRCID( "LIN_MEAN1_DEF" ),WizardDialogSetFunc::SetDefLinearMean1 )
EVT_BUTTON( XRCID( "LIN_SIZE1_DEF" ),WizardDialogSetFunc::SetDefLinearSigma1 )
EVT_BUTTON( XRCID( "LIN_MEAN2_DEF" ),WizardDialogSetFunc::SetDefLinearMean2 )
EVT_BUTTON( XRCID( "LIN_SIZE2_DEF" ),WizardDialogSetFunc::SetDefLinearSigma2 )

EVT_BUTTON( XRCID( "AD_Y_MIN_DEFAULT" ),WizardDialogSetFunc::SetDefAdLayerYMin )
EVT_BUTTON( XRCID( "AD_Y_MAX_DEFAULT" ),WizardDialogSetFunc::SetDefAdLayerYMax )
EVT_BUTTON( XRCID( "AD_CONST_DEFAULT" ),WizardDialogSetFunc::SetDefAdLayerConst )
EVT_BUTTON( XRCID( "AD_VISC_DEFAULT" ),WizardDialogSetFunc::SetDefAdLayerVisc )
EVT_BUTTON( XRCID( "AD_BREAK_DEFAULT" ),WizardDialogSetFunc::SetDefAdLayerBreak )

EVT_BUTTON( XRCID( "WH_Y_MIN_DEF" ),WizardDialogSetFunc::SetDefHorLayerYMin )
EVT_BUTTON( XRCID( "WH_Y_MAX_DEF" ),WizardDialogSetFunc::SetDefHorLayerYMax )
EVT_BUTTON( XRCID( "WH_CONST_DEF" ),WizardDialogSetFunc::SetDefHorLayerConst )
EVT_BUTTON( XRCID( "WH_VISC_DEF" ),WizardDialogSetFunc::SetDefHorLayerVisc )
EVT_BUTTON( XRCID( "WH_BREAK_DEF" ),WizardDialogSetFunc::SetDefHorLayerBreak )

EVT_BUTTON( XRCID( "SW_BOTH_DEF" ),WizardDialogSetFunc::SetDefSetWallBoth )
EVT_BUTTON( XRCID( "SW_CONST_DEF" ),WizardDialogSetFunc::SetDefSetWallConst )

EVT_BUTTON( XRCID( "GAUSS_MEAN_DEFAULT" ),WizardDialogSetFunc::SetDefGaussMean )
EVT_BUTTON( XRCID( "GAUSS_SIGMA_DEFAULT" ),WizardDialogSetFunc::SetDefGaussSigma )

EVT_BUTTON( XRCID( "WA_CONST_DEF" ),WizardDialogSetFunc::SetDefWAConst )
EVT_BUTTON( XRCID( "WA_VISC_DEF" ),WizardDialogSetFunc::SetDefWAVisc )
EVT_BUTTON( XRCID( "WA_BREAK_DEF" ),WizardDialogSetFunc::SetDefWABreak )

EVT_BUTTON( XRCID( "MG_CONST_DEF" ),WizardDialogSetFunc::SetDefMGConst )
EVT_BUTTON( XRCID( "MG_BREAK_DEF" ),WizardDialogSetFunc::SetDefMGBreak )

EVT_BUTTON( XRCID( "TS_DEF" ),WizardDialogSetFunc::SetDefTimeStep )
EVT_BUTTON( XRCID( "RT_DEF" ),WizardDialogSetFunc::SetDefMultRelax )

EVT_BUTTON( XRCID( "ID_BUTTON_WEAKEN_GRAIN" ),WizardDialogSetFunc::SetWeakenGrain)

EVT_BUTTON( XRCID( "ID_BUTTON_SET_DEFAULT" ),WizardDialogSetFunc::CallSetSettings)

//gaussian checkboxes
//EVT_UPDATE_UI(XRCID( "GAUSS_CHECK" ), WizardDialogSetFunc::OnUpdateCheckboxGauss )
//EVT_UPDATE_UI(XRCID( "LIN_CHECK" ), WizardDialogSetFunc::OnUpdateCheckboxLin )
END_EVENT_TABLE()

void WizardDialogSetFunc::CallSetSettings( wxCommandEvent& WXUNUSED(event) )
{
   SetSettings();
}

WizardDialogSetFunc::WizardDialogSetFunc(wxWindow* parent, process_set prc, usdat* userdata )
{

  userdat = userdata;
  master = parent;

  switch (prc)
    {
    case EDIT_MIKE_SETTINGS:
      wxXmlResource::Get()->LoadDialog(this, parent, wxT("NB_Set"));
      // wxXmlResource::Get()->LoadDialog(this, parent, wxT("Distribution"));
      proc = EDIT_MIKE_SETTINGS;
      SetSettings();
      break;
    case EDITYOUNGGAUSS:
      wxXmlResource::Get()->LoadDialog(this, parent, wxT("GaussParams"));
      break;
    case EDITGAUSSPARAMS:
      wxXmlResource::Get()->LoadDialog(this, parent, wxT("GaussParams"));
      break;
    case EDITSPRINGGAUSS:
      wxXmlResource::Get()->LoadDialog(this, parent, wxT("GaussParams"));
      break;
    case EDITRATEGAUSS:
      wxXmlResource::Get()->LoadDialog(this, parent, wxT("GaussParams"));
      break;
    case WALL_EDIT:
      wxXmlResource::Get()->LoadDialog(this, parent, wxT("SetWallParams"));
      break;
    case WEAKEN_EDIT:
      wxXmlResource::Get()->LoadDialog(this, parent, wxT("WeakenLayer"));
      break;
    case WEAKEN_ALL_EDIT:
      wxXmlResource::Get()->LoadDialog(this, parent, wxT("WeakenAll"));
      break;
    case  EDIT_GRAIN_BOUNDARIES:
      wxXmlResource::Get()->LoadDialog(this, parent, wxT("MakeGrainBoundaries"));
      break;
    case EDIT_2ND_HOR_LAYER:
      wxXmlResource::Get()->LoadDialog(this, parent, wxT("WeakenAdditionalLayer"));
      break;
    case EDITLINEARPARAMS:
      wxXmlResource::Get()->LoadDialog(this, parent, wxT("LinearParams"));
      break;
    case WEAKEN_GRAIN:
		 wxXmlResource::Get()->LoadDialog(this, parent, wxT("WeakenAll"));
      break;
    }
}

WizardDialogSetFunc::~WizardDialogSetFunc()
{}

//-----------------------------------------------------------------------------
// Private members (including the event handlers)
//-----------------------------------------------------------------------------

void WizardDialogSetFunc::Cancel( wxCommandEvent& WXUNUSED(event) )
{
  this->Destroy();
}

void WizardDialogSetFunc::OnOk( wxCommandEvent& WXUNUSED(event))
{
  switch (proc)
    {
    case EDIT_MIKE_SETTINGS:
      CloseEditMikeSettings();
      userdat->SaveData();
      break;
    case EDITGAUSSPARAMS:
      CloseGaussParams();
      break;
    case EDITSPRINGGAUSS:
      CloseSpringGauss();
      break;
    case EDITYOUNGGAUSS:
      CloseYoungGauss();
      break;
    case EDITRATEGAUSS:
      CloseRateGauss();
      break;
    case WALL_EDIT:
      CloseWallEdit();
      break;
    case WEAKEN_EDIT:
      CloseWeakenEdit();
      break;
    case WEAKEN_ALL_EDIT:
      CloseWeakenAllEdit();
      break;
    case  EDIT_GRAIN_BOUNDARIES:
      CloseGrainBoundaryEdit();
      break;
    case EDIT_2ND_HOR_LAYER:
      CloseAddLayer();
      break;
    case EDITLINEARPARAMS:
      CloseGaussParams();
      break;
    case WEAKEN_GRAIN:
      CloseWeakenGrain();
      break;
    }
}

// initialise
void WizardDialogSetFunc::EditGaussStrengthParas( wxCommandEvent& WXUNUSED(event) )
{
  wxVariant word;

  if (XRCCTRL(*this, "STRENGTH_DISTRI", wxRadioBox)->GetSelection() == 2)
    {
      WizardDialogSetFunc dialog(this, EDITGAUSSPARAMS, userdat );
      dialog.proc = EDITGAUSSPARAMS;

      word = userdat->gauss_strength[1];
      XRCCTRL(*this, "GAUSS_SIGMA", wxTextCtrl)->SetValue(word);
      word= userdat->gauss_strength[0];
      XRCCTRL(*this, "GAUSS_MEAN", wxTextCtrl)->SetValue(word);
      dialog.ShowModal();
    }
  else if (XRCCTRL(*this, "STRENGTH_DISTRI", wxRadioBox)->GetSelection() == 1)
    {
      WizardDialogSetFunc dialog(this, EDITLINEARPARAMS, userdat );
      dialog.proc = EDITLINEARPARAMS;

      word = userdat->linear_strength[0];
      XRCCTRL(*this, "ID_TEXTCTRL_LINEAR_MEAN1", wxTextCtrl)->SetValue(word);
      word= userdat->linear_strength[1];
      XRCCTRL(*this, "ID_TEXTCTRL_LINEAR_SIZE1", wxTextCtrl)->SetValue(word);
      word = userdat->linear_strength[2];
      XRCCTRL(*this, "ID_TEXTCTRL_LINEAR_MEAN2", wxTextCtrl)->SetValue(word);
      word= userdat->linear_strength[3];
      XRCCTRL(*this, "ID_TEXTCTRL_LINEAR_SIZE2", wxTextCtrl)->SetValue(word);
      dialog.ShowModal();
    }
  else
    {
      wxMessageDialog dialog(this, _T("Choose a type first!"), _T("Error!"), wxOK );
      dialog.ShowModal();
    }
}

void WizardDialogSetFunc::EditGaussRateParas( wxCommandEvent& WXUNUSED(event) )
{
  wxVariant word;

  WizardDialogSetFunc dialog(this, EDITGAUSSPARAMS, userdat );
  dialog.proc = EDITRATEGAUSS;

  word = userdat->gauss_rate[0];
  XRCCTRL(*this, "GAUSS_MEAN", wxTextCtrl)->SetValue(word);
  word = userdat->gauss_rate[1];
  XRCCTRL(*this, "GAUSS_SIGMA", wxTextCtrl)->SetValue(word);

  dialog.ShowModal();
}

void WizardDialogSetFunc::EditGaussSpringParas( wxCommandEvent& WXUNUSED(event) )
{
  wxVariant word;

  WizardDialogSetFunc dialog(this, EDITGAUSSPARAMS, userdat );
  dialog.proc = EDITSPRINGGAUSS;

  word = userdat->gauss_spring[0];
  XRCCTRL(*this, "GAUSS_MEAN", wxTextCtrl)->SetValue(word);
  word = userdat->gauss_spring[1];
  XRCCTRL(*this, "GAUSS_SIGMA", wxTextCtrl)->SetValue(word);

  dialog.ShowModal();
}

void WizardDialogSetFunc::EditGaussYoungParas( wxCommandEvent& WXUNUSED(event) )
{
  wxVariant word;

  WizardDialogSetFunc dialog(this, EDITGAUSSPARAMS, userdat );
  dialog.proc = EDITYOUNGGAUSS;

  word = userdat->gauss_young[0];
  XRCCTRL(*this, "GAUSS_MEAN", wxTextCtrl)->SetValue(word);
  word = userdat->gauss_young[1];
  XRCCTRL(*this, "GAUSS_SIGMA", wxTextCtrl)->SetValue(word);

  dialog.ShowModal();
}

void WizardDialogSetFunc::EditWalls( wxCommandEvent& WXUNUSED(event) )
{
  wxVariant word;

  WizardDialogSetFunc dialog(this, WALL_EDIT, userdat );
  dialog.proc = WALL_EDIT;

  XRCCTRL(*this, "ID_RADIOBOX_SW_BOTH_EDIT", wxRadioBox)->SetSelection(int(userdat->walls[0]));
  word = userdat->walls[1];
  XRCCTRL(*this, "SW_CONST_EDIT", wxTextCtrl)->SetValue(word);

  dialog.ShowModal();
}

void WizardDialogSetFunc::WeakenHoriz( wxCommandEvent& WXUNUSED(event) )
{
  wxVariant word;

  WizardDialogSetFunc dialog(this, WEAKEN_EDIT, userdat );
  dialog.proc = WEAKEN_EDIT;

  word = userdat->weaken_hor[0];
  XRCCTRL(*this, "WH_Y_MIN", wxTextCtrl)->SetValue(word);
  word = userdat->weaken_hor[1];
  XRCCTRL(*this, "WH_Y_MAX", wxTextCtrl)->SetValue(word);
  word = userdat->weaken_hor[2];
  XRCCTRL(*this, "WH_CONST", wxTextCtrl)->SetValue(word);
  word = userdat->weaken_hor[3];
  XRCCTRL(*this, "WH_VISC", wxTextCtrl)->SetValue(word);
  word = userdat->weaken_hor[4];
  XRCCTRL(*this, "WH_BREAKSTRENGTH", wxTextCtrl)->SetValue(word);

  XRCCTRL(*this, "ID_CHECKBOX_2ND_LAYER", wxCheckBox)->SetValue(userdat->extra_hor_layer_proc);

  dialog.ShowModal();
}

void WizardDialogSetFunc::WeakenEdit( wxCommandEvent& WXUNUSED(event) )
{
  wxVariant word;

  WizardDialogSetFunc dialog(this, WEAKEN_ALL_EDIT, userdat );
  dialog.proc = WEAKEN_ALL_EDIT;

  word = userdat->weaken_all[0];
  XRCCTRL(*this, "WA_CONST", wxTextCtrl)->SetValue(word);
  word = userdat->weaken_all[1];
  XRCCTRL(*this, "WA_VISC", wxTextCtrl)->SetValue(word);
  word = userdat->weaken_all[2];
  XRCCTRL(*this, "WA_BREAKSTRENGTH", wxTextCtrl)->SetValue(word);

  dialog.ShowModal();
}

void WizardDialogSetFunc::EditGrainBoundaries( wxCommandEvent& WXUNUSED(event))
{
  wxVariant word;

  WizardDialogSetFunc dialog(this,  EDIT_GRAIN_BOUNDARIES, userdat );
  dialog.proc = EDIT_GRAIN_BOUNDARIES;

  word = userdat->make_gb[0];
  XRCCTRL(*this, "GB_CONST", wxTextCtrl)->SetValue(word);
  word = userdat->make_gb[1];
  XRCCTRL(*this, "GB_BREAKSTRENGTH", wxTextCtrl)->SetValue(word);

  dialog.ShowModal();
}


// on exit functions (called in destructor)
void WizardDialogSetFunc::CloseRunFunc()
{}

void WizardDialogSetFunc::CloseNameEdit()
{}

void WizardDialogSetFunc::CloseStatEdit()
{}

void WizardDialogSetFunc::SetSettings()
{
  wxVariant word;

  // both erad from transition[]
  // activation energy
  // first convert to GPa
  word = userdat->transition[0];
  XRCCTRL(*this, "ID_TEXTCTRL_ACTIVATION_ENERGY", wxTextCtrl)->SetValue(word);
  // start reactions at ... GPa (convert to pascal on save!)
  userdat->transition[1] /= 1e9;
  word = userdat->transition[1];
  XRCCTRL(*this, "ID_TEXTCTRL_START_GPA", wxTextCtrl)->SetValue(word);

  //the distribution-type for strength
  if (userdat->gauss_strength_proc)
    XRCCTRL(*this, "STRENGTH_DISTRI", wxRadioBox)->SetSelection(2);
  else if (userdat->linear_strength_proc)
    XRCCTRL(*this, "STRENGTH_DISTRI", wxRadioBox)->SetSelection(1);
  else
    XRCCTRL(*this, "STRENGTH_DISTRI", wxRadioBox)->SetSelection(0);

  //rate distrib
  XRCCTRL(*this, "RATE_DISTRI", wxCheckBox)->SetValue(userdat->gauss_rate_proc);

  //spring distrib
  XRCCTRL(*this, "SPRING_DISTRI", wxCheckBox)->SetValue(userdat->gauss_spring_proc);

  //young distrib
  XRCCTRL(*this, "YOUNG_DISTRI", wxCheckBox)->SetValue(userdat->gauss_young_proc);

  //   //Relaxthreshold

  word = userdat->multiply_relax[0];
  XRCCTRL(*this, "MULT_RELAX", wxTextCtrl)->SetValue(word);
  XRCCTRL(*this, "CHECK_MULT_RELAX", wxCheckBox)->SetValue(userdat->multiply_relax_proc);

  //   //timestep
  word = userdat->timestep[0];
  XRCCTRL(*this, "TIMESTEP", wxTextCtrl)->SetValue(word);
  XRCCTRL(*this, "CHECK_TIMESTEP", wxCheckBox)->SetValue(userdat->timestep_proc);

  //   //set walls
  XRCCTRL(*this, "CHECK_SET_WALLS", wxCheckBox)->SetValue(userdat->walls_proc);

  //   //wweaken horizontal
  XRCCTRL(*this, "CHECK_WEAKEN_HORIZ", wxCheckBox)->SetValue(userdat->weaken_hor_proc);

  //   //weaken all
  XRCCTRL(*this, "CHECK_WEAKEN_ALL", wxCheckBox)->SetValue(userdat->weaken_all_proc);

  //weaken grain
  XRCCTRL(*this, "ID_CHECKBOX_WEAKEN_GRAIN", wxCheckBox)->SetValue(userdat->weaken_grain_proc);
  XRCCTRL(*this, "ID_SPINCTRL_WEAKEN_GRAIN", wxSpinCtrl)->SetValue(int(userdat->weaken_grain[3]));

  //   //make gb
  XRCCTRL(*this, "CHECK_MAKE_GB", wxCheckBox)->SetValue(userdat->make_gb_proc);

  //   //mineral
  XRCCTRL(*this, "ID_MINERAL", wxRadioBox)->SetSelection(int(userdat->mineral[0]));

  //   // x-row
  XRCCTRL(*this, "CHECK_X_ROW", wxCheckBox)->SetValue(userdat->dissolve_x_proc);
  word = userdat->dissolve_x[0];
  XRCCTRL(*this, "DIS_UPPER", wxTextCtrl)->SetValue(word);
  word = userdat->dissolve_x[1];
  XRCCTRL(*this, "DIS_LOWER", wxTextCtrl)->SetValue(word);

  // y-row
  XRCCTRL(*this, "CHECK_Y_ROW", wxCheckBox)->SetValue(userdat->dissolve_y_proc);
  word = userdat->dissolve_y[0];
  XRCCTRL(*this, "DIS_LEFT", wxTextCtrl)->SetValue(word);
  word = userdat->dissolve_y[1];
  XRCCTRL(*this, "DIS_RIGHT", wxTextCtrl)->SetValue(word);

  // box size
  XRCCTRL(*this, "CHECK_BOX_SIZE", wxCheckBox)->SetValue(userdat->set_box_size_proc);
  word = userdat->set_box_size[0];
  XRCCTRL(*this, "BOX_SIZE", wxTextCtrl)->SetValue(word);

  // plot frac
  XRCCTRL(*this, "CHECK_PLOT", wxCheckBox)->SetValue(userdat->plot_frac_proc);
  XRCCTRL(*this, "PLOT_NB", wxSpinCtrl)->SetValue(int(userdat->plot_frac[0]));

  XRCCTRL(*this, "ID_CHECKBOX_TRANSITION", wxCheckBox)->SetValue(userdat->transition_proc);
  XRCCTRL(*this, "ID_CHECKBOX_TRANSITION_HEAT", wxCheckBox)->SetValue(userdat->transition_heat_proc);
  
  XRCCTRL(*this, "ID_CHECKBOX_HEAT_CONDUCTION", wxCheckBox)->SetValue(userdat->heat_conduction_proc);
  XRCCTRL(*this, "ID_SPINCTRL_HEAT_CONDUCTION_NB", wxSpinCtrl)->SetValue(int(userdat->heat_conduction[0]));
  XRCCTRL(*this, "ID_SPINCTRL_HEAT_CONDUCTION_T", wxSpinCtrl)->SetValue(int(userdat->heat_conduction[1]));

}

void WizardDialogSetFunc::CloseGaussParams()
{
  bool a,c,b,d;


  if (XRCCTRL(*master, "STRENGTH_DISTRI", wxRadioBox)->GetSelection() == 2)
    {
      if (XRCCTRL(*this, "GAUSS_MEAN", wxTextCtrl)->GetValue().ToDouble(&userdat->gauss_strength[0]) && XRCCTRL(*this, "GAUSS_SIGMA", wxTextCtrl)->GetValue().ToDouble(&userdat->gauss_strength[1]))
        {
          EndModal(true);
        }
      else
        {
          wxMessageDialog msgDlg(this, _("Numbers only!"), _("Error!"), wxOK );
          msgDlg.ShowModal();
        }
    }
  else if (XRCCTRL(*master, "STRENGTH_DISTRI", wxRadioBox)->GetSelection() == 1)
    {
      a = XRCCTRL(*this, "ID_TEXTCTRL_LINEAR_MEAN1", wxTextCtrl)->GetValue().ToDouble(&userdat->linear_strength[0]);
      b = XRCCTRL(*this, "ID_TEXTCTRL_LINEAR_SIZE1", wxTextCtrl)->GetValue().ToDouble(&userdat->linear_strength[1]);
      c = XRCCTRL(*this, "ID_TEXTCTRL_LINEAR_MEAN2", wxTextCtrl)->GetValue().ToDouble(&userdat->linear_strength[2]);
      d = XRCCTRL(*this, "ID_TEXTCTRL_LINEAR_SIZE2", wxTextCtrl)->GetValue().ToDouble(&userdat->linear_strength[3]);
      if (a && b && c && d)
        {
          EndModal(true);
        }
      else
        {
          wxMessageDialog msgDlg(this, _("Numbers only!"), _("Error!"), wxOK );
          msgDlg.ShowModal();
        }
    }
}

void WizardDialogSetFunc::CloseSpringGauss()
{
  if (XRCCTRL(*this, "GAUSS_MEAN", wxTextCtrl)->GetValue().ToDouble(&userdat->gauss_spring[0]) && XRCCTRL(*this, "GAUSS_SIGMA", wxTextCtrl)->GetValue().ToDouble(&userdat->gauss_spring[1]))
    EndModal(true);
  else
    {
      wxMessageDialog msgDlg(this, _("Numbers only!"), _("Error!"), wxOK );
      msgDlg.ShowModal();
    }
}

void WizardDialogSetFunc::CloseRateGauss()
{
  if (XRCCTRL(*this, "GAUSS_MEAN", wxTextCtrl)->GetValue().ToDouble(&userdat->gauss_rate[0]) && XRCCTRL(*this, "GAUSS_SIGMA", wxTextCtrl)->GetValue().ToDouble(&userdat->gauss_rate[1]))
    EndModal(true);
  else
    {
      wxMessageDialog msgDlg(this, _("Numbers only!"), _("Error!"), wxOK );
      msgDlg.ShowModal();
    }
}

void WizardDialogSetFunc::CloseYoungGauss()
{
  if (XRCCTRL(*this, "GAUSS_MEAN", wxTextCtrl)->GetValue().ToDouble(&userdat->gauss_young[0]) && XRCCTRL(*this, "GAUSS_SIGMA", wxTextCtrl)->GetValue().ToDouble(&userdat->gauss_young[1]))
    EndModal(true);
  else
    {
      wxMessageDialog msgDlg(this, _("Numbers only!"), _("Error!"), wxOK );
      msgDlg.ShowModal();
    }
}

void WizardDialogSetFunc::CloseWallEdit()
{
  userdat->walls[0] = XRCCTRL(*this, "ID_RADIOBOX_SW_BOTH_EDIT", wxRadioBox)->GetSelection();

  if (XRCCTRL(*this, "SW_CONST_EDIT", wxTextCtrl)->GetValue().ToDouble(&userdat->walls[1]))
    EndModal(true);
  else
    {
      wxMessageDialog msgDlg(this, _("Numbers only!"), _("Error!"), wxOK );
      msgDlg.ShowModal();
    }
}

void WizardDialogSetFunc::CloseWeakenEdit()
{
  bool a,b,c,d,e;

  a = XRCCTRL(*this, "WH_Y_MIN", wxTextCtrl)->GetValue().ToDouble(&userdat->weaken_hor[0]);
  b = XRCCTRL(*this, "WH_Y_MAX", wxTextCtrl)->GetValue().ToDouble(&userdat->weaken_hor[1]);
  c = XRCCTRL(*this, "WH_CONST", wxTextCtrl)->GetValue().ToDouble(&userdat->weaken_hor[2]);
  d = XRCCTRL(*this, "WH_VISC", wxTextCtrl)->GetValue().ToDouble(&userdat->weaken_hor[3]);
  e = XRCCTRL(*this, "WH_BREAKSTRENGTH", wxTextCtrl)->GetValue().ToDouble(&userdat->weaken_hor[4]);

  userdat->extra_hor_layer_proc = XRCCTRL(*this, "ID_CHECKBOX_2ND_LAYER", wxCheckBox)->IsChecked();

  if (a && b && c && d && e)
    EndModal(true);
  else
    {
      wxMessageDialog msgDlg(this, _("Numbers only!"), _("Error!"), wxOK );
      msgDlg.ShowModal();
    }
}

void WizardDialogSetFunc::CloseWeakenAllEdit()
{
  bool a,b,c;

  a = XRCCTRL(*this, "WA_CONST", wxTextCtrl)->GetValue().ToDouble(&userdat->weaken_all[0]);
  b = XRCCTRL(*this, "WA_VISC", wxTextCtrl)->GetValue().ToDouble(&userdat->weaken_all[1]);
  c = XRCCTRL(*this, "WA_BREAKSTRENGTH", wxTextCtrl)->GetValue().ToDouble(&userdat->weaken_all[2]);

  if (a && b && c)
    EndModal(true);
  else
    {
      wxMessageDialog msgDlg(this, _("Numbers only!"), _("Error!"), wxOK );
      msgDlg.ShowModal();
    }
}

void WizardDialogSetFunc::CloseGrainBoundaryEdit()
{
  if (XRCCTRL(*this, "GB_CONST", wxTextCtrl)->GetValue().ToDouble(&userdat->make_gb[0]) && XRCCTRL(*this, "GB_BREAKSTRENGTH", wxTextCtrl)->GetValue().ToDouble(&userdat->make_gb[1]))
    EndModal(true);
  else
    {
      wxMessageDialog msgDlg(this, _("Numbers only!"), _("Error!"), wxOK );
      msgDlg.ShowModal();
    }
}

void WizardDialogSetFunc::CloseEditMikeSettings()
{
	bool a = true,b = true,c = true,d = true,e = true,f = true,g = true,h = true,i = true,k = true;

  // both written to transition[]:::
  // start reactions at ... GPa (convert to pascal on save!)
  k = XRCCTRL(*this, "ID_TEXTCTRL_START_GPA", wxTextCtrl)->GetValue().ToDouble(&userdat->transition[1]);
  userdat->transition[1] *= 1e9;
  //activation energy
  i = XRCCTRL(*this, "ID_TEXTCTRL_ACTIVATION_ENERGY", wxTextCtrl)->GetValue().ToDouble(&userdat->transition[0]);
			  
  //the distribution-type for strength
  switch (XRCCTRL(*this, "STRENGTH_DISTRI", wxRadioBox)->GetSelection())
    {
    case 0:
      userdat->gauss_strength_proc = false;
      userdat->linear_strength_proc = false;
      break;
    case 1:
      userdat->gauss_strength_proc = false;
      userdat->linear_strength_proc = true;
      break;
    case 2:
      userdat->gauss_strength_proc = true;
      userdat->linear_strength_proc = false;
      break;
    }

  //rate distrib
  userdat->gauss_rate_proc = XRCCTRL(*this, "RATE_DISTRI", wxCheckBox)->IsChecked();
  //spring distrib
  userdat->gauss_spring_proc = XRCCTRL(*this, "SPRING_DISTRI", wxCheckBox)->IsChecked();
  //young distrib
  userdat->gauss_young_proc = XRCCTRL(*this, "YOUNG_DISTRI", wxCheckBox)->IsChecked();

  //Relaxthreshold
  a = XRCCTRL(*this, "MULT_RELAX", wxTextCtrl)->GetValue().ToDouble(&userdat->multiply_relax[0]);
	//cout << userdat->multiply_relax[0] << endl;
  userdat->multiply_relax_proc = XRCCTRL(*this, "CHECK_MULT_RELAX", wxCheckBox)->IsChecked();

  //timestep
  b = XRCCTRL(*this, "TIMESTEP", wxTextCtrl)->GetValue().ToDouble(&userdat->timestep[0]);
  userdat->timestep_proc = XRCCTRL(*this, "CHECK_TIMESTEP", wxCheckBox)->IsChecked();

  //set walls
  userdat->walls_proc = XRCCTRL(*this, "CHECK_SET_WALLS", wxCheckBox)->IsChecked();

  //wweaken horizontal
  userdat->weaken_hor_proc = XRCCTRL(*this, "CHECK_WEAKEN_HORIZ", wxCheckBox)->IsChecked();

  //weaken all
  userdat->weaken_all_proc = XRCCTRL(*this, "CHECK_WEAKEN_ALL", wxCheckBox)->IsChecked();

  //weaken grain
  userdat->weaken_grain_proc = XRCCTRL(*this, "ID_CHECKBOX_WEAKEN_GRAIN", wxCheckBox)->IsChecked();
  userdat->weaken_grain[3] = XRCCTRL(*this, "ID_SPINCTRL_WEAKEN_GRAIN", wxSpinCtrl)->GetValue();

  //make gb
  userdat->make_gb_proc = XRCCTRL(*this, "CHECK_MAKE_GB", wxCheckBox)->IsChecked();

  //mineral
  switch (XRCCTRL(*this, "ID_MINERAL", wxRadioBox)->GetSelection())
    {
    case 3:
      userdat->set_min = true;
      userdat->mineral[0] = 3;
      break;
    case 1:
      userdat->set_min = true;
      userdat->mineral[0] = 1;
      break;
    case 2:
      userdat->set_min = true;
      userdat->mineral[0] = 2;
      break;
    default:
      userdat->set_min = false;
      userdat->mineral[0] = 0;
      break;
    }

  // x-row
  userdat->dissolve_x_proc = XRCCTRL(*this, "CHECK_X_ROW", wxCheckBox)->IsChecked();
  c = XRCCTRL(*this, "DIS_UPPER", wxTextCtrl)->GetValue().ToDouble(&userdat->dissolve_x[0]);
  d = XRCCTRL(*this, "DIS_LOWER", wxTextCtrl)->GetValue().ToDouble(&userdat->dissolve_x[1]);

  // y-row
  userdat->dissolve_y_proc = XRCCTRL(*this, "CHECK_Y_ROW", wxCheckBox)->IsChecked();
  e = XRCCTRL(*this, "DIS_LEFT", wxTextCtrl)->GetValue().ToDouble(&userdat->dissolve_y[0]);
  f = XRCCTRL(*this, "DIS_RIGHT", wxTextCtrl)->GetValue().ToDouble(&userdat->dissolve_y[1]);

  // box size
  userdat->set_box_size_proc = XRCCTRL(*this, "CHECK_BOX_SIZE", wxCheckBox)->IsChecked();
  g = XRCCTRL(*this, "BOX_SIZE", wxTextCtrl)->GetValue().ToDouble(&userdat->set_box_size[0]);

  // plot frac
  userdat->plot_frac_proc = XRCCTRL(*this, "CHECK_PLOT", wxCheckBox)->IsChecked();
  userdat->plot_frac[0] = XRCCTRL(*this, "PLOT_NB", wxSpinCtrl)->GetValue();

  userdat->transition_proc = XRCCTRL(*this, "ID_CHECKBOX_TRANSITION", wxCheckBox)->GetValue();
  userdat->transition_heat_proc = XRCCTRL(*this, "ID_CHECKBOX_TRANSITION_HEAT", wxCheckBox)->GetValue();
  
  userdat->heat_conduction_proc = XRCCTRL(*this, "ID_CHECKBOX_HEAT_CONDUCTION", wxCheckBox)->GetValue();
  userdat->heat_conduction[0] = XRCCTRL(*this, "ID_SPINCTRL_HEAT_CONDUCTION_NB", wxSpinCtrl)->GetValue();
  userdat->heat_conduction[1] = XRCCTRL(*this, "ID_SPINCTRL_HEAT_CONDUCTION_T", wxSpinCtrl)->GetValue();
	
	//cout << a << b << c << d << e << f << g << h << i << k << endl;
  
  if (a && b && c && e && f && g && h && i && k)
    EndModal(true);
  else
    {
      wxMessageDialog msgDlg(this, _("Numbers only!"), _("Error!"), wxOK );
      msgDlg.ShowModal();
    }
}

void WizardDialogSetFunc::ExtraHorLayer(wxCommandEvent& WXUNUSED(event))
{
  wxVariant word;
  WizardDialogSetFunc dialog(this,  EDIT_2ND_HOR_LAYER, userdat );
  dialog.proc = EDIT_2ND_HOR_LAYER;

  word = userdat->extra_hor_layer[0];
  XRCCTRL(*this, "ID_TEXTCTRL_2ND_LAY_Y_MIN", wxTextCtrl)->SetValue(word);
  word = userdat->extra_hor_layer[1];
  XRCCTRL(*this, "ID_TEXTCTRL_2ND_LAY_Y_MAX", wxTextCtrl)->SetValue(word);
  word = userdat->extra_hor_layer[2];
  XRCCTRL(*this, "ID_TEXTCTRL_2ND_LAY_CONSTANT", wxTextCtrl)->SetValue(word);
  word = userdat->extra_hor_layer[3];
  XRCCTRL(*this, "ID_TEXTCTRL_2ND_LAY_VISCOSITY", wxTextCtrl)->SetValue(word);
  word = userdat->extra_hor_layer[4];
  XRCCTRL(*this, "ID_TEXTCTRL_2ND_LAY_BREAK_STRENGTH", wxTextCtrl)->SetValue(word);

  dialog.ShowModal();
}

void WizardDialogSetFunc::CloseAddLayer()
{
  bool a,b,c,d,e;

  a = XRCCTRL(*this, "ID_TEXTCTRL_2ND_LAY_Y_MIN", wxTextCtrl)->GetValue().ToDouble(&userdat->extra_hor_layer[0]);
  b = XRCCTRL(*this, "ID_TEXTCTRL_2ND_LAY_Y_MAX", wxTextCtrl)->GetValue().ToDouble(&userdat->extra_hor_layer[1]);
  c = XRCCTRL(*this, "ID_TEXTCTRL_2ND_LAY_CONSTANT", wxTextCtrl)->GetValue().ToDouble(&userdat->extra_hor_layer[2]);
  d = XRCCTRL(*this, "ID_TEXTCTRL_2ND_LAY_VISCOSITY", wxTextCtrl)->GetValue().ToDouble(&userdat->extra_hor_layer[3]);
  e = XRCCTRL(*this, "ID_TEXTCTRL_2ND_LAY_BREAK_STRENGTH", wxTextCtrl)->GetValue().ToDouble(&userdat->extra_hor_layer[4]);

  if (a && b && c && d && e)
    EndModal(true);
  else
    {
      wxMessageDialog msgDlg(this, _("Numbers only!"), _("Error!"), wxOK );
      msgDlg.ShowModal();
    }
}

// set the defaults, if a button is clicked...
void WizardDialogSetFunc::SetDefLinearMean1( wxCommandEvent& WXUNUSED(event))
{
  wxVariant word;
  word = userdat->linear_strength[0];
  XRCCTRL(*this, "ID_TEXTCTRL_LINEAR_MEAN1", wxTextCtrl)->SetValue(word);
}

void WizardDialogSetFunc::SetDefLinearSigma1 ( wxCommandEvent& WXUNUSED(event))
{
  wxVariant word;
  word = userdat->linear_strength[1];
  XRCCTRL(*this, "ID_TEXTCTRL_LINEAR_SIZE1", wxTextCtrl)->SetValue(word);
}

void WizardDialogSetFunc::SetDefLinearMean2 ( wxCommandEvent& WXUNUSED(event))
{
  wxVariant word;
  word = userdat->linear_strength[2];
  XRCCTRL(*this, "ID_TEXTCTRL_LINEAR_MEAN2", wxTextCtrl)->SetValue(word);
}

void WizardDialogSetFunc::SetDefLinearSigma2 ( wxCommandEvent& WXUNUSED(event))
{
  wxVariant word;
  word = userdat->linear_strength[3];
  XRCCTRL(*this, "ID_TEXTCTRL_LINEAR_SIZE2", wxTextCtrl)->SetValue(word);
}

void WizardDialogSetFunc::SetDefAdLayerYMin( wxCommandEvent& WXUNUSED(event))
{
  wxVariant word;
  word = userdat->extra_hor_layer[0];
  XRCCTRL(*this, "ID_TEXTCTRL_2ND_LAY_Y_MIN", wxTextCtrl)->SetValue(word);
}

void WizardDialogSetFunc::SetDefAdLayerYMax( wxCommandEvent& WXUNUSED(event))
{
  wxVariant word;
  word = userdat->extra_hor_layer[1];
  XRCCTRL(*this, "ID_TEXTCTRL_2ND_LAY_Y_MAX", wxTextCtrl)->SetValue(word);
}

void WizardDialogSetFunc::SetDefAdLayerConst ( wxCommandEvent& WXUNUSED(event))
{
  wxVariant word;
  word = userdat->extra_hor_layer[2];
  XRCCTRL(*this, "ID_TEXTCTRL_2ND_LAY_CONSTANT", wxTextCtrl)->SetValue(word);
}

void WizardDialogSetFunc::SetDefAdLayerVisc ( wxCommandEvent& WXUNUSED(event))
{
  wxVariant word;
  word = userdat->extra_hor_layer[3];
  XRCCTRL(*this, "ID_TEXTCTRL_2ND_LAY_VISCOSITY", wxTextCtrl)->SetValue(word);
}

void WizardDialogSetFunc::SetDefAdLayerBreak ( wxCommandEvent& WXUNUSED(event))
{
  wxVariant word;
  word = userdat->extra_hor_layer[4];
  XRCCTRL(*this, "ID_TEXTCTRL_2ND_LAY_BREAK_STRENGTH", wxTextCtrl)->SetValue(word);
}

void WizardDialogSetFunc::SetDefHorLayerYMin (wxCommandEvent& WXUNUSED(event))
{
  wxVariant word;
  word = userdat->weaken_hor[0];
  XRCCTRL(*this, "WH_Y_MIN", wxTextCtrl)->SetValue(word);
}

void WizardDialogSetFunc::SetDefHorLayerYMax (wxCommandEvent& WXUNUSED(event))
{
  wxVariant word;
  word = userdat->weaken_hor[1];
  XRCCTRL(*this, "WH_Y_MAX", wxTextCtrl)->SetValue(word);
}

void WizardDialogSetFunc::SetDefHorLayerConst (wxCommandEvent& WXUNUSED(event))
{
  wxVariant word;
  word = userdat->weaken_hor[1];
  XRCCTRL(*this, "WH_CONST", wxTextCtrl)->SetValue(word);
}

void WizardDialogSetFunc::SetDefHorLayerVisc (wxCommandEvent& WXUNUSED(event))
{
  wxVariant word;
  word = userdat->weaken_hor[2];
  XRCCTRL(*this, "WH_VISC", wxTextCtrl)->SetValue(word);
}

void WizardDialogSetFunc::SetDefHorLayerBreak (wxCommandEvent& WXUNUSED(event))
{
  wxVariant word;
  word = userdat->weaken_hor[3];
  XRCCTRL(*this, "WH_BREAKSTRENGTH", wxTextCtrl)->SetValue(word);
}

void WizardDialogSetFunc::SetDefSetWallBoth ( wxCommandEvent& WXUNUSED(event))
{
  XRCCTRL(*this, "ID_RADIOBOX_SW_BOTH_EDIT", wxRadioBox)->SetSelection(int(userdat->walls[0]));
}

void WizardDialogSetFunc::SetDefSetWallConst ( wxCommandEvent& WXUNUSED(event))
{
  wxVariant word;

  word = userdat->walls[1];
  XRCCTRL(*this, "SW_CONST_EDIT", wxTextCtrl)->SetValue(word);
}

void WizardDialogSetFunc::SetDefGaussMean ( wxCommandEvent& WXUNUSED(event))
{
  wxVariant word;
  switch (proc)
    {
    case EDITGAUSSPARAMS:
      word= userdat->gauss_strength[0];
      break;
    case EDITSPRINGGAUSS:
      word = userdat->gauss_spring[0];
      break;
    case EDITRATEGAUSS:
      word = userdat->gauss_rate[0];
      break;
    }
  XRCCTRL(*this, "GAUSS_MEAN", wxTextCtrl)->SetValue(word);
}

void WizardDialogSetFunc::SetDefGaussSigma ( wxCommandEvent& WXUNUSED(event))
{
  wxVariant word;
  switch (proc)
    {
    case EDITGAUSSPARAMS:
      word= userdat->gauss_strength[1];
      break;
    case EDITSPRINGGAUSS:
      word = userdat->gauss_spring[1];
      break;
    case EDITRATEGAUSS:
      word = userdat->gauss_rate[1];
      break;
    }
  XRCCTRL(*this, "GAUSS_SIGMA", wxTextCtrl)->SetValue(word);
}

void WizardDialogSetFunc::SetDefWAConst (wxCommandEvent& WXUNUSED(event))
{
  wxVariant word;

  word = userdat->weaken_all[0];
  XRCCTRL(*this, "WA_CONST", wxTextCtrl)->SetValue(word);
}

void WizardDialogSetFunc::SetDefWAVisc (wxCommandEvent& WXUNUSED(event))
{
  wxVariant word;

  word = userdat->weaken_all[1];
  XRCCTRL(*this, "WA_VISC", wxTextCtrl)->SetValue(word);
}

void WizardDialogSetFunc::SetDefWABreak (wxCommandEvent& WXUNUSED(event))
{
  wxVariant word;

  word = userdat->weaken_all[2];
  XRCCTRL(*this, "WA_BREAKSTRENGTH", wxTextCtrl)->SetValue(word);
}

void WizardDialogSetFunc::SetDefMGConst (wxCommandEvent& WXUNUSED(event))
{
  wxVariant word;

  word = userdat->make_gb[0];
  XRCCTRL(*this, "GB_CONST", wxTextCtrl)->SetValue(word);
}

void WizardDialogSetFunc::SetDefMGBreak (wxCommandEvent& WXUNUSED(event))
{
  wxVariant word;
  word = userdat->make_gb[1];
  XRCCTRL(*this, "GB_BREAKSTRENGTH", wxTextCtrl)->SetValue(word);
}

void WizardDialogSetFunc::SetDefTimeStep (wxCommandEvent& WXUNUSED(event))
{
  wxVariant word;

  word = userdat->timestep[0];
  XRCCTRL(*this, "TIMESTEP", wxTextCtrl)->SetValue(word);
}

void WizardDialogSetFunc::SetDefMultRelax (wxCommandEvent& WXUNUSED(event))
{
  wxVariant word;

  word = userdat->multiply_relax[0];
  XRCCTRL(*this, "MULT_RELAX", wxTextCtrl)->SetValue(word);
}

void WizardDialogSetFunc::SetWeakenGrain(wxCommandEvent& WXUNUSED(event))
{
  wxVariant word;

  WizardDialogSetFunc dialog(this, WEAKEN_GRAIN, userdat );
  dialog.proc = WEAKEN_GRAIN;

  word = userdat->weaken_grain[0];
  XRCCTRL(*this, "WA_CONST", wxTextCtrl)->SetValue(word);
  word = userdat->weaken_grain[1];
  XRCCTRL(*this, "WA_VISC", wxTextCtrl)->SetValue(word);
  word = userdat->weaken_grain[2];
  XRCCTRL(*this, "WA_BREAKSTRENGTH", wxTextCtrl)->SetValue(word);

  dialog.ShowModal();
}

void WizardDialogSetFunc::CloseWeakenGrain()
{
  bool a,b,c;

  a = XRCCTRL(*this, "WA_CONST", wxTextCtrl)->GetValue().ToDouble(&userdat->weaken_grain[0]);
  b = XRCCTRL(*this, "WA_VISC", wxTextCtrl)->GetValue().ToDouble(&userdat->weaken_grain[1]);
  c = XRCCTRL(*this, "WA_BREAKSTRENGTH", wxTextCtrl)->GetValue().ToDouble(&userdat->weaken_grain[2]);

  if (a && b && c)
    EndModal(true);
  else
    {
      wxMessageDialog msgDlg(this, _("Numbers only!"), _("Error!"), wxOK );
      msgDlg.ShowModal();
    }
}
