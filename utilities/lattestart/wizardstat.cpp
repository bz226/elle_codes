
//-----------------------------------------------------------------------------
// GCC implementation
//-----------------------------------------------------------------------------

#ifdef __GNUG__
    #pragma implementation "wizardstat.h"
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

#include "wizardstat.h"

#include "wx/xrc/xmlres.h"              // XRC XML resouces

BEGIN_EVENT_TABLE(WizardDialogStat, wxDialog)
EVT_BUTTON( XRCID( "wxID_CANCEL" ),WizardDialogStat::Cancel )
EVT_BUTTON( XRCID( "wxID_OK" ),WizardDialogStat::OnOk )
EVT_BUTTON( XRCID( "ID_BUTTON_EDIT_STRESS_BOX" ),WizardDialogStat::SetStressBox )
EVT_BUTTON( XRCID( "EDIT_XSTRESS" ),WizardDialogStat::SetXStress )

//gaussian checkboxes
//EVT_UPDATE_UI(XRCID( "GAUSS_CHECK" ), WizardDialogStat::OnUpdateCheckboxGauss )
//EVT_UPDATE_UI(XRCID( "LIN_CHECK" ), WizardDialogStat::OnUpdateCheckboxLin )
END_EVENT_TABLE()

WizardDialogStat::WizardDialogStat(wxWindow* parent, process_stat prc, usdat* userdata )
{

  userdat = userdata;

  switch (prc)
    {
    case EDITSTATISTICS:
      wxXmlResource::Get()->LoadDialog(this, parent, wxT("Statistics"));
      proc = EDITSTATISTICS;
      SetStatistics();
      break;
    case STRESS_BOX:
      wxXmlResource::Get()->LoadDialog(this, parent, wxT("EditStatisticStressBox"));
      proc = STRESS_BOX;
      break;
    case XSTRESS:
      wxXmlResource::Get()->LoadDialog(this, parent, wxT("EditXYStress"));
      proc = XSTRESS;
      break;
    case YSTRESS:
      wxXmlResource::Get()->LoadDialog(this, parent, wxT("EditXYStress"));
      proc = YSTRESS;
      break;
    }
}

WizardDialogStat::~WizardDialogStat()
{}

//-----------------------------------------------------------------------------
// Private members (including the event handlers)
//-----------------------------------------------------------------------------

void WizardDialogStat::Cancel( wxCommandEvent& WXUNUSED(event) )
{
  this->Destroy();
}

void WizardDialogStat::OnOk( wxCommandEvent& WXUNUSED(event))
{
  switch (proc)
    {
    case EDITSTATISTICS:
      CloseStatistics();
      userdat->SaveData();
      break;
    case STRESS_BOX:
      CloseStressBox();
      break;
    case XSTRESS:
      CloseXStress();
      break;
    case YSTRESS:
      CloseYStress();
      break;
    }

  EndModal(true);
}

void WizardDialogStat::SetStatistics()
{
  XRCCTRL(*this, "ID_CHECKBOX_STRESS_BOX", wxCheckBox)->SetValue(userdat->stress_box_proc);
  XRCCTRL(*this, "ID_CHECKBOX_STRESS_GRAIN", wxCheckBox)->SetValue(userdat->stress_grain_proc);
  XRCCTRL(*this, "ID_CHECKBOX_STRESS_TWO_GRAIN", wxCheckBox)->SetValue(userdat->stress_two_grains_proc);
  XRCCTRL(*this, "ID_CHECKBOX_XSTRESS", wxCheckBox)->SetValue(userdat->xstress_proc);

  XRCCTRL(*this, "ID_SPINCTRL_GRAIN_NB",wxSpinCtrl)->SetValue(int(userdat->stress_grain[0]));

  XRCCTRL(*this, "ID_SPINCTRL_GRAIN_ONE",wxSpinCtrl)->SetValue(int(userdat->stress_two_grains[0]));
  XRCCTRL(*this, "ID_SPINCTRL_GRAIN_TWO",wxSpinCtrl)->SetValue(int(userdat->stress_two_grains[1]));

  XRCCTRL(*this, "ID_CHECKBOX_STATISTIC_SURFACE",wxCheckBox)->SetValue(userdat->surface_proc);
}

void WizardDialogStat::CloseStatistics()
{
  userdat->stress_box_proc = XRCCTRL(*this, "ID_CHECKBOX_STRESS_BOX", wxCheckBox)->GetValue();
  userdat->stress_grain_proc = XRCCTRL(*this, "ID_CHECKBOX_STRESS_GRAIN", wxCheckBox)->GetValue();
  userdat->stress_two_grains_proc = XRCCTRL(*this, "ID_CHECKBOX_STRESS_TWO_GRAIN", wxCheckBox)->GetValue();
  userdat->xstress_proc = XRCCTRL(*this, "ID_CHECKBOX_XSTRESS", wxCheckBox)->GetValue();

  userdat->stress_grain[0] = XRCCTRL(*this, "ID_SPINCTRL_GRAIN_NB",wxSpinCtrl)->GetValue();

  userdat->stress_two_grains[0] = XRCCTRL(*this, "ID_SPINCTRL_GRAIN_ONE",wxSpinCtrl)->GetValue();
  userdat->stress_two_grains[1] = XRCCTRL(*this, "ID_SPINCTRL_GRAIN_TWO",wxSpinCtrl)->GetValue();

  userdat->surface_proc = XRCCTRL(*this, "ID_CHECKBOX_STATISTIC_SURFACE",wxCheckBox)->GetValue();
}

void WizardDialogStat::SetStressBox( wxCommandEvent& WXUNUSED(event) )
{
  wxVariant word;
  WizardDialogStat dialog(this, STRESS_BOX, userdat );

  word = userdat->stress_box[0];
  XRCCTRL(*this, "ID_TEXTCTRL_STRESS_BOX_Y_MIN",wxTextCtrl)->SetValue(word);
  word = userdat->stress_box[1];
  XRCCTRL(*this, "ID_TEXTCTRL_STRESS_BOX_Y_MAX",wxTextCtrl)->SetValue(word);
  word = userdat->stress_box[2];
  XRCCTRL(*this, "ID_TEXTCTRL_STRESS_BOX_X_MIN",wxTextCtrl)->SetValue(word);
  word = userdat->stress_box[3];
  XRCCTRL(*this, "ID_TEXTCTRL_STRESS_BOX_X_MAX",wxTextCtrl)->SetValue(word);

  dialog.ShowModal();
}

void WizardDialogStat::CloseStressBox()
{
  XRCCTRL(*this, "ID_TEXTCTRL_STRESS_BOX_Y_MIN",wxTextCtrl)->GetValue().ToDouble(&userdat->stress_box[0]);
  XRCCTRL(*this, "ID_TEXTCTRL_STRESS_BOX_Y_MAX",wxTextCtrl)->GetValue().ToDouble(&userdat->stress_box[1]);
  XRCCTRL(*this, "ID_TEXTCTRL_STRESS_BOX_X_MIN",wxTextCtrl)->GetValue().ToDouble(&userdat->stress_box[2]);
  XRCCTRL(*this, "ID_TEXTCTRL_STRESS_BOX_X_MAX",wxTextCtrl)->GetValue().ToDouble(&userdat->stress_box[3]);
}

void WizardDialogStat::SetXStress( wxCommandEvent& WXUNUSED(event) )
{
  wxVariant word;
  WizardDialogStat dialog(this, XSTRESS, userdat );

  word = userdat->xstress[0];
  XRCCTRL(*this, "ID_TEXTCTRL_STRESS_XY_Y_MIN",wxTextCtrl)->SetValue(word);
  word = userdat->xstress[1];
  XRCCTRL(*this, "ID_TEXTCTRL_STRESS_XY_Y_MAX",wxTextCtrl)->SetValue(word);
  word = userdat->xstress[2];
  XRCCTRL(*this, "ID_TEXTCTRL_STRESS_XY_X_MIN",wxTextCtrl)->SetValue(word);
  word = userdat->xstress[3];
  XRCCTRL(*this, "ID_TEXTCTRL_STRESS_XY_X_MAX",wxTextCtrl)->SetValue(word);

  XRCCTRL(*this, "ID_RADIOBOX_XY_YX", wxRadioBox)->SetSelection(int(userdat->xstress[4]));

  dialog.ShowModal();
}

void WizardDialogStat::CloseYStress()
{
  XRCCTRL(*this, "ID_TEXTCTRL_STRESS_XY_Y_MIN",wxTextCtrl)->GetValue().ToDouble(&userdat->ystress[0]);
  XRCCTRL(*this, "ID_TEXTCTRL_STRESS_XY_Y_MAX",wxTextCtrl)->GetValue().ToDouble(&userdat->ystress[1]);
  XRCCTRL(*this, "ID_TEXTCTRL_STRESS_XY_X_MIN",wxTextCtrl)->GetValue().ToDouble(&userdat->ystress[2]);
  XRCCTRL(*this, "ID_TEXTCTRL_STRESS_XY_X_MAX",wxTextCtrl)->GetValue().ToDouble(&userdat->ystress[3]);
  userdat->ystress[4] = XRCCTRL(*this, "ID_RADIOBOX_XY_YX", wxRadioBox)->GetSelection();
}

void WizardDialogStat::CloseXStress()
{
  XRCCTRL(*this, "ID_TEXTCTRL_STRESS_XY_Y_MIN",wxTextCtrl)->GetValue().ToDouble(&userdat->xstress[0]);
  XRCCTRL(*this, "ID_TEXTCTRL_STRESS_XY_Y_MAX",wxTextCtrl)->GetValue().ToDouble(&userdat->xstress[1]);
  XRCCTRL(*this, "ID_TEXTCTRL_STRESS_XY_X_MIN",wxTextCtrl)->GetValue().ToDouble(&userdat->xstress[2]);
  XRCCTRL(*this, "ID_TEXTCTRL_STRESS_XY_X_MAX",wxTextCtrl)->GetValue().ToDouble(&userdat->xstress[3]);
  userdat->xstress[4] = XRCCTRL(*this, "ID_RADIOBOX_XY_YX", wxRadioBox)->GetSelection();
}
