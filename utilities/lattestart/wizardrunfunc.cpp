
//-----------------------------------------------------------------------------
// GCC implementation
//-----------------------------------------------------------------------------

#ifdef __GNUG__
#pragma implementation "wizardrunfunc.h"
#endif

//-----------------------------------------------------------------------------
// Standard wxWidgets headers
//-----------------------------------------------------------------------------

// For compilers that support precompilation, includes "wx/wx.h".
#include "wx/wxprec.h"

#ifdef __BORLANDC__
    #pragma hdrstop
#endif

// For all others, include the necessary headers (this file is usually all you
// need because it includes almost all "standard" wxWidgets headers)
#ifndef WX_PRECOMP
    #include "wx/wx.h"
#endif

#include "wizardrunfunc.h"

#include "wx/xrc/xmlres.h"              // XRC XML resouces

BEGIN_EVENT_TABLE(WizardDialogRunFunc, wxDialog)
EVT_BUTTON( XRCID( "wxID_CANCEL" ),WizardDialogRunFunc::Cancel )
EVT_BUTTON( XRCID( "ID_BUTTON_EXTRADEFORM" ),WizardDialogRunFunc::ExtraDeform )
EVT_BUTTON( XRCID( "wxID_OK" ),WizardDialogRunFunc::OnOk )

EVT_BUTTON( XRCID( "STEPSIZE_DEFAULT" ),WizardDialogRunFunc::SetStepsizeDef )
EVT_BUTTON( XRCID( "VISCOSITY_DEFAULT" ),WizardDialogRunFunc::SetViscDef )

EVT_BUTTON( XRCID( "STY_STRESS_DEF" ),WizardDialogRunFunc::SetStyStressDef )
EVT_BUTTON( XRCID( "STY_SHRINK_DEF" ),WizardDialogRunFunc::SetStyShrinkDef )

EVT_BUTTON( XRCID( "SHRINK_NB_DEF" ),WizardDialogRunFunc::SetShrinkNbDef )
EVT_BUTTON( XRCID( "SHRINK_SHRINK_DEF" ),WizardDialogRunFunc::SetShrinkShrinkDef )
EVT_BUTTON( XRCID( "SHRINK_BOX_DEF" ),WizardDialogRunFunc::SetShrinkBoxDef )

EVT_BUTTON( XRCID( "ID_BUTTON_RUN_DEFAULT" ),WizardDialogRunFunc::CallSetRunFunc)

//gaussian checkboxes
//EVT_UPDATE_UI(XRCID( "GAUSS_CHECK" ), WizardDialogRunFunc::OnUpdateCheckboxGauss )
//EVT_UPDATE_UI(XRCID( "LIN_CHECK" ), WizardDialogRunFunc::OnUpdateCheckboxLin )
END_EVENT_TABLE()

void WizardDialogRunFunc::CallSetRunFunc( wxCommandEvent& WXUNUSED(event))
{
   SetRunFunc();
}

WizardDialogRunFunc::WizardDialogRunFunc(wxWindow* parent, process_run prc, usdat* userdata )
{

  userdat = userdata;

  switch (prc)
    {
    case EDIT_MIKE_RUNFUNCTIONS:
      wxXmlResource::Get()->LoadDialog(this, parent, wxT("NB_Run"));
      // wxXmlResource::Get()->LoadDialog(this, parent, wxT("Distribution"));
      proc = EDIT_MIKE_RUNFUNCTIONS;
      SetRunFunc();
      break;
    case EDIT_EXTRADEFORMATION:
      wxXmlResource::Get()->LoadDialog(this, parent, wxT("ExtraSettingsDeformation"));
      proc = EDIT_EXTRADEFORMATION;
      break;
    }
}

WizardDialogRunFunc::~WizardDialogRunFunc()
{}

//-----------------------------------------------------------------------------
// Private members (including the event handlers)
//-----------------------------------------------------------------------------

void WizardDialogRunFunc::Cancel( wxCommandEvent& WXUNUSED(event) )
{
  this->Destroy();
}

void WizardDialogRunFunc::OnOk( wxCommandEvent& WXUNUSED(event))
{
  switch (proc)
    {
    case EDIT_MIKE_RUNFUNCTIONS:
      CloseRunFunc();
      userdat->SaveData();
      EndModal(true);
      break;
    case EDIT_EXTRADEFORMATION:
      CloseExtraDeform();
      userdat->SaveData();
      break;
    }
}

// start with defaults
void WizardDialogRunFunc::SetRunFunc()
{
  wxVariant word;

  XRCCTRL(*this, "ID_CHECKBOX_COMPACTION", wxCheckBox)->SetValue(userdat->def_unilateral_proc);
  XRCCTRL(*this, "ID_CHECKBOX_PURESHEAR", wxCheckBox)->SetValue(userdat->def_pureshear_proc);
  XRCCTRL(*this, "ID_CHECKBOX_BILATERAL", wxCheckBox)->SetValue(userdat->def_bilateral_proc);
  XRCCTRL(*this, "ID_CHECKBOX_VISCOSITY_ENABLE", wxCheckBox)->SetValue(userdat->viscosity_proc);

  word = userdat->stepsize[0];
  XRCCTRL(*this, "ID_TEXTCTRL_STEPSIZE", wxTextCtrl)->SetValue(word);

  word = userdat->viscosity[0];
  XRCCTRL(*this, "ID_TEXTCTRL_VISCOSITY", wxTextCtrl)->SetValue(word);

  XRCCTRL(*this, "ID_CHECKBOX_VISCOSITY_ENABLE", wxCheckBox)->SetValue(userdat->viscosity_proc);

  XRCCTRL(*this, "ID_RADIOBOX_DISSOLUTION_STRESS",wxRadioBox)->SetSelection(int(userdat->stylolithe_stress_shrink[0]));
  XRCCTRL(*this, "ID_RADIOBOX_DIS_SHRINK", wxRadioBox)->SetSelection(int(userdat->stylolithe_stress_shrink[1]));
  XRCCTRL(*this, "ID_CHECKBOX_STYLOLITHEDISSOLUTION",wxCheckBox)->SetValue(userdat->stylolithe_dissolution);
  XRCCTRL(*this, "ID_SPINCTRL_DISSOLUTION_PLOT",wxSpinCtrl)->SetValue(int(userdat->dis_plot[0]));

  XRCCTRL(*this, "ID_CHECKBOX_DISSOLUTIONSTRAIN",wxCheckBox)->SetValue(userdat->dissolution_strain);

  XRCCTRL(*this, "ID_CHECKBOX_SHRINK_GRAIN", wxCheckBox)->SetValue(userdat->shrink_grain_proc);
  XRCCTRL(*this, "ID_SPINCTRL_GRAINNB",wxSpinCtrl)->SetValue(int(userdat->shrink_grain[0]));
  word = userdat->shrink_grain[1];
  XRCCTRL(*this, "ID_TEXTCTRL_GRAINSHRINK",wxTextCtrl)->SetValue(word);

  XRCCTRL(*this, "ID_CHECKBOX_SHRINK_BOX", wxCheckBox)->SetValue(userdat->shrink_box);
  word = userdat->box_shrinkage[0];
  XRCCTRL(*this, "ID_TEXTCTRL_SHRINK_BOX", wxTextCtrl)->SetValue(word);

  XRCCTRL(*this, "ID_CHECKBOX_SINOSOIDAL", wxCheckBox)->SetValue(userdat->sin_anisotropy_proc);
  word = userdat->sin_anisotropy[0];
  XRCCTRL(*this, "ID_TEXTCTRL_SIN_WAVELENGTH", wxTextCtrl)->SetValue(word);
  word = userdat->sin_anisotropy[1];
  XRCCTRL(*this, "ID_TEXTCTRL_SIN_AMPLITUDE", wxTextCtrl)->SetValue(word);
}

void WizardDialogRunFunc::CloseRunFunc()
{
  userdat->def_unilateral_proc = (XRCCTRL(*this, "ID_CHECKBOX_COMPACTION", wxCheckBox)->GetValue());
  userdat->def_pureshear_proc = (XRCCTRL(*this, "ID_CHECKBOX_PURESHEAR", wxCheckBox)->GetValue());
  userdat->def_bilateral_proc = (XRCCTRL(*this, "ID_CHECKBOX_BILATERAL", wxCheckBox)->GetValue());
  userdat->viscosity_proc = (XRCCTRL(*this, "ID_CHECKBOX_VISCOSITY_ENABLE", wxCheckBox)->GetValue());

  XRCCTRL(*this, "ID_TEXTCTRL_STEPSIZE", wxTextCtrl)->GetValue().ToDouble(&userdat->stepsize[0]);

  userdat->viscosity_proc = XRCCTRL(*this, "ID_CHECKBOX_VISCOSITY_ENABLE", wxCheckBox)->GetValue();
  XRCCTRL(*this, "ID_TEXTCTRL_VISCOSITY", wxTextCtrl)->GetValue().ToDouble(&userdat->viscosity[0]);

  userdat->stylolithe_dissolution = XRCCTRL(*this, "ID_CHECKBOX_STYLOLITHEDISSOLUTION",wxCheckBox)->GetValue();
  userdat->dis_plot[0] = XRCCTRL(*this, "ID_SPINCTRL_DISSOLUTION_PLOT",wxSpinCtrl)->GetValue();
  userdat->stylolithe_stress_shrink[0] = XRCCTRL(*this, "ID_RADIOBOX_DISSOLUTION_STRESS",wxRadioBox)->GetSelection();
  userdat->stylolithe_stress_shrink[1] = XRCCTRL(*this, "ID_RADIOBOX_DIS_SHRINK",wxRadioBox)->GetSelection();
  userdat->dis_plot[0] = XRCCTRL(*this, "ID_SPINCTRL_DISSOLUTION_PLOT",wxSpinCtrl)->GetValue();

  userdat->dissolution_strain = XRCCTRL(*this, "ID_CHECKBOX_DISSOLUTIONSTRAIN",wxCheckBox)->GetValue();

  userdat->shrink_grain_proc = XRCCTRL(*this, "ID_CHECKBOX_SHRINK_GRAIN", wxCheckBox)->GetValue();
  userdat->shrink_grain[0] = XRCCTRL(*this, "ID_SPINCTRL_GRAINNB",wxSpinCtrl)->GetValue();
  XRCCTRL(*this, "ID_TEXTCTRL_GRAINSHRINK",wxTextCtrl)->GetValue().ToDouble(&userdat->shrink_grain[1]);

  userdat->shrink_box = XRCCTRL(*this, "ID_CHECKBOX_SHRINK_BOX", wxCheckBox)->GetValue();
  XRCCTRL(*this, "ID_TEXTCTRL_SHRINK_BOX", wxTextCtrl)->GetValue().ToDouble(&userdat->box_shrinkage[0]);

  userdat->sin_anisotropy_proc = XRCCTRL(*this, "ID_CHECKBOX_SINOSOIDAL", wxCheckBox)->GetValue();
  XRCCTRL(*this, "ID_TEXTCTRL_SIN_WAVELENGTH", wxTextCtrl)->GetValue().ToDouble(&userdat->sin_anisotropy[0]);
  XRCCTRL(*this, "ID_TEXTCTRL_SIN_AMPLITUDE", wxTextCtrl)->GetValue().ToDouble(&userdat->sin_anisotropy[1]);

  EndModal(true);
}

void WizardDialogRunFunc::ExtraDeform( wxCommandEvent& WXUNUSED(event))
{
  wxVariant word;
  WizardDialogRunFunc dialog(this, EDIT_EXTRADEFORMATION, userdat );

  word = userdat->extra_deformation[0];
  XRCCTRL(*this, "ID_TEXTCTRL_EXTRASTEPSIZE", wxTextCtrl)->SetValue(word);
  XRCCTRL(*this, "ID_SPINCTRL_EXTRATIMESTEP", wxSpinCtrl)->SetValue(int(userdat->extra_deformation[1]));
  XRCCTRL(*this, "ID_CHECKBOX_EXTRADEFORM", wxCheckBox)->SetValue(userdat->extra_deform);
  if (userdat->extra_def_unilateral_proc)
     XRCCTRL(*this, "ID_RADIOBOX_EXTRA_DEFORM", wxRadioBox)->SetSelection(0);
  else if (userdat->extra_def_bilateral_proc)
     XRCCTRL(*this, "ID_RADIOBOX_EXTRA_DEFORM", wxRadioBox)->SetSelection(1);
  else if (userdat->extra_def_pureshear_proc)
     XRCCTRL(*this, "ID_RADIOBOX_EXTRA_DEFORM", wxRadioBox)->SetSelection(2);
  
  dialog.ShowModal();
}

void WizardDialogRunFunc::CloseExtraDeform()
{
  if (XRCCTRL(*this, "ID_TEXTCTRL_EXTRASTEPSIZE", wxTextCtrl)->GetValue().ToDouble(&userdat->extra_deformation[0]))
    {
      userdat->extra_deformation[1] = XRCCTRL(*this, "ID_SPINCTRL_EXTRATIMESTEP", wxSpinCtrl)->GetValue();

      userdat->extra_deform = XRCCTRL(*this, "ID_CHECKBOX_EXTRADEFORM", wxCheckBox)->GetValue();
      
      switch (XRCCTRL(*this, "ID_RADIOBOX_EXTRA_DEFORM", wxRadioBox)->GetSelection())
        {
        case 0:
          userdat->extra_def_unilateral_proc = true;
          userdat->extra_def_bilateral_proc = false;
          userdat->extra_def_pureshear_proc = false;
          break;
        case 1:
          userdat->extra_def_unilateral_proc = false;
          userdat->extra_def_bilateral_proc = true;
          userdat->extra_def_pureshear_proc = false;
          break;
        case 2:
          userdat->extra_def_unilateral_proc = false;
          userdat->extra_def_bilateral_proc = false;
          userdat->extra_def_pureshear_proc = true;
          break;
        }
      
      EndModal(true);
    }
  else
    {
      wxMessageDialog msgDlg(this, _("Numbers only!"), _("Error!"), wxOK );
      msgDlg.ShowModal();
    }
}

void WizardDialogRunFunc::SetStepsizeDef ( wxCommandEvent& WXUNUSED(event))
{
	wxVariant word;
	
	word = userdat->stepsize[0];
	XRCCTRL(*this, "ID_TEXTCTRL_STEPSIZE", wxTextCtrl)->SetValue(word);
}

void WizardDialogRunFunc::SetViscDef ( wxCommandEvent& WXUNUSED(event))
{
	wxVariant word;
	
	word = userdat->viscosity[0];
	XRCCTRL(*this, "ID_TEXTCTRL_VISCOSITY", wxTextCtrl)->SetValue(word);	
}

void WizardDialogRunFunc::SetStyStressDef (wxCommandEvent& WXUNUSED(event))
{
	XRCCTRL(*this, "ID_RADIOBOX_DISSOLUTION_STRESS",wxRadioBox)->SetSelection(int(userdat->stylolithe_stress_shrink[0]));
}

void WizardDialogRunFunc::SetStyShrinkDef (wxCommandEvent& WXUNUSED(event))
{
   XRCCTRL(*this, "ID_RADIOBOX_DIS_SHRINK",wxRadioBox)->SetSelection(int(userdat->stylolithe_stress_shrink[1]));
}

void WizardDialogRunFunc::SetShrinkNbDef (wxCommandEvent& WXUNUSED(event))
{
	XRCCTRL(*this, "ID_SPINCTRL_GRAINNB",wxSpinCtrl)->SetValue(int(userdat->shrink_grain[0]));
}

void WizardDialogRunFunc::SetShrinkShrinkDef (wxCommandEvent& WXUNUSED(event))
{
	wxVariant word;

	word = userdat->shrink_grain[1];
	XRCCTRL(*this, "ID_TEXTCTRL_GRAINSHRINK",wxTextCtrl)->SetValue(word);
}

void WizardDialogRunFunc::SetShrinkBoxDef (wxCommandEvent& WXUNUSED(event))
{
	wxVariant word;
	
	word = userdat->box_shrinkage[0];
	XRCCTRL(*this, "ID_TEXTCTRL_SHRINK_BOX", wxTextCtrl)->SetValue(word);
}
