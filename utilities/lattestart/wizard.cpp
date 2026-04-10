
//-----------------------------------------------------------------------------
// GCC implementation
//-----------------------------------------------------------------------------

#ifdef __GNUG__
    #pragma implementation "wizard.h"
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

#include "wizard.h"

#include "wx/xrc/xmlres.h"              // XRC XML resouces

BEGIN_EVENT_TABLE(WizardDialog, wxDialog)
EVT_BUTTON( XRCID( "wxID_CANCEL" ),WizardDialog::Cancel )
EVT_BUTTON( XRCID( "wxID_OK" ),WizardDialog::OnOk )

EVT_BUTTON( XRCID( "GSP" ),WizardDialog::EditGaussStrengthParas )
EVT_BUTTON( XRCID( "GRP" ),WizardDialog::EditGaussRateParas )
EVT_BUTTON( XRCID( "GSpP" ),WizardDialog::EditGaussSpringParas )

EVT_BUTTON( XRCID( "EDIT_WALLS" ),WizardDialog::EditWalls )
EVT_BUTTON( XRCID( "WEAKEN_HORIZ" ),WizardDialog::WeakenHoriz )
EVT_BUTTON( XRCID( "WEAKEN_ALL" ),WizardDialog::WeakenEdit )
EVT_BUTTON( XRCID( "EDIT_GB" ),WizardDialog::EditGrainBoundaries )

//gaussian checkboxes
//EVT_UPDATE_UI(XRCID( "GAUSS_CHECK" ), WizardDialog::OnUpdateCheckboxGauss )
//EVT_UPDATE_UI(XRCID( "LIN_CHECK" ), WizardDialog::OnUpdateCheckboxLin )
END_EVENT_TABLE()

WizardDialog::WizardDialog(wxWindow* parent, process prc)
{

  //userdat.OpenData();

  switch (prc)
    {
    case EDIT_MIKE_SETTINGS:
      wxXmlResource::Get()->LoadDialog(this, parent, wxT("NB_Set"));
      // wxXmlResource::Get()->LoadDialog(this, parent, wxT("Distribution"));
      proc = EDIT_MIKE_SETTINGS;
      userdat.OpenData();
      SetSettings();
      break;
    case EDIT_MIKE_RUNFUNCTIONS:
      wxXmlResource::Get()->LoadDialog(this, parent, wxT("NB_Run"));
      break;
    case EDITNAME:
      wxXmlResource::Get()->LoadDialog(this, parent, wxT("ChangeName"));
      break;
    case EDITSTATISTICS:
      wxXmlResource::Get()->LoadDialog(this, parent, wxT("Statistics"));
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
    }
}

WizardDialog::~WizardDialog()
{}

//-----------------------------------------------------------------------------
// Private members (including the event handlers)
//-----------------------------------------------------------------------------

void WizardDialog::Cancel( wxCommandEvent& WXUNUSED(event) )
{
  this->Destroy();
}

void WizardDialog::OnOk( wxCommandEvent& WXUNUSED(event))
{
  switch (proc)
    {
    case EDIT_MIKE_SETTINGS:
      CloseEditMikeSettings();
      userdat.SaveData();
      break;
    case EDIT_MIKE_RUNFUNCTIONS:
      CloseRunFunc();
      break;
    case EDITNAME:
      CloseNameEdit();
      break;
    case EDITSTATISTICS:
      CloseStatEdit();
      break;
    case EDITGAUSSPARAMS:
      CloseGaussParams();
      break;
    case EDITSPRINGGAUSS:
      CloseSpringGauss();
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
    }
}

// initialise
void WizardDialog::EditGaussStrengthParas( wxCommandEvent& WXUNUSED(event) )
{
  WizardDialog dialog(this, EDITGAUSSPARAMS );
  dialog.proc = EDITGAUSSPARAMS;
  dialog.ShowModal();
}

void WizardDialog::EditGaussRateParas( wxCommandEvent& WXUNUSED(event) )
{
  WizardDialog dialog(this, EDITGAUSSPARAMS );
  dialog.proc = EDITRATEGAUSS;
  dialog.ShowModal();
}

void WizardDialog::EditGaussSpringParas( wxCommandEvent& WXUNUSED(event) )
{
  WizardDialog dialog(this, EDITGAUSSPARAMS );
  dialog.proc = EDITSPRINGGAUSS;
  dialog.ShowModal();
}

void WizardDialog::EditWalls( wxCommandEvent& WXUNUSED(event) )
{
  WizardDialog dialog(this, WALL_EDIT );
  dialog.proc = WALL_EDIT;
  dialog.ShowModal();
}

void WizardDialog::WeakenHoriz( wxCommandEvent& WXUNUSED(event) )
{
  WizardDialog dialog(this, WEAKEN_EDIT );
  dialog.proc = WEAKEN_EDIT;
  dialog.ShowModal();
}

void WizardDialog::WeakenEdit( wxCommandEvent& WXUNUSED(event) )
{
  WizardDialog dialog(this, WEAKEN_ALL_EDIT );
  dialog.proc = WEAKEN_ALL_EDIT;
  dialog.ShowModal();
}

void WizardDialog::EditGrainBoundaries( wxCommandEvent& WXUNUSED(event))
{
  WizardDialog dialog(this,  EDIT_GRAIN_BOUNDARIES );
  dialog.proc = EDIT_GRAIN_BOUNDARIES;
  dialog.ShowModal();
}


// on exit functions (called in destructor)
void WizardDialog::CloseRunFunc()
{}

void WizardDialog::CloseNameEdit()
{}

void WizardDialog::CloseStatEdit()
{}

void WizardDialog::SetSettings()
{
   wxString word;
   string word2;
   
  //the distribution-type for strength
   if (userdat.gauss_strength_proc)
      XRCCTRL(*this, "STRENGTH_DISTRI", wxRadioBox)->SetSelection(2);
   else if (userdat.linear_strength_proc)
      XRCCTRL(*this, "STRENGTH_DISTRI", wxRadioBox)->SetSelection(1);
   else
      XRCCTRL(*this, "STRENGTH_DISTRI", wxRadioBox)->SetSelection(0);

   //rate distrib
   if(userdat.gauss_rate_proc)
      XRCCTRL(*this, "RATE_DISTRI", wxCheckBox)->SetValue(true);
   
   //spring distrib
   if (userdat.gauss_spring_proc)
      XRCCTRL(*this, "SPRING_DISTRI", wxCheckBox)->SetValue(true);

   if (userdat.multiply_relax_proc)
      XRCCTRL(*this, "CHECK_MULT_RELAX", wxCheckBox)->SetValue(true);

}

void WizardDialog::CloseGaussParams()
{
  if (XRCCTRL(*this, "GAUSS_MEAN", wxTextCtrl)->GetValue().ToDouble(&userdat.gauss_strength[0]) && XRCCTRL(*this, "GAUSS_SIGMA", wxTextCtrl)->GetValue().ToDouble(&userdat.gauss_strength[1]))
    {
      userdat.gauss_strength_proc = true;
      EndModal(true);
    }
  else
    {
      wxMessageDialog msgDlg(this, _("Numbers only!"), _("Error!"), wxOK );
      msgDlg.ShowModal();
    }

  //cout << userdat.gauss_strength[1] << endl;
}

void WizardDialog::CloseSpringGauss()
{
  if (XRCCTRL(*this, "GAUSS_MEAN", wxTextCtrl)->GetValue().ToDouble(&userdat.gauss_spring[0]) && XRCCTRL(*this, "GAUSS_SIGMA", wxTextCtrl)->GetValue().ToDouble(&userdat.gauss_spring[1]))
    EndModal(true);
  else
    {
      wxMessageDialog msgDlg(this, _("Numbers only!"), _("Error!"), wxOK );
      msgDlg.ShowModal();
    }
}

void WizardDialog::CloseRateGauss()
{
  if (XRCCTRL(*this, "GAUSS_MEAN", wxTextCtrl)->GetValue().ToDouble(&userdat.gauss_rate[0]) && XRCCTRL(*this, "GAUSS_SIGMA", wxTextCtrl)->GetValue().ToDouble(&userdat.gauss_rate[1]))
    EndModal(true);
  else
    {
      wxMessageDialog msgDlg(this, _("Numbers only!"), _("Error!"), wxOK );
      msgDlg.ShowModal();
    }
}

void WizardDialog::CloseWallEdit()
{
  if (XRCCTRL(*this, "SW_BOTH_EDIT", wxTextCtrl)->GetValue().ToDouble(&userdat.walls[0]) && XRCCTRL(*this, "SW_CONST_EDIT", wxTextCtrl)->GetValue().ToDouble(&userdat.walls[1]))
    EndModal(true);
  else
    {
      wxMessageDialog msgDlg(this, _("Numbers only!"), _("Error!"), wxOK );
      msgDlg.ShowModal();
    }
}

void WizardDialog::CloseWeakenEdit()
{
  bool a,b,c,d,e;

  a = XRCCTRL(*this, "WH_Y_MIN", wxTextCtrl)->GetValue().ToDouble(&userdat.weaken_hor[0]);
  b = XRCCTRL(*this, "WH_Y_MAX", wxTextCtrl)->GetValue().ToDouble(&userdat.weaken_hor[1]);
  c = XRCCTRL(*this, "WH_CONST", wxTextCtrl)->GetValue().ToDouble(&userdat.weaken_hor[2]);
  d = XRCCTRL(*this, "WH_VISC", wxTextCtrl)->GetValue().ToDouble(&userdat.weaken_hor[3]);
  e = XRCCTRL(*this, "WH_BREAKSTRENGTH", wxTextCtrl)->GetValue().ToDouble(&userdat.weaken_hor[1]);

  if (a && b && c && d && e)
    EndModal(true);
  else
    {
      wxMessageDialog msgDlg(this, _("Numbers only!"), _("Error!"), wxOK );
      msgDlg.ShowModal();
    }
}

void WizardDialog::CloseWeakenAllEdit()
{
  bool a,b,c;

  a = XRCCTRL(*this, "WA_CONST", wxTextCtrl)->GetValue().ToDouble(&userdat.weaken_all[0]);
  b = XRCCTRL(*this, "WA_VISC", wxTextCtrl)->GetValue().ToDouble(&userdat.weaken_all[1]);
  c = XRCCTRL(*this, "WA_BREAKSTRENGTH", wxTextCtrl)->GetValue().ToDouble(&userdat.weaken_all[2]);

  if (a && b && c)
    EndModal(true);
  else
    {
      wxMessageDialog msgDlg(this, _("Numbers only!"), _("Error!"), wxOK );
      msgDlg.ShowModal();
    }
}

void WizardDialog::CloseGrainBoundaryEdit()
{
  if (XRCCTRL(*this, "GB_CONST", wxTextCtrl)->GetValue().ToDouble(&userdat.make_gb[0]) && XRCCTRL(*this, "GB_BREAKSTRENGTH", wxTextCtrl)->GetValue().ToDouble(&userdat.make_gb[1]))
    EndModal(true);
  else
    {
      wxMessageDialog msgDlg(this, _("Numbers only!"), _("Error!"), wxOK );
      msgDlg.ShowModal();
    }
}

void WizardDialog::CloseEditMikeSettings()
{
  bool a,b,c,d,e,f,g,h;

  //the distribution-type for strength
  switch (XRCCTRL(*this, "STRENGTH_DISTRI", wxRadioBox)->GetSelection())
    {
    case 0:
      userdat.gauss_strength_proc = false;
      userdat.linear_strength_proc = false;
      break;
    case 1:
      userdat.gauss_strength_proc = false;
      userdat.linear_strength_proc = true;
      break;
    case 2:
      userdat.gauss_strength_proc = true;
      userdat.linear_strength_proc = false;
      break;
    }

  //rate distrib
  userdat.gauss_rate_proc = XRCCTRL(*this, "RATE_DISTRI", wxCheckBox)->IsChecked();
  //spring distrib
  userdat.gauss_spring_proc = XRCCTRL(*this, "SPRING_DISTRI", wxCheckBox)->IsChecked();

  //Relaxthreshold
  a = XRCCTRL(*this, "MULT_RELAX", wxTextCtrl)->GetValue().ToDouble(&userdat.multiply_relax[0]);
  userdat.multiply_relax_proc = XRCCTRL(*this, "CHECK_MULT_RELAX", wxCheckBox)->IsChecked();

  //timestep
  b = XRCCTRL(*this, "TIMESTEP", wxTextCtrl)->GetValue().ToDouble(&userdat.timestep[0]);
  userdat.timestep_proc = XRCCTRL(*this, "CHECK_TIMESTEP", wxCheckBox)->IsChecked();

  //set walls
  userdat.walls_proc = XRCCTRL(*this, "CHECK_SET_WALLS", wxCheckBox)->IsChecked();

  //wweaken horizontal
  userdat.weaken_hor_proc = XRCCTRL(*this, "CHECK_WEAKEN_HORIZ", wxCheckBox)->IsChecked();

  //weaken all
  userdat.weaken_all_proc = XRCCTRL(*this, "CHECK_WEAKEN_ALL", wxCheckBox)->IsChecked();

  //make gb
  userdat.make_gb_proc = XRCCTRL(*this, "CHECK_MAKE_GB", wxCheckBox)->IsChecked();

  //mineral
  switch (XRCCTRL(*this, "ID_MINERAL", wxRadioBox)->GetSelection() == 0)
    {
    case 0:
      userdat.set_min = false;
    case 3:
      userdat.set_min = true;
      userdat.mineral[0] = 3;
    case 1:
      userdat.set_min = true;
      userdat.mineral[0] = 1;
    case 2:
      userdat.set_min = true;
      userdat.mineral[0] = 2;
    }

  // x-row
  userdat.dissolve_x_proc = XRCCTRL(*this, "CHECK_X_ROW", wxCheckBox)->IsChecked();
  c = XRCCTRL(*this, "DIS_UPPER", wxTextCtrl)->GetValue().ToDouble(&userdat.dissolve_x[0]);
  d = XRCCTRL(*this, "DIS_LOWER", wxTextCtrl)->GetValue().ToDouble(&userdat.dissolve_x[1]);

  // y-row
  userdat.dissolve_y_proc = XRCCTRL(*this, "CHECK_Y_ROW", wxCheckBox)->IsChecked();
  e = XRCCTRL(*this, "DIS_LEFT", wxTextCtrl)->GetValue().ToDouble(&userdat.dissolve_y[0]);
  f = XRCCTRL(*this, "DIS_RIGHT", wxTextCtrl)->GetValue().ToDouble(&userdat.dissolve_y[1]);

  // box size
  userdat.set_box_size_proc = XRCCTRL(*this, "CHECK_BOX_SIZE", wxCheckBox)->IsChecked();
  g = XRCCTRL(*this, "BOX_SIZE", wxTextCtrl)->GetValue().ToDouble(&userdat.set_box_size[0]);

  // plot frac
  userdat.plot_frac_proc = XRCCTRL(*this, "CHECK_PLOT", wxCheckBox)->IsChecked();
  h = XRCCTRL(*this, "PLOT_NB", wxTextCtrl)->GetValue().ToDouble(&userdat.plot_frac[0]);

  if (a && b && c && e && f && g && h)
    EndModal(true);
  else
    {
      wxMessageDialog msgDlg(this, _("Numbers only!"), _("Error!"), wxOK );
      msgDlg.ShowModal();
    }
}

