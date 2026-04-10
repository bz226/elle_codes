
//-----------------------------------------------------------------------------
// GCC implementation
//-----------------------------------------------------------------------------

#ifdef __GNUG__
#pragma implementation "newfile.h"
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

//-----------------------------------------------------------------------------
// Header of this .cpp file
//-----------------------------------------------------------------------------

#include "newfile.h"

#include "wx/xrc/xmlres.h"              // XRC XML resouces
#include "stdio.h"

// For functions to manipulate our wxTreeCtrl and wxListCtrl
#include "wx/treectrl.h"
#include "wx/listctrl.h"
#include "wx/radiobox.h"

//-----------------------------------------------------------------------------
// Event table: connect the events to the handler functions to process them
//-----------------------------------------------------------------------------

BEGIN_EVENT_TABLE(NewFile, wxDialog)
EVT_BUTTON(XRCID("ID_BUTTON_ES"),  NewFile::ShowEditSettingsDialog)
EVT_BUTTON(XRCID("ID_BUTTON_ER"), NewFile::ShowRunFunctionsDialog)
EVT_BUTTON(XRCID("ID_BUTTON_NC"), NewFile::ShowNameChangeDialog)
EVT_BUTTON(XRCID("ID_BUTTON_SS"), NewFile::ShowStatisticsDialog)
EVT_RADIOBOX(XRCID("ID_RADIOBOX_DEFAULTS"), NewFile::SetDefaults)
EVT_BUTTON( XRCID("ID_BUTTON_EXTERNAL_PROCESSES"), NewFile::ExternalProcess)
EVT_BUTTON( XRCID("ID_BUTTON_LATTICE_GAS"), NewFile::EditLatticeGas)
EVT_BUTTON(XRCID("wxID_OK"), NewFile::OnOk)
EVT_BUTTON(XRCID("wxID_CANCEL"), NewFile::OnCancel)
END_EVENT_TABLE()

NewFile::NewFile( wxWindow* parent, wxString archive_out_path )
{
  master = parent;

  wxXmlResource::Get()->LoadDialog(this, parent, wxT("BasicSettings"));

	userdata = new usdat;
  
  userdata->SetBoudins();
  
   userdata->archive_out_path = archive_out_path;
}

void NewFile::SetButtons()
{
	  // if its a lattice-process, deactivate all other buttons
	if (userdata->set_fluid_lattice_gas_grain_proc || userdata->set_walls_lattice_gas_proc)
	{
		XRCCTRL(*this, "ID_BUTTON_ES", wxButton)->Enable(false);
		XRCCTRL(*this, "ID_BUTTON_ER", wxButton)->Enable(false);
		XRCCTRL(*this, "ID_BUTTON_SS", wxButton)->Enable(false);
		XRCCTRL(*this, "ID_BUTTON_EXTERNAL_PROCESSES", wxButton)->Enable(false);
	}
}

NewFile::~NewFile()
{
	delete userdata;
}

void NewFile::ShowEditSettingsDialog( wxCommandEvent& WXUNUSED(event) )
{
  WizardDialogSetFunc dialog( this, EDIT_MIKE_SETTINGS, userdata );

  dialog.proc = EDIT_MIKE_SETTINGS;
  dialog.ShowModal();
}

void NewFile::ShowRunFunctionsDialog( wxCommandEvent& WXUNUSED(event) )
{
  WizardDialogRunFunc dialog( this, EDIT_MIKE_RUNFUNCTIONS, userdata );

  dialog.proc = EDIT_MIKE_RUNFUNCTIONS;
  dialog.ShowModal();
}

void NewFile::ShowStatisticsDialog( wxCommandEvent& WXUNUSED(event) )
{
  WizardDialogStat dialog( this, EDITSTATISTICS, userdata );

  dialog.proc = EDITSTATISTICS;
  dialog.ShowModal();
}

void NewFile::ShowNameChangeDialog( wxCommandEvent& WXUNUSED(event) )
{
  WizardDialogName dialog( this, EDITNAME, userdata );
  dialog.proc = EDITNAME;
  dialog.ShowModal();
}

void NewFile::SetDefaults( wxCommandEvent& WXUNUSED(event) )
{
  switch (XRCCTRL(*this, "ID_RADIOBOX_DEFAULTS", wxRadioBox)->GetSelection())
    {
    case 0:
      userdata->SetBoudins();
      //XRCCTRL(*this, "ID_TEXT_NAME", wxStaticText)->SetLabel(_T("Boudins"));
      //userdata->name = _T("Boudins");
      break;
    case 1:
      userdata->SetFractures();
      //userdata->name = _T("Fracturing");
      //XRCCTRL(*this, "ID_TEXT_NAME", wxStaticText)->SetLabel(_T("Fracturing"));
      break;
    case 2:
      userdata->SetInclusions();
      //userdata->name = _T("Expanding inclusions");
      //XRCCTRL(*this, "ID_TEXT_NAME", wxStaticText)->SetLabel(_T("Inclusions"));
      break;
    case 3:
      userdata->SetShrinkage();
      //userdata->name = _T("Shrinkage");
      //XRCCTRL(*this, "ID_TEXT_NAME", wxStaticText)->SetLabel(_T("Shrinkage"));
      break;
    case 4:
      userdata->SetGrooves();
      //userdata->name = _T("Grooves");
      //XRCCTRL(*this, "ID_TEXT_NAME", wxStaticText)->SetLabel(_T("Grooves"));
      break;
    case 5:
      userdata->SetStylolithes();
      //userdata->name = _T("Stylolites");
      //XRCCTRL(*this, "ID_TEXT_NAME", wxStaticText)->SetLabel(_T("Stylolites"));
      break;
    case 6:
      userdata->SetPhaseTransition();
      //userdata->name = _T("Phase Transition");
      //XRCCTRL(*this, "ID_TEXT_NAME", wxStaticText)->SetLabel(_T("Phase-transition"));
      break;
    case 7:
      userdata->SetHeatFlow();
      //userdata->name = _T("Heat flow");
      //XRCCTRL(*this, "ID_TEXT_NAME", wxStaticText)->SetLabel(_T("Heat flow"));
      break;
    case 8:
      userdata->SetGrainGrowth();
      //userdata->name = _T("Grain growth");
      //XRCCTRL(*this, "ID_TEXT_NAME", wxStaticText)->SetLabel(_T("Grain growth"));
      break;
    case 9:
      userdata->SetDiffusion();
      //userdata->name = _T("Diffusion");
      //XRCCTRL(*this, "ID_TEXT_NAME", wxStaticText)->SetLabel(_T("Diffusion"));
      break;
    case 10:
      userdata->SetFluidFlow();
      //userdata->name = _T("Fluid flow");
      //XRCCTRL(*this, "ID_TEXT_NAME", wxStaticText)->SetLabel(_T("Fluid flow"));
      break;
    case 11:
      userdata->SetViscousRelax();
      //userdata->name = _T("Viscoelastic relaxation");
      //XRCCTRL(*this, "ID_TEXT_NAME", wxStaticText)->SetLabel(_T("Viscoelastic relax"));
      break;
    case 12:
      userdata->SetGrainGrowthFracture();
      //userdata->name = _T("Graingrowth with fracturing");
      //XRCCTRL(*this, "ID_TEXT_NAME", wxStaticText)->SetLabel(_T("Graingrowth with fracturing"));
      break;
    }
}

void NewFile::OnOk( wxCommandEvent& WXUNUSED(event))
{
  userdata->SaveData();
  //XRCCTRL(*master, "ID_TEXT_MAIN_NAME", wxStaticText)->SetLabel(userdata->name);
  EndModal(true);
}

void NewFile::OnCancel( wxCommandEvent& WXUNUSED(event))
{
  Destroy();
}

void NewFile::ExternalProcess( wxCommandEvent& WXUNUSED(event))
{
  WizardDialogExternalProcess dialog( this, EDITEXTPROC, userdata );

  dialog.proc = EDITEXTPROC;
  dialog.ShowModal();
}

void NewFile::EditLatticeGas( wxCommandEvent& WXUNUSED(event))
{
  LatticeGas dialog( this, EDITLATTICEGAS, userdata, true );

  SetButtons();

  dialog.proc = EDITLATTICEGAS;
  dialog.ShowModal();
}
