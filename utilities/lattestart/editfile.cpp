
//-----------------------------------------------------------------------------
// GCC implementation
//-----------------------------------------------------------------------------

#ifdef __GNUG__
#pragma implementation "editfile.h"
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

#include "editfile.h"

#include "wx/xrc/xmlres.h"              // XRC XML resouces
#include "stdio.h"

// For functions to manipulate our wxTreeCtrl and wxListCtrl
#include "wx/treectrl.h"
#include "wx/listctrl.h"

//-----------------------------------------------------------------------------
// Event table: connect the events to the handler functions to process them
//-----------------------------------------------------------------------------

BEGIN_EVENT_TABLE(EditFile, wxDialog)
EVT_BUTTON(XRCID("ID_BUTTON_EDIT_SETTINGS"),  EditFile::ShowEditSettingsDialog)
EVT_BUTTON(XRCID("ID_BUTTON_EDIT_RUNFUNCTIONS"), EditFile::ShowRunFunctionsDialog)
EVT_BUTTON(XRCID("ID_BUTTON_CHANGE_NAME"), EditFile::ShowNameChangeDialog)
EVT_BUTTON(XRCID("ID_BUTTON_STATISTICS"), EditFile::ShowStatisticsDialog)
EVT_BUTTON( XRCID("ID_BUTTON_EXTERNAL_PROCESSES"), EditFile::ExternalProcess)
EVT_BUTTON( XRCID("ID_BUTTON_LATTICE"), EditFile::EditLatticeGas)
EVT_BUTTON( XRCID("wxID_OK"), EditFile::OnOk)

END_EVENT_TABLE()

EditFile::EditFile( wxWindow* parent, wxString archive_path, wxString archive_out_path )
{
  master = parent;

  wxXmlResource::Get()->LoadDialog(this, parent, wxT("BasicSettingsExistingFile"));

  userdata = new usdat;

  userdata->CleanData();
  userdata->archive_path = archive_path;
  userdata->archive_out_path = archive_out_path;
  userdata->OpenData();
  // 	if (userdata->name)
  // 		XRCCTRL(*this, "ID_TEXT_NAME", wxStaticText)->SetLabel(userdata->name);

  SetButtons();
}

void EditFile::SetButtons()
{
	  // if its a lattice-process, deactivate all other buttons
	if (userdata->set_fluid_lattice_gas_grain_proc || userdata->set_walls_lattice_gas_proc)
	{
		XRCCTRL(*this, "ID_BUTTON_EDIT_RUNFUNCTIONS", wxButton)->Enable(false);
		XRCCTRL(*this, "ID_BUTTON_STATISTICS", wxButton)->Enable(false);
		XRCCTRL(*this, "ID_BUTTON_EXTERNAL_PROCESSES", wxButton)->Enable(false);
		XRCCTRL(*this, "ID_BUTTON_EXTERNAL_PROCESSES", wxButton)->Enable(false);
		XRCCTRL(*this, "ID_BUTTON_EDIT_RUNFUNCTIONS", wxButton)->Enable(false);
		XRCCTRL(*this, "ID_BUTTON_EDIT_SETTINGS", wxButton)->Enable(false);
	}
	else
	{
		XRCCTRL(*this, "ID_BUTTON_EDIT_RUNFUNCTIONS", wxButton)->Enable(true);
		XRCCTRL(*this, "ID_BUTTON_STATISTICS", wxButton)->Enable(true);
		XRCCTRL(*this, "ID_BUTTON_EXTERNAL_PROCESSES", wxButton)->Enable(true);
		XRCCTRL(*this, "ID_BUTTON_EXTERNAL_PROCESSES", wxButton)->Enable(true);
		XRCCTRL(*this, "ID_BUTTON_EDIT_RUNFUNCTIONS", wxButton)->Enable(true);
		XRCCTRL(*this, "ID_BUTTON_EDIT_SETTINGS", wxButton)->Enable(true);
	}
}

EditFile::~EditFile()
{
  delete userdata;
}

void EditFile::ShowEditSettingsDialog( wxCommandEvent& WXUNUSED(event) )
{
  WizardDialogSetFunc dialog( this, EDIT_MIKE_SETTINGS, userdata );

  dialog.proc = EDIT_MIKE_SETTINGS;
  dialog.ShowModal();
}

void EditFile::OnOk( wxCommandEvent& WXUNUSED(event))
{
  //control has been removed...
  //XRCCTRL(*master, "ID_TEXT_MAIN_NAME", wxStaticText)->SetLabel(userdata->name);
  //userdata->SaveData();
  EndModal(true);
}

void EditFile::ShowRunFunctionsDialog( wxCommandEvent& WXUNUSED(event) )
{
  WizardDialogRunFunc dialog( this, EDIT_MIKE_RUNFUNCTIONS, userdata );

  dialog.proc = EDIT_MIKE_RUNFUNCTIONS;
  dialog.ShowModal();
}

void EditFile::ShowStatisticsDialog( wxCommandEvent& WXUNUSED(event) )
{
  WizardDialogStat dialog( this, EDITSTATISTICS, userdata );

  dialog.proc = EDITSTATISTICS;
  dialog.ShowModal();
}

void EditFile::ShowNameChangeDialog( wxCommandEvent& WXUNUSED(event) )
{
  WizardDialogName dialog( this, EDITNAME, userdata );
  dialog.proc = EDITNAME;
  dialog.ShowModal();
}

void EditFile::ExternalProcess( wxCommandEvent& WXUNUSED(event))
{
  WizardDialogExternalProcess dialog( this, EDITEXTPROC, userdata );

  dialog.proc = EDITEXTPROC;
  dialog.ShowModal();
}

void EditFile::EditLatticeGas( wxCommandEvent& WXUNUSED(event))
{
	LatticeGas dialog( this, EDITLATTICEGAS, userdata, false );

  dialog.proc = EDITLATTICEGAS;
  dialog.ShowModal();
}
