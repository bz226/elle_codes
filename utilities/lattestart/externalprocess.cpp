
//-----------------------------------------------------------------------------
// GCC implementation
//-----------------------------------------------------------------------------

#ifdef __GNUG__
#pragma implementation "externalprocess.h"
#endif

//-----------------------------------------------------------------------------
// Standard wxWidgets headers
//-----------------------------------------------------------------------------

// For compilers that support precompilation, includes "wx/wx.h".
#include "wx/wxprec.h"
#include "wx/variant.h"
#include "wx/textctrl.h"

#ifdef __BORLANDC__
#pragma hdrstop
#endif

// For all others, include the necessary headers (this file is usually all you
// need because it includes almost all "standard" wxWidgets headers)
#ifndef WX_PRECOMP
#include "wx/wx.h"
#endif

#include "externalprocess.h"

#include "wx/xrc/xmlres.h"              // XRC XML resouces

BEGIN_EVENT_TABLE(WizardDialogExternalProcess, wxDialog)
	EVT_BUTTON( XRCID( "wxID_CANCEL" ),WizardDialogExternalProcess::Cancel )
	EVT_BUTTON( XRCID( "wxID_OK" ),WizardDialogExternalProcess::OnOk )
END_EVENT_TABLE()



WizardDialogExternalProcess::WizardDialogExternalProcess( wxWindow* parent, process_external_process prc, usdat* userdata )
{
  userdat = userdata;

  master = parent;

  switch (prc)
    {
    case EDITEXTPROC:
      wxXmlResource::Get()->LoadDialog(this, parent, wxT("ExternalProcesses"));
      proc = EDITEXTPROC;
      SetEditExtProc();
      break;
    }
}

WizardDialogExternalProcess::~WizardDialogExternalProcess()
{}

//-----------------------------------------------------------------------------
// Private members (including the event handlers)
//-----------------------------------------------------------------------------

void WizardDialogExternalProcess::Cancel( wxCommandEvent& WXUNUSED(event) )
{
  this->Destroy();
}

void WizardDialogExternalProcess::OnOk( wxCommandEvent& WXUNUSED(event))
{
  switch (proc)
    {
    case EDITEXTPROC:
      CloseEditExtProc();
      userdat->SaveData();
      break;
    }
  EndModal(true);
}

void WizardDialogExternalProcess::SetEditExtProc()
{
  XRCCTRL(*this, "ID_CHECKBOX_GG", wxCheckBox)->SetValue(userdat->graingrowth_proc);
  XRCCTRL(*this, "ID_CHECKBOX_MELT", wxCheckBox)->SetValue(userdat->melt_proc);
  XRCCTRL(*this, "ID_CHECKBOX_MELT", wxCheckBox)->Enable(false);
}

void WizardDialogExternalProcess::CloseEditExtProc()
{
  userdat->graingrowth_proc = XRCCTRL(*this, "ID_CHECKBOX_GG", wxCheckBox)->GetValue();
  userdat->add_doubles_proc = XRCCTRL(*this, "ID_CHECKBOX_GG", wxCheckBox)->GetValue();
  userdat->melt_proc = XRCCTRL(*this, "ID_CHECKBOX_MELT", wxCheckBox)->GetValue();
}
