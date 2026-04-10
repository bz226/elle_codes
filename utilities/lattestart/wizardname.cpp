
//-----------------------------------------------------------------------------
// GCC implementation
//-----------------------------------------------------------------------------

#ifdef __GNUG__
#pragma implementation "wizardname.h"
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

#include "wizardname.h"

#include "wx/xrc/xmlres.h"              // XRC XML resouces

BEGIN_EVENT_TABLE(WizardDialogName, wxDialog)
	EVT_BUTTON( XRCID( "wxID_CANCEL" ),WizardDialogName::Cancel )
	EVT_BUTTON( XRCID( "wxID_OK" ),WizardDialogName::OnOk )
END_EVENT_TABLE()



WizardDialogName::WizardDialogName( wxWindow* parent, process_name prc, usdat* userdata )
{
  userdat = userdata;

  master = parent;

  switch (prc)
    {
    case EDITNAME:
      wxXmlResource::Get()->LoadDialog(this, parent, wxT("ChangeName"));
      // wxXmlResource::Get()->LoadDialog(this, parent, wxT("Distribution"));
      proc = EDITNAME;
      SetEditName();
      break;
    }
}

WizardDialogName::~WizardDialogName()
{
}

//-----------------------------------------------------------------------------
// Private members (including the event handlers)
//-----------------------------------------------------------------------------

void WizardDialogName::Cancel( wxCommandEvent& WXUNUSED(event) )
{
  this->Destroy();
}

void WizardDialogName::OnOk( wxCommandEvent& WXUNUSED(event))
{
  switch (proc)
    {
    case EDITNAME:
		//userdat->name.Empty();
      CloseEditName();
      userdat->SaveData();
      break;
    }
    //XRCCTRL(*frame, "ID_TEXT_NAME", wxTextCtrl)->SetValue(userdat->name);
  EndModal(true);
}

void WizardDialogName::CloseEditName()
{
	//userdat->name = XRCCTRL(*this, "ID_TEXTCTRL_NAME", wxTextCtrl)->GetValue();
	//if (userdat->name)
	//XRCCTRL(*master, "ID_TEXT_NAME", wxStaticText)->SetLabel(userdat->name);
}

void WizardDialogName::SetEditName()
{
	//XRCCTRL(*this, "ID_TEXTCTRL_NAME", wxTextCtrl)->SetValue(userdat->name);
}
