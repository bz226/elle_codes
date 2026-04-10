
//-----------------------------------------------------------------------------
// GCC implementation
//-----------------------------------------------------------------------------

#ifdef __GNUG__
#pragma implementation "latticegas.h"
#endif

//-----------------------------------------------------------------------------
// Standard wxWidgets headers
//-----------------------------------------------------------------------------

// For compilers that support precompilation, includes "wx/wx.h".
#include "wx/wxprec.h"
#include "wx/variant.h"
#include "wx/textctrl.h"
#include "wx/spinctrl.h"

#ifdef __BORLANDC__
#pragma hdrstop
#endif

// For all others, include the necessary headers (this file is usually all you
// need because it includes almost all "standard" wxWidgets headers)
#ifndef WX_PRECOMP
#include "wx/wx.h"
#endif

#include "latticegas.h"

#include "wx/xrc/xmlres.h"              // XRC XML resouces

BEGIN_EVENT_TABLE(LatticeGas, wxDialog)
EVT_BUTTON( XRCID( "wxID_CANCEL" ),LatticeGas::Cancel )
EVT_BUTTON( XRCID( "wxID_OK" ),LatticeGas::OnOk )
EVT_CHECKBOX( XRCID( "ID_CHECKBOX_DIFFUSION" ),LatticeGas::UnCheckOne)
EVT_CHECKBOX( XRCID( "ID_CHECKBOX_FLUID_FLOW" ),LatticeGas::UnCheckTwo)
END_EVENT_TABLE()



LatticeGas::LatticeGas( wxWindow* parent, process_latticegas prc, usdat* userdata, bool CalledFromNewFile )
{

  CalledByNewFile = CalledFromNewFile;

  userdat = userdata;

  master = parent;

  switch (prc)
    {
    case EDITLATTICEGAS:
      wxXmlResource::Get()->LoadDialog(this, parent, wxT("LatticeGas"));
      // wxXmlResource::Get()->LoadDialog(this, parent, wxT("Distribution"));
      proc = EDITLATTICEGAS;
      SetLatticeGas();
      break;
    }
}

LatticeGas::~LatticeGas()
{
  // if its a lattice-process, deactivate all other buttons
  if (!CalledByNewFile)
    {
      if (userdat->set_fluid_lattice_gas_grain_proc || userdat->set_walls_lattice_gas_proc)
        {
          XRCCTRL(*master, "ID_BUTTON_EDIT_RUNFUNCTIONS", wxButton)->Enable(false);
          XRCCTRL(*master, "ID_BUTTON_STATISTICS", wxButton)->Enable(false);
          XRCCTRL(*master, "ID_BUTTON_EXTERNAL_PROCESSES", wxButton)->Enable(false);
          XRCCTRL(*master, "ID_BUTTON_EXTERNAL_PROCESSES", wxButton)->Enable(false);
          XRCCTRL(*master, "ID_BUTTON_EDIT_RUNFUNCTIONS", wxButton)->Enable(false);
          XRCCTRL(*master, "ID_BUTTON_EDIT_SETTINGS", wxButton)->Enable(false);
        }
      else
        {
          XRCCTRL(*master, "ID_BUTTON_EDIT_RUNFUNCTIONS", wxButton)->Enable(true);
          XRCCTRL(*master, "ID_BUTTON_STATISTICS", wxButton)->Enable(true);
          XRCCTRL(*master, "ID_BUTTON_EXTERNAL_PROCESSES", wxButton)->Enable(true);
          XRCCTRL(*master, "ID_BUTTON_EXTERNAL_PROCESSES", wxButton)->Enable(true);
          XRCCTRL(*master, "ID_BUTTON_EDIT_RUNFUNCTIONS", wxButton)->Enable(true);
          XRCCTRL(*master, "ID_BUTTON_EDIT_SETTINGS", wxButton)->Enable(true);
        }
    }
  else
    {
      if (userdat->set_fluid_lattice_gas_grain_proc || userdat->set_walls_lattice_gas_proc)
        {
          XRCCTRL(*master, "ID_BUTTON_ES", wxButton)->Enable(false);
          XRCCTRL(*master, "ID_BUTTON_ER", wxButton)->Enable(false);
          XRCCTRL(*master, "ID_BUTTON_SS", wxButton)->Enable(false);
          XRCCTRL(*master, "ID_BUTTON_EXTERNAL_PROCESSES", wxButton)->Enable(false);
        }
      else
        {
          XRCCTRL(*master, "ID_BUTTON_ES", wxButton)->Enable(true);
          XRCCTRL(*master, "ID_BUTTON_ER", wxButton)->Enable(true);
          XRCCTRL(*master, "ID_BUTTON_SS", wxButton)->Enable(true);
          XRCCTRL(*master, "ID_BUTTON_EXTERNAL_PROCESSES", wxButton)->Enable(true);
        }
    }
}

//-----------------------------------------------------------------------------
// Private members (including the event handlers)
//-----------------------------------------------------------------------------

void LatticeGas::Cancel( wxCommandEvent& WXUNUSED(event) )
{
  this->Destroy();
}

void LatticeGas::OnOk( wxCommandEvent& WXUNUSED(event))
{
  switch (proc)
    {
    case EDITLATTICEGAS:
      CloseLattice();
      userdat->SaveData();
      break;
    }
  //XRCCTRL(*frame, "ID_TEXT_NAME", wxTextCtrl)->SetValue(userdat->name);
  EndModal(true);
}

void LatticeGas::CloseLattice()
{
  // IST NICHT MEHR REALISIERT!
  XRCCTRL(*this, "ID_TEXTCTRL_GAS_PERCENT", wxTextCtrl)->GetValue().ToDouble(&userdat->fluid_lattice_gas_random[0]);
  userdat->fluid_lattice_gas_random_proc = false;

  //setfunc
  userdat->set_fluid_lattice_gas_grain_proc = XRCCTRL(*this, "ID_CHECKBOX_DIFFUSION", wxCheckBox)->IsChecked();
  //runfunc
  //userdat->lattice_fluid_gas_percent_proc();
  XRCCTRL(*this, "ID_TEXTCTRL_GAS_POROSITY", wxTextCtrl)->GetValue().ToDouble(&userdat->set_fluid_lattice_gas_grain[0]);
  userdat->set_fluid_lattice_gas_grain[1] = int(XRCCTRL(*this, "ID_SPINCTRL_GRAIN_AMOUNT", wxSpinCtrl)->GetValue());

  //only runfuncs necessary
  userdat->set_walls_lattice_gas_proc = XRCCTRL(*this, "ID_CHECKBOX_FLUID_FLOW", wxCheckBox)->IsChecked();
  XRCCTRL(*this, "ID_TEXTCTRL_GAS_PUMP", wxTextCtrl)->GetValue().ToDouble(&userdat->set_walls_lattice_gas[0]);

  if (userdat->set_fluid_lattice_gas_grain_proc || userdat->set_walls_lattice_gas_proc)
    userdat->lattice_fluid_gas_percent_proc = true;

}

void LatticeGas::SetLatticeGas()
{
  wxVariant word;

  word = userdat->fluid_lattice_gas_random[0];
  XRCCTRL(*this, "ID_TEXTCTRL_GAS_PERCENT", wxTextCtrl)->SetValue(word);

  XRCCTRL(*this, "ID_CHECKBOX_DIFFUSION", wxCheckBox)->SetValue(userdat->set_fluid_lattice_gas_grain_proc);
  word = userdat->set_fluid_lattice_gas_grain[0];
  XRCCTRL(*this, "ID_TEXTCTRL_GAS_POROSITY", wxTextCtrl)->SetValue(word);
  XRCCTRL(*this, "ID_SPINCTRL_GRAIN_AMOUNT", wxSpinCtrl)->SetValue(int(userdat->set_fluid_lattice_gas_grain[1]));

  XRCCTRL(*this, "ID_CHECKBOX_FLUID_FLOW", wxCheckBox)->SetValue(userdat->set_walls_lattice_gas_proc);
  word = userdat->set_walls_lattice_gas[0];
  XRCCTRL(*this, "ID_TEXTCTRL_GAS_PUMP", wxTextCtrl)->SetValue(word);
}

void LatticeGas::UnCheckOne(wxCommandEvent& WXUNUSED(event))
{
  if (XRCCTRL(*this, "ID_CHECKBOX_DIFFUSION", wxCheckBox)->IsChecked() && XRCCTRL(*this, "ID_CHECKBOX_FLUID_FLOW", wxCheckBox)->IsChecked())
    XRCCTRL(*this, "ID_CHECKBOX_FLUID_FLOW", wxCheckBox)->SetValue(0);
}

void LatticeGas::UnCheckTwo(wxCommandEvent& WXUNUSED(event))
{
  if (XRCCTRL(*this, "ID_CHECKBOX_DIFFUSION", wxCheckBox)->IsChecked() && XRCCTRL(*this, "ID_CHECKBOX_FLUID_FLOW", wxCheckBox)->IsChecked())
    XRCCTRL(*this, "ID_CHECKBOX_DIFFUSION", wxCheckBox)->SetValue(0);
}
