// Written in 2004
// Author: Dr. J.K. Becker
// Copyright: Dr. J.K. Becker (becker@jkbecker.de)

#include "simwiz.h"

IMPLEMENT_CLASS( SimWiz, wxFrame )

BEGIN_EVENT_TABLE(SimWiz, wxFrame)
  EVT_TOOL(Sok,SimWiz::OnOK)
  EVT_TOOL(Scancel,SimWiz::OnCancel)
END_EVENT_TABLE()

SimWiz::SimWiz(wxWindow* parent):wxFrame(parent,SIMWIZWIN,"Simulation wizard",wxPoint(100,100),wxSize(430,380),wxCAPTION|wxSYSTEM_MENU|wxRESIZE_BORDER)
{
  toolbarbitmap[0]=new wxBitmap(save_xpm);
  toolbarbitmap[1]=new wxBitmap(fileopen_xpm);
  toolbarbitmap[2]=new wxBitmap(ok_xpm);
  toolbarbitmap[3]=new wxBitmap(cancel_xpm);
  wxToolBar *toolbar=CreateToolBar(wxNO_BORDER, -1, "ToolBar");
  //toolbar->AddTool(Ssave, "Save",*(toolbarbitmap[0]),"Save", wxITEM_NORMAL);
  //toolbar->AddTool(Sopen, "Open",*(toolbarbitmap[1]),"Open", wxITEM_NORMAL);
  toolbar->AddTool(Sok, "OK",*(toolbarbitmap[2]),"OK", wxITEM_NORMAL);
  toolbar->AddTool(Scancel, "Cancel",*(toolbarbitmap[3]),"CANCEL", wxITEM_NORMAL);
  toolbar->Realize();
  book=new wxNotebook(this,-1,wxDefaultPosition,wxSize(425,500));
  wxPanel *p=(wxPanel *) NULL;
  p = CreateGeneralPage();
  book->AddPage( p,"General" , FALSE);
  p = CreateElleMeltPage();
  book->AddPage( p,"Melt" , FALSE);
}
wxPanel* SimWiz::CreateGeneralPage(void)
{
//mit welchem elle-file soll die sim denn gemacht werden? file-open-dialog!
  wxPanel *panel=new wxPanel(book);
  wxBoxSizer* base=new wxBoxSizer(wxHORIZONTAL);
  wxBoxSizer* left = new wxBoxSizer(wxVERTICAL);
  wxBoxSizer* right = new wxBoxSizer(wxVERTICAL);
  panel->SetSizer(base);
  panel->SetAutoLayout(TRUE);
  base->Add(left, 0, wxALIGN_CENTER_HORIZONTAL|wxALL, 5);
  base->Add(right, 0, wxALIGN_CENTER_HORIZONTAL|wxALL, 5);

  wxStaticText *l1=new wxStaticText(panel,-1,"Stages",wxDefaultPosition,wxSize(100,23));
  left->Add(l1, 0, wxALIGN_LEFT|wxALL, 5);

  wxTextCtrl *stages=new wxTextCtrl(panel, Sstages, _("10"),wxDefaultPosition, wxDefaultSize);
  right->Add(stages, 0, wxALIGN_RIGHT|wxALL, 5);

  wxStaticText *l2=new wxStaticText(panel,-1,"Save interval",wxDefaultPosition,wxSize(100,23));
  left->Add(l2, 0, wxALIGN_LEFT|wxALL, 5);

  wxTextCtrl *sinterv=new wxTextCtrl(panel, Ssinterv, _("10"),wxDefaultPosition, wxDefaultSize);
  right->Add(sinterv, 0, wxALIGN_RIGHT|wxALL, 5);

  wxStaticText *l3=new wxStaticText(panel,-1,"Save file root",wxDefaultPosition,wxSize(100,23));
  left->Add(l3, 0, wxALIGN_LEFT|wxALL, 5);

  wxTextCtrl *sfiler=new wxTextCtrl(panel, Ssfiler, _("simulation"),wxDefaultPosition, wxDefaultSize);
  right->Add(sfiler, 0, wxALIGN_RIGHT|wxALL, 5);

  wxStaticText *l14=new wxStaticText(panel,-1,"Save pics root",wxDefaultPosition,wxSize(130,23));
  left->Add(l14, 0, wxALIGN_LEFT|wxALL, 5);

  wxButton* spic = new wxButton(panel, Ssapic, _("SavePics"), wxDefaultPosition, wxDefaultSize, 0);
  right->Add(spic, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);

  wxStaticText *l4=new wxStaticText(panel,-1,"Switch distance",wxDefaultPosition,wxSize(130,23));
  left->Add(l4, 0, wxALIGN_LEFT|wxALL, 5);

  wxTextCtrl *switchd=new wxTextCtrl(panel, Sswitchd, _("0.005"),wxDefaultPosition, wxDefaultSize);
  right->Add(switchd, 0, wxALIGN_RIGHT|wxALL, 5);

  wxStaticText *l5=new wxStaticText(panel,-1,"Speed up",wxDefaultPosition,wxSize(100,23));
  left->Add(l5, 0, wxALIGN_LEFT|wxALL, 5);

  wxTextCtrl *sup=new wxTextCtrl(panel, Ssup, _("1"),wxDefaultPosition, wxDefaultSize);
  right->Add(sup, 0, wxALIGN_RIGHT|wxALL, 5);

  return panel;
}
wxPanel* SimWiz::CreateElleMeltPage(void)
{
  wxPanel *panel=new wxPanel(book);
  wxBoxSizer* base = new wxBoxSizer(wxVERTICAL);
  wxBoxSizer* top =new wxBoxSizer(wxHORIZONTAL);
  wxBoxSizer* bottom =new wxBoxSizer(wxHORIZONTAL);
  wxBoxSizer* left=new wxBoxSizer(wxVERTICAL);
  wxBoxSizer *right=new wxBoxSizer(wxVERTICAL);
  panel->SetSizer(base);
  panel->SetAutoLayout(TRUE);
  bottom->Add(left,0,wxALIGN_LEFT|wxALL,5);
  bottom->Add(right,0,wxALIGN_RIGHT|wxALL,5);
  base->Add(top, 0, wxALIGN_CENTER_HORIZONTAL|wxALL, 5);
  base->Add(bottom, 0, wxALIGN_CENTER_HORIZONTAL|wxALL, 5);

  wxCheckBox *elle_melt=new wxCheckBox(panel,Selle_melt,"Use elle_melt",wxDefaultPosition,wxDefaultSize);
  top->Add(elle_melt,0,wxALIGN_LEFT|wxALL,5);

  wxStaticText* tx1=new wxStaticText(panel,-1,"Mobility",wxDefaultPosition,wxSize(130,23));
  wxTextCtrl* mobility = new wxTextCtrl(panel, Smobility, _("1e-11"),wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER);
  left->Add(tx1,0,wxALIGN_LEFT|wxALL,5);
  right->Add(mobility,0,wxALIGN_RIGHT|wxALL,5);

  wxStaticText* tx2=new wxStaticText(panel,-1,"Magic number",wxDefaultPosition,wxSize(130,23));
  wxTextCtrl* magic_number = new wxTextCtrl(panel, Smagic_number, _("0.01"),wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER);
  left->Add(tx2,0,wxALIGN_LEFT|wxALL,5);
  right->Add(magic_number,0,wxALIGN_RIGHT|wxALL,5);

  wxStaticText* tx3=new wxStaticText(panel,-1,"Melt percent",wxDefaultPosition,wxSize(130,23));
  wxTextCtrl* melt_percent = new wxTextCtrl(panel, Smelt_percent, _("-1"),wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER);
  left->Add(tx3,0,wxALIGN_LEFT|wxALL,5);
  right->Add(melt_percent,0,wxALIGN_RIGHT|wxALL,5);

  wxStaticText* tx4=new wxStaticText(panel,-1,"solid-solid",wxDefaultPosition,wxSize(130,23));
  wxTextCtrl* xl_xl = new wxTextCtrl(panel, Sxl_xl, _("1"),wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER);
  left->Add(tx4,0,wxALIGN_LEFT|wxALL,5);
  right->Add(xl_xl,0,wxALIGN_RIGHT|wxALL,5);

  wxStaticText* tx5=new wxStaticText(panel,-1,"solid-liquid",wxDefaultPosition,wxSize(130,23));
  wxTextCtrl* liq_xl = new wxTextCtrl(panel, Sliq_xl, _("1"),wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER);
  left->Add(tx5,0,wxALIGN_LEFT|wxALL,5);
  right->Add(liq_xl,0,wxALIGN_RIGHT|wxALL,5);

  wxStaticText* tx6=new wxStaticText(panel,-1,"liquid-liquid",wxDefaultPosition,wxSize(130,23));
  wxTextCtrl* liq_liq = new wxTextCtrl(panel, Sliq_liq, _("0"),wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER);
  left->Add(tx6,0,wxALIGN_LEFT|wxALL,5);
  right->Add(liq_liq,0,wxALIGN_RIGHT|wxALL,5);

  wxStaticText* tx7=new wxStaticText(panel,-1,"incremental change",wxDefaultPosition,wxSize(130,23));
  wxTextCtrl* inc_change = new wxTextCtrl(panel, Sinc_change, _("0"),wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER);
  left->Add(tx7,0,wxALIGN_LEFT|wxALL,5);
  right->Add(inc_change,0,wxALIGN_RIGHT|wxALL,5);

  return panel;
}

void SimWiz::OnOK(wxCommandEvent & event)
{
  //control.RunSimulation();
}

void SimWiz::OnCancel(wxCommandEvent & event)
{
  Close(true);
}
