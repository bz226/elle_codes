#ifndef _E_simwiz_h
#define _E_simwiz_h

#include "wx/wx.h"
#include "wx/notebook.h"
//#include "control.h"
#include "icons.h"
#include "attrib.h"
  #include "defs.h"

enum
{
  Ssave,
  Sopen,
  Selle_melt,
  Smobility,
  Smagic_number,
  Smelt_percent,
  Sxl_xl,
  Sliq_xl,
  Sliq_liq,
  Sinc_change,
  Sok,
  Scancel,
  Sdialog,
  Sstages,
  Ssinterv,
  Ssup,
  Sswitchd,
  Ssfiler,
  Ssapic,
};

class SimWiz:public wxFrame
{
        DECLARE_CLASS(SimWiz)
 public:
  SimWiz(wxWindow *parent);
 private:
  wxBitmap *toolbarbitmap[4];
  wxNotebook *book;
  wxPanel* CreateElleMeltPage();
  wxPanel* CreateGeneralPage();
  //Control control;
  UserData elle_melt;
  void OnOK(wxCommandEvent & event);
  void OnCancel(wxCommandEvent & event);
 DECLARE_EVENT_TABLE()
};

#endif //simwiz
