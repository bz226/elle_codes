/////////////////////////////////////////////////////////////////////////////
// Name:        runoptsdialog.h
// Purpose:
// Author:      Dr. J.K. Becker
// Modified by:
// Created:
// RCS-ID:
// Copyright: Dr. J.K. Becker
// Licence: GPL
/////////////////////////////////////////////////////////////////////////////

#ifndef _E_RunDia_H_
  #define _E_RunDia_H_
  #include "wx/wx.h"
    #include "wx/grid.h"
  #include <math.h>
  #include <stdlib.h>
#include "runopts.h"

  #define ID_DIALOG 10000
  #define ID_stages 10001
  #define ID_sinterv 10002
  #define ID_sup 10003
  #define ID_switchd 10004
  #define ID_sfiler 10005

  #define ID_OK 10006
  #define ID_RDEFAULT 10008
  #define ID_Cancel 10007
  #define ID_SaPic 10009
  #define ID_logfile 10010
  #define ID_logfileappend 10011
  #define ID_loglevel0 10012
  #define ID_loglevel1 10013
  #define ID_loglevel2 10014
  #define ID_dfreq 10015

class RunDia : public wxDialog
{
  DECLARE_CLASS( RunDia )DECLARE_EVENT_TABLE()

  public : RunDia( wxWindow * parent, wxWindowID id = -1, const wxString & caption = _( "Set run options" ),
       const wxPoint & pos = wxDefaultPosition, const wxSize & size = wxDefaultSize, long style =
       wxCAPTION | wxRESIZE_BORDER | wxSYSTEM_MENU );
private:
  void OnLogFile( wxCommandEvent & event );
  void CreateControls();
  void OnLogLevel0( wxCommandEvent & event );
  void OnLogLevel1( wxCommandEvent & event );
  void OnLogLevel2( wxCommandEvent & event );
  void OnOk( wxCommandEvent & event );
  void OnReset( wxCommandEvent & event );
  void OnCancel( wxCommandEvent & event );
  void OnSavePics( wxCommandEvent & event );
  void Init();
  void OnCellEdit( wxGridEvent & event );
  int vstages, vsfreq, loglevel,displayfrequency;
  double vsdist, vsup;
  wxString vsfiler;
  wxTextCtrl * stages, * sinterv, * sfiler, * switchd, * sup,*dfreq;
  wxCheckBox * logfile, * logf, * loglevel0, * loglevel1, * loglevel2;
  wxGrid *ugrid;
};

#endif // _E_RunDia_H_
