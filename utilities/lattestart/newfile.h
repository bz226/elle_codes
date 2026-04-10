//-----------------------------------------------------------------------------
// Name:        myframe.h
// Purpose:     XML resources sample: A derived frame, called MyFrame
// Author:      Robert O'Connor (rob@medicalmnemonics.com), Vaclav Slavik
// RCS-ID:      $Id$
// Copyright:   (c) Robert O'Connor and Vaclav Slavik
// Licence:     wxWindows licence
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// Begin single inclusion of this .h file condition
//-----------------------------------------------------------------------------

#ifndef _NEWFILE_H_
#define _NEWFILE_H_

//-----------------------------------------------------------------------------
// GCC interface
//-----------------------------------------------------------------------------

#if defined(__GNUG__) && !defined(__APPLE__)
#pragma interface "newfile.h"
#endif

//-----------------------------------------------------------------------------
// Headers
//-----------------------------------------------------------------------------

#include "wx/dialog.h"
#include "wizardsetfunc.h"
#include "wizardrunfunc.h"
#include "wizardstat.h"
#include "wizardname.h"
#include "externalprocess.h"
#include "latticegas.h"

//-----------------------------------------------------------------------------
// Class definition: NewFile
//-----------------------------------------------------------------------------

// Define a new frame type: this is going to be our main frame
class NewFile : public wxDialog
  {

  public:

    // Constructor.
    NewFile( wxWindow* parent , wxString );
    ~NewFile();
    wxWindow* master;

  private:
    usdat* userdata;

    void ShowEditSettingsDialog( wxCommandEvent& event );
    void ShowRunFunctionsDialog( wxCommandEvent& event );
    void ShowNameChangeDialog( wxCommandEvent& event );
    void ShowStatisticsDialog( wxCommandEvent& event );
    void SetDefaults( wxCommandEvent& event );
    void OnOk( wxCommandEvent& event);
    void OnCancel( wxCommandEvent& event);
    void ExternalProcess( wxCommandEvent& event);
    void EditLatticeGas( wxCommandEvent& event);
    void SetButtons();

    // Any class wishing to process wxWidgets events must use this macro
    DECLARE_EVENT_TABLE()

  };
#endif
