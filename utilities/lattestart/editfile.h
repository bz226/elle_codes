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

#ifndef _EDITFILE_H_
#define _EDITFILE_H_

//-----------------------------------------------------------------------------
// GCC interface
//-----------------------------------------------------------------------------

#if defined(__GNUG__) && !defined(__APPLE__)
#pragma interface "editfile.h"
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
// Class definition: MyFrame
//-----------------------------------------------------------------------------

// Define a new frame type: this is going to be our main frame
class EditFile : public wxDialog
  {

  public:

    // Constructor
    EditFile( wxWindow* parent , wxString, wxString);
    ~EditFile();
    wxWindow* master;

  private:
    usdat* userdata;

    void ShowEditSettingsDialog( wxCommandEvent& event );
    void ShowRunFunctionsDialog( wxCommandEvent& event );
    void ShowNameChangeDialog( wxCommandEvent& event );
    void ShowStatisticsDialog( wxCommandEvent& event );
    void ExternalProcess( wxCommandEvent& event );
    void EditLatticeGas( wxCommandEvent& event);

    void OnOk( wxCommandEvent& WXUNUSED(event));
	 
	 void SetButtons();

    // Any class wishing to process wxWidgets events must use this macro
    DECLARE_EVENT_TABLE()

  };
#endif

