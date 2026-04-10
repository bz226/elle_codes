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

#ifndef _MYFRAME_H_
#define _MYFRAME_H_

//-----------------------------------------------------------------------------
// GCC interface
//-----------------------------------------------------------------------------

#if defined(__GNUG__) && !defined(__APPLE__)
 #pragma interface "myframe.h"
#endif

//-----------------------------------------------------------------------------
// Headers
//-----------------------------------------------------------------------------

#include "wx/frame.h"
#include "wx/string.h"

#include "newfile.h"
#include "editfile.h"
#include "wx/process.h"

#include "wx/numdlg.h"
#include "wx/ffile.h"
#include "wx/utils.h"

#include "wx/txtstrm.h"
#include "wx/wfstream.h"
#include "wx/zipstrm.h"
#include "wx/sstream.h"

#include "wx/spinctrl.h"

#include "wx/xrc/xmlres.h"              // XRC XML resouces
#include "stdio.h"

// For functions to manipulate our wxTreeCtrl and wxListCtrl
#include "wx/treectrl.h"
#include "wx/listctrl.h"
#include "wx/string.h"

#include "wx/file.h"
#include "wx/filedlg.h"

#include "wx/event.h"

#include "wx/zipstrm.h"

enum ext_proc
{
	LATTE_PROC,
	ELLE_TO_PS_PROC,
	PS_TO_GIF_PROC,
	CONVERT_MOVIE
};

//-----------------------------------------------------------------------------
// Class definition: MyFrame
//-----------------------------------------------------------------------------

// Define a new frame type: this is going to be our main frame
class MyFrame : public wxFrame
{

public:

	// Constructor
	MyFrame( wxWindow* parent = ( wxWindow * ) NULL );
	~MyFrame();

private:

	NewFile* newdialog;
	EditFile* editdialog;

	wxString ellefile;
	wxString elle_path;
	wxString elle_dir;

	wxString archive;
	wxString archive_path;
	wxString archive_dir;

	wxString archive_out;
	wxString archive_out_path;
	wxString archive_out_dir;
	wxString appdirectory;

	wxPathList pathlist;

	bool SPSIsRunning;

	wxProcess *callback, *movie_callback, *convert_callback;

	bool userselectedarchive;

	long code, movie_code;
	bool IsRunning;

	bool CheckArchiveDir();
	bool CheckArchive( wxString );
	void OpenArchiveDescription( wxString );
	void SaveDesc();

	void ChooseElleFile( wxCommandEvent& event );
	void ChooseZipFile( wxCommandEvent& event );
	void ShowEditExistingDialog( wxCommandEvent& event );
	void ShowCreateNewFileDialog( wxCommandEvent& event );
	void OnOk( wxCommandEvent& event );
	void StartRun( wxCommandEvent& event );
	void OnCancel( wxCommandEvent& event );
	void BrowseZip( wxCommandEvent& event );
	void SelectArchive( wxCommandEvent& event );
	void SelectElleFile( wxCommandEvent& event );
	void RunTerminated( wxProcessEvent& event );
	void SaveStdout( wxCommandEvent& event );
	void SelectWorkdir( wxCommandEvent& event );

	void SelectExecutable( wxCommandEvent& event );

	/// Handle idle events
	void OnIdle( wxIdleEvent& event );

	/// Make a movie of a past run
	void MakeMovie( wxCommandEvent& WXUNUSED( event ) );
	void MakeGif();
	void ConvertElle();
	void ConvertElleTerminated(wxProcessEvent& event);
	void MakeGifTerminated(wxProcessEvent& event);

	void ChooseMovieStartFile( wxCommandEvent& );
	void ChooseMovieEndFile( wxCommandEvent& event );
	void ChooseSaveFileName( wxCommandEvent& );

	long elle_counter, elle_stepsize, start_file_nb, end_file_nb;
	wxString elle_filename, start_file_dir, end_file_dir, save_file_dir, gif_save_file, moviemake_dir;

	// pathes
	wxString showelleps_path, mencoder_path, latte_path, convert_path;

	/// Convert Movies (depending on mencoder)
	void ChooseConversionFile( wxCommandEvent& event );
	void StartConversion( wxCommandEvent& event );
	void ConvertMovieTerminated( wxProcessEvent& event );
	void GifChecked( wxCommandEvent& event );
	void AviChecked( wxCommandEvent& event );
	void MpegChecked( wxCommandEvent& event );

	wxString convert_file;
	bool is_gif, is_avi, is_mpeg, CIsRunning;
	long convert_code;

	// Any class wishing to process wxWidgets events must use this macro
	DECLARE_EVENT_TABLE()

};
#endif
