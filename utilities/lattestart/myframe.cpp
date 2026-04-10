
//-----------------------------------------------------------------------------
// GCC implementation
//-----------------------------------------------------------------------------

#ifdef __GNUG__
 #pragma implementation "myframe.h"
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
#include "myframe.h"

// definiert den wxZipEntryPtr "smartpointer", was auch immer das sein mag
#include <wx/ptr_scpd.h>
wxDEFINE_SCOPED_PTR_TYPE( wxZipEntry );

//-----------------------------------------------------------------------------
// Event table: connect the events to the handler functions to process them
//-----------------------------------------------------------------------------

BEGIN_EVENT_TABLE( MyFrame, wxFrame )
EVT_BUTTON( XRCID( "ID_BUTTON_EDITEXISTINGFILE" ), MyFrame::ShowEditExistingDialog )
EVT_BUTTON( XRCID( "ID_BUTTON_CREATENEWFILE" ), MyFrame::ShowCreateNewFileDialog )
EVT_CHOICE( XRCID( "ID_CHOICE_ELLE_FILE" ), MyFrame::ChooseElleFile )
EVT_CHOICE( XRCID( "ID_CHOICE_ZIP_FILE" ), MyFrame::ChooseZipFile )
EVT_BUTTON( XRCID( "wxID_OK" ), MyFrame::OnOk )
EVT_BUTTON( XRCID( "wxID_CANCEL" ), MyFrame::OnCancel )
EVT_BUTTON( XRCID( "ID_BUTTON_BROWSE_ZIP" ), MyFrame::BrowseZip )
EVT_BUTTON( XRCID( "ID_BUTTON_START" ), MyFrame::StartRun )
EVT_BUTTON( XRCID( "ID_BUTTON_BROWSEWORKINGDIR" ), MyFrame::SelectElleFile )
EVT_BUTTON( XRCID( "ID_BUTTON_BROWSEARCHIVEDIR" ), MyFrame::SelectArchive )

EVT_BUTTON( XRCID( "ID_BUTTON_STARTMOVIEMAKE" ), MyFrame::MakeMovie )
EVT_BUTTON( XRCID( "ID_BUTTON_SELECTSTART" ), MyFrame::ChooseMovieStartFile )
EVT_BUTTON( XRCID( "ID_BUTTON_SELECTEND" ), MyFrame::ChooseMovieEndFile )
EVT_BUTTON( XRCID( "ID_BUTTON_SELECTSAVENAME" ), MyFrame::ChooseSaveFileName )

EVT_BUTTON( XRCID( "ID_BUTTON_CONVERTFILE" ), MyFrame::ChooseConversionFile )
EVT_BUTTON( XRCID( "ID_BUTTON_CONVERSIONSTART" ), MyFrame::StartConversion )
EVT_BUTTON( XRCID( "ID_BUTTON_SAVE_STDOUT" ), MyFrame::SaveStdout )

EVT_BUTTON( XRCID( "ID_BUTTON_SELECTWORKDIR" ), MyFrame::SelectWorkdir )
EVT_BUTTON( XRCID( "ID_BUTTON_BROWSE_LATTE_DIR" ), MyFrame::SelectExecutable )

EVT_IDLE( MyFrame::OnIdle )

EVT_END_PROCESS( ELLE_TO_PS_PROC, MyFrame::ConvertElleTerminated )
EVT_END_PROCESS( PS_TO_GIF_PROC, MyFrame::MakeGifTerminated )
EVT_END_PROCESS( CONVERT_MOVIE, MyFrame::ConvertMovieTerminated )

EVT_END_PROCESS( LATTE_PROC, MyFrame::RunTerminated )

EVT_CHECKBOX( XRCID( "ID_CHECKBOX_GIF" ), MyFrame::GifChecked )
EVT_CHECKBOX( XRCID( "ID_CHECKBOX_AVI" ), MyFrame::AviChecked )
EVT_CHECKBOX( XRCID( "ID_CHECKBOX_MPEG" ), MyFrame::MpegChecked )

END_EVENT_TABLE( )

MyFrame::MyFrame( wxWindow* parent )
{

    wxXmlResource::Get() ->LoadFrame( this, parent, wxT( "Main_NoteBook" ) );
    IsRunning = false;
    SPSIsRunning = false;

    pathlist.AddEnvList( wxT( "PATH" ) );
    pathlist.Add( wxT( "../../bin/" ) );
    pathlist.Add( wxT( "../../binx/" ) );

    if ( !wxFileExists( wxGetCwd() + wxFILE_SEP_PATH + wxT( "latte_wx" ) ) )
    {
        XRCCTRL( *this, "ID_CHECKBOX_LOCAL_DIR", wxCheckBox ) -> SetValue( false );
        XRCCTRL( *this, "ID_CHECKBOX_LOCAL_DIR", wxCheckBox ) -> Enable( false );
    }

    // pfade setzen fuer:
    // showelleps, mencoder, latte (mit disablen der control und warnhinweis), imagemaker
    // showelleps_path, mencoder_path, latte_path, convert_path

    mencoder_path = pathlist.FindAbsoluteValidPath( wxT( "mencoder" ) );

    if ( mencoder_path == wxEmptyString )
    {
        XRCCTRL( *this, "ID_BUTTON_CONVERTFILE", wxButton ) ->Enable( false );
        XRCCTRL( *this, "ID_BUTTON_CONVERSIONSTART", wxButton ) ->Enable( false );
        XRCCTRL( *this, "ID_BUTTON_SAVE_STDOUT", wxButton ) ->Enable( false );
    }

    showelleps_path = pathlist.FindAbsoluteValidPath( wxT( "showelleps" ) );
    convert_path = pathlist.FindAbsoluteValidPath( wxT( "convert" ) );
    //cout << showelleps_path.ToAscii() << endl;
    if ( showelleps_path == wxEmptyString || convert_path == wxEmptyString )
    {
        XRCCTRL( *this, "ID_BUTTON_SELECTSTART", wxButton ) ->Enable( false );
        XRCCTRL( *this, "ID_BUTTON_SELECTEND", wxButton ) ->Enable( false );
        XRCCTRL( *this, "ID_BUTTON_SELECTSAVENAME", wxButton ) ->Enable( false );
        XRCCTRL( *this, "ID_BUTTON_STARTMOVIEMAKE", wxButton ) ->Enable( false );
    }

    latte_path = wxEmptyString;

    //cout << latte_path.ToAscii() << endl;

    archive = wxEmptyString;
    archive_dir = wxGetCwd();
    archive_path = wxEmptyString;

    archive_out = wxEmptyString;
    archive_out_dir = wxGetCwd();
    archive_out_path = wxEmptyString;

    ellefile = wxEmptyString;
    elle_dir = wxGetCwd();
    elle_path = wxEmptyString;

    userselectedarchive = false;

    is_gif = is_avi = is_mpeg = false;

    start_file_dir = wxGetCwd();
    end_file_dir = wxGetCwd();
    moviemake_dir = wxGetCwd();

    appdirectory = wxGetCwd();
    XRCCTRL( *this, "ID_TEXTCTRL_APPDIR", wxTextCtrl ) ->Clear();
    XRCCTRL( *this, "ID_TEXTCTRL_APPDIR", wxTextCtrl ) ->AppendText( appdirectory );

    XRCCTRL( *this, "ID_TEXTCTRL_ARCHIVEDIR", wxTextCtrl ) ->Clear();
}

// check if there's output of latte in your idle time
// and of course: give it out
void MyFrame::OnIdle( wxIdleEvent& event )
{
    wxString msg = wxEmptyString;

    if ( IsRunning && callback )
    {
        if ( callback->IsInputAvailable() )
        {
            wxTextInputStream tis( *callback->GetInputStream() );

            // this assumes that the output is always line buffered

            msg << tis.ReadLine() << wxT( "\n" );

            XRCCTRL( *this, "ID_TEXTCTRL_STDOUT", wxTextCtrl ) ->AppendText( msg );
        }
    }

    msg = wxEmptyString;

    //cout << SPSIsRunning << endl;

    if ( CIsRunning && convert_callback )
    {
        if ( convert_callback->IsInputAvailable() )
        {
            wxTextInputStream tis2( *convert_callback->GetInputStream() );

            msg << tis2.ReadLine() << wxT( "\n" );

            XRCCTRL( *this, "ID_TEXTCTRL_CONVERSIONSTDOUT", wxTextCtrl ) ->AppendText( msg );
        }
    }
}

void MyFrame::SelectWorkdir( wxCommandEvent& WXUNUSED( event ) )
{
    wxDirDialog dir( this , wxT( "Choose a directory" ), wxGetCwd() );
    int cont = dir.ShowModal();

    if ( cont == wxID_OK )
    {
        appdirectory = dir.GetPath();

        XRCCTRL( *this, "ID_TEXTCTRL_APPDIR", wxTextCtrl ) ->Clear();
        XRCCTRL( *this, "ID_TEXTCTRL_APPDIR", wxTextCtrl ) ->AppendText( appdirectory );

        XRCCTRL( *this, "ID_TEXTCTRL_WORKINGDIR", wxTextCtrl ) ->Clear();

        wxSetWorkingDirectory( appdirectory );

        if ( !wxFileExists( wxGetCwd() + wxFILE_SEP_PATH + wxT( "latte_wx" ) ) )
        {
            XRCCTRL( *this, "ID_CHECKBOX_LOCAL_DIR", wxCheckBox ) -> SetValue( false );
            XRCCTRL( *this, "ID_CHECKBOX_LOCAL_DIR", wxCheckBox ) -> Enable( false );
        }
        else
        {
            XRCCTRL( *this, "ID_CHECKBOX_LOCAL_DIR", wxCheckBox ) -> SetValue( true );
            XRCCTRL( *this, "ID_CHECKBOX_LOCAL_DIR", wxCheckBox ) -> Enable( true );
        }
    }
}

void MyFrame::SelectElleFile( wxCommandEvent& WXUNUSED( event ) )
{
    wxFileDialog dialog( this, wxT( "Choose elle-file" ), appdirectory, wxEmptyString, _T( "elle-files (*.elle)|*.elle" ), wxOPEN );

    int cont = dialog.ShowModal();

    if ( cont == wxID_OK )
    {
        elle_path = dialog.GetPath();
        ellefile = dialog.GetFilename();
        elle_dir = dialog.GetDirectory();

        XRCCTRL( *this, "ID_TEXTCTRL_WORKINGDIR", wxTextCtrl ) ->Clear();
        XRCCTRL( *this, "ID_TEXTCTRL_WORKINGDIR", wxTextCtrl ) ->AppendText( elle_path );

        archive_out_path = elle_path.BeforeLast( '.' ) + wxT( ".zip" );
        archive_out = wxFileNameFromPath( archive_out_path );
        archive_out_dir = elle_dir;

        if ( !userselectedarchive )
        {
            wxDir dir( archive_dir );
            // archive existiert schon?
            if ( dir.HasFiles( archive_out ) )
            {
                if ( CheckArchive( archive_out_path ) )
                {
                    XRCCTRL( *this, "ID_TEXTCTRL_ARCHIVEDIR", wxTextCtrl ) -> Clear();
                    XRCCTRL( *this, "ID_TEXTCTRL_ARCHIVEDIR", wxTextCtrl ) -> AppendText( archive_out_path );
                    //wxMessageDialog dialog( this, wxT( "An archive of this name already exists \n you may edit existing preferences \n or replace them " ), wxT( "Info" ), wxID_OK );
                    //dialog.ShowModal();

                    OpenArchiveDescription( archive_out_path );

                    XRCCTRL( *this, "ID_BUTTON_EDITEXISTINGFILE", wxButton ) -> Enable ( true );
                }
                else
                {
                    XRCCTRL( *this, "ID_TEXTCTRL_ARCHIVEDIR", wxTextCtrl ) -> Clear();
                    XRCCTRL( *this, "ID_TEXTCTRL_ARCHIVEDIR", wxTextCtrl ) -> AppendText( archive_path );
                    wxMessageDialog dialog( this, wxT( "An archive already exists \n but doesnt contain preferences." ), wxT( "Info" ), wxID_OK );
                    dialog.ShowModal();

                    XRCCTRL( *this, "ID_BUTTON_EDITEXISTINGFILE", wxButton ) -> Enable ( false );
                    XRCCTRL( *this, "ID_TEXTCTRL_ARCHIVEDIR", wxTextCtrl ) ->Clear();

                    XRCCTRL( *this, "ID_BUTTON_EDITEXISTINGFILE", wxButton ) -> Enable ( false );
                }
            }
            else
            {
                XRCCTRL( *this, "ID_TEXTCTRL_ARCHIVEDIR", wxTextCtrl ) ->Clear();

                //wxMessageDialog dialog( this, archive_path + wxT( " will be created in current elle-directory" ), wxT( "Create new archive" ), wxID_OK );
                //dialog.ShowModal();

                XRCCTRL( *this, "ID_BUTTON_EDITEXISTINGFILE", wxButton ) -> Enable ( false );
            }

            XRCCTRL( *this, "ID_TEXTCTRL_ARCHIVEOUT", wxTextCtrl ) ->Clear();
            XRCCTRL( *this, "ID_TEXTCTRL_ARCHIVEOUT", wxTextCtrl ) ->AppendText( archive_out_path );
        }
    }
}

void MyFrame::SelectArchive( wxCommandEvent& WXUNUSED( event ) )
{
    wxFileDialog dialog( this, wxT( "Choose zip-archive" ), archive_dir, wxEmptyString, _T( "ZIP archives (*.zip)|*.zip" ), wxOPEN );
    int cont;

    cont = dialog.ShowModal();

    if ( cont == wxID_OK )
    {
        archive_path = dialog.GetPath();

        if ( !CheckArchive( archive_path ) )
        {
            wxMessageDialog message( this, wxT( "This archive doesnt contain preferences to import!" ), wxT( "Warning" ), wxID_OK );
            message.ShowModal();
        }
        else
        {
            XRCCTRL( *this, "ID_TEXTCTRL_ARCHIVEDIR", wxTextCtrl ) ->Clear();
            XRCCTRL( *this, "ID_TEXTCTRL_ARCHIVEDIR", wxTextCtrl ) ->AppendText( archive_path );
            XRCCTRL( *this, "ID_BUTTON_EDITEXISTINGFILE", wxButton ) ->Enable( true );

            userselectedarchive = true;

            archive_path = dialog.GetPath();
            archive = dialog.GetFilename();
            archive_dir = dialog.GetDirectory();

            OpenArchiveDescription( archive_path );
        }
    }
}

MyFrame::~MyFrame()
{}

void MyFrame::BrowseZip( wxCommandEvent& WXUNUSED( event ) )
{

    wxFileDialog dialog( this, _T( "Choose a ZIP archive" ), wxGetCwd(), wxEmptyString, _T( "ZIP archives (*.zip)|*.zip" ), wxOPEN );

    if ( dialog.ShowModal() == wxID_OK )
    {
        archive = dialog.GetFilename();
        archive_dir = dialog.GetDirectory();
        archive_path = dialog.GetPath();

        if ( !CheckArchive( archive_path ) )
        {
            wxMessageDialog dialog( this, wxT( "zip-file doesn't contain pref-file to import!" ), wxT( "Error importing!" ), wxID_OK );
            dialog.ShowModal();

            archive = archive_out;
            archive_dir = archive_out_dir;
            archive_path = archive_out_path;

            XRCCTRL( *this, "ID_TEXTCTRL_DESC", wxTextCtrl ) ->Clear();
        }
        else
        {
            if ( archive_dir.Trim() == wxGetCwd().Trim() )
            {
                XRCCTRL( *this, "ID_CHOICE_ZIP_FILE", wxChoice ) ->SetStringSelection( archive );
            }
            else
            {
                XRCCTRL( *this, "ID_CHOICE_ZIP_FILE", wxChoice ) ->SetStringSelection( archive_path );
            }
        }
        XRCCTRL( *this, "ID_BUTTON_EDITEXISTINGFILE", wxButton ) ->Enable( true );
    }
}

void MyFrame::ShowCreateNewFileDialog( wxCommandEvent& WXUNUSED( event ) )
{
    bool overwrite = true;

    if ( archive_out_path.IsEmpty() )
    {
        if ( !elle_path.IsEmpty() )
            archive_out_path = elle_path;
        else
        {
            wxMessageDialog dialog( this, _T( "Choose an elle-file from this directory first!\n Otherwise an archive can't be created." ), _T( "Error" ), wxOK );
            dialog.ShowModal();
            return ;
        }
    }

    if ( wxFileExists( archive_out_path ) )
    {
        if ( CheckArchive( archive_out_path ) )
        {
            wxMessageDialog message( this, _T( "Really overwrite settings\n in an existing archive?" ), _T( "An archive already exists!" ), wxOK | wxCANCEL );
            int nb = message.ShowModal();

            if ( nb == wxID_OK )
                overwrite = true;
            else
                overwrite = false;
        }
    }

    if ( overwrite )
    {
        newdialog = new NewFile( this , archive_out_path );
        newdialog->ShowModal();

        if ( wxFileExists( archive_out_path ) )
        {
            if ( CheckArchive( archive_out_path ) )
            {
                XRCCTRL( *this, "ID_BUTTON_EDITEXISTINGFILE", wxButton ) ->Enable( true );
            }
        }
    }
}

void MyFrame::ShowEditExistingDialog( wxCommandEvent& WXUNUSED( event ) )
{

    if ( archive == wxEmptyString && archive_out != wxEmptyString )
    {
        archive = archive_out;
        archive_dir = archive_out_dir;
        archive_path = archive_out_path;
    }

    if ( archive != wxEmptyString )
    {
        editdialog = new EditFile( this , archive_path, archive_out_path );

        editdialog->ShowModal();
    }
    else
    {
        wxMessageDialog dialog( this, _T( "A pref-file doesn't yet exist!\nYou'll have to create one." ), _T( "Error!" ), wxOK );
        dialog.ShowModal();
        return ;
    }
}

void MyFrame::ChooseElleFile( wxCommandEvent& WXUNUSED( event ) )
{
    if ( XRCCTRL( *this, "ID_CHOICE_ELLE_FILE", wxChoice ) ->GetSelection() != 0 )
    {
        ellefile = XRCCTRL( *this, "ID_CHOICE_ELLE_FILE", wxChoice ) ->GetStringSelection();
        elle_dir = wxGetCwd();
        elle_path = elle_dir + wxFILE_SEP_PATH + ellefile;

        archive_out = ellefile.BeforeLast( '.' ) + wxT( ".zip" );
        archive_out_dir = elle_dir;
        archive_out_path = archive_out_dir + wxFILE_SEP_PATH + archive_out;

        if ( archive_path == wxEmptyString )
        {
            archive_path = archive_out_path;
            archive_dir = archive_out_dir;
            archive = archive_out;

            OpenArchiveDescription( archive_out_path );
            XRCCTRL( *this, "ID_CHOICE_ZIP_FILE", wxChoice ) ->SetStringSelection( archive );
        }

        if ( archive == wxT( "None selected" ) )
            archive == wxEmptyString;

        if ( archive == wxEmptyString )
        {
            archive = ellefile.BeforeLast( '.' ) + wxT( ".zip" );
            archive_dir = elle_dir;
            archive_path = archive_out_dir + wxFILE_SEP_PATH + archive_out;
        }
    }
    else
    {
        ellefile = wxEmptyString;
        elle_dir = wxGetCwd();
        elle_path = wxEmptyString;

        if ( XRCCTRL( *this, "ID_CHOICE_ZIP_FILE", wxChoice ) ->GetSelection() == 0 )
        {
            ellefile = wxEmptyString;
            elle_dir = wxGetCwd();
            elle_path = wxEmptyString;

            archive = wxEmptyString;
            archive_path = wxEmptyString;
            archive_dir = elle_dir;

            archive_out = wxEmptyString;
            archive_out_path = wxEmptyString;
            archive_out_dir = elle_dir;

            XRCCTRL( *this, "ID_TEXTCTRL_DESC", wxTextCtrl ) ->Clear();

            XRCCTRL( *this, "ID_TEXT_OUT_ARCHIVE", wxStaticText ) ->SetLabel( wxT( "none" ) );

        }
    }

    XRCCTRL( *this, "ID_TEXT_OUT_ARCHIVE", wxStaticText ) ->SetLabel( archive_out );

    CheckArchive( archive_out_path );
}

void MyFrame::ChooseZipFile( wxCommandEvent& WXUNUSED( event ) )
{}

// check, whether the in_archive contains a latte_pref file
bool MyFrame::CheckArchive( wxString test_archive_path )
{

    if ( test_archive_path.AfterLast( '.' ) == wxT( "zip" ) )
    {

        wxZipEntry * entry;

        wxFFileInputStream in( test_archive_path );
        wxZipInputStream zip( in );

        while ( entry = zip.GetNextEntry(), entry != NULL )
        {
            if ( entry->GetName() == wxT( "latte_pref" ) )
            {
                if ( zip.OpenEntry( *entry ) )
                {
                    return true;
                }
            }
        }
    }


    XRCCTRL( *this, "ID_TEXTCTRL_DESC", wxTextCtrl ) ->Clear();
    return false;
}

// does dir contain archives at all?
// CAUTION, this is still a DUMMY function that has to be adapted
bool MyFrame::CheckArchiveDir()
{
    //cout << archive_path.ToAscii() << " ende" << endl;

    wxDir dir( archive_dir );

    if ( !dir.IsOpened() )
    {
        return ( false );
    }

    wxString filename;

    bool cont = dir.GetFirst( &filename );
    while ( cont )
    {
        if ( filename.AfterLast( '.' ) == wxT( "zip" ) )
        {

            wxZipEntry * entry;

            //cout << filename.ToAscii() << endl;

            wxFFileInputStream in( filename );
            wxZipInputStream zip( in );

            while ( entry = zip.GetNextEntry(), entry != NULL )
            {
                if ( entry->GetName() == wxT( "latte_pref" ) )
                {
                    if ( zip.OpenEntry( *entry ) )
                    {
                        //OpenArchiveDescription();
                        return true;
                    }
                }
            }
        }
        cont = dir.GetNext( &filename );
    }

    XRCCTRL( *this, "ID_TEXTCTRL_DESC", wxTextCtrl ) ->Clear();
    return false;
}

void MyFrame::OpenArchiveDescription( wxString temp )
{
    wxZipEntry * entry;

    wxFFileInputStream in( temp );
    wxZipInputStream zip( in );
    wxTextInputStream text( zip );

    XRCCTRL( *this, "ID_TEXTCTRL_DESC", wxTextCtrl ) ->Clear();

    while ( entry = zip.GetNextEntry(), entry != NULL )
    {
        if ( entry->GetName() == wxT( "latte_desc" ) )
        {
            if ( zip.OpenEntry( *entry ) )
            {
                while ( !zip.Eof() )
                    XRCCTRL( *this, "ID_TEXTCTRL_DESC", wxTextCtrl ) ->AppendText( text.ReadLine() + wxT( "\n" ) );
            }
        }
    }
}

void MyFrame::StartRun( wxCommandEvent& WXUNUSED( event ) )
{

    if ( !XRCCTRL( *this, "ID_CHECKBOX_LOCAL_DIR", wxCheckBox ) ->IsChecked() )
    {
        if ( !wxFileExists( latte_path ) )
        {
            //cout << latte_path.ToAscii() << endl;
            wxMessageDialog dialog( this, wxT( "No global latte-executable found! \nAdd it in the 'misc-settings' dialoge." ), wxT( "Error" ), wxID_OK );
            dialog.ShowModal();
            return ;
        }
    }
    else
    {
        if ( !wxFileExists( wxT( "elle_latte" ) ) )
        {
            wxMessageDialog dialog( this, wxT( "No local latte-executable found!\nIf it exists in this directory, \nplease rename it to 'elle_latte'" ), wxT( "Error" ), wxID_OK );
            dialog.ShowModal();
            return ;
        }
    }

    if ( !IsRunning )
    {
        wxString cmd, s;

        if ( !ellefile )
        {
            wxMessageDialog dialog( this, wxT( "No *.elle file!" ), wxT( "You must open an elle data-file first!" ) );
            dialog.ShowModal();
            return ;
        }

        //check, if it should run in the local dir, and additionally if it should be running in a shell
        if ( XRCCTRL( *this, "ID_CHECKBOX_LOCAL_DIR", wxCheckBox ) ->IsChecked() )
        {
            cmd = _T( "./elle_latte" );
        }
        else
        {
            cmd = latte_path;
        }

        cmd.Append( _T( " -u 0 -i " ) );
        cmd.Append( ellefile );
        cmd.Append( _T( " -f " ) );
        s.Printf( _T( "%d" ), int( XRCCTRL( *this, "ID_SPINCTRL_SAVE_INTERVAL", wxSpinCtrl ) ->GetValue() ) );
        cmd.Append( s );

        code = 0;

        callback = new wxProcess( this, LATTE_PROC );
        callback->Redirect();

        //cout << appdirectory.ToAscii() << endl;
        wxSetWorkingDirectory( appdirectory );

        code = ( wxExecute ( cmd, wxEXEC_ASYNC, callback ) );
        IsRunning = ( code != 0 );

        if ( IsRunning )
        {
            XRCCTRL( *this, "ID_BUTTON_BROWSEWORKINGDIR", wxButton ) ->Enable( false );
            XRCCTRL( *this, "ID_BUTTON_BROWSEARCHIVEDIR", wxButton ) ->Enable( false );
            XRCCTRL( *this, "ID_BUTTON_CREATENEWFILE", wxButton ) ->Enable( false );
            XRCCTRL( *this, "ID_BUTTON_EDITEXISTINGFILE", wxButton ) ->Enable( false );
            XRCCTRL( *this, "ID_BUTTON_START", wxButton ) ->SetLabel( _T( "Kill run" ) );
            XRCCTRL( *this, "ID_TEXTCTRL_STDOUT", wxTextCtrl ) ->Clear();
            XRCCTRL( *this, "ID_BUTTON_SELECTWORKDIR", wxButton ) ->Enable( false );
        }
    }
    else
    {
        if ( callback->Kill( code, wxSIGTERM ) == wxKILL_OK )
        {
            XRCCTRL( *this, "ID_BUTTON_BROWSEWORKINGDIR", wxButton ) ->Enable( true );
            XRCCTRL( *this, "ID_BUTTON_BROWSEARCHIVEDIR", wxButton ) ->Enable( true );
            XRCCTRL( *this, "ID_BUTTON_CREATENEWFILE", wxButton ) ->Enable( true );
            XRCCTRL( *this, "ID_BUTTON_EDITEXISTINGFILE", wxButton ) ->Enable( true );
            XRCCTRL( *this, "ID_BUTTON_START", wxButton ) ->SetLabel( _T( "Start run" ) );
            XRCCTRL( *this, "ID_BUTTON_SELECTWORKDIR", wxButton ) ->Enable( true );
            IsRunning = false;
        }
    }
}

void MyFrame::RunTerminated( wxProcessEvent& WXUNUSED( event ) )
{
    XRCCTRL( *this, "ID_BUTTON_BROWSEWORKINGDIR", wxButton ) ->Enable( true );
    XRCCTRL( *this, "ID_BUTTON_BROWSEARCHIVEDIR", wxButton ) ->Enable( true );
    XRCCTRL( *this, "ID_BUTTON_CREATENEWFILE", wxButton ) ->Enable( true );

    XRCCTRL( *this, "ID_BUTTON_START", wxButton ) ->SetLabel( _T( "Start run" ) );
    XRCCTRL( *this, "ID_BUTTON_SELECTWORKDIR", wxButton ) ->Enable( true );
    IsRunning = false;
}

void MyFrame::OnCancel( wxCommandEvent& WXUNUSED( event ) )
{
    wxKill( code );
    Destroy();
}

void MyFrame::OnOk( wxCommandEvent& WXUNUSED( event ) )
{
    if ( archive_out_path != wxEmptyString )
        SaveDesc();

    wxKill( code );
    Destroy();
}

void MyFrame::SaveDesc()
{
    wxString archive_tmp;

    if ( archive_out_path != wxEmptyString )
    {
        wxZipEntry * entry;
        wxFFileOutputStream out( _T( "tmp.zip" ) );
        wxZipOutputStream zip_out( out );
        wxTextOutputStream txt_out( zip_out );

        // copy old archive
        if ( archive_path == wxEmptyString )
            archive_tmp = archive_out_path;
        else
            archive_tmp = archive_path;

        if ( archive_tmp.AfterLast( '.' ) == wxT( "zip" ) && wxFileExists( archive_tmp ) )
        {
            wxFFileInputStream in( archive_tmp );
            wxZipInputStream zip_in( in );

            zip_out.PutNextEntry( _T( "latte_desc" ) );
            txt_out << XRCCTRL( *this, "ID_TEXTCTRL_DESC", wxTextCtrl ) ->GetValue();

            while ( entry = zip_in.GetNextEntry(), entry != NULL )
            {
                if ( entry->GetName() != wxT( "latte_desc" ) )
                {
                    zip_out.CopyEntry( entry, zip_in );
                }
            }
        }
        else
        {
            zip_out.PutNextEntry( _T( "latte_desc" ) );
            txt_out << XRCCTRL( *this, "ID_TEXTCTRL_DESC", wxTextCtrl ) ->GetValue();
        }

        zip_out.CloseEntry();
        zip_out.Close();
        out.Close();

        //cout << archive_out_path.ToAscii() << endl;

        wxRenameFile( wxGetCwd() + wxFILE_SEP_PATH + wxT( "tmp.zip" ), archive_out_path );
    }
}

void MyFrame::SaveStdout ( wxCommandEvent& WXUNUSED( event ) )
{
    wxFileDialog dialog( this, wxT( "Choose elle-file" ), elle_dir, wxEmptyString, _T( "elle-files (*.elle)|*.elle" ), wxSAVE );

    dialog.ShowModal();

    ofstream fout( ( dialog.GetPath() + wxT( ".sto" ) ).ToAscii() );

    fout << XRCCTRL( *this, "ID_TEXTCTRL_STDOUT", wxTextCtrl ) ->GetValue().ToAscii();

    fout.close();
}

void MyFrame::SelectExecutable( wxCommandEvent& WXUNUSED( event ) )
{
    wxDirDialog dialog( this, wxT( "Choose directory of latte-executable" ), wxGetCwd() );
    dialog.ShowModal();
#ifdef __WIN32__ //JKBecker
    latte_path = ( dialog.GetPath() + wxFILE_SEP_PATH + wxT( "elle_latte.exe" ) );
	#else
    latte_path = ( dialog.GetPath() + wxFILE_SEP_PATH + wxT( "elle_latte" ) );
#endif
    if ( wxFileExists( latte_path ) )
    {
        XRCCTRL( *this, "ID_TEXTCTRL_LATTEEX", wxTextCtrl ) -> Clear();
        XRCCTRL( *this, "ID_TEXTCTRL_LATTEEX", wxTextCtrl ) -> AppendText( latte_path.BeforeLast( wxFILE_SEP_PATH ) );
    }
    else
        latte_path = wxEmptyString;
}

/******************   The movie make routines   *********************/
/**************   Should have an own modules one day   *****************/

// this will call the appropriate steps in appropriate time, nothing else
void MyFrame::MakeMovie( wxCommandEvent& WXUNUSED( event ) )
{

    wxString start_file, end_file, save_file;
    wxString k, buffer, startnumber = wxEmptyString, endnumber = wxEmptyString, save_movie_file; //buffer

    if ( start_file_dir != end_file_dir )
    {
        wxMessageDialog dialog( this, wxT( " Start- and end-files for the \nmovie are in different directories " ), wxT( "Error" ), wxID_OK );
        dialog.ShowModal();

        return ;
    }

    if ( save_file_dir != end_file_dir )
    {
        wxMessageDialog dialog( this, wxT( " Save-path has to be in the \nsame directory as elle-files! " ), wxT( "Error" ), wxID_OK );
        dialog.ShowModal();

        return ;
    }

    // if abfragen (alles im lot mit den user-eingaben?)

    // setzen des elle_counter, der elle_stepsize (beides global definiert)
    // definieren des elle_filename (==rumpf) und addition der start-nr und .elle-endung
    elle_stepsize = XRCCTRL( *this, "ID_SPINCTRL_MOVIEINTERVAL", wxSpinCtrl ) -> GetValue();

    start_file = wxFileNameFromPath ( XRCCTRL( *this, "ID_TEXTCTRL_SELECTSTART", wxTextCtrl ) -> GetValue() );
    end_file = wxFileNameFromPath ( XRCCTRL( *this, "ID_TEXTCTRL_SELECTEND", wxTextCtrl ) -> GetValue() );
    save_movie_file = wxFileNameFromPath ( XRCCTRL( *this, "ID_TEXTCTRL_SELECTSAVENAME", wxTextCtrl ) -> GetValue() );

    if ( end_file == start_file )
    {
        wxMessageDialog dialog( this, wxT( " Start- and end-files for the \nmovie are identical " ), wxT( "Error" ), wxID_OK );
        dialog.ShowModal();

        return ;
    }

    if ( save_movie_file.Trim( 1 ).Trim( 0 ) == wxEmptyString )
    {
        wxMessageDialog dialog( this, wxT( " Set a filename for the movie! " ), wxT( "Error" ), wxID_OK );
        dialog.ShowModal();

        return ;
    }

    if ( start_file.Trim( 1 ).Trim( 0 ).IsEmpty() )
    {
        wxMessageDialog dialog( this, wxT( " Set a start-file! " ), wxT( "Error" ), wxID_OK );
        dialog.ShowModal();

        return ;
    }

    start_file = start_file.BeforeLast( '.' );

    buffer = start_file.Last();
    while ( buffer.IsNumber() )
    {
        startnumber.Prepend( buffer );
        start_file.RemoveLast();
        buffer = start_file.Last();
    }

    startnumber.ToLong( &elle_counter );

    if ( end_file.Trim( 1 ).Trim( 0 ).IsEmpty() )
    {
        wxMessageDialog dialog( this, wxT( " Set a end-file! " ), wxT( "Error" ), wxID_OK );
        dialog.ShowModal();

        return ;
    }

    end_file = end_file.BeforeLast( '.' );

    buffer = end_file.Last();
    while ( buffer.IsNumber() )
    {
        endnumber.Prepend( buffer );
        end_file.RemoveLast();
        buffer = end_file.Last();
    }

    if ( end_file != start_file )
    {
        wxMessageDialog dialog( this, wxT( " Start- and end-file are \nof different series! " ), wxT( "Error" ), wxID_OK );
        dialog.ShowModal();

        return ;
    }

    gif_save_file = save_file_dir + wxFILE_SEP_PATH + save_movie_file;

    if ( gif_save_file.Trim().AfterLast( '.' ) != wxT( "gif" ) )
    {
        gif_save_file += wxT( ".gif" );
        save_movie_file += wxT( ".gif" );

        XRCCTRL( *this , "ID_TEXTCTRL_SELECTSAVENAME" , wxTextCtrl ) -> Clear();
        XRCCTRL( *this , "ID_TEXTCTRL_SELECTSAVENAME" , wxTextCtrl ) -> AppendText( save_movie_file );
    }

    // überprüfen, ob alle files verfügbar sind, evt stepsize anpassen, evt erneut überprüfen

    wxDir dir( start_file_dir );
    //cout << start_file_dir.ToAscii() << endl;
    long i, j, m;
    wxVariant word;

    startnumber.ToLong( &i );
    endnumber.ToLong( &j );
    end_file_nb = j;
    start_file_nb = i;

    word = i + 1;
    m = 1;

    wxString nb;

    nb = word.MakeString();

    if ( elle_counter < 10 )
        nb.Prepend( wxT( "00" ) );
    else if ( elle_counter < 100 )
        nb.Prepend( wxT( "0" ) );

    while ( !dir.HasFiles( start_file + nb + wxT( ".elle" ) ) )
    {
        m++;
        word = i + m;

        nb = word.MakeString();

        if ( ( i + m ) < 10 )
            nb.Prepend( wxT( "00" ) );
        else if ( ( i + m ) < 100 )
            nb.Prepend( wxT( "0" ) );

    }

    elle_stepsize *= m;

    elle_filename = start_file;
    XRCCTRL( *this, "ID_TEXTCTRL_MAKEMOVIE", wxTextCtrl ) ->Clear();
    ConvertElle();
}

// converts the elle-files to ps. will depend heavily on the onidle-event
void MyFrame::ConvertElle()
{
    wxVariant word = elle_counter;
    wxString nb;

    nb = word.MakeString();

    if ( elle_counter < 10 )
        nb.Prepend( wxT( "00" ) );
    else if ( elle_counter < 100 )
        nb.Prepend( wxT( "0" ) );

    wxSetWorkingDirectory( moviemake_dir );

    wxString cmd = showelleps_path;
    cmd.Append( wxT( " -i " ) );
    cmd += elle_filename + nb + wxT( ".elle" );

    movie_callback = new wxProcess( this, ELLE_TO_PS_PROC );
    movie_callback->Redirect();

    movie_code = ( wxExecute ( cmd, wxEXEC_ASYNC, movie_callback ) );
    //ShowellePS = SPS!
    SPSIsRunning = ( movie_code != 0 );
}

// called by onterminate-event, restarts showelleps
void MyFrame::ConvertElleTerminated( wxProcessEvent& WXUNUSED( event ) )
{
    SPSIsRunning = false;
    wxVariant word = elle_counter;
    wxString nb;

    wxString cmd = wxEmptyString;

    nb = word.MakeString();

    if ( elle_counter < 10 )
        nb.Prepend( wxT( "00" ) );
    else if ( elle_counter < 100 )
        nb.Prepend( wxT( "0" ) );

    cmd += elle_filename + nb + wxT( ".elle to postscript\n" ) ;
    cmd.Prepend( wxT( "Converted " ) );
    XRCCTRL( *this, "ID_TEXTCTRL_MAKEMOVIE", wxTextCtrl ) ->AppendText( cmd );

    if ( elle_counter < end_file_nb )
    {
        elle_counter += elle_stepsize;
        ConvertElle();
    }
    else
        MakeGif();
}

// calls image-magick
void MyFrame::MakeGif()
{
    wxString cmd;
    wxVariant word;

    movie_callback = new wxProcess( this, PS_TO_GIF_PROC );
    movie_callback->Redirect();

    long loop = ( ( end_file_nb - start_file_nb ) / elle_stepsize ) + 1;
    word = loop;

    cmd = wxT( "convert -loop " ) + word.MakeString() + wxT( " *.ps " ) + gif_save_file;

    movie_code = ( wxExecute ( cmd, wxEXEC_ASYNC, movie_callback ) );
    //ShowellePS = SPS!
    SPSIsRunning = ( movie_code != 0 );
}

void MyFrame::MakeGifTerminated( wxProcessEvent& WXUNUSED( event ) )
{
    long i;
    wxVariant word;
    wxString nb;

    SPSIsRunning = false;
    XRCCTRL( *this, "ID_TEXTCTRL_MAKEMOVIE", wxTextCtrl ) ->AppendText( wxT( "\nConversion to gif finished!\n" ) );
    XRCCTRL( *this, "ID_TEXTCTRL_MAKEMOVIE", wxTextCtrl ) ->AppendText( wxT( "File saved as " ) + gif_save_file );

    if ( XRCCTRL( *this, "ID_CHECKBOX_DELETEPS", wxCheckBox ) ->IsChecked() )
    {
        for ( i = start_file_nb; i <= end_file_nb; i += elle_stepsize )
        {
            word = i;
            nb = word.MakeString();

            if ( i < 10 )
                nb.Prepend( wxT( "00" ) );
            else if ( i < 100 )
                nb.Prepend( wxT( "0" ) );

            wxRemoveFile( elle_filename + nb + wxT( ".ps" ) );
        }

        XRCCTRL( *this, "ID_TEXTCTRL_MAKEMOVIE", wxTextCtrl ) ->AppendText( wxT( "\n\nRemoved post-script files\n" ) );
    }
}

void MyFrame::ChooseMovieStartFile( wxCommandEvent& WXUNUSED( event ) )
{
    wxFileDialog dialog( this, wxT( "Choose elle-file" ), moviemake_dir, wxEmptyString, _T( "elle-files (*.elle)|*.elle" ), wxOPEN );

    int cont = dialog.ShowModal();

    if ( cont == wxID_OK )
    {
        if ( !( dialog.GetDirectory() != end_file_dir && end_file_dir != wxEmptyString ) )
        {
            XRCCTRL( *this , "ID_TEXTCTRL_SELECTSTART" , wxTextCtrl ) -> Clear();
            XRCCTRL( *this , "ID_TEXTCTRL_SELECTSTART" , wxTextCtrl ) -> AppendText( dialog.GetPath() );

            start_file_dir = dialog.GetDirectory();
            moviemake_dir = start_file_dir;
        }
        else
        {
            wxMessageDialog dialog( this, wxT( "Start- and end-files must be in the same directories!" ), wxT( "Sorry" ), wxID_OK );
            dialog.ShowModal();
        }
    }
}

void MyFrame::ChooseMovieEndFile( wxCommandEvent& WXUNUSED( event ) )
{

    wxFileDialog dialog( this, wxT( "Choose elle-file" ), moviemake_dir, wxEmptyString, _T( "elle-files (*.elle)|*.elle" ), wxOPEN );

    int cont = dialog.ShowModal();

    if ( cont == wxID_OK )
    {
        if ( !( dialog.GetDirectory() != start_file_dir && start_file_dir != wxEmptyString ) )
        {
            XRCCTRL( *this , "ID_TEXTCTRL_SELECTEND" , wxTextCtrl ) -> Clear();
            XRCCTRL( *this , "ID_TEXTCTRL_SELECTEND" , wxTextCtrl ) -> AppendText( dialog.GetPath() );

            end_file_dir = dialog.GetDirectory();
            moviemake_dir = end_file_dir;
        }
        else
        {
            wxMessageDialog dialog( this, wxT( "Start- and end-files must be in the same directories!" ), wxT( "Sorry" ), wxID_OK );
            dialog.ShowModal();
        }
    }
}

void MyFrame::ChooseSaveFileName( wxCommandEvent& WXUNUSED( event ) )
{
    wxFileDialog dialog( this, wxT( "Choose filename" ), moviemake_dir, wxEmptyString, _T( "gif-files (*.gif)|*.gif" ), wxSAVE );

    int cont = dialog.ShowModal();

    if ( cont == wxID_OK )
    {
        XRCCTRL( *this , "ID_TEXTCTRL_SELECTSAVENAME" , wxTextCtrl ) -> Clear();
        XRCCTRL( *this , "ID_TEXTCTRL_SELECTSAVENAME" , wxTextCtrl ) -> AppendText( dialog.GetPath() );

        save_file_dir = dialog.GetDirectory();
    }
}

/******************* Conversion routines **************************/
/****************** depends on mencoder!  *************************/

void MyFrame::ChooseConversionFile( wxCommandEvent& WXUNUSED( event ) )
{
    wxFileDialog dialog( this, wxT( "Choose movie-file" ), elle_dir, wxEmptyString, _T( "movie-files (*.gif;*.avi; *.mpg)|*.gif;*.avi;*.mpg" ), wxOPEN );

    int cont = dialog.ShowModal();

    is_gif = is_avi = is_mpeg = false;

    if ( cont == wxID_OK )
    {

        convert_file = dialog.GetPath();

        if ( convert_file != wxEmptyString )
        {
            XRCCTRL ( *this, "ID_TEXTCTRL_CONVERTFILE", wxTextCtrl ) -> Clear();
            XRCCTRL ( *this, "ID_TEXTCTRL_CONVERTFILE", wxTextCtrl ) -> AppendText( convert_file );
        }
        else
            return ;

        if ( convert_file.AfterLast( '.' ).Trim() == wxT( "gif" ) )
        {
            XRCCTRL( *this, "ID_CHECKBOX_GIF", wxCheckBox ) -> SetValue( false );
            XRCCTRL( *this, "ID_CHECKBOX_GIF", wxCheckBox ) -> Enable( false );

            XRCCTRL( *this, "ID_CHECKBOX_AVI", wxCheckBox ) -> SetValue( false );
            XRCCTRL( *this, "ID_CHECKBOX_AVI", wxCheckBox ) -> Enable( true );

            XRCCTRL( *this, "ID_CHECKBOX_MPEG", wxCheckBox ) -> SetValue( false );
            XRCCTRL( *this, "ID_CHECKBOX_MPEG", wxCheckBox ) -> Enable( true );

            is_gif = true;
        }
        else if ( convert_file.AfterLast( '.' ).Trim() == wxT( "avi" ) )
        {
            XRCCTRL( *this, "ID_CHECKBOX_GIF", wxCheckBox ) -> SetValue( false );
            XRCCTRL( *this, "ID_CHECKBOX_GIF", wxCheckBox ) -> Enable( true );

            XRCCTRL( *this, "ID_CHECKBOX_AVI", wxCheckBox ) -> SetValue( false );
            XRCCTRL( *this, "ID_CHECKBOX_AVI", wxCheckBox ) -> Enable( false );

            XRCCTRL( *this, "ID_CHECKBOX_MPEG", wxCheckBox ) -> SetValue( false );
            XRCCTRL( *this, "ID_CHECKBOX_MPEG", wxCheckBox ) -> Enable( true );

            is_avi = true;
        }
        else if ( convert_file.AfterLast( '.' ).Trim() == wxT( "mpg" ) )
        {
            XRCCTRL( *this, "ID_CHECKBOX_GIF", wxCheckBox ) -> SetValue( false );
            XRCCTRL( *this, "ID_CHECKBOX_GIF", wxCheckBox ) -> Enable( true );

            XRCCTRL( *this, "ID_CHECKBOX_AVI", wxCheckBox ) -> SetValue( false );
            XRCCTRL( *this, "ID_CHECKBOX_AVI", wxCheckBox ) -> Enable( true );

            XRCCTRL( *this, "ID_CHECKBOX_MPEG", wxCheckBox ) -> SetValue( false );
            XRCCTRL( *this, "ID_CHECKBOX_MPEG", wxCheckBox ) -> Enable( false );

            is_mpeg = true;
        }
        else
        {
            XRCCTRL( *this, "ID_CHECKBOX_GIF", wxCheckBox ) -> SetValue( false );
            XRCCTRL( *this, "ID_CHECKBOX_GIF", wxCheckBox ) -> Enable( false );

            XRCCTRL( *this, "ID_CHECKBOX_AVI", wxCheckBox ) -> SetValue( false );
            XRCCTRL( *this, "ID_CHECKBOX_AVI", wxCheckBox ) -> Enable( false );

            XRCCTRL( *this, "ID_CHECKBOX_MPEG", wxCheckBox ) -> SetValue( false );
            XRCCTRL( *this, "ID_CHECKBOX_MPEG", wxCheckBox ) -> Enable( false );
        }
    }
}

void MyFrame::StartConversion( wxCommandEvent& WXUNUSED( event ) )
{
    wxString cmd = wxT( "mencoder " ), file2, file1;
    bool to_avi = XRCCTRL( *this, "ID_CHECKBOX_AVI", wxCheckBox ) -> IsChecked();
    bool to_gif = XRCCTRL( *this, "ID_CHECKBOX_GIF", wxCheckBox ) -> IsChecked();
    bool to_mpeg = XRCCTRL( *this, "ID_CHECKBOX_MPEG", wxCheckBox ) -> IsChecked();

    XRCCTRL( *this, "ID_TEXTCTRL_CONVERSIONSTDOUT", wxTextCtrl ) ->Clear();

    if ( to_gif || to_avi || to_mpeg )
    {
        if ( is_mpeg || is_avi || is_gif )
        {
            file1 = XRCCTRL ( *this, "ID_TEXTCTRL_CONVERTFILE", wxTextCtrl ) -> GetValue().Trim();
            file2 = file1.BeforeLast( '.' );

            if ( to_avi )
            {
                file2.Append( wxT( ".avi " ) );
                // /usr/bin/mencoder file1 -o file2 -ovc lavc
            }
            else if ( to_mpeg )
            {
                file2.Append( wxT( ".mpg" ) );
            }
            else if ( to_gif )
            {
                file2.Append( wxT( ".gif" ) );
            }

            cmd.Append( file1 + wxT( " -o " ) + file2 + wxT( " -ovc lavc" ) );

            //cout << cmd.ToAscii() << endl;

            convert_callback = new wxProcess( this, CONVERT_MOVIE );
            convert_callback->Redirect();

            convert_code = ( wxExecute ( cmd, wxEXEC_ASYNC, convert_callback ) );
            CIsRunning = ( convert_code != 0 );

            XRCCTRL( *this, "ID_BUTTON_CONVERSIONSTART", wxButton ) ->Enable( false );
            XRCCTRL( *this, "ID_BUTTON_CONVERTFILE", wxButton ) ->Enable( false );
            XRCCTRL( *this, "ID_CHECKBOX_GIF", wxCheckBox ) ->Enable( false );
            XRCCTRL( *this, "ID_CHECKBOX_AVI", wxCheckBox ) ->Enable( false );
            XRCCTRL( *this, "ID_CHECKBOX_MPEG", wxCheckBox ) ->Enable( false );
        }
        else
        {
            wxMessageDialog dialog( this, wxT( "You didn't choose a file to convert, \nor it's of the wrong format.\nNote that movie-files are recognized by the file-ending! " ), wxT( "Problem" ), wxID_OK );
            dialog.ShowModal();

            return ;
        }
    }
    else
    {
        wxMessageDialog dialog( this, wxT( " You didn't choose a file-type \n to convert to. " ), wxT( "Problem" ), wxID_OK );
        dialog.ShowModal();

        return ;
    }
}

void MyFrame::ConvertMovieTerminated( wxProcessEvent& WXUNUSED( event ) )
{
    CIsRunning = false;

    XRCCTRL( *this, "ID_BUTTON_CONVERSIONSTART", wxButton ) ->Enable( true );
    XRCCTRL( *this, "ID_BUTTON_CONVERTFILE", wxButton ) ->Enable( true );
    XRCCTRL( *this, "ID_CHECKBOX_GIF", wxCheckBox ) ->Enable( true );
    XRCCTRL( *this, "ID_CHECKBOX_AVI", wxCheckBox ) ->Enable( true );
    XRCCTRL( *this, "ID_CHECKBOX_MPEG", wxCheckBox ) ->Enable( true );

    XRCCTRL( *this, "ID_TEXTCTRL_CONVERSIONSTDOUT", wxTextCtrl ) ->AppendText( wxT( "\n\n Finished! \n" ) );
}

void MyFrame::GifChecked( wxCommandEvent& WXUNUSED( event ) )
{

    if ( XRCCTRL( *this, "ID_CHECKBOX_GIF", wxCheckBox ) ->IsChecked() )
    {
        XRCCTRL( *this, "ID_CHECKBOX_AVI", wxCheckBox ) ->Enable( false );
        XRCCTRL( *this, "ID_CHECKBOX_MPEG", wxCheckBox ) ->Enable( false );
    }
    else
    {
        if ( XRCCTRL ( *this, "ID_TEXTCTRL_CONVERTFILE", wxTextCtrl ) -> GetValue().Trim().AfterLast( '.' ) == wxT( "mpg" ) )
            XRCCTRL( *this, "ID_CHECKBOX_AVI", wxCheckBox ) ->Enable( true );
        else if ( XRCCTRL ( *this, "ID_TEXTCTRL_CONVERTFILE", wxTextCtrl ) -> GetValue().Trim().AfterLast( '.' ) == wxT( "avi" ) )
            XRCCTRL( *this, "ID_CHECKBOX_MPEG", wxCheckBox ) ->Enable( true );
        else
        {
            XRCCTRL( *this, "ID_CHECKBOX_AVI", wxCheckBox ) ->Enable( true );
            XRCCTRL( *this, "ID_CHECKBOX_MPEG", wxCheckBox ) ->Enable( true );
        }
    }
}

void MyFrame::AviChecked( wxCommandEvent& WXUNUSED( event ) )
{
    if ( XRCCTRL( *this, "ID_CHECKBOX_AVI", wxCheckBox ) ->IsChecked() )
    {
        XRCCTRL( *this, "ID_CHECKBOX_GIF", wxCheckBox ) ->Enable( false );
        XRCCTRL( *this, "ID_CHECKBOX_MPEG", wxCheckBox ) ->Enable( false );
    }
    else
    {
        if ( XRCCTRL ( *this, "ID_TEXTCTRL_CONVERTFILE", wxTextCtrl ) -> GetValue().Trim().AfterLast( '.' ) == wxT( "mpg" ) )
            XRCCTRL( *this, "ID_CHECKBOX_GIF", wxCheckBox ) ->Enable( true );
        else if ( XRCCTRL ( *this, "ID_TEXTCTRL_CONVERTFILE", wxTextCtrl ) -> GetValue().Trim().AfterLast( '.' ) == wxT( "gif" ) )
            XRCCTRL( *this, "ID_CHECKBOX_MPEG", wxCheckBox ) ->Enable( true );
        else
        {
            XRCCTRL( *this, "ID_CHECKBOX_GIF", wxCheckBox ) ->Enable( true );
            XRCCTRL( *this, "ID_CHECKBOX_MPEG", wxCheckBox ) ->Enable( true );
        }
    }
}

void MyFrame::MpegChecked( wxCommandEvent& WXUNUSED( event ) )
{
    if ( XRCCTRL( *this, "ID_CHECKBOX_MPEG", wxCheckBox ) ->IsChecked() )
    {
        XRCCTRL( *this, "ID_CHECKBOX_GIF", wxCheckBox ) ->Enable( false );
        XRCCTRL( *this, "ID_CHECKBOX_AVI", wxCheckBox ) ->Enable( false );
    }
    else
    {
        if ( XRCCTRL ( *this, "ID_TEXTCTRL_CONVERTFILE", wxTextCtrl ) -> GetValue().Trim().AfterLast( '.' ) == wxT( "gif" ) )
            XRCCTRL( *this, "ID_CHECKBOX_AVI", wxCheckBox ) ->Enable( true );
        else if ( XRCCTRL ( *this, "ID_TEXTCTRL_CONVERTFILE", wxTextCtrl ) -> GetValue().Trim().AfterLast( '.' ) == wxT( "avi" ) )
            XRCCTRL( *this, "ID_CHECKBOX_GIF", wxCheckBox ) ->Enable( true );
        else
        {
            XRCCTRL( *this, "ID_CHECKBOX_AVI", wxCheckBox ) ->Enable( true );
            XRCCTRL( *this, "ID_CHECKBOX_GIF", wxCheckBox ) ->Enable( true );
        }
    }
}
