// Written in 2003
// Author: Dr. J.K. Becker
// Copyright: Dr. J.K. Becker (becker@jkbecker.de)

#include "showelle.h"

extern Settings * Options();
extern DSettings * GetDisplayOptions();

/*extern void LogMessage( int loglevel, char msg[4096] );*/
extern void Error();


IMPLEMENT_CLASS( ShowelleFrame, wxFrame )

BEGIN_EVENT_TABLE( ShowelleFrame, wxFrame )
EVT_MENU( ID_Run, ShowelleFrame::OnRun )
EVT_CLOSE( ShowelleFrame::OnCloseWindow )
EVT_SIZE( ShowelleFrame::OnSize )
EVT_ERASE_BACKGROUND( ShowelleFrame::OnEraseBackground)
EVT_MENU( ID_Wizard, ShowelleFrame::OnSimWiz )
EVT_MENU( ID_PSPrint, ShowelleFrame::OnPSPrint )
EVT_MENU( ID_ReRun, ShowelleFrame::OnReRun )
EVT_MENU( ID_RunDia, ShowelleFrame::OnRunDia )
EVT_MENU( ID_Quit, ShowelleFrame::OnQuit )
EVT_MENU( ID_Save, ShowelleFrame::OnSave )
EVT_MENU( ID_SaveAs, ShowelleFrame::OnSaveAs )
EVT_MENU( ID_About, ShowelleFrame::OnAbout )
EVT_MENU( ID_Open, ShowelleFrame::OnOpen )
EVT_MENU( ID_GSaveRun, ShowelleFrame::OnSaveRun )
EVT_MENU( ID_Redraw, ShowelleFrame::OnRedraw )
EVT_MENU( ID_Prefs, ShowelleFrame::OnPrefs )
EVT_MENU( ID_GSaveSingle, ShowelleFrame::OnSaveSingle )
EVT_MENU( ID_Overlay_always, ShowelleFrame::OnOverlayAlways )
EVT_MENU( ID_Zoom, ShowelleFrame::OnZoom )
EVT_MENU( ID_Log, ShowelleFrame::OnShowLog )
EVT_MENU( ID_OnlineHelp, ShowelleFrame::OnOnline )
EVT_MENU( ID_Table_New_Bnode, ShowelleFrame::OnNewTableBnode )
EVT_MENU( ID_Table_New_Unode, ShowelleFrame::OnNewTableUnode )
EVT_MENU( ID_Table_New_Flynn, ShowelleFrame::OnNewTableFlynn )
EVT_MENU( ID_SaveStereo, ShowelleFrame::OnSaveStereo )
// LE EVT_PAINT( ShowelleFrame::OnPaint )
EVT_TOOL( Tool_zoomin, ShowelleFrame::OnToolIn )
EVT_TOOL( Tool_zoomout, ShowelleFrame::OnToolOut )
EVT_TOOL( Tool_stop, ShowelleFrame::OnToolStop )
EVT_TOOL( Tool_getbnode, ShowelleFrame::OnGetInfoBnode )
EVT_TOOL( Tool_getunode, ShowelleFrame::OnGetInfoUnode )
EVT_TOOL( Tool_label, ShowelleFrame::OnLabel )
END_EVENT_TABLE()

ShowelleFrame::ShowelleFrame( const wxString & title, const wxPoint & pos, const wxSize & size ) :
        wxFrame( (wxFrame *)NULL, wxID_ANY, title, pos, size)
{
    wxInitAllImageHandlers();
    panel = 0;
    int width, height;
    wxIcon icon;
    icon.LoadFile( "icons/elle_logo.xpm", wxBITMAP_TYPE_XPM );
    SetIcon( icon );
    GetClientSize( & width, & height );
    panel = new Canvas( this, width-10, height-10 );
    logframe = new wxLogWindow( this, "Logging window", false, true );
    wxMenu * menuFile = new wxMenu;
    /*Supposed to be in only when it is working. it is not so I took it out
    *
    * menuFile->Append( ID_Wizard, "&Simulation wizard" );
     */
    menuFile->Append( ID_Open, "&Open..." );
    menuFile->Append( ID_Save, "&Save..." );
    menuFile->Append( ID_SaveAs, "Save as..." );
    menuFile->Append( ID_PSPrint, "Print.." );
    menuFile->AppendSeparator();
    menuFile->Append( ID_Log, "Show Log..." );
    menuFile->AppendSeparator();
    menuFile->Append( ID_Quit, "E&xit" );
    wxMenu * menuRun = new wxMenu;
    menuRun->Append( ID_Run, "&Run..." );
    menuRun->Append( ID_ReRun, "Rer&un..." );
    menuRun->Append( ID_RunDia, "Run options.." );

    wxMenu * menuHelp = new wxMenu;
    menuHelp->Append( ID_About, "&About..." );
    menuHelp->Append( ID_OnlineHelp, "&Help..." );

    wxMenu * menuGSave = new wxMenu;
    menuGSave->Append( ID_GSaveSingle, "Save single pic" );
    menuGSave->Append( ID_GSaveRun, "Save whole run" );
    menuGSave->Append( ID_SaveStereo, "Save Stereonet" );

    menuGraphic = new wxMenu;
    menuGraphic->Append( ID_Prefs, "&Preferences" );
    menuGraphic->AppendSeparator();
    menuGraphic->Append( ID_Zoom, "&Zoom" );
    menuGraphic->AppendSeparator();

    menuGraphic->Append(ID_Overlay_always,"Always &overlay","Always overlay stages",true);

    menuGraphic->AppendSeparator();
    menuGraphic->Append( ID_SPic, "&Save", menuGSave );
    menuGraphic->AppendSeparator();
    menuGraphic->Append( ID_Redraw, "&Redraw" );

    wxMenu * menuData = new wxMenu;
    menuData->Append( ID_Table_New_Flynn, "&Flynns" );
    menuData->Append( ID_Table_New_Bnode, "&Bnodes" );
    menuData->Append( ID_Table_New_Unode, "&Unodes" );
    wxMenuBar * menuBar = new wxMenuBar;
    menuBar->Append( menuFile, "&File" );
    menuBar->Append( menuRun, "&Run" );
    menuBar->Append( menuGraphic, "&Graphics" );
    menuBar->Append( menuData, "&Data" );
    menuBar->Append( menuHelp, "&Help" );

    menuGraphic->Check(ID_Overlay_always,false);

    SetMenuBar( menuBar );
    CreateStatusBar( 3 );
    int stat[3] =
        {
            -1, -1,155
        };
    SetStatusWidths( 3, stat );
    SetStatusText( "Ready for action", 0 );
    if (strlen(ElleFile())>0)
    {
        wxString shortname = ElleFile();
        SetStatusText(shortname.AfterLast( E_DIR_SEPARATOR ),1);
        GetMenuBar()->EnableTop( 1, true );
        GetMenuBar()->EnableTop( 2, true );
    }
    else {
        SetStatusText("No File Open",1);
        GetMenuBar()->EnableTop( 1, false );
        GetMenuBar()->EnableTop( 2, false );
    }

    wxBoxSizer *b=new wxBoxSizer(wxVERTICAL);

    b->Add(panel,0,wxALIGN_LEFT|wxALIGN_TOP|wxALL,5);//CENTER_HORIZONTAL | wxALL, 5 );


    SetSizer( b );
    SetAutoLayout( TRUE );
    panel->Show(true);
    Show(true);
    panel->Init();
}


/*
 * called by Close()
 */
void ShowelleFrame::OnCloseWindow(wxCloseEvent& event)
{
#ifdef __WIN32__
    exit(0);
		Close(true);
    wxSafeYield();
    Destroy();
#else
    wxExit();
#endif //__win32__
}
#if XY
void ShowelleFrame::OnCloseWindow(wxCloseEvent& event)
  {
//    if (MyDataHasBeenModified())
 //   {
//      wxMessageDialog* dialog = new wxMessageDialog(this,
//        "Save changed data?", "My app", wxYES_NO|wxCANCEL);

//      int ans = dialog->ShowModal();
//      dialog->Destroy();
      int ans = wxID_NO;
      exit(0);

      switch (ans)
      {
        case wxID_YES:      // Save, then destroy, quitting app
          SaveMyData();
          this->Destroy();
          break;
        case wxID_NO:       // Don't save; just destroy, quitting app
          this->Destroy();
          break;
        case wxID_CANCEL:   // Do nothing - so don't quit app.
        default:
          if (!event.CanVeto()) // Test if we can veto this deletion
            this->Destroy();    // If not, destroy the window anyway.
          else
            event.Veto();     // Notify the calling code that we didn't delete the frame.
          break;
      }
//    }
}
#endif

void ShowelleFrame::OnQuit( wxCommandEvent & event )
{
    extern DSettings *display_options;
    extern Settings *user_options;
    delete panel;
    delete display_options;
    delete user_options;
    exit(0);
    Close( true );
    wxExit();
}

void ShowelleFrame::RepaintCanvas( bool c )
{
    if ( c )
    {
        //GLSwapBuffers();
        EllePlotRegions(0);
    }
    else
    {
        EllePlotRegions( 0 );
    }

}

void ShowelleFrame::OnShowLog( wxCommandEvent & event )
{
    logframe->Show( true );
}

void ShowelleFrame::OnRunDia( wxCommandEvent & event )
{
    RunDia * dialog = new RunDia( this );
    dialog->ShowModal();
}

void ShowelleFrame::OnSaveStereo( wxCommandEvent & event )
{
    //Defined in utilities/view/plotaxes.elle.h
    wxString filename,sdir;
    DSettings * doptions = GetDisplayOptions();
    sdir = doptions->GetSaveFileDir();
    filename = wxFileSelector( "Choose File", sdir, "", ".ps",
                               "Postscript files (*.ps)|*.ps|All Files (*)|*" ,wxSAVE);
    if(filename.Find('.')==-1)
        filename.Append(".ps");
    PlotAxes(filename);
}

//run simulation and disable menus so that the user can not screw things up
//after finishing, enable menus again
void ShowelleFrame::OnRun( wxCommandEvent &event )
{
    Settings * user_options = GetUserOptions();
    GetMenuBar()->Enable( ID_Open, false );
    GetMenuBar()->Enable( ID_Save, false );
    GetMenuBar()->Enable( ID_SaveAs, false );
    GetMenuBar()->Enable( ID_Run, false );
    GetMenuBar()->Enable( ID_ReRun, false );
    GetMenuBar()->Enable( ID_RunDia, false );
    GetMenuBar()->EnableTop( 2, false );
    GetMenuBar()->EnableTop( 3, false );
    if ( user_options->GetLogFile() )
    {
        FILE * logfile;
        if ( ( logfile = fopen( "logfile.txt", "r" ) ) != NULL )
            if ( !user_options->GetLogAppend() )
            {
                FILE * logfile = fopen( "logfile.txt", "w" );
                Log( 0, "New logfile created" );
                fclose( logfile );
            }
            else
            {
                Log( 0, "---------Using old logfile to append---------" );
            }
    }
    ElleRunFunc run;
    run = ElleRunFunction();
    ( * run ) ();
    GetMenuBar()->Enable( ID_Open, true );
    GetMenuBar()->Enable( ID_Save, true );
    GetMenuBar()->Enable( ID_SaveAs, true );
    GetMenuBar()->Enable( ID_Run, true );
    GetMenuBar()->Enable( ID_ReRun, true );
    GetMenuBar()->Enable( ID_RunDia, true );
    GetMenuBar()->EnableTop( 2, true );
    GetMenuBar()->EnableTop( 3, true );
}

//reset and start simulation
void ShowelleFrame::OnReRun(wxCommandEvent &event)
{
    ElleRunFunc init;
    if ( ( init = ElleInitFunction() ) ) ( * init ) ();
// LE  Reinitialise rainbow vals
    DSettings * doptions = GetDisplayOptions();
    doptions->SetRainStages(0);
    doptions->SetRainColor(0);

    panel->ClearScreen();
    EllePlotRegions( 0 );
    OnRun(event);
}

void ShowelleFrame::OnSave(wxCommandEvent & event)
{
    wxString filename, sdir, fileend;
    DSettings * doptions = GetDisplayOptions();
    sdir = doptions->GetSaveFileDir();
    filename=ElleFile();
    if ( !filename.IsEmpty() )
    {
        filename.Prepend(sdir);
        if(filename.Find('.')==-1)
            filename.Append(".elle");
        if ( filename.AfterLast('.') != "zip" )
        {
            ElleWriteData( filename.c_str() );
            filename=filename.BeforeLast('.');
            filename.Append(".zip");
            SaveArchive(filename);
        }
        else
        {
            cppFile sfile;
            sfile.SetFilename( filename );
            sfile.SaveZipFile();
        }
    }
    else
        OnSaveAs(event);
}

void ShowelleFrame::SaveArchive(wxString filename)
{
    DSettings * doptions = GetDisplayOptions();
    doptions->SaveArchive(filename);
}

void ShowelleFrame::OnSaveAs(wxCommandEvent & event)
{
    wxString filename, sdir;
    DSettings * doptions = GetDisplayOptions();
    sdir = doptions->GetSaveFileDir();
    filename = wxFileSelector( "Choose File", "", "", "", "ELLE files (*.elle)|*.elle|All Files (*)|*" ,wxSAVE);
    if ( !filename.IsEmpty() )
    {
        if(filename.Find('.')==-1)
            filename.Append(".elle");
        sdir = filename.BeforeLast( E_DIR_SEPARATOR );
        doptions->SetSaveFileDir( sdir );
        ElleWriteData( filename.c_str() );
        filename=filename.BeforeLast('.');
        filename.Append(".zip");
        SaveArchive(filename);
    }
}

void ShowelleFrame::OnOnline( wxCommandEvent & event )
{
    wxFrame *frame=new wxFrame(this,18,"Help-Window",wxDefaultPosition,wxDefaultSize);
    wxHtmlWindow *help=new wxHtmlWindow(frame, 19, wxDefaultPosition, wxSize(640,600),wxHW_DEFAULT_STYLE,"Help-Window");
    help->LoadPage("codo.html");
    frame->Fit();
    frame->Show(true);
}

void ShowelleFrame::OnAbout( wxCommandEvent & event )
{
    wxString message;
    message = "This is Elle-3.0.\n\nFor further questions and comments please\nsend emails to the developers\n(including the log-file!):\n\nDr. J.K. Becker\n\nLynn Evans\n\nDr. M. Jessell\n\nProf. P. Bons\n\nDr. D. Koehn\n\nGo to http://www.microstructure.info for current\naddresses and more information.";
    wxMessageBox( message, "About", wxOK );
}

void ShowelleFrame::CreateToolbarZoom( void )
{
    tool = new wxFrame( this, ZOOMWIN, "Zoom", wxDefaultPosition, wxSize( 250, 150 ) );
    tooltext = new wxTextCtrl( tool, -1, "", wxDefaultPosition, wxSize( 180, 50 ), wxTE_MULTILINE | wxTE_READONLY );
    toolbaricon[0] = new wxBitmap( vmagp_xpm );
    toolbaricon[1] = new wxBitmap( vmagm_xpm );
    toolbaricon[2] = new wxBitmap( stop_xpm );
    toolbaricon[3] = new wxBitmap( getinfobnode );
    toolbaricon[4] = new wxBitmap( getinfounodes );
    toolbaricon[5] = new wxBitmap( labelall );
    wxToolBar * toolbar = tool->CreateToolBar( wxNO_BORDER, -1, "ToolBar" );
    toolbar->AddTool( Tool_zoomin, "Zoom in", * ( toolbaricon[0] ), "Zoom in", wxITEM_NORMAL );
    toolbar->AddTool( Tool_zoomout, "Zoom out", * ( toolbaricon[1] ), "Zoom out", wxITEM_NORMAL );
    toolbar->AddTool( Tool_stop, "Stop zooming", * ( toolbaricon[2] ), "Stop zooming", wxITEM_NORMAL );
    toolbar->AddTool( Tool_getbnode, "Get info for Bnode", * ( toolbaricon[3] ), "Get info for Bnode", wxITEM_NORMAL );
    toolbar->AddTool( Tool_getunode, "Get info for Unode", * ( toolbaricon[4] ), "Get info for Unode", wxITEM_NORMAL );
    toolbar->AddTool( Tool_label, "Label", * ( toolbaricon[5] ), "Label nodes and flynns", wxITEM_NORMAL );
    toolbar->Realize();
    tool->Show( true );
    panel->Refresh( false, NULL );
}

void ShowelleFrame::OnZoom( wxCommandEvent & event )
{
    if ( FindWindow( ZOOMWIN ) == NULL )
    {
        panel->enablezoom = true;
        CreateToolbarZoom();
    }
    else
        FindWindow( ZOOMWIN )->Raise();
}

void ShowelleFrame::OnToolIn( wxCommandEvent & event )
{
    panel->ChangeView( 5 );
}



void ShowelleFrame::OnToolOut( wxCommandEvent & event )
{
    panel->ChangeView( 4 );
}

void ShowelleFrame::OnToolStop( wxCommandEvent & event )
{
    panel->enablezoom = false;
    panel->mzoom = false;
    panel->SetCursor( wxCURSOR_ARROW );
    panel->UnZoom();
    tool->Destroy();
    panel->Refresh( false, NULL );
}

void ShowelleFrame::OnGetInfoBnode( wxCommandEvent & event )
{
    panel->SetCursor( wxCURSOR_BULLSEYE );
    panel->getinfobnode = true;
}

void ShowelleFrame::OnGetInfoUnode( wxCommandEvent & event )
{
    panel->SetCursor( wxCURSOR_BULLSEYE );
    panel->getinfounode = true;
}

void ShowelleFrame::OnLabel( wxCommandEvent & event )
{
    panel->ElleShowFlynnNumbers();
    panel->ElleShowNodeNumbers();
    panel->Refresh( false, NULL );
}

void ShowelleFrame::GetInfoBnode( double mousex, double mousey )
{
    double x1, x2, y1, y2, unodeval;
    int n, t;
    Coords xy;
    wxString text = "",tex="";

    char attribid[1000];
    int * active = 0, max;

    SetCursor( wxCURSOR_ARROW );
    //the coordinates I get from zooming are not correct, I have to
    //take the zoomfactor into account!
    //we can't use exact coordinates, we use a bounding box of 0.1
    x1 = mousex - 0.005;
    x2 = mousex + 0.005;
    y1 = mousey - 0.005;
    y2 = mousey + 0.005;
    for ( n = 0; n < ElleMaxNodes(); n++ )
    {
        if ( ElleNodeIsActive( n ) )
        {
            ElleNodePosition( n, & xy );
            if ( xy.x > x1 && x2 > xy.x && xy.y > y1 && y2 > xy.y )
            {
                ElleNodeAttributeList( & active, & max );
                panel->GLDrawSingleBNode( xy.x,xy.y, 255, 255, 0 );
                text.Printf( "Bnode Number: %d\n   Position: X- %lf Y- %lf\n", n, xy.x, xy.y );
                for ( t = 0; t < max; t++ )
                {
                    id_match( FileKeys, active[t], attribid );
                    tex.Printf( "%s : %lf\n", attribid, ElleNodeAttribute( n, active[t] ) );
                    text.Append(tex);
                }
                if ( active )
                    delete active;
                if ( max == 0 ) {
                    tex = "No bnode-attributes defined\n";
                    text.Append(tex);
                }
                break;
            }
        }
    }
    if ( text.IsEmpty() )
        text = "No Hit!";
    text.Append( "\n" );
    tooltext->AppendText( text );
    Refresh(true,NULL);
}

void ShowelleFrame::GetInfoUnode( double mousex, double mousey )
{
    double x1, x2, y1, y2, unodeval;
    int n, t;
    Coords xy;
    wxString text = "",tex="";
    char attribid[1000];
    int * active = 0, max;
    SetCursor( wxCURSOR_ARROW );
    //the coordinates I get from zooming are not correct, I have to
    //take the zoomfactor into account!
    x1 = mousex - 0.005;
    x2 = mousex + 0.005;
    y1 = mousey - 0.005;
    y2 = mousey + 0.005;

    ElleUnodeAttributeList( & active, & max );
        bool fin = true;
        for ( n = 0; n < ElleMaxUnodes() && fin; n++ )
        {
            ElleGetUnodePosition( n, & xy );
            if ( xy.x > x1 && x2 > xy.x && xy.y > y1 && y2 > xy.y )
            {
                panel->GLDrawSingleBNode( xy.x,xy.y, 255, 255, 0 );
                text.Printf( "Unode Number: %d\n   Position: X- %lf  Y- %lf\n", n, xy.x, xy.y );
                for ( t = 0; t < max; t++ )
                {
                    id_match( FileKeys, active[t], attribid );
                    ElleGetUnodeAttribute( n, & unodeval, active[t] );
                    tex.Printf( "%s : %lf\n", attribid, unodeval );
                    text.Append(tex);
                }
                if ( active )
                    delete active;
                if ( max == 0 ) {
                    tex = "No unode-attributes defined\n";
                    text.Append(tex);
                }
                fin = false;
            }
        }
        if ( text.IsEmpty() )
            text = "No Hit!";

    text.Append( "\n" );
    tooltext->AppendText( text );
    Refresh(true,NULL);
}

void ShowelleFrame::OnOpen( wxCommandEvent & event )
{
    ///Open the file-dialog and get the filename of the the elle file to open.
    wxString filename, fileend;
    wxString filedir,title;
    DSettings * doptions = GetDisplayOptions();
    filedir = doptions->GetLoadFileDir();
    filename = wxFileSelector( "Choose File", filedir, "", "",
                               "ELLE files (*.elle)|*.elle|ELLE-Archive files (*.zip)|*.zip|All Files (*)|*" ,wxOPEN);
    if ( !filename.IsEmpty() )
    {
        filedir = filename.BeforeLast( E_DIR_SEPARATOR );
        doptions->SetLoadFileDir( filedir );
        fileend = filename.AfterLast( '.' );
        //LE set the filename in the elle runoptions
        ElleSetFile((char *)filename.c_str());       //<<<<<----
        if ( fileend != "zip" )
            LoadFile( filename );
        else if(fileend=="zip")
            LoadArchive(filename);
        //this enables the menu-items in the graphics menu.
        GetMenuBar()->EnableTop( 1, true );
        GetMenuBar()->EnableTop( 2, true );
        wxString shortname = ElleFile();
        SetStatusText(shortname.AfterLast( E_DIR_SEPARATOR ),1);
        //LE SetStatusText( filename.AfterLast(E_PATH_SEPARATOR), 1 );
        title.Printf("Opening ");
        title.Append(filename);
        Log( 0, (char *)(title.c_str()) );
    }
    else
    {
        Log( 0, "No file chosen" );
    }
    EllePlotRegions(0);
}

void ShowelleFrame::OnSaveRun( wxCommandEvent & event )
{
    ///Pictures of a whole run are saved here
    wxString suffix,savefilename,message;
    wxArrayString f;
    wxDir dir(wxGetCwd());
    int n;
    FILE * fp;
    wxString dird = wxFileSelector( "Choose file", dir.GetName(), "", "","ELLE files (*.elle)|*.elle|All Files (*)|*" ,wxOPEN);
	suffix=dird.AfterFirst('.');
    suffix=suffix.AfterFirst('.');
    if(suffix.IsEmpty())
        suffix="*.elle";
    else
        suffix.Prepend("*.");
    dird=dird.BeforeLast(E_DIR_SEPARATOR);
    if ( !dird.IsEmpty() )
    {
        if(dir.Open(dird))
        {

            fp = fopen( "filelist.txt", "w" );
            fprintf(fp,"mencoder \"mf://\\\n");
            dir.GetAllFiles(dird,&f,suffix);
            message.Printf("You are about to make pictures out of %ld files! Is that what you want?",f.GetCount());
            if(wxMessageBox(message, _T("Interesting question..."), wxYES|wxNO)==wxYES)
            {
                wxBeginBusyCursor();
                for(n=0;n<f.GetCount();n++)
                {
                /*    ElleRunFunc init;
                    if ( ( init = ElleInitFunction() ) )
                        ( * init ) ();
                    */
		ElleReinit(); //loescht die ganzen Arrays
                    ElleReadData( ( char * ) (f.Item(n)).c_str() );
                    EllePlotRegions( 0 );
                    panel->Refresh( false, NULL );
                    panel->Update();
                    savefilename=f.Item(n).BeforeLast('.');
                    savefilename.Append(".png");
                    panel->OnSaveSingle( savefilename );
                    fprintf(fp,"%s,\\\n",(savefilename.AfterLast(E_DIR_SEPARATOR)).c_str());
                }
                fprintf(fp,"\" -mf type=png -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o movie.wmf");
                fclose(fp);
                wxEndBusyCursor();
            }
        }
    }
}
void ShowelleFrame::OnPrefs( wxCommandEvent & event )
{
    if ( FindWindow( PREFSWIN ) != NULL )
        FindWindow( PREFSWIN )->Raise();
    else
    {
        prefs=new Preferences( this );
        prefs->Show(true);
    }
}

void ShowelleFrame::OnRedraw( wxCommandEvent & event )
{
    EllePlotRegions( 0 );
    panel->Refresh( false, NULL );
}

void ShowelleFrame::OnSaveSingle( wxCommandEvent & event )
{
    wxString savefilename = wxFileSelector( "Choose filename to save the pic", "", "", "",
                                            "PNG files (*.png)|*.png|JPEG files (*.jpg)|*.jpg|PCX files (*.pcx)|*.pcx|Ps files (*.ps)|*.ps|BMP files (*.bmp)|*.bmp|All Files (*)|*",wxSAVE );
    if ( !savefilename.IsEmpty() )
    {
        if(savefilename.AfterLast('.')=="ps")
            PSPrint();
        else
            panel->OnSaveSingle( savefilename );
    }
}

void ShowelleFrame::OnOverlayAlways(wxCommandEvent& event)
{
    Settings * user_options = GetUserOptions();
    if(event.IsChecked())
        user_options->SetOverlayOption(1);
    else
        user_options->SetOverlayOption(0);
}

void ShowelleFrame::LoadArchive(wxString filename)
{
    int t;
    Settings * user_options = GetUserOptions();
    DSettings * doptions = GetDisplayOptions();
    doptions->LoadArchive(filename,true);
    user_options->GetOverlayOption(&t);
    if(t==1)
        menuGraphic->Check(ID_Overlay_always,true);
    else
        menuGraphic->Check(ID_Overlay_always,false);
}
void ShowelleFrame::LoadFile( wxString filename )
{
    DSettings * doptions = GetDisplayOptions();
    /*int vsize,hsize;
    panel->GetSize(&hsize,&vsize);
    panel->SetVirtualSize(wxSize(hsize,vsize));
    */
	ElleRunFunc init;
    if ( ( init = ElleInitFunction() ) )
        ( * init ) ();
#if XY
    ElleReinit(); //loescht die ganzen Arrays
    ElleReadData( ( char * ) filename.c_str() );
    //panel->Refresh(false,NULL);
#endif
    filename=filename.BeforeLast('.');
    filename.Append(".zip");
    if(wxFileExists(filename))
    {
        LoadArchive(filename);
    }
    else
    {
        filename = doptions->GetLoadFileDir();
        filename.Append(E_DIR_SEPARATOR);
        filename.Append("defaults.zip");
        if(wxFileExists(filename))
            LoadArchive(filename);
    }
}

void ShowelleFrame::OnPSPrint( wxCommandEvent & event )
{
    PSPrint();
}
void ShowelleFrame::PSPrint(void )
{
    wxPrintData *printData=new wxPrintData;
    wxPageSetupDialogData *pagesdata=new wxPageSetupDialogData();
    //pagesdata->SetPaperId(wxPAPER_NONE);
    //pagesdata->SetPaperSize(wxSize(220,220));
    //pagesdata->EnableMargins(true);
    //pagesdata->SetMarginTopLeft(wxPoint(10,10));
    //pagesdata->SetMarginBottomRight(wxPoint(100,100));
    //wxPageSetupDialog *pd=new wxPageSetupDialog(this,pagesdata);
    //pd->ShowModal();
    (*printData)=pagesdata->GetPrintData();
    wxPrintDialogData printDialogData(*printData);
    //printDialogData.SetPrintData(*printData);
    printDialogData.SetMaxPage(1);
    printDialogData.EnablePageNumbers(false);
    printDialogData.SetPrintToFile(true);
    wxPrintPreview *preview = new wxPrintPreview(new PrintPS(panel), new PrintPS(panel), &printDialogData);
    if (!preview->Ok())
    {
        delete preview;
        wxMessageBox(_T("There was a problem previewing.\nPerhaps your current printer is not set correctly?"), _T("Previewing"), wxOK);
    }
    else
    {
        wxPreviewFrame *psframe = new wxPreviewFrame(preview, this, _T("Isn't this great: A preview"), wxPoint(100, 100), wxSize(600, 650));
        psframe->Centre(wxBOTH);
        psframe->Initialize();
        psframe->Show();
    }
}

void ShowelleFrame::OnNewTableUnode( wxCommandEvent & event )
{
    if ( FindWindow( FLYNNTABLE ) != NULL )
        FindWindow( FLYNNTABLE )->Close();
    if ( FindWindow( BNODETABLE ) != NULL )
        FindWindow( BNODETABLE )->Close();
    if ( FindWindow( UNODETABLE ) != NULL )
        FindWindow( UNODETABLE )->Raise();
    else
    {
        table = new TableData( panel, UNODETABLE, "UNode-Data" );
        int x, count, * welche, n, y;
        Coords xy;
        double unodeval;
        char attribid[1000];
        table->AddCol();
        table->SetColHeader( 1, "x" );
        table->AddCol();
        table->SetColHeader( 2, "y" );
        ElleUnodeAttributeList( & welche, & count );
        if ( count == 0 )
        {
            table->AddRow();
            table->SetValueS( 0, 0, "NO UNODE ATTRIBUTES DEFINED" );
        }
        else
        {
            for ( x = 0; x < count; x++ )
            {
                id_match( FileKeys, welche[x], attribid );
                table->AddCol();
                table->SetColHeader( x + 3, attribid );
            }
            for ( n = 0; n < ElleMaxUnodes(); n++ )
            {
                table->AddRow();
                for ( y = 0; y < count; y++ )
                {
                    table->SetValueI( n, 0, n );
                    ElleGetUnodePosition( n, & xy );
                    table->SetValueD( n, 1, xy.x );
                    table->SetValueD( n, 2, xy.y );
                    ElleGetUnodeAttribute( n, & unodeval, welche[y] );
                    table->SetValueD( n, y + 3, unodeval );
                }
            }
        }
        if ( welche )
            free( welche );
        table->AutoSize();
        table->Show( true );
    }
}

/* LE
void ShowelleFrame::OnPaint( wxPaintEvent & event )
{
    wxPaintDC dc( this );
}
*/

void ShowelleFrame::OnEraseBackground( wxEraseEvent & event )
{
}

void ShowelleFrame::OnNewTableBnode( wxCommandEvent & event )
{
    if ( FindWindow( UNODETABLE ) != NULL )
        FindWindow( UNODETABLE )->Close();
    if ( FindWindow( FLYNNTABLE ) != NULL )
        FindWindow( FLYNNTABLE )->Close();
    if ( FindWindow( BNODETABLE ) != NULL )
        FindWindow( BNODETABLE )->Raise();
    else
    {
        table = new TableData( panel, BNODETABLE, "BNode-Data" );
        int a = 0, x, count, * welche, n, y;
        Coords xy;
        char attribid[255];
        table->AddCol();
        table->SetColHeader( 1, "x" );
        table->AddCol();
        table->SetColHeader( 2, "y" );
        ElleNodeAttributeList( & welche, & count );
        if ( count == 0 )
        {
            table->AddRow();
            table->SetValueS( 0, 0, "NO NODE ATTRIBUTES DEFINED" );
        }
        else
        {
            for ( x = 0; x < count; x++ )
            {
                id_match( FileKeys, welche[x], attribid );
                table->AddCol();
                table->SetColHeader( x + 3, attribid );
            }
            for ( n = 0; n < ElleMaxNodes(); n++ )
            {
                if ( ElleNodeIsActive( n ) )
                {
                    table->AddRow();
                    for ( y = 0; y < count; y++ )
                    {
                        ElleNodePosition( n, & xy );
                        table->SetValueI( a, 0, n );
                        table->SetValueD( a, 1, xy.x );
                        table->SetValueD( a, 2, xy.y );
                        table->SetValueD( a, y + 3, ElleNodeAttribute( n, welche[y] ) );
                        a++;
                    }
                }
            }
        }
        if ( welche )
            delete welche;
        table->AutoSize();
        table->Show( true );
    }
}

void ShowelleFrame::OnNewTableFlynn( wxCommandEvent & event )
{
    if ( FindWindow( BNODETABLE ) != NULL )
        FindWindow( BNODETABLE )->Close();
    if ( FindWindow( UNODETABLE ) != NULL )
        FindWindow( UNODETABLE )->Close();
    if ( FindWindow( FLYNNTABLE ) != NULL )
        FindWindow( FLYNNTABLE )->Raise();
    else
    {
        table = new TableData( panel, FLYNNTABLE, "Flynn-Data" );
        int x, a = 0, ccount, * welche, n, ival;
        double val, b, c;
        double orientation, dip;
        float fval;
        char attribid[255];
        ElleFlynnDfltAttributeList( & welche, & ccount );
        for ( x = 0; x < ccount; x++ )
        {
            table->AddCol();
            if ( !id_match( FileKeys, welche[x], attribid ) )
                if ( !id_match( FlynnStrainKeys, welche[x], attribid ) )
                if ( !id_match( StressKeys, welche[x], attribid ) )
                    if ( !id_match( FlynnAgeKeys, welche[x], attribid ) );
            table->SetColHeader( x + 1, attribid );
        }
        for ( n = 0; n < ElleMaxFlynns(); n++ )
        {
            if ( ElleFlynnIsActive( n ) )
            {
                table->AddRow();
                table->SetValueI( a, 0, n );
                //LE ElleFlynnDfltAttributeList( & welche, & ccount );
                for ( x = 0; x < ccount; x++ )
                {
                    switch ( welche[x] )
                    {
                    case EXPAND:
                    case MINERAL:
                    case GRAIN:
                        ElleGetFlynnIntAttribute( n, & ival, welche[x] );
                        val = ( double )ival;
                        table->SetValueI( a, x + 1, ( int )val );
                        break;
                    case VISCOSITY:
                    case ENERGY:
                    case AGE:
                    case CYCLE:
                    case DISLOCDEN:
                    case F_ATTRIB_A:
                    case F_ATTRIB_B:
                    case F_ATTRIB_C:
                    case F_ATTRIB_D:
                    case F_ATTRIB_E:
                    case F_ATTRIB_F:
                    case F_ATTRIB_I:
                    case F_ATTRIB_J:
                    case F_ATTRIB_K:
                    case TAU_XX:
                    case TAU_YY:
                    case TAU_ZZ:
                    case TAU_XY:
                    case TAU_1:
                    case PRESSURE:
                        ElleGetFlynnRealAttribute( n, & val, welche[x] );
                        table->SetValueD( a, x + 1, val );
                        break;
                    case F_INCR_S:
                    case F_BULK_S:
                    case INCR_S:
                    case BULK_S:
                    case E_XX:
                    case E_XY:
                    case E_YX:
                    case E_YY:
                    case E_ZZ:
                        //ElleGetFlynnStrain( n, & val, welche[x] );
                        //table->SetValueD( a, x + 1, val );
                        break;
                        //If E3_ALPHA is read, BETA and GAMMA are automatically printed too
                        //More or less ugly fix...
                    case E3_ALPHA:
                    case EULER_3:
                        ElleGetFlynnEuler3( n, & val, & b, & c );
                        //val = fmod( ( val + 90.0 ), 30.0 ) / 15;
                        //if ( val > 1.0 ) val = 2.0 - val;
                        table->SetColHeader( x + 1, "E3_ALPHA" );
                        table->SetColHeader( x + 2, "E3_BETA" );
                        table->SetColHeader( x + 3, "E3_GAMMA" );
                        table->SetValueD( a, x + 1, val );
                        table->SetValueD( a, x + 2, b );
                        table->SetValueD( a, x + 3, c );
                        break;
                    }
                }
                a++;
            }
        }
        if ( welche )
            free( welche );
        table->AutoSize();
        table->Show( true );
    }
}

void ShowelleFrame::OnSimWiz( wxCommandEvent & event )
{
    SimWiz * wiz = new SimWiz( this );
    wiz->Show( true );
}

void ShowelleFrame::OnSize( wxSizeEvent & event )
{
    panel->OnSize(event);
    //SetClientSize(panel->GetSize());
    Refresh(true,NULL);
    Update();
}

bool PrintPS::OnPrintPage(int p)
{
    int w,h;

    CellData unitcell;
    ElleCellBBox(&unitcell);
    pscan->psDC = GetDC();    
	if(unitcell.xlength>=unitcell.ylength)
    {
        pscan->psDC->SetUserScale((1/unitcell.xlength)*unitcell.xlength,(1/unitcell.xlength)*unitcell.ylength);
        pscan->psDC->SetClippingRegion(0,0,300,300);
    }
    else
    {
        pscan->psDC->SetUserScale((1/unitcell.ylength)*unitcell.xlength,(1/unitcell.ylength)*unitcell.ylength);
        pscan->psDC->SetClippingRegion(0,0,300,300);
    }
    if(pscan->psDC )
    {
        pscan->DrawToPS(true);
        EllePlotRegions(0);
        pscan->DrawToPS(false);
    }
    EllePlotRegions(0);
}

void PrintPS::GetPageInfo(int *minPage, int *maxPage,int *pageFrom, int *pageTo)
{
    *minPage = 1; *maxPage =1;
    *pageFrom = 1; *pageTo = 1;
}

bool PrintPS::HasPage(int pageNum)
{
    return (pageNum ==1);
}
