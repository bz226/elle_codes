// Written in 2003
// Author: Dr. J.K. Becker
// Copyright: Dr. J.K. Becker (becker@jkbecker.de)

#include "wxstart.h"

int ElleReadImage(char *fname, int ***image, int *rows, int *cols,
                  int decimate);

Settings * user_options = new Settings();
DSettings * display_options = new DSettings();

bool Showelle::OnInit()
{
    mframe=0; //LE
    #ifdef __WIN32__
#else
    setlocale(LC_ALL,"C");
#endif //__win32__
    FILE * file;
    wxChar cc[255];
    wxString mc,title;
    ElleRunFunc init;
    Init_Data();
    mframe = new ShowelleFrame( ElleAppName(), wxPoint( 150, 50 ), wxSize( 450, 500 ) );
    SetTopWindow( mframe );
    if ( ( init = ElleInitFunction() ) )
        ( * init ) ();
    for(int n=0;n<argc;n++){
        mc.Printf("%s",argv[n]);
        if(mc=="-u")
            display_options->SetUCmdArgs(true);
    }
    if (strlen(ElleFile())>0)
    {
        Log( 0, (const char *)ElleFile() );
        wxString shortname = ElleFile();
        //LE
        wxString filedir = shortname.BeforeLast(E_DIR_SEPARATOR);
        if (filedir.Length() ==0) filedir = ".";
        display_options->SetLoadFileDir(filedir);

        mframe->SetStatusText(shortname.AfterLast(E_DIR_SEPARATOR),1);
        wxString filename=shortname.BeforeLast('.');
        filename.Append(".zip");
        // LE this code is the same as that called from OnOpen
        //    so we can have default settings for all files in
        //    a directory
        if(wxFileExists(filename) )
        {
            if (argc<2)
                mframe->LoadArchive(filename);
            else
                mframe->LoadArchive(filename);
        }
        else
        {
            filename = display_options->GetLoadFileDir();
            filename.Append(E_DIR_SEPARATOR);
            filename.Append("defaults.zip");
            if(wxFileExists(filename) )
            {
                if (argc<2)
                    mframe->LoadArchive(filename);
                else
                    mframe->LoadArchive(filename);
            }
            else
                Log(0,"Could not load zip-archive, because it does not exist... Sorry!");
        }
        mframe->GetMenuBar()->EnableTop( 1, true );
        mframe->GetMenuBar()->EnableTop( 2, true );
        //LE should we set user scale based on unticell dimensions??
        EllePlotRegions( 0 );
        //LE mframe->panel->Refresh(true,NULL);
    }
    Log( 0, "Done initialisation" );
    return true;
}


IMPLEMENT_APP_NO_MAIN( Showelle )

int SetupApp( int argc, char * argv[] )
{
#ifdef __WIN32__
    HINSTANCE hInstance = GetModuleHandle(0);
    int nCmdShow = SW_SHOWNORMAL;
    bool enterLoop = true;
    //wxEntry(hInstance, 0, "", nCmdShow); // wxwin261

    /* from src/msw/app.cpp in wxwin2.4.2 */
    // take everything into a try-except block to be able to call
    // OnFatalException() if necessary
#if wxUSE_ON_FATAL_EXCEPTION
    __try {
#endif
        wxhInstance = (HINSTANCE) hInstance;

        if (!wxEntryStart(argc,argv))
        return 0;

        // create the application object or ensure that one already
        // exists
        if (!wxTheApp)
            wxCHECK_MSG( wxTheApp, 0, wxT("You have to define an instance of wxApp!") );

            // save the WinMain() parameters
            wxTheApp->m_nCmdShow = nCmdShow;

            // We really don't want timestamps by default, because it means
            // we can't simply double-click on the error message and get to
            // that
            // line in the source. So VC++ at least, let's have a sensible
            // default.
#ifdef __VISUALC__
            wxLog::SetTimestamp(NULL);
#endif

            // init the app
            int retValue = /*wxEntryInitGui() &&*/ wxTheApp->OnInit() ? 0 :
                                                   -1;
            bool flag = wxTheApp->GetExitOnFrameDelete();
            wxTheApp->SetExitOnFrameDelete(true);

            if ( retValue == 0 )
            {
                if ( enterLoop )
                    {
                        // run the main loop
                        wxTheApp->OnRun();             }
                    else
                    {
                        // we want to initialize, but not run or exit
                        // immediately.
                        return 1;
                    }
                }
        //else: app initialization failed, so we skipped OnRun()

        wxWindow *topWindow = wxTheApp->GetTopWindow();
        if ( topWindow )
    {
        // Forcibly delete the window.
        if ( topWindow->IsKindOf(CLASSINFO(wxFrame)) ||
                    topWindow->IsKindOf(CLASSINFO(wxDialog)) )
            {
                topWindow->Close(TRUE);
                // DeletePendingObjects is protected so assume there aren't any
                //wxTheApp->DeletePendingObjects();
            }
            else
            {
                delete topWindow;
                wxTheApp->SetTopWindow(NULL);
            }
        }

        retValue = wxTheApp->OnExit();

        //wxEntryCleanup(); //problem:this tries to delete [] argv

        return retValue;
#if wxUSE_ON_FATAL_EXCEPTION
    }
    __except ( gs_handleExceptions ? EXCEPTION_EXECUTE_HANDLER
               : EXCEPTION_CONTINUE_SEARCH ) {
        if ( wxTheApp )
        {
            // give the user a chance to do something special about this
            wxTheApp->OnFatalException();
        }

        ::ExitProcess(3); // the same exit code as abort()

        // NOTREACHED
    }
#endif // wxUSE_ON_FATAL_EXCEPTION
#else
    wxEntry( argc, argv );
#endif

}

int Run_App( FILE * bla )
{
}


// these functions are in here for backward compability with the old elle code
//and will be replaced by new ones over time....

int FillPolygon( int index, float * x, float * y, int * num, int * col, int * outline )
{
    char logbuf[4096];
    double( * coords ) [3] = new double[* num] [3],theta,phi;
    double val[3];
    int n, set, r, g, b, lr, lg, lb,raincolor;
    bool color,rainbow;
    Coords_3D coordval;
    for ( n = 0; n < * num; n++ )
    {
        coords[n] [0] = x[n] ;
        coords[n] [1] = y[n] ;
        coords[n] [2] = 0;
    }
    if (index!=-1) {
      if (*outline != 2) {
        if ( display_options->GetShowArgs( FLYNNS) == EULER_RGB ) {
            ElleGetFlynnEuler3( index, &val[0], &val[1], &val[2]);
            r = (int)(255*val[0]/360);
            g = (int)(255*val[1]/180);
            b = (int)(255*val[2]/360);
            // wxColour(), SetColour() expect unsigned chars
            // conversion of int to uchar gives equivalent values
            // regardless of angle range (0->360 or -180->180)
            // e.g. -150 -> r=-106 -> uchar r=149
            //      210  -> r=149  -> uchar r=149
            color=true;
        }
        else /*if ( display_options->GetShowArgs( FLYNNS) != CAXIS )*/ {
            ElleFlynnGetColor( index, & color, & r, & g, & b );
        }
/*        else
        {
            ElleGetFlynnCAxis( index, & coordval );
            CartesianToPolar( coordval.x, coordval.y, coordval.z, & theta, & phi );
            sprintf(logbuf,"flynn:%d theta:%lf,phi:%lf\n",index,theta,phi);
            Log(2,(const char *)logbuf);
        }*/
      }
    if ( !color || index==-1)
        display_options->CmapGetColor( * col, & r, & g, & b, & set);
    //LE  no idea why this is here??
    //else
     //   wxGetApp().mframe->panel->GLDrawPolygon( coords, * num, r, g, b ); //fill
    }
    display_options->GetLineColor( & lr, & lg, & lb );
    rainbow=user_options->GetRainbow();
    if(rainbow)
    {
        if(display_options->GetRainStages()!=ElleCount())
            if(display_options->GetRainStages()+display_options->GetRainChange()==ElleCount())
            {
                display_options->IncrementRainColor();
                display_options->SetRainStages(ElleCount());
            }
        raincolor=display_options->GetRainColor();
        display_options->CmapGetColor( raincolor, & lr, &lg, &lb, & set);
        *outline=2;
    }
    switch ( * outline )
    {
    case 0:
        wxGetApp().mframe->panel->GLDrawPolygon( coords, * num, r, g, b ); //fill
        break;
    case 1:
        wxGetApp().mframe->panel->GLDrawPolygon( coords, * num, r, g, b ); //fill
        /*wxGetApp().mframe->panel->GLDrawLines( coords, * num, lr, lg,
lb ); //boundary attribute//and outline*/
        break;
    case 2:
        wxGetApp().mframe->panel->GLDrawLines( coords, * num, lr, lg, lb ); //boundary attribute//outline
        break;
    case 3:
        wxGetApp().mframe->panel->GLDrawLinesBound( coords, * num, r, g, b ); //boundary attribute
        break;
    default: //do nothing
        break;
    }
    delete[] coords;
    if ( display_options->ShowFlynnNumbers() )
        wxGetApp().mframe->panel->ElleShowFlynnNumbers( index, x, y, * num );
    if ( display_options->ShowNodeNumbers() )
        wxGetApp().mframe->panel->ElleShowNodeNumbers();
    return ( 0 );
}

int drawsegment_( float * x1, float * y1, float * x2, float * y2, int * col )
{
    char logbuf[4096];
    double( * coords ) [3] = new double[2] [3];
    int set, r, g, b, lr, lg, lb;
    int val;
    coords[0] [0] = *x1 ;
    coords[0] [1] = *y1 ;
    coords[0] [2] = 0;
    coords[1] [0] = *x2 ;
    coords[1] [1] = *y2 ;
    coords[1] [2] = 0;
    display_options->CmapGetColor( * col, & r, & g, & b, & set);
    wxGetApp().mframe->panel->GLDrawLinesBound( coords, 2, r, g, b );
    delete[] coords;
    return ( 0 );
}

int drawcircle_( int dd, float * x, float * y, float * size, int * col, int * outline )
{
    int r, g, b, set
        ;
    double xx, yy;
    xx = * x ;
    yy = * y ;
    if ( * col == -4 || * col == -3 ) //ElleNodeIsTriple( dd ) )
        if ( * col == -3 )
            display_options->GetDNodeColor( & r, & g, & b );
        else
            display_options->GetTNodeColor( & r, & g, & b );
    else
        display_options->CmapGetColor( * col, & r, & g, & b, & set
                                         );
    wxGetApp().mframe->panel->GLDrawSingleBNode( xx, yy, r, g, b );
    return ( 0 );
}

int DrawNode( int dd, float * x, float * y, float * size, int * col, int * outline )
{
    int r, g, b, set;
    double xx, yy;
    xx = * x ; yy = * y ;
    if ( * col == -4 || * col == -3) //ElleNodeIsTriple( dd ) )
        if ( * col == -3 )
            display_options->GetDNodeColor( & r, & g, & b );
        else
            display_options->GetTNodeColor( & r, & g, & b );
    else
        display_options->CmapGetColor( * col, & r, & g, & b, & set );
    wxGetApp().mframe->panel->GLDrawSingleBNode( xx, yy, r, g, b );
    return ( 0 );
}

void DrawUnodes( int col, int range, int attrib, double min, double max )
{
    wxGetApp().mframe->panel->ElleDrawUNodes( attrib);
}

void DrawSegments( int col, int range, int attrib, double min, double max )
{
    wxGetApp().mframe->panel->ElleDrawSegments( attrib);
}

int plotu_( float * x, float * y, int * w )
{
    return ( 0 );
}

int setpencolor_( int * col )
{
    return ( 0 );
}

int drawlabel_( float *x, float *y,int *x_justify, int *y_justify,
                int *mode,char *label, int *len )
{
}

void SetStage( int stages )
{
    char cc[255];
    int val;
    div_t rest;
    wxString filename,tmp,path;
    sprintf( cc, "Stage: %d", stages );
    wxGetApp().mframe->SetStatusText( cc,0 );
    if(display_options->GetSavePic())
    {
        ElleGetRunOptionValue(RO_SVFREQ,&val);
        rest=div(stages,val);
        if(rest.rem==0)
        {
            filename=display_options->GetSavePicFName();
            tmp.Printf("%c%d",E_DIR_SEPARATOR,stages);
            path=filename.BeforeLast(E_DIR_SEPARATOR);
            filename=filename.AfterLast(E_DIR_SEPARATOR);
            filename.Prepend(tmp);
            filename.Prepend(path);
            sprintf(cc,"Filename pics: %s",filename.c_str());
            Log(0,(const char *)cc);
            wxGetApp().mframe->panel->OnSaveSingle(filename);
        }
	}
}

void RescaleBox( float, float, float, float )
{
    wxGetApp().mframe->panel->CheckUnitCell();
}

/*! only called from EllePlotRegions if !overlay */
void clearbg_()
{
#if wxCHECK_VERSION(2, 6, 2)
    if (wxGetApp().IsMainLoopRunning())
#else // replacement code for old version
    if (wxGetApp().Initialized() && wxGetApp().mframe)
#endif
    {
        wxGetApp().mframe->panel->ClearScreen();
    }
}

void update_()
{
#if wxCHECK_VERSION(2, 6, 2)
    if (wxGetApp().IsMainLoopRunning())
#else // replacement code for old version
    if (wxGetApp().Initialized() && wxGetApp().mframe &&
            wxGetApp().mframe->panel )
#endif
    {
        wxGetApp().mframe->panel->Refresh(false,NULL);
    }
    wxGetApp().Yield(true);
}

void warning_msg( int i, const char * msg )
{
    wxString mesg;
    mesg = msg;
    wxLogMessage( wxT("This comes from warning_msg in wxstart.cc!!!!!!!!!!!!!!!!"));
    wxLogMessage( wxT("%s"), msg );
}

Settings * GetUserOptions()
{
    return ( user_options );
}

DSettings * GetDisplayOptions()
{
    return ( display_options );
}

void Log( int level, const char msg[4096] )
{
    if ( display_options->GetLogLevel() >= level && display_options->GetLogLevel()!=-1)
    {
        int c = ElleCount();
        char cc[100];
        sprintf( cc, "Stage: %d", c );
        wxLogMessage( wxT("%s"), cc );
        wxLogMessage( wxT("%s"), msg );
        if ( user_options->GetLogFile() )
        {
            FILE * logfile;
            if ( ( logfile = fopen( wxT("logfile.txt"), wxT("a") ) ) != NULL )
            {
                fprintf( logfile, "%s -- %s\n", cc, msg );
                fclose( logfile );
            }
            else
            {
                wxLogMessage( wxT("Could not find logfile to save, not saving anything!"));
                display_options->SetLogLevel(-1);
            }

        }
    }
	if (level<0) {
		wxLogError(wxT("%s"), msg );
		/*Error();*/
	}
}

void Error()
{
    wxGetApp().mframe->SetStatusText( wxT("There were errors"), 1 );
    wxGetApp().mframe->logframe->Show( true );
}

void SavePic()
{
    wxString file = display_options->GetSavePicFName(), name, path;
    char f[4096];
    int i;
    if ( file != wxT("" ))
    {
        name = file.AfterLast( E_DIR_SEPARATOR );
        path = file.BeforeLast( E_DIR_SEPARATOR );
        path.Append( wxT("/"));
        file = wxT("");
        i = ElleCount();
        file << i;
        name.Prepend( file );
        path.Append( name );
        sprintf( f, wxT("%s"), path.c_str() );
        Log( 1, (const char *)f );
        wxGetApp().mframe->panel->OnSaveSingle( path );
    }
}

/*********************************************************

  ReadImage reads in a pixmap image format file
  and decimates image before stroing in array
  uses wxImage

**********************************************************/

int ElleReadImage(char *fname, int ***image, int *rows, int *cols,
                  int decimate)
{
    FILE *in; // image file
    int err=0;
    int i,j;
    unsigned long colr; // 8-bit colour
    int colsP,rowsP; // rows,cols in image
    unsigned long maxvalP; // number of colours
    wxImage pixmap;
    wxInitAllImageHandlers();
    i=pixmap.LoadFile(wxT(fname),wxBITMAP_TYPE_ANY);
	if ( i== 0)
    {
        Log(0,(const char *)fname);
        return(OPEN_ERR);
    }
    colsP=(int)pixmap.GetWidth();
    rowsP=(int)pixmap.GetHeight();
    maxvalP = 16;
    printf("Cols:%d Rows:%d\n",colsP,rowsP);
    *image=imatrix(0,rowsP/decimate,0,colsP/decimate);
    if (*image==0) return(MALLOC_ERR);

    for(i=0;i<rowsP;i++)
    {
        for(j=0;j<colsP;j++)
        {
            colr=pixmap.GetRed(j,i)+pixmap.GetGreen(j,i)*maxvalP+pixmap.GetBlue(j,i)*maxvalP*maxvalP;

            if(i%decimate==0 && j%decimate==0)
                (*image)[i/decimate][j/decimate]=(int)colr;
        }
    }

    for(i=0;i<rowsP/decimate;i++)
        (*image)[i][colsP/decimate]=(*image)[i][0];

    for(j=0;j<colsP/decimate;j++)
        (*image)[rowsP/decimate][j]=(*image)[0][j];

    (*image)[rowsP/decimate][colsP/decimate]=(*image)[0][0];



    *rows=(rowsP/decimate);
    *cols=(colsP/decimate);

    return(err);

}
