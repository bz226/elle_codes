/////////////////////////////////////////////////////////////////////////////
// Name:        dsettings.cc
// Purpose:
// Author:      Dr. J.K. Becker
// Modified by:
// Created:
// RCS-ID:
// Copyright: Dr. J.K. Becker
// Licence: GPL
/////////////////////////////////////////////////////////////////////////////

#include "dsettings.h"
#include <fstream>
#include <iostream>
#include <stdio.h>
#include "settings.h"
#include "interface.h"

extern Settings * GetUserOptions( void );


DSettings::DSettings( void )
{
    CmapLoad( "default.cmap" );
    cmap[256] [0] = 0;
    cmap[256] [1] = 0;
    cmap[256] [2] = 0;
    cmap[256] [3] = 0;

    cmap[257] [0] = 255;
    cmap[257] [1] = 255;
    cmap[257] [2] = 255;
    cmap[257] [3] = 0;

    cmap[258] [0] = 255;
    cmap[258] [1] = 0;
    cmap[258] [2] = 0;
    cmap[258] [3] = 0;

    cmap[259] [0] = 0;
    cmap[259] [1] = 255;
    cmap[259] [2] = 0;
    cmap[259] [3] = 0;

    cmap[260] [0] = 0;
    cmap[260] [1] = 0;
    cmap[260] [2] = 255;
    cmap[260] [3] = 0;

    cmap[261] [0] = 255;
    cmap[261] [1] = 0;
    cmap[261] [2] = 255;
    cmap[261] [3] = 0;

    cmap[262] [0] = 255;
    cmap[262] [1] = 255;
    cmap[262] [2] = 0;
    cmap[262] [3] = 0;

    cmap[263] [0] = 0;
    cmap[263] [1] = 255;
    cmap[263] [2] = 255;
    cmap[263] [3] = 0;

    CmapResetToDefault();
    savepicfname = "";
    loglevel=0;
    rangeflagunodes = false;
    unodesrangeflagmaxvalue = 0;
    unodesrangeflagminvalue = 0;
    notrangeflagunodes = false;
    unodesnotrangeflagmaxvalue = 0;
    unodesnotrangeflagminvalue = 0;
    rangeflagsegments = false;
    segmentsrangeflagmaxvalue = 0;
    segmentsrangeflagminvalue = 0;
    notrangeflagsegments = false;
    segmentsnotrangeflagmaxvalue = 0;
    segmentsnotrangeflagminvalue = 0;
    nodesize = 1;
    linesize = 1;
    unodesize = 1;
    segmentsize = 1;

    dnodecolor[0] = 0;
    dnodecolor[1] = 0;
    dnodecolor[2] = 255;
    tnodecolor[0] = 255;
    tnodecolor[1] = 0;
    tnodecolor[2] = 0;
    linecolor[0] = 255;
    linecolor[1] = 255;
    linecolor[2] = 255;
    loadfiledir = "";
    savefiledir = "";
    shownodenumbers = false;
    showflynnnumbers = false;
    savepics=false;
    cstages=0;
    raincolor=0;
    rainchange=10;
    fclampcolor=bclampcolor=nclampcolor=uclampcolor=sclampcolor=ucmdargs=false;
    bccmin=bccmax=nccmin=nccmax=uccmin=uccmax=sccmax=sccmin=fccmax=fccmin=0.0;
}

void DSettings::SetUCmdArgs(bool t)
{
	ucmdargs=t;
}

void DSettings::SetRainStages(int s)
{
    cstages=s;
}

int DSettings::GetRainStages()
{
    return cstages;
}

void DSettings::IncrementRainColor()
{
    raincolor++;
    if(raincolor>255)
        raincolor=0;
}

void DSettings::SetRainColor(int i)
{
    raincolor = i;
    if(raincolor>255)
        raincolor=0;
}

int DSettings::GetRainColor()
{
    return raincolor;
}

void DSettings::SetRainChange(int s)
{
    rainchange=s;
}

int DSettings::GetRainChange()
{
    return rainchange;
}

void DSettings::SetSavePicFName(wxString t)
{
    savepicfname=t;
}
wxString DSettings::GetSavePicFName()
{
    return savepicfname;
}

void DSettings::SetSavePic(bool t)
{
    savepics=t;
}
bool DSettings::GetSavePic()
{
    return savepics;
}
void DSettings::SetLogLevel(int l)
{
    loglevel=l;
}
/*char * DSettings::GetFilename()
 {
    return filename;
 }
 int DSettings::SetFilename(char *filen)
 {
    filename=filen;
    return 1;
 }*/
int DSettings::GetLogLevel()
{
    return loglevel;
}
void DSettings::SetUnodesRangeFlag( bool var )
{
    rangeflagunodes = var;
}
void DSettings::SetSegmentsRangeFlag( bool var )
{
    rangeflagsegments = var;
}

void DSettings::ShowNodeNumbers( bool c )
{
    shownodenumbers = c;
}

void DSettings::ShowFlynnNumbers( bool c )
{
    showflynnnumbers = c;
}

bool DSettings::ShowFlynnNumbers()
{
    return showflynnnumbers;
}

bool DSettings::ShowNodeNumbers()
{
    return shownodenumbers;
}
void DSettings::LoadArchive(wxString filename,bool run)
{
    wxZipEntry *entry;
    wxFFileInputStream in(filename);
    wxZipInputStream zip(in);
    wxTextInputStream text( zip );
    while (entry=zip.GetNextEntry(),entry!= NULL) {
        if(entry->GetName()=="dsettings.txt")
            if(zip.OpenEntry(*entry))
                LoadArchiveDSettings(&text);
        if(entry->GetName()=="cmap.cmap")
            if(zip.OpenEntry(*entry))
                LoadArchiveCmap(&text);
        if(entry->GetName()=="runopts.txt")
            if(zip.OpenEntry(*entry))
                LoadArchiveRunOpts(&text);
        if(entry->GetName()=="userdata.txt"&&run)
            if(zip.OpenEntry(*entry))
                LoadArchiveUserData(&text);
    }
}

void DSettings::LoadArchiveDSettings(wxTextInputStream *text)
{
    wxString line,wxline;
    int val[3], n = 0,f_arg,n_arg,b_arg,u_arg;
    double min,max;
    bool tc;
    Settings * user_options = GetUserOptions();
    //Read comments in first three lines
    text->ReadLine();
    text->ReadLine();
    text->ReadLine();
    //LineColor
    line = text->ReadLine();
    wxline = line.AfterFirst( ' ' );
    val[0] = atoi( ( wxline.BeforeFirst( ' ' ) ).c_str() );
    wxline = wxline.AfterFirst( ' ' );
    val[1] = atoi( ( wxline.BeforeFirst( ' ' ) ).c_str() );
    val[2] = atoi( ( wxline.AfterFirst( ' ' ) ).c_str() );
    SetLineColor( val[0], val[1], val[2] );
    //LineSize
    line = text->ReadLine();
    val[0] = atoi( ( line.AfterFirst( ' ' ) ).c_str() );
    SetLineSize( val[0] );
    //LineArgs
    line = text->ReadLine();
    line = line.AfterLast( ' ' );
    //LE
    b_arg=atoi( line.c_str() ) ;
    SetShowArgs( BOUNDARIES, true, atoi( line.c_str() ) );
    line = text->ReadLine();
    line = line.AfterLast( ' ' );
    val[0]=atoi( line.c_str() );
    SetRainChange(val[0]);
    line=text->ReadLine();
    line=line.AfterLast(' ');
    if(line=="TRUE")
        user_options->SetOverlayOption(1);
    else
        user_options->SetOverlayOption(0);

    //NodeSize
    line = text->ReadLine();
    line = line.AfterLast( ' ' );
    SetNodeSize( atoi( line.c_str() ) );
    //BNodeColor
    line = text->ReadLine();
    wxline = line.AfterFirst( ' ' );
    val[0] = atoi( ( wxline.BeforeFirst( ' ' ) ).c_str() );
    wxline = wxline.AfterFirst( ' ' );
    val[1] = atoi( ( wxline.BeforeFirst( ' ' ) ).c_str() );
    val[2] = atoi( ( wxline.AfterFirst( ' ' ) ).c_str() );

    SetDNodeColor( val[0], val[1], val[2] );
    //TNodeColor
    line = text->ReadLine();
    wxline = line.AfterFirst( ' ' );
    val[0] = atoi( ( wxline.BeforeFirst( ' ' ) ).c_str() );
    wxline = wxline.AfterFirst( ' ' );
    val[1] = atoi( ( wxline.BeforeFirst( ' ' ) ).c_str() );
    val[2] = atoi( ( wxline.AfterFirst( ' ' ) ).c_str() );

    SetTNodeColor( val[0], val[1], val[2] );
    //NodeArgs
    line = text->ReadLine();
    line = line.AfterLast( ' ' );
    //LE
    n_arg=atoi( line.c_str() ) ;
    SetShowArgs( BNODES, true, atoi( line.c_str() ) );
    //FlynnArgs
    line = text->ReadLine();
    line = line.AfterLast( ' ' );
    //LE
    f_arg=atoi( line.c_str() ) ;
    SetShowArgs( FLYNNS, true, atoi( line.c_str() ) );
    //UNodeArgs
    line = text->ReadLine();
    line = line.AfterLast( ' ' );
    u_arg=atoi( line.c_str() ) ;
    SetShowArgs( UNODES, true, atoi( line.c_str() ) );
    //Unode triangulation
    line = text->ReadLine();
    line = line.AfterLast( ' ' );
    if(line=="TRUE")
        tc=true;
    else
        tc=false;
    user_options->draw_nodes= ((tc==true) ? 1 : 0);

    //UnodesSize
    line = text->ReadLine();
    line = line.AfterLast( ' ' );
    SetUNodeSize(atoi( line.c_str() ));
    //UnodesNotRangeFlag
    line = text->ReadLine();
    line = line.AfterLast( ' ' );
    if ( line == "TRUE" ) {
        SetUnodesNotRangeFlag( true );
        line = text->ReadLine();
        line = line.AfterLast( ' ' );
        SetUnodesNotRangeFlagMaxValue( atof( line.c_str() ) );
        line = text->ReadLine();
        line = line.AfterLast( ' ' );
        SetUnodesNotRangeFlagMinValue( atof( line.c_str() ) );
    } else {
        SetUnodesNotRangeFlag( false );
        line=text->ReadLine();
        line=text->ReadLine();
    }
    //UnodesRangeFlag
    line = text->ReadLine();
    line = line.AfterLast( ' ' );
    if ( line == "TRUE" ) {
        SetUnodesRangeFlag( true );
        line = text->ReadLine();
        line = line.AfterLast( ' ' );
        SetUnodesRangeFlagMaxValue( atof( line.c_str() ) );
        line = text->ReadLine();
        line = line.AfterLast( ' ' );
        SetUnodesRangeFlagMinValue( atof( line.c_str() ) );
    } else {
        SetUnodesRangeFlag( false );
        line=text->ReadLine();
        line=text->ReadLine();
    }
    //FlynnClampColor
    line = text->ReadLine();
    line = line.AfterLast( ' ' );
    if ( line == "TRUE" )
        tc=true;
    else
        tc=false;
    line = text->ReadLine();
    line = line.AfterLast( ' ' );
    min=atof( line.c_str() );
    line = text->ReadLine();
    line = line.AfterLast( ' ' );
    max=atof( line.c_str() );
    SetClampColor(0,tc,min,max);
    user_options->SetFlynnAttribOption(f_arg,min,max);

    //Boundary clamp color
    line = text->ReadLine();
    line = line.AfterLast( ' ' );
    if ( line == "TRUE" )
        tc=true;
    else
        tc=false;
    line = text->ReadLine();
    line = line.AfterLast( ' ' );
    min=atof( line.c_str() );
    line = text->ReadLine();
    line = line.AfterLast( ' ' );
    max=atof( line.c_str() );
    SetClampColor(1,tc,min,max);
    user_options->SetBndAttribOption(b_arg,min,max);
    //nodes clamp color
    line = text->ReadLine();
    line = line.AfterLast( ' ' );
    if ( line == "TRUE" )
        tc=true;
    else
        tc=false;
    line = text->ReadLine();
    line = line.AfterLast( ' ' );
    min=atof( line.c_str() );
    line = text->ReadLine();
    line = line.AfterLast( ' ' );
    max=atof( line.c_str() );
    SetClampColor(2,tc,min,max);
    user_options->SetNodeAttribOption(n_arg,min,max);
    //unode clamp color
    line = text->ReadLine();
    line = line.AfterLast( ' ' );
    if ( line == "TRUE" )
        tc=true;
    else
        tc=false;
    line = text->ReadLine();
    line = line.AfterLast( ' ' );
    min=atof( line.c_str() );
    line = text->ReadLine();
    line = line.AfterLast( ' ' );
    max=atof( line.c_str() );
    SetClampColor(3,tc,min,max);
    user_options->SetUnodeAttribOption(u_arg,min,max);
    //Set LoadFileDir
    line = text->ReadLine();
    line = line.AfterLast( ' ' );
    wxline = line.c_str();
    //LE Do we want to do this?
    SetLoadFileDir( wxline );
    //LE What about File?
    line = text->ReadLine();
    //Set SaveFileDir
    line = text->ReadLine();
    line = line.AfterLast( ' ' );
    wxline = line.c_str();
    //LE Do we want to do this?
    SetSaveFileDir( wxline );

    //LE preferences window can be open while new file is opened
    // Need to move triangulate to dsettings or check whether preferences exists

}

void DSettings::LoadArchiveCmap(wxTextInputStream *text)
{
	int n, set, r, g, b;
	text->ReadLine();
    for ( n = 0; n < 256; n++ ) //there are 256 colors in the file
    {
        r=(int)text->ReadDouble();
        g=(int)text->ReadDouble();
        b=(int)text->ReadDouble();
        set=(int)text->ReadDouble();
        CmapChangeColor( n, r, g, b, set);
    }
}

void DSettings::SaveArchiveDSettings(wxTextOutputStream &txt)
{
    string ntest, name;
    int r,g,b,set
    ;
    double min,max;
    bool test;
    Settings * user_options = GetUserOptions();
    txt<< "-----------------------------------\n" ;
    txt<< "DISPLAY OPTIONS\n" ;
    txt<< "-----------------------------------\n" ;
    GetLineColor( & r, & g, & b) ;
    txt<< "Linecolor: "<<r<<" "<<g<<" "<<b<<"\n";
    txt<< "Linesize: "<< GetLineSize()<<"\n";
    txt<< "Lineargs: "<< GetShowArgs( BOUNDARIES ) <<"\n";
    txt<< "Rainbow-change: "<< GetRainChange() <<"\n";
    txt<< "Always overlay: ";
    user_options->GetOverlayOption(&r);
    if(r)
        txt<<"TRUE"<<"\n";
    else
        txt<<"FALSE"<<"\n";
    txt<< "Nodesize: "<<GetNodeSize() <<"\n";
    GetDNodeColor( & r, & g, & b );
    txt<< "BNodecolor: "<< r<<" " <<g<<" " << b <<"\n";
    GetTNodeColor( & r, & g, & b );
    txt<< "TNodecolor: "<< r<<" " << g<<" " << b <<"\n";
    txt<< "Nodeargs: "<< GetShowArgs( BNODES ) <<"\n";
    txt<< "Flynnargs: "<< GetShowArgs( FLYNNS ) <<"\n";
    txt<< "UNodeargs: "<< GetShowArgs( UNODES ) <<"\n";
    user_options->GetUnodePlotOption(&set
                                    );
    txt<< "UNodeTriangulate: ";
    if ( set
                !=0 )
            txt<<"TRUE\n" ;
    else
        txt<<"FALSE\n";
    txt<< "UNodeSize: "<< GetUNodeSize()<<"\n";
    txt<< "UNodesNotRangeFlag: ";
    test = GetUnodesNotRangeFlag();
    if(test)
        txt<< "TRUE\n";
    else
        txt<<"FALSE\n";
    txt<< "UNodesNotRangeFlagMax: "<< GetUnodesNotRangeFlagMaxValue() <<"\n";
    txt<< "UNodesNotRangeFlagMin: "<< GetUnodesNotRangeFlagMinValue() <<"\n";
    test = GetUnodesRangeFlag();
    txt<< "UNodesRangeFlag: ";
    if ( test )
        txt<<"TRUE\n" ;
    else
        txt<<"FALSE\n";
    txt<< "UNodesRangeFlagMax: "<< GetUnodesRangeFlagMaxValue() <<"\n";
    txt<< "UNodesRangeFlagMin: "<< GetUnodesRangeFlagMinValue() <<"\n";

    test=GetClampColor(0,&min,&max);
    txt<< "Flynn-clamp-color: ";
    if ( test )
        txt<<"TRUE\n" ;
    else
        txt<<"FALSE\n";
    txt<<"Min: "<<min<<"\n";
    txt<<"Max: "<<max<<"\n";

    test=GetClampColor(1,&min,&max);
    txt<< "Boundary-clamp-color: ";
    if ( test )
        txt<<"TRUE\n" ;
    else
        txt<<"FALSE\n";
    txt<<"Min: "<<min<<"\n";
    txt<<"Max: "<<max<<"\n";

    test=GetClampColor(2,&min,&max);
    txt<< "Node-clamp-color: ";
    if ( test )
        txt<<"TRUE\n" ;
    else
        txt<<"FALSE\n";
    txt<<"Min: "<<min<<"\n";
    txt<<"Max: "<<max<<"\n";

    test=GetClampColor(3,&min,&max);
    txt<< "Unode-clamp-color: ";
    if ( test )
        txt<<"TRUE\n" ;
    else
        txt<<"FALSE\n";
    txt<<"Min: "<<min<<"\n";
    txt<<"Max: "<<max<<"\n";

    txt<< "LoadFileDir: "<< GetLoadFileDir() <<"\n";
    txt<< "File: "<< ElleFile()<<"\n";
    txt<< "SaveFileDir: "<< GetSaveFileDir() <<"\n";

}

void DSettings::SaveArchiveCmap(wxTextOutputStream &txt)
{
    int r, g, b,set
    ;
    txt<<"Colormap for Elle-files... Please do not edit\n";
    for (int n = 0; n < 256; n++ ) {
        CmapGetColor( n, & r, & g, & b, & set
                        );
        txt<< r<<" "<< g<<" "<< b<<" "<< set
            <<"\n";
    }
}

void DSettings::SaveArchive(wxString filename)
{
    wxFFileOutputStream out(filename);
    wxZipOutputStream zip(out);
    wxTextOutputStream txt(zip);

    zip.PutNextEntry(_T("dsettings.txt"));
    SaveArchiveDSettings(txt);

    zip.PutNextEntry(_T("cmap.cmap"));
    SaveArchiveCmap(txt);

    zip.PutNextEntry(_T("runopts.txt"));
    SaveArchiveRunOpts(txt);

    zip.PutNextEntry(_T("userdata.txt"));
    SaveArchiveUserData(txt);

    zip.CloseEntry();
    zip.Close();
    out.Close();
}
void DSettings::LoadArchiveUserData(wxTextInputStream *text)
{
    UserData ud;
    int i;
    wxString line;
    text->ReadLine();
    text->ReadLine();
    text->ReadLine();
    for (i=0;i<MAX_U_DATA;i++) {
        line = text->ReadLine();
        line = line.AfterLast( ' ' );
        ud[i]=atof(line.c_str());
    }
    if(!ucmdargs)
    ElleSetUserData(ud);
}

void DSettings::LoadArchiveRunOpts(wxTextInputStream *text)
{
    wxString line;
    double dval;
    int ival;
    Settings * user_options = GetUserOptions();
    line=text->ReadLine();
    line=text->ReadLine();
    line=text->ReadLine();
    line=text->ReadLine();
    ival=atoi(line.AfterLast(' '));
    ElleSetSaveFrequency( ival );
    line=text->ReadLine();
    ival=atoi(line.AfterLast(' '));
    ElleSetStages( ival );
    line=text->ReadLine();
    dval=atof(line.AfterLast(' '));
    ElleSetSwitchdistance( dval );
    line=text->ReadLine();
    dval=atof(line.AfterLast(' '));
    ElleSetSpeedup( dval );
    line=text->ReadLine();
    line=line.AfterLast(' ');
    ElleSetSaveFileRoot( ( char * ) ( line.c_str() ) );
    line=text->ReadLine();
    line=line.AfterLast(' ');
    ival=atoi(line.c_str());
    user_options->SetOptionValue(O_UPDFREQ,(double*)&ival);
    line=text->ReadLine();
    line=line.AfterLast(' ');
    if(line=="TRUE")
        user_options->SetLogFile(true );
    else
        user_options->SetLogFile(false);
    line=text->ReadLine();
    line=line.AfterLast(' ');
    if(line=="TRUE")
        user_options->SetLogAppend(true );
    else
        user_options->SetLogAppend(false);
    line=text->ReadLine();
    line=line.AfterLast(' ');
    ival=atoi(line.c_str());
    SetLogLevel( ival );
}

void DSettings::SaveArchiveRunOpts(wxTextOutputStream &text)
{
    int vstages, vsfreq,displayfrequency,n;
    double vsdist, vsup;
    wxString val;
    UserData ud;
    ElleUserData(ud);
    text<< "-----------------------------------\n";
    text<< "RUNTIME OPTIONS\n" ;
    text<< "-----------------------------------\n" ;

    ElleGetRunOptionValue( RO_SVFREQ, & vsfreq );
    text<< "RO_SVFREQ: "<< vsfreq <<"\n";

    ElleGetRunOptionValue( RO_STAGES, & vstages );
    text<< "RO_STAGES: "<< vstages<<"\n" ;

    ElleGetRunOptionValue( RO_SWITCHDIST, ( int * ) & vsdist );
    text<< "RO_SWITCHDIST: "<< vsdist <<"\n";

    ElleGetRunOptionValue( RO_SPEEDUP, ( int * ) & vsup );
    text<< "RO_SPEEDUP: "<< vsup <<"\n";

    text<< "ELLSAVEFILEROOT: "<< ElleSaveFileRoot() <<"\n";

    Settings * user_options = GetUserOptions();

    user_options->GetUpdFreqOption(&displayfrequency);
    text<< "Disp_Freq: "<<displayfrequency<<"\n";
    text<< "Log_to_file: ";
    if(user_options->GetLogFile())
        text<<"TRUE" <<"\n";
    else
        text<<"FALSE" <<"\n";
    text<<"Append_to_log: ";
    if(user_options->GetLogAppend())
        text<<"TRUE" <<"\n";
    else
        text<<"FALSE" <<"\n";
    text<<"Loglevel: "<<GetLogLevel() <<"\n";
}

void DSettings::SaveArchiveUserData(wxTextOutputStream &text)
{
    UserData ud;
    int i;
    ElleUserData(ud);
    wxString test;
    text<< "-----------------------------------\n" ;
    text<< "UserData\n" ;
    text<< "-----------------------------------\n" ;
    for (i=0;i<MAX_U_DATA;i++) {
        text<< "UserData-"<<i<<": ";
        test.Printf("%E",ud[i]);
        text<<test<<"\n";
        //text<< ud[i]  <<"\n";
    }
}


void DSettings::SaveDSettings()
{
    wxString filename;
    filename=ElleFile();
    filename=filename.AfterLast( E_DIR_SEPARATOR );
    filename=filename.BeforeLast('.');
    filename.Append(".zip");
    filename = wxFileSelector( "Choose filename to save preferences", GetSaveFileDir(), filename, "zip",
                               "Preference-Files (*.zip)|*.zip|All Files (*)|*",wxSAVE );
    FILE * out;
    if ( !filename.IsEmpty() && filename.AfterLast( '.' )=="zip")
        SaveArchive(filename);
    else {
        filename.Append(".zip");
        SaveArchive(filename);
    }
}

bool DSettings::LoadDSettings()
{
    wxString filename;
    filename = wxFileSelector( "Choose filename to load preferences", "", "", "",
                               "Preference-Files (*.zip)|*.zip|All Files (*)|*",wxOPEN );
    if ( !filename.IsEmpty() && filename.AfterLast( '.' )=="zip") {
        LoadArchive(filename,true);
        return true;
    } else
        return false;
}

bool DSettings::GetUnodesRangeFlag()
{
    return rangeflagunodes;
}

double DSettings::GetUnodesRangeFlagMaxValue()
{
    return unodesrangeflagmaxvalue;
}

void DSettings::SetUnodesRangeFlagMaxValue( double x )
{
    unodesrangeflagmaxvalue = x;
}

double DSettings::GetUnodesRangeFlagMinValue()
{
    return unodesrangeflagminvalue;
}

void DSettings::SetUnodesRangeFlagMinValue( double x )
{
    unodesrangeflagminvalue = x;
}

void DSettings::SetUnodesNotRangeFlag( bool var )
{
    notrangeflagunodes = var;
}

bool DSettings::GetUnodesNotRangeFlag()
{
    return notrangeflagunodes;
}

double DSettings::GetUnodesNotRangeFlagMaxValue()
{
    return unodesnotrangeflagmaxvalue;
}

void DSettings::SetUnodesNotRangeFlagMaxValue( double x )
{
    unodesnotrangeflagmaxvalue = x;
}

double DSettings::GetUnodesNotRangeFlagMinValue()
{
    return unodesnotrangeflagminvalue;
}

void DSettings::SetUnodesNotRangeFlagMinValue( double x )
{
    unodesnotrangeflagminvalue = x;
}

bool DSettings::GetSegmentsRangeFlag()
{
    return rangeflagsegments;
}

double DSettings::GetSegmentsRangeFlagMaxValue()
{
    return segmentsrangeflagmaxvalue;
}

void DSettings::SetSegmentsRangeFlagMaxValue( double x )
{
    segmentsrangeflagmaxvalue = x;
}

double DSettings::GetSegmentsRangeFlagMinValue()
{
    return segmentsrangeflagminvalue;
}

void DSettings::SetSegmentsRangeFlagMinValue( double x )
{
    segmentsrangeflagminvalue = x;
}

void DSettings::SetSegmentsNotRangeFlag( bool var )
{
    notrangeflagsegments = var;
}

bool DSettings::GetSegmentsNotRangeFlag()
{
    return notrangeflagsegments;
}

double DSettings::GetSegmentsNotRangeFlagMaxValue()
{
    return segmentsnotrangeflagmaxvalue;
}

void DSettings::SetSegmentsNotRangeFlagMaxValue( double x )
{
    segmentsnotrangeflagmaxvalue = x;
}

double DSettings::GetSegmentsNotRangeFlagMinValue()
{
    return segmentsnotrangeflagminvalue;
}

void DSettings::SetSegmentsNotRangeFlagMinValue( double x )
{
    segmentsnotrangeflagminvalue = x;
}

int DSettings::GetNodeSize()
{
    return nodesize;
}

void DSettings::SetNodeSize( int size )
{
    nodesize = size;
}
void DSettings::SetUNodeSize(int size)
{
    unodesize=size;
}

int DSettings::GetUNodeSize()
{
    return unodesize;
}
void DSettings::SetSegmentSize(int size)
{
    segmentsize=size;
}

int DSettings::GetSegmentSize()
{
    return segmentsize;
}


int DSettings::GetLineSize()
{
    return linesize;
}

void DSettings::SetLineSize( int size )
{
    linesize = size;
}

void DSettings::GetDNodeColor( int * r, int * g, int * b )
{
    * r = dnodecolor[0];
    * g = dnodecolor[1];
    * b = dnodecolor[2];
}

void DSettings::SetDNodeColor( int r, int g, int b )
{
    dnodecolor[0] = r;
    dnodecolor[1] = g;
    dnodecolor[2] = b;
}

void DSettings::GetTNodeColor( int * r, int * g, int * b )
{
    * r = tnodecolor[0];
    * g = tnodecolor[1];
    * b = tnodecolor[2];
}

void DSettings::SetTNodeColor( int r, int g, int b )
{
    tnodecolor[0] = r;
    tnodecolor[1] = g;
    tnodecolor[2] = b;
}

void DSettings::SetLineColor( int r, int g, int b )
{
    linecolor[0] = r;
    linecolor[1] = g;
    linecolor[2] = b;
}

void DSettings::GetLineColor( int * r, int * g, int * b )
{
    * r = linecolor[0];
    * g = linecolor[1];
    * b = linecolor[2];
}

void DSettings::CmapChangeColor( int in, int r, int g, int b, int set)
{
    int ro,go,bo,x,n,steps;
    double rdif,gdif,bdif;
    cmap[in] [0] = r;
    cmap[in] [1] = g;
    cmap[in] [2] = b;
    if(set !=0)
        set=1;
    cmap[in] [3] = set;
    if(in!=0 && set==1)
        {
            x=in-1;
            while(x>0 && cmap[x][3]!=1)
                x--;
            steps=in-x;
            if(steps!=0)
            {
                ro=cmap[x][0];
                go=cmap[x][1];
                bo=cmap[x][2];
                rdif=cmap[in][0]-ro;
                gdif=cmap[in][1]-go;
                bdif=cmap[in][2]-bo;
                if(rdif!=0)
                    rdif=rdif/steps;
                if(gdif!=0)
                    gdif=gdif/steps;
                if(bdif!=0)
                    bdif=bdif/steps;
                         for(n=1;n <=steps;n++)
                         {
                             cmap[in-n][0]=(int)(cmap[in][0]-(n*rdif));
                             cmap[in-n][1]=(int)(cmap[in][1]-(n*gdif));
                             cmap[in-n][2]=(int)(cmap[in][2]-(n*bdif));
                         }
                 }
             }
         if(in!=256 && set==1)
         {
		 x=in+1;
            while(x<257 && cmap[x][3]!=1)
                x++;
            steps=x-in;
            if(steps!=0)
            {
                ro=cmap[x][0];
                go=cmap[x][1];
                bo=cmap[x][2];
                rdif=cmap[in][0]-ro;
                gdif=cmap[in][1]-go;
                bdif=cmap[in][2]-bo;
                if(rdif!=0)
                    rdif=rdif/steps;
                if(gdif!=0)
                    gdif=gdif/steps;
                if(bdif!=0)
                    bdif=bdif/steps;
                         for(n=1;n <=steps;n++)
                         {
                             cmap[in+n][0]=(int)(cmap[in][0]-(n*rdif));
                             cmap[in+n][1]=(int)(cmap[in][1]-(n*gdif));
                             cmap[in+n][2]=(int)(cmap[in][2]-(n*bdif));
                         }
                 }
	 }
}

void DSettings::CmapGetColor( int in, int * r, int * g, int * b, int * set)
{
    * r = cmap[in] [0];
    * g = cmap[in] [1];
    * b = cmap[in] [2];
    * set = cmap[in] [3];
}

void DSettings::CmapResetToDefault()
{
    //Initialisiere Color-Map
    cmap[0] [0] = 0; //r
    cmap[0] [1] = 0; //g
    cmap[0] [2] = 255; //b
    cmap[0] [3] = 1; //wenn 1 dann set, wenn 0 dann ignore

    cmap[255] [0] = 255; //r
    cmap[255] [1] = 0; //g
    cmap[255] [2] = 0; //b
    cmap[255] [3] = 1; //wenn 1 dann set, wenn 0 dann ignore

    for ( int n = 1; n < 255; n++ ) {
        cmap[n] [0] = n;
        cmap[n] [1] = 0;
        cmap[n] [2] = 255 - n;
        cmap[n] [3] = 0;
    }
}

int DSettings::CmapLoad( string filename )
{
    int n, set, pos, pos1, r, g, b;
    string line, separators=" ,\t";
    ifstream ffile( filename.c_str() );
    if ( ffile )
    {
        std::getline( ffile,line,'\n' );
//LE ffile.getline(char *,50) crashed program if more than
//     50 chars before the \n
// Needs error checking and simplifying
        for ( n = 0; n < 256; n++ ) //there are 256 colors in the file
        {
            std::getline( ffile,line,'\n' );
            pos = 0;
            pos1 = 0;
            pos = line.find_first_not_of( separators );
            pos1 = line.find_first_of( separators, pos );
            r = atoi( ( line.substr( pos, pos1 ) ).c_str() );
            pos = line.find_first_not_of( separators, pos1 );
            pos1 = line.find_first_of( separators, pos );
            g = atoi( ( line.substr( pos, pos1 ) ).c_str() );
            pos = line.find_first_not_of( separators, pos1 );
            pos1 = line.find_first_of( separators, pos );
            b = atoi( ( line.substr( pos, pos1 ) ).c_str() );
            pos = line.find_first_not_of( separators, pos1 );
            pos1 = line.find_first_of( separators, pos );
            set = atoi( ( line.substr( pos, pos1 ) ).c_str() );
            CmapChangeColor( n, r, g, b, set);
        }
        ffile.close();
    }
    /*Taken out, otherwise a default.cmap file is always written. Pain in the ass.
     */
    /*else
    {
      CmapResetToDefault();
      CmapSave( "default.cmap" );
    }*/
}

int DSettings::CmapSave( string filename )
{
    int n, set, r, g, b;
    FILE * file;
    if ( ( file = fopen( filename.c_str(), "w" ) ) ) {
        fprintf( file, "Colormap for Elle-files... Please do not edit\n" );
        for ( n = 0; n < 256; n++ ) {
            CmapGetColor( n, & r, & g, & b, & set
                            );
            fprintf( file, "%d %d %d %d\n", r, g, b, set
                       );
        }
        fclose( file );
    }
}

void DSettings::SetShowArgs( int key, bool state, int attrib )
{
    Settings * options = GetUserOptions();
    double amax, amin;
    switch ( key ) {
    case BNODES:
        if ( state ) {
            /*
            Alternative call for the 4 lines below.
                        options->SetOptionValue(O_NODEATTRIB,(XtPointer)&attrib);
            */
            options->node_attrib = attrib;
            ElleFindNodeAttributeRange( attrib, & amin, & amax );
            options->node_limits[0] = amin;
            options->node_limits[1] = amax;
        } else
            options->node_attrib = NONE;
        break;
    case BOUNDARIES:
        if ( state ) {
            /*
            Alternative call for the 7 lines below.
                        options->SetOptionValue(O_BNDATTRIB,(XtPointer)&attrib);
            */
            options->boundary_attrib = attrib;
            if (options->boundary_attrib==MISORIENTATION) {
                ElleFindBndAttributeRange(options->boundary_attrib,
                                          &amin, &amax);
                options->bnd_limits[0] = amin;
                options->bnd_limits[1] = amax;
            }
        } else
            options->boundary_attrib = NONE;
        break;
    case FLYNNS:
        if ( state ) {
            options->flynn_attrib = attrib;
            if (attrib==EULER_RGB) {
                amin=0; amax=256;
            }
            else {
                ElleFindFlynnAttributeRange( attrib, & amin, & amax );
            }
            options->flynn_limits[0] = amin;
            options->flynn_limits[1] = amax;
        } else
            options->flynn_attrib = NONE;
        break;
    case UNODES:
        if ( state ) {
            options->unode_attrib = attrib;
            if (attrib==EULER_RGB) {
                amin=0; amax=256;
            }
            else {
                ElleFindUnodeAttributeRange(attrib, &amin, &amax);
            }
            options->unode_limits[0]=amin;
            options->unode_limits[1]=amax;
        } else
            options->unode_attrib = NONE;
        break;
    case SEGMENTS:
        if ( state ) {
            options->segment_attrib = attrib;
            ElleFindSegmentAttributeRange(attrib, &amin, &amax);
            options->segment_limits[0]=amin;
            options->segment_limits[1]=amax;
        } else
            options->segment_attrib = NONE;
        break;

    default:
        break;
    }
}


int DSettings::GetShowArgs( int key )
{
    Settings * options = GetUserOptions();
    switch ( key ) {
    case BNODES:
        return ( options->node_attrib );
        break;
    case BOUNDARIES:
        return ( options->boundary_attrib );
        break;
    case FLYNNS:
        return ( options->flynn_attrib );
        break;
    case UNODES:
        return ( options->unode_attrib );
        break;
    case SEGMENTS:
        return ( options->segment_attrib );
        break;
    default:
        break;
    }
}

void DSettings::SetLoadFileDir( wxString filed )
{
    loadfiledir = filed;
}

wxString DSettings::GetLoadFileDir()
{
    return loadfiledir;
}

void DSettings::SetSaveFileDir( wxString filed )
{
    savefiledir = filed;
}

wxString DSettings::GetSaveFileDir()
{
    return savefiledir;
}

void DSettings::SetClampColor(int x,bool f)
{
    if(x==0)
        fclampcolor=f;
    if(x==1)
        bclampcolor=f;
    if(x==2)
        nclampcolor=f;
    if(x==3)
        uclampcolor=f;
    if(x==4)
        sclampcolor=f;
}


bool DSettings::GetClampColor(int x)
{
    if(x==0)
        return fclampcolor;
    if(x==1)
        return bclampcolor;
    if(x==2)
        return nclampcolor;
    if(x==3)
        return uclampcolor;
    if(x==4)
        return sclampcolor;
}

void DSettings::SetClampColor(int x,bool f,double min,double max)
{
    if(x==0) {
        fclampcolor=f;
        fccmin=min;
        fccmax=max;
    }
    if(x==1) {
        bclampcolor=f;
        bccmin=min;
        bccmax=max;
    }
    if(x==2) {
        nclampcolor=f;
        nccmin=min;
        nccmax=max;
    }
    if(x==3) {
        uclampcolor=f;
        uccmin=min;
        uccmax=max;
    }
    if(x==4) {
        sclampcolor=f;
        sccmin=min;
        sccmax=max;
    }
}
bool DSettings::GetClampColor(int x,double *min,double *max)
{
    //0=Flynns,1=boundaries,2=nodes,3=unodes,4=segments
    if(x==0) {
        *min=fccmin;
        *max=fccmax;
        return fclampcolor;
    }
    if(x==1) {
        *min=bccmin;
        *max=bccmax;
        return bclampcolor;
    }
    if(x==2) {
        *min=nccmin;
        *max=nccmax;
        return nclampcolor;
    }
    if(x==3) {
        *min=uccmin;
        *max=uccmax;
        return uclampcolor;
    }
    if(x==4) {
        *min=sccmin;
        *max=sccmax;
        return sclampcolor;
    }
}
