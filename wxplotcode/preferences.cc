// Written in 2003
// Author: Dr. J.K. Becker
// Copyright: Dr. J.K. Becker (becker@jkbecker.de)

#include "preferences.h"
#include "dsettings.h"
#include "display.h"
#include "showelle.h"

extern DSettings * GetDisplayOptions();

IMPLEMENT_CLASS( Preferences, wxDialog )

BEGIN_EVENT_TABLE( Preferences, wxDialog )
EVT_BUTTON( ID_PApply, Preferences::OnApply )
EVT_BUTTON( ID_PCancel, Preferences::OnCancel )
EVT_BUTTON( ID_POk, Preferences::OnOk )
EVT_BUTTON( ID_PLoad, Preferences::OnLoadPrefs )
EVT_BUTTON( ID_PSave, Preferences::OnSavePrefs )
EVT_BUTTON( ID_PDColor, Preferences::OnDNodeColor )
EVT_BUTTON( ID_PLColor, Preferences::OnLineColor )
EVT_BUTTON( ID_PTColor, Preferences::OnTNodeColor )
EVT_SPINCTRL( ID_Pbnodesize, Preferences::OnBNodeSize )
EVT_SPINCTRL( ID_Plinesize, Preferences::OnLineSize )
EVT_CHECKBOX( ID_Pflynnshownumbers, Preferences::OnFlynnShowNumbers )
EVT_SPINCTRL( ID_Punodesize, Preferences::OnUNodeSize )
EVT_SPINCTRL( ID_Psegmentsize, Preferences::OnSegmentSize )
EVT_SPINCTRL( ID_Prainchange, Preferences::OnRainChange )
EVT_CHECKBOX( ID_Pnodeshownumbers, Preferences::OnNodeShowNumbers )
EVT_CHECKBOX( ID_Prangeflagunodes, Preferences::OnRangeFlagUnodes )
EVT_CHECKBOX( ID_Prangeflagsegments, Preferences::OnRangeFlagSegments )
EVT_CLOSE( Preferences::OnClose )
// LE EVT_PAINT( Preferences::OnPaint )
EVT_CHECKBOX( ID_Pnotrangeflagunodes, Preferences::OnNotRangeFlagUnodes )
EVT_CHECKBOX( ID_Pnotrangeflagsegments, Preferences::OnNotRangeFlagSegments )
EVT_BUTTON( ID_Pcolorsingleflynn, Preferences::OnColorSingleFlynn )
EVT_BUTTON( ID_PResetSingleFlynnColor, Preferences::OnResetSingleFlynn )
EVT_BUTTON( ID_Unodes3d, Preferences::OnUnodes3d )
EVT_CHECKBOX(ID_Punodetriangles, Preferences::OnUNodesTriangulate )
EVT_TEXT( ID_punoderangeflagmaxval, Preferences::OnUnodesRangeFlagTxtMax )
EVT_TEXT( ID_punoderangeflagminval, Preferences::OnUnodesRangeFlagTxtMin )
EVT_TEXT( ID_punodenotrangeflagmaxval, Preferences::OnUnodesNotRangeFlagTxtMax )
EVT_TEXT( ID_psegmentrangeflagmaxval, Preferences::OnSegmentsRangeFlagTxtMax )
EVT_TEXT( ID_psegmentrangeflagminval, Preferences::OnSegmentsRangeFlagTxtMin )
EVT_TEXT( ID_psegmentnotrangeflagmaxval, Preferences::OnSegmentsNotRangeFlagTxtMax )
EVT_LISTBOX( ID_pbnodelistbox, Preferences::OnListBox )
EVT_LISTBOX( ID_pflynnlistbox, Preferences::OnListBox )
EVT_LISTBOX( ID_punodelistbox, Preferences::OnListBox )
EVT_LISTBOX( ID_psegmentlistbox, Preferences::OnListBox )
EVT_LISTBOX( ID_plinelistbox, Preferences::OnListBox )
EVT_TEXT( ID_punodenotrangeflagminval, Preferences::OnUnodesNotRangeFlagTxtMin )
EVT_TEXT( ID_psegmentnotrangeflagminval, Preferences::OnSegmentsNotRangeFlagTxtMin )
EVT_TEXT_ENTER( ID_TEXTCTRL, Preferences::ColormapOnChangeColorPercent )
EVT_TEXT_ENTER( ID_CTEXT, Preferences::ColormapOnChangeColorValue )
EVT_BUTTON( ID_DEFAULT, Preferences::ColormapOnDefault )
EVT_BUTTON( ID_CLOAD, Preferences::ColormapOnLoad )
EVT_BUTTON( ID_CSAVE, Preferences::ColormapOnSave )
EVT_CHOICE( ID_sflynn, Preferences::ColormapOnRangeFlynn )
EVT_CHOICE( ID_sbnode, Preferences::ColormapOnRangeBNode )
EVT_CHOICE( ID_sunode, Preferences::ColormapOnRangeUNode )
EVT_CHOICE( ID_ssegment, Preferences::ColormapOnRangeSegment )
EVT_CHECKBOX( ID_FClampColor, Preferences::OnFClampColor )
EVT_CHECKBOX( ID_BClampColor, Preferences::OnBClampColor )
EVT_CHECKBOX( ID_NClampColor, Preferences::OnNClampColor )
EVT_CHECKBOX( ID_UClampColor, Preferences::OnUClampColor )
EVT_CHECKBOX( ID_SClampColor, Preferences::OnSClampColor )
EVT_TEXT( ID_FCCMin, Preferences::OnFCCMin )
EVT_TEXT( ID_FCCMax, Preferences::OnFCCMax)
EVT_TEXT( ID_BCCMin, Preferences::OnBCCMin )
EVT_TEXT( ID_BCCMax, Preferences::OnBCCMax)
EVT_TEXT( ID_NCCMin, Preferences::OnNCCMin )
EVT_TEXT( ID_NCCMax, Preferences::OnNCCMax)
EVT_TEXT( ID_UCCMin, Preferences::OnUCCMin )
EVT_TEXT( ID_UCCMax, Preferences::OnUCCMax)
EVT_TEXT( ID_SCCMin, Preferences::OnSCCMin )
EVT_TEXT( ID_SCCMax, Preferences::OnSCCMax)
END_EVENT_TABLE()

IMPLEMENT_CLASS( CBarPane, wxPanel )

BEGIN_EVENT_TABLE( CBarPane, wxPanel )
EVT_PAINT(CBarPane::paintEvent)
END_EVENT_TABLE()

IMPLEMENT_CLASS( LinePane, wxPanel )

BEGIN_EVENT_TABLE( LinePane, wxPanel )
EVT_PAINT(LinePane::paintEvent)
END_EVENT_TABLE()

IMPLEMENT_CLASS( DotPane, wxPanel )

BEGIN_EVENT_TABLE( DotPane, wxPanel )
EVT_PAINT(DotPane::paintEvent)
END_EVENT_TABLE()


CBarPane::CBarPane(wxWindow *parent, const wxSize barsize) :
wxPanel(parent,-1,wxDefaultPosition,barsize)
{
}

LinePane::LinePane(wxWindow *parent, const wxSize linesize, int wdth,
wxColor col) : wxPanel(parent,-1,wxDefaultPosition,linesize)
{
    linewdth = wdth;
    linecol = col;
}

DotPane::DotPane(wxWindow *parent, const wxSize panesize, int size,
wxColor col) : wxPanel(parent,-1,wxDefaultPosition,panesize)
{
    dotsize = size;
    dotcol = col;
}

// Initialize everyting. Set all default values for colors and diameters and so on and show the
//preferences window. Oh, the buttons and stuff are defined here too!
Preferences::Preferences( wxWindow * parent ) :wxDialog( parent, PREFSWIN, wxT("Preferences"), wxPoint( 100, 100 ), wxSize( 500, 480 ),wxCAPTION | wxSYSTEM_MENU )
{
    wxMemoryDC bnodedc, linedc, tnodedc;
    DSettings * doptions = GetDisplayOptions();
    int r, g, b, tr, tg, tb;
    Settings * user_options = GetUserOptions();
    if(user_options->draw_nodes)
        triangulate=true;
    else
        triangulate=false;

    unodesize=doptions->GetUNodeSize();
    segmentsize=doptions->GetSegmentSize();
    attrib = doptions->GetShowArgs( FLYNNS );
    useunodea = doptions->GetShowArgs( UNODES );
    usebnodea = doptions->GetShowArgs( BNODES );
    usesegmenta = doptions->GetShowArgs( SEGMENTS );
    rainchange=doptions->GetRainChange();
    urangevalmax = doptions->GetUnodesRangeFlagMaxValue();
    urangevalmin = doptions->GetUnodesRangeFlagMinValue();
    prangeflagunodes = doptions->GetUnodesRangeFlag();
    unotrangevalmax = doptions->GetUnodesNotRangeFlagMaxValue();
    unotrangevalmin = doptions->GetUnodesNotRangeFlagMinValue();;
    pnotrangeflagunodes = doptions->GetUnodesNotRangeFlag();
    unodes3d = false;
    //LE triangulate=false;
    srangevalmax = doptions->GetSegmentsRangeFlagMaxValue();
    srangevalmin = doptions->GetSegmentsRangeFlagMinValue();
    prangeflagsegments = doptions->GetSegmentsRangeFlag();
    snotrangevalmax = doptions->GetSegmentsNotRangeFlagMaxValue();
    snotrangevalmin = doptions->GetSegmentsNotRangeFlagMinValue();;
    pnotrangeflagsegments = doptions->GetSegmentsNotRangeFlag();

    fclampcolor=doptions->GetClampColor(0,&fccmin,&fccmax);
    bclampcolor=doptions->GetClampColor(1,&bccmin,&bccmax);
    nclampcolor=doptions->GetClampColor(2,&nccmin,&nccmax);
    uclampcolor=doptions->GetClampColor(3,&uccmin,&uccmax);
    sclampcolor=doptions->GetClampColor(4,&sccmin,&sccmax);

    wxBoxSizer * base = new wxBoxSizer( wxVERTICAL );

    wxBoxSizer * erster = new wxBoxSizer( wxHORIZONTAL );
    base->Add( erster, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    wxButton * ok = new wxButton( this, ID_POk, wxT("OK"), wxDefaultPosition, wxDefaultSize );
    wxButton * apply = new wxButton( this, ID_PApply, wxT("Apply"), wxDefaultPosition, wxDefaultSize );
    wxButton * cancel = new wxButton( this, ID_PCancel, wxT("Cancel"), wxDefaultPosition, wxDefaultSize );
    wxButton * save = new wxButton( this, ID_PSave, wxT("Save"), wxDefaultPosition, wxDefaultSize );
    wxButton * load = new wxButton( this, ID_PLoad, wxT("Load"), wxDefaultPosition, wxDefaultSize );
    erster->Add( ok, 0, wxALIGN_LEFT | wxALL, 5 );
    erster->Add( apply, 0, wxALIGN_LEFT | wxALL, 5 );
    erster->Add( cancel, 0, wxALIGN_LEFT | wxALL, 5 );
    erster->Add( save, 0, wxALIGN_LEFT | wxALL, 5 );
    erster->Add( load, 0, wxALIGN_LEFT | wxALL, 5 );
    wxBoxSizer * zweiter = new wxBoxSizer( wxHORIZONTAL );
    base->Add( zweiter, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );

    book = new wxNotebook( this, -1, wxDefaultPosition, wxSize(500,500) );
    wxPanel * p = ( wxPanel * ) NULL;
    p = CreateFlynnPage();
    book->AddPage( p, wxT("Flynns"), FALSE );
    p = CreateLinePage();
    book->AddPage( p, wxT("Boundaries"), FALSE );
    p = CreateBNodePage();
    book->AddPage( p, wxT("BNodes"), FALSE );
    p = CreateUNodePage();
    book->AddPage( p, wxT("UNodes"), FALSE );
    p = CreateSegmentPage();
    book->AddPage( p, wxT("Segments"), FALSE );
    p=ColormapCreateControls();
    book->AddPage( p, wxT("Colormap"), FALSE );
    zweiter->Add( book, 0, wxALIGN_LEFT | wxALL, 5 );
    this->SetSizer( base );
    this->Fit();
    this->SetAutoLayout( TRUE );
    this->Refresh( true, NULL ); //LE
}

void Preferences::OnListBox( wxCommandEvent & event )
{
    int in;
    wxString attrib;
    double max=0, min=0;

    // LE
    /*
     * This fn is called by all the attribute list boxes
     * if an attribute is chosen, the min max revert
     * to the range in the file
     */
    switch(event.GetId())
    {
    case ID_pbnodelistbox:
        if ( ( in = cbnode->GetSelection() ) != 0 )
        {
            attrib = cbnode->GetString( in );
		in = GetAttributeInt( attrib,BNODES );

            ElleFindNodeAttributeRange( in, & min, & max );
            attrib = "";
            attrib.Printf( "%lf", max );
            Nccmax->SetValue( attrib );
            attrib = "";
            attrib.Printf( "%lf", min );
            Nccmin->SetValue( attrib );
        }
        break;
    case ID_pflynnlistbox:
        if ( ( in = cflynn->GetSelection() ) != 0 )
        {
            attrib = cflynn->GetString( in );
		in = GetAttributeInt( attrib,FLYNNS );
            if (in==EULER_RGB) {
              min=0;
              max=256;
            }
            else {
            ElleFindFlynnAttributeRange( in, & min, & max );
            }
            attrib = "";
            attrib.Printf( "%lf", max );
            Fccmax->SetValue( attrib );
            attrib = "";
            attrib.Printf( "%lf", min );
            Fccmin->SetValue( attrib );
        }
        break;
    case ID_punodelistbox:
        if ( ( in = cunode->GetSelection() ) != 0 )
        {
            attrib = cunode->GetString( in );
        in = GetAttributeInt( attrib,UNODES );
            if (in==EULER_RGB) {
              min=0;
              max=256;
            }
            else {
            ElleFindUnodeAttributeRange( in, & min, & max );
            }
            attrib = "";
            attrib.Printf( "%-10.3lg", max );
            uvalmax->SetValue( attrib );
            Uccmax->SetValue( attrib );
            attrib = "";
            attrib.Printf( "%-10.3lg", min );
            uvalmin->SetValue( attrib );
            Uccmin->SetValue( attrib );
        }
        break;
    case ID_psegmentlistbox:
        if ( ( in = csegment->GetSelection() ) != 0 )
        {
            attrib = csegment->GetString( in );
        in = GetAttributeInt( attrib,SEGMENTS );
            ElleFindSegmentAttributeRange( in, & min, & max );
            attrib = "";
            attrib.Printf( "%-10.3lg", max );
            svalmax->SetValue( attrib );
            Sccmax->SetValue( attrib );
            attrib = "";
            attrib.Printf( "%-10.3lg", min );
            svalmin->SetValue( attrib );
            Sccmin->SetValue( attrib );
        }
        break;
    case ID_plinelistbox:
        if ( ( in = cline->GetSelection() ) != 0 )
        {
            attrib = cline->GetString( in );
		in = GetAttributeInt( attrib,BOUNDARIES );

            ElleFindBndAttributeRange( in, & min, & max );
            attrib = "";
            attrib.Printf( "%lf", max );
            Bccmax->SetValue( attrib );
            attrib = "";
            attrib.Printf( "%lf", min );
            Bccmin->SetValue( attrib );
        }
        break;
    default: break;
    }
}

void Preferences::OnLoadPrefs( wxCommandEvent & event )
{
    DSettings * doptions = GetDisplayOptions();
    doptions->LoadDSettings();
    if(doptions->LoadDSettings())
    {
        cbar->paintNow();
        linecolsize->paintNow();
        ddotcolsize->paintNow();
        tdotcolsize->paintNow();
        EllePlotRegions( 0 );
        wxMessageBox( wxT("Preferences loaded successfully!"), wxT("This is an important message!"), wxOK | wxICON_INFORMATION );
        Show(false);
    }
}

void Preferences::OnSavePrefs( wxCommandEvent & event )
{
    DSettings * doptions = GetDisplayOptions();
    doptions->SaveDSettings();
}

void Preferences::OnClose(wxCloseEvent &event)
{
    Destroy();
}

wxPanel * Preferences::CreateUNodePage( void )
{
    wxPanel * pppanel = new wxPanel( book );
    wxString mm;
    int x, count, * welche=0;
    wxStaticText * stext = new wxStaticText( pppanel, -1, wxT("UNode size") );
    mm.Printf( "%d", unodesize );
    wxSpinCtrl * unodespin = new wxSpinCtrl( pppanel, ID_Punodesize, mm, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 10, 2 );
    cunode = new wxListBox( pppanel, ID_punodelistbox, wxDefaultPosition, wxDefaultSize, 0, NULL, wxLB_SINGLE);
    ///////////////////////////////////////////////////////
    //append attributes to list selector for unodes
    cunode->Append( wxT("NONE") );
    if ( ElleUnodesActive() ) {
        ElleUnodeAttributeList( & welche, & count );
        cunode->Append( wxT("LOCATION") );
        for ( x = 0; x < count; x++ )
        {
		if ( welche[x]==E3_ALPHA||welche[x]==EULER_3)
          cunode->Append( GetAttributeName( EULER_RGB,UNODES ) );
		cunode->Append( GetAttributeName( welche[x],UNODES ) );
        }
    }
    else {
        useunodea = NONE;
    }
    wxString name = GetAttributeName( useunodea,UNODES );
    int number = cunode->FindString( name );
    cunode->SetSelection( number, true );
    if (welche) free( welche );
    ///////////////////////////////////////////////////////////
    wxCheckBox * utriangle = new wxCheckBox( pppanel,
ID_Punodetriangles, wxT("Triangulate unodes"), wxDefaultPosition, wxDefaultSize );
    utriangle->SetValue( triangulate );
    //wxButton * single = new wxButton( pppanel, ID_Unodes3d, wxT("Show UNode-surface"), wxDefaultPosition, wxDefaultSize );
    wxBoxSizer * fuenfter = new wxBoxSizer( wxVERTICAL );
    wxBoxSizer * unflag = new wxBoxSizer( wxHORIZONTAL );
    wxBoxSizer * left = new wxBoxSizer( wxHORIZONTAL );
    wxBoxSizer * left1 = new wxBoxSizer( wxVERTICAL );
    //show only unodes with between these values
    wxCheckBox * flagunodes =new wxCheckBox( pppanel,
ID_Prangeflagunodes, wxT("Only draw unodes with values between"), wxDefaultPosition, wxDefaultSize );
    flagunodes->SetValue( prangeflagunodes );
    mm = wxT("");
    mm.Printf( "%-10.3lg", urangevalmin );
    uvalmin = new wxTextCtrl( pppanel, ID_punoderangeflagminval, mm, wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
    uvalmin->SetValue( mm );
    mm = wxT("");
    mm.Printf( "%-10.3lg", urangevalmax );
    uvalmax = new wxTextCtrl( pppanel, ID_punoderangeflagmaxval, mm, wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
    uvalmax->SetValue( mm );
    wxStaticText * tx = new wxStaticText( pppanel, -1, wxT("and") );

    unflag->Add( uvalmin, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    unflag->Add( tx, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    unflag->Add( uvalmax, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );

    //do NOT show unodes with between these values
    wxBoxSizer * notunflag = new wxBoxSizer( wxHORIZONTAL );
    wxCheckBox * notflagunodes =new wxCheckBox( pppanel,
				ID_Pnotrangeflagunodes,
				wxT("Do NOT draw unodes with values between"),
				wxDefaultPosition, wxDefaultSize );
    notflagunodes->SetValue( pnotrangeflagunodes );
    mm = wxT("");
    mm.Printf( "%-10.3lg", unotrangevalmin );
    wxTextCtrl * notuvalmin =new wxTextCtrl( pppanel, ID_punodenotrangeflagminval, mm, wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
    notuvalmin->SetValue( mm );
    mm = wxT("");
    mm.Printf( "%-10.3lg", unotrangevalmax );
    wxTextCtrl * notuvalmax =new wxTextCtrl( pppanel, ID_punodenotrangeflagmaxval, mm, wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
    notuvalmax->SetValue( mm );
    wxStaticText * nottx = new wxStaticText( pppanel, -1, wxT("and") );

    notunflag->Add( notuvalmin, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    notunflag->Add( nottx, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    notunflag->Add( notuvalmax, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );

    left1->Add( stext, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    left1->Add( unodespin, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    left->Add( left1, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    left->Add( cunode, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    fuenfter->Add( left, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    fuenfter->Add( utriangle, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    //fuenfter->Add( single, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    fuenfter->Add( flagunodes, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    fuenfter->Add( unflag, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    fuenfter->Add( notflagunodes, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    fuenfter->Add( notunflag, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );

    wxBoxSizer * v2 = new wxBoxSizer( wxHORIZONTAL );
    wxCheckBox * ccolor =new wxCheckBox( pppanel, ID_UClampColor,
wxT("Clamp color between"), wxDefaultPosition, wxDefaultSize );
    ccolor->SetValue( uclampcolor );
    Uccmin = new wxTextCtrl( pppanel, ID_UCCMin, mm, wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
    mm.Printf("%-10.3lg",uccmin);
    Uccmin->SetValue( mm );
    Uccmax = new wxTextCtrl( pppanel, ID_UCCMax, mm, wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
    mm.Printf("%-10.3lg",uccmax);
    Uccmax->SetValue( mm );
    wxStaticText * tx1 = new wxStaticText( pppanel, -1, wxT("and") );

    v2->Add( ccolor, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    v2->Add( Uccmin, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    v2->Add( tx1, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    v2->Add( Uccmax, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    fuenfter->Add( v2, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );

    pppanel->SetSizer( fuenfter );
    return pppanel;
}

void Preferences::OnUNodesTriangulate(wxCommandEvent &event)
{
    triangulate = event.IsChecked();
}

void Preferences::OnUnodesRangeFlagTxtMax( wxCommandEvent & event )
{
    ( ( ( wxTextCtrl * ) event.GetEventObject() )->GetValue() ).ToDouble( & urangevalmax );
}

void Preferences::OnUnodesRangeFlagTxtMin( wxCommandEvent & event )
{
    ( ( ( wxTextCtrl * ) event.GetEventObject() )->GetValue() ).ToDouble( & urangevalmin );
}

void Preferences::OnUnodesNotRangeFlagTxtMax( wxCommandEvent & event )
{
    ( ( ( wxTextCtrl * ) event.GetEventObject() )->GetValue() ).ToDouble( & unotrangevalmax );
}

void Preferences::OnUnodesNotRangeFlagTxtMin( wxCommandEvent & event )
{
    ( ( ( wxTextCtrl * ) event.GetEventObject() )->GetValue() ).ToDouble( & unotrangevalmin );
}

void Preferences::OnUnodes3d(wxCommandEvent & event)
{
    unodes3d = true;
    DSettings * doptions = GetDisplayOptions();
    useunodea = GetAttributeInt( cunode->GetStringSelection(),UNODES );
}

wxPanel * Preferences::CreateSegmentPage( void )
{
    wxPanel * pppanel = new wxPanel( book );
    wxString mm;
    int x, count, * welche=0;
    wxStaticText * stext = new wxStaticText( pppanel, -1, wxT("Segment size") );
    mm.Printf( "%d", segmentsize );
    wxSpinCtrl * segmentspin = new wxSpinCtrl( pppanel, ID_Psegmentsize, mm, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 10, 2 );
    csegment = new wxListBox( pppanel, ID_psegmentlistbox, wxDefaultPosition, wxDefaultSize, 0, NULL, wxLB_SINGLE);
    ///////////////////////////////////////////////////////
    //append attributes to list selector for segments
    csegment->Append( wxT("NONE") );
    if ( ElleMaxSegments()>0 ) {
        ElleSegmentAttributeList( & welche, & count );
        csegment->Append( wxT("LOCATION") );
        for ( x = 0; x < count; x++ )
        {
		csegment->Append( GetAttributeName( welche[x],SEGMENTS ) );
        }
    }
    else {
        usesegmenta = NONE;
    }
    wxString name = GetAttributeName( usesegmenta,SEGMENTS );
    int number = csegment->FindString( name );
    csegment->SetSelection( number, true );
    if (welche) free( welche );
    ///////////////////////////////////////////////////////////
    wxBoxSizer * fuenfter = new wxBoxSizer( wxVERTICAL );
    wxBoxSizer * snflag = new wxBoxSizer( wxHORIZONTAL );
    wxBoxSizer * left = new wxBoxSizer( wxHORIZONTAL );
    wxBoxSizer * left1 = new wxBoxSizer( wxVERTICAL );
    //show only segments with between these values
    wxCheckBox * flagsegments =new wxCheckBox( pppanel,
ID_Prangeflagsegments, wxT("Only draw segments with values between"), wxDefaultPosition, wxDefaultSize );
    flagsegments->SetValue( prangeflagsegments );
    mm = wxT("");
    mm.Printf( "%-10.3lg", srangevalmin );
    svalmin = new wxTextCtrl( pppanel, ID_psegmentrangeflagminval, mm, wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
    svalmin->SetValue( mm );
    mm = wxT("");
    mm.Printf( "%-10.3lg", srangevalmax );
    svalmax = new wxTextCtrl( pppanel, ID_psegmentrangeflagmaxval, mm, wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
    svalmax->SetValue( mm );
    wxStaticText * tx = new wxStaticText( pppanel, -1, wxT("and") );

    snflag->Add( svalmin, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    snflag->Add( tx, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    snflag->Add( svalmax, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );

    //do NOT show segments with between these values
    wxBoxSizer * notsnflag = new wxBoxSizer( wxHORIZONTAL );
    wxCheckBox * notflagsegments =new wxCheckBox( pppanel,
				ID_Pnotrangeflagsegments,
				wxT("Do NOT draw segments with values between"),
				wxDefaultPosition, wxDefaultSize );
    notflagsegments->SetValue( pnotrangeflagsegments );
    mm = wxT("");
    mm.Printf( "%-10.3lg", snotrangevalmin );
    wxTextCtrl * notsvalmin =new wxTextCtrl( pppanel, ID_psegmentnotrangeflagminval, mm, wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
    notsvalmin->SetValue( mm );
    mm = wxT("");
    mm.Printf( "%-10.3lg", snotrangevalmax );
    wxTextCtrl * notsvalmax =new wxTextCtrl( pppanel, ID_psegmentnotrangeflagmaxval, mm, wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
    notsvalmax->SetValue( mm );
    wxStaticText * nottx = new wxStaticText( pppanel, -1, wxT("and") );

    notsnflag->Add( notsvalmin, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    notsnflag->Add( nottx, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    notsnflag->Add( notsvalmax, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );

    left1->Add( stext, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    left1->Add( segmentspin, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    left->Add( left1, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    left->Add( csegment, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    fuenfter->Add( left, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    //fuenfter->Add( single, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    fuenfter->Add( flagsegments, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    fuenfter->Add( snflag, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    fuenfter->Add( notflagsegments, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    fuenfter->Add( notsnflag, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );

    wxBoxSizer * v2 = new wxBoxSizer( wxHORIZONTAL );
    wxCheckBox * ccolor =new wxCheckBox( pppanel, ID_SClampColor,
wxT("Clamp color between"), wxDefaultPosition, wxDefaultSize );
    ccolor->SetValue( sclampcolor );
    Sccmin = new wxTextCtrl( pppanel, ID_SCCMin, mm, wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
    mm.Printf("%-10.3lg",sccmin);
    Sccmin->SetValue( mm );
    Sccmax = new wxTextCtrl( pppanel, ID_SCCMax, mm, wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
    mm.Printf("%-10.3lg",sccmax);
    Sccmax->SetValue( mm );
    wxStaticText * tx1 = new wxStaticText( pppanel, -1, wxT("and") );

    v2->Add( ccolor, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    v2->Add( Sccmin, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    v2->Add( tx1, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    v2->Add( Sccmax, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    fuenfter->Add( v2, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );

    pppanel->SetSizer( fuenfter );
    return pppanel;
}

void Preferences::OnSegmentsRangeFlagTxtMax( wxCommandEvent & event )
{
    ( ( ( wxTextCtrl * ) event.GetEventObject() )->GetValue() ).ToDouble( & srangevalmax );
}

void Preferences::OnSegmentsRangeFlagTxtMin( wxCommandEvent & event )
{
    ( ( ( wxTextCtrl * ) event.GetEventObject() )->GetValue() ).ToDouble( & srangevalmin );
}

void Preferences::OnSegmentsNotRangeFlagTxtMax( wxCommandEvent & event )
{
    ( ( ( wxTextCtrl * ) event.GetEventObject() )->GetValue() ).ToDouble( & snotrangevalmax );
}

void Preferences::OnSegmentsNotRangeFlagTxtMin( wxCommandEvent & event )
{
    ( ( ( wxTextCtrl * ) event.GetEventObject() )->GetValue() ).ToDouble( & snotrangevalmin );
}

wxPanel * Preferences::CreateFlynnPage( void )
{

    wxPanel * fpanel = new wxPanel( book );
    wxBoxSizer * vierter = new wxBoxSizer( wxVERTICAL );
    wxBoxSizer * v1 = new wxBoxSizer( wxHORIZONTAL );
    wxBoxSizer * v2 = new wxBoxSizer( wxHORIZONTAL );
    wxCheckBox * pfshownumbers = new wxCheckBox( fpanel,
ID_Pflynnshownumbers, wxT("Show Numbers"), wxDefaultPosition, wxDefaultSize );
    pfshownumbers->SetValue( false );
    pflynnshownumbers = false;
    wxButton * single = new wxButton( fpanel, ID_Pcolorsingleflynn,
wxT("Color Single\nFlynn"), wxDefaultPosition, wxDefaultSize );
    wxButton * rsingle = new
                         wxButton( fpanel, ID_PResetSingleFlynnColor,
wxT("Reset Single\nFlynn"), wxDefaultPosition, wxDefaultSize );
    cflynn = new wxListBox( fpanel, ID_pflynnlistbox, wxDefaultPosition, wxDefaultSize, 0, NULL, wxLB_SINGLE );
    //////////////////////////////////////////////////////////
    int x, count, * welche=0;
    ElleFlynnDfltAttributeList( & welche, & count );
    cflynn->Append( wxT("NONE") );
    for ( x = 0; x < count; x++ )
    {
		if ( welche[x]==E3_ALPHA||welche[x]==EULER_3)
          cflynn->Append( GetAttributeName( EULER_RGB,FLYNNS ) );
		cflynn->Append( GetAttributeName( welche[x],FLYNNS ) );
    }
	wxString name = GetAttributeName( attrib,FLYNNS );
    int number = cflynn->FindString( name );
    cflynn->SetSelection( number, true );
    if (welche) free( welche );
    ////////////////////////////////////////////////////////////
    wxString mm;
    wxCheckBox * ccolor =new wxCheckBox( fpanel, ID_FClampColor,
wxT("Clamp color between"), wxDefaultPosition, wxDefaultSize );
    ccolor->SetValue( fclampcolor );
    Fccmin = new wxTextCtrl( fpanel, ID_FCCMin, mm, wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
    mm.Printf("%lf",fccmin);
    Fccmin->SetValue( mm );
    Fccmax = new wxTextCtrl( fpanel, ID_FCCMax, mm, wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
    mm.Printf("%lf",fccmax);
    Fccmax->SetValue( mm );
    wxStaticText * tx = new wxStaticText( fpanel, -1, wxT("and") );

    v2->Add( ccolor, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    v2->Add( Fccmin, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    v2->Add( tx, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    v2->Add( Fccmax, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );

    vierter->Add( pfshownumbers, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    vierter->Add( cflynn, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    v1->Add( single, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    v1->Add( rsingle, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    vierter->Add( v1, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    vierter->Add( v2, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    fpanel->SetSizer( vierter );
    return fpanel;
}

wxPanel * Preferences::CreateLinePage( void )
{
    int wdth, r, g, b;
    wxColour col;
    DSettings * doptions = GetDisplayOptions();
    wxPanel * lpanel = new wxPanel( book );

    wxBoxSizer * dritter = new wxBoxSizer( wxVERTICAL );
    wxBoxSizer * v2 = new wxBoxSizer( wxHORIZONTAL );
    wxBoxSizer * v1 = new wxBoxSizer( wxVERTICAL );
    wxBoxSizer * d1 = new wxBoxSizer( wxVERTICAL );
    wxBoxSizer * d2 = new wxBoxSizer( wxHORIZONTAL );
    dritter->Add( d2, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    d2->Add( d1, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    wxButton * linecolor = new wxButton( lpanel, ID_PLColor,
wxT("Line-Color"), wxDefaultPosition, wxDefaultSize );
    wdth = doptions->GetLineSize();
    doptions->GetLineColor( &r, &g, &b );
    col.Set(r,g,b);
    linecolsize = new LinePane( lpanel,wxSize(50,50),wdth,col );
    wxString ls;
    ls.Printf( "%d", doptions->GetLineSize() );
    wxStaticText * linetx = new wxStaticText( lpanel, -1, wxT("Line width") );
    wxSpinCtrl * linespin = new wxSpinCtrl( lpanel, ID_Plinesize, ls, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 10, wdth );
    d1->Add( linecolor, 0, wxALIGN_LEFT | wxALL, 5 );
    d2->Add( linecolsize, 0, wxALIGN_LEFT | wxALL, 5 );
    d1->Add( linetx, 0, wxALIGN_LEFT | wxALL, 5 );
    d1->Add( linespin, 0, wxALIGN_LEFT | wxALL, 5 );

    cline = new wxListBox( lpanel, ID_plinelistbox, wxDefaultPosition, wxDefaultSize, 0, NULL, wxLB_SINGLE );
    v1->Add( cline, 0, wxALIGN_LEFT | wxALL, 5 );
    cline->Append( wxT("NONE") );
    int dumd=0, dumt=0;
    ElleNumberOfNodes(&dumd,&dumt);
    if ( (dumd+dumt)>0 ) {
    cline->Append( wxT("STANDARD") );
    // need to check for EULER of CAXIS
    cline->Append( wxT("MISORIENTATION") );
    cline->Append( wxT("RAINBOW") );
    }
	wxString name = GetAttributeName( doptions->GetShowArgs( BOUNDARIES),BOUNDARIES );
    int number = cline->FindString( name );
    cline->SetSelection( number, true );
    if ( doptions->GetShowArgs( BOUNDARIES ) == NONE )
        plineshow = false;
    else
        plineshow = true;
    wxString mm;
    wxStaticText * raintx = new wxStaticText( lpanel, -1, wxT("Change line color every xth stage") );
    mm.Printf( "%d", rainchange );
    wxSpinCtrl *rchange = new wxSpinCtrl( lpanel, ID_Prainchange, mm, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 1000, 10 );
    v1->Add( raintx, 0, wxALIGN_LEFT | wxALL, 5 );
    v1->Add( rchange, 0, wxALIGN_LEFT | wxALL, 5 );

    wxCheckBox * ccolor =new wxCheckBox( lpanel, ID_BClampColor,
wxT("Clamp color between"), wxDefaultPosition, wxDefaultSize );
    ccolor->SetValue( bclampcolor );
    Bccmin = new wxTextCtrl( lpanel, ID_BCCMin, mm, wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
    mm.Printf("%lf",bccmin);
    Bccmin->SetValue( mm );
    Bccmax = new wxTextCtrl( lpanel, ID_BCCMax, mm, wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
    mm.Printf("%lf",bccmax);
    Bccmax->SetValue( mm );
    wxStaticText * tx = new wxStaticText( lpanel, -1, wxT("and") );

    v2->Add( ccolor, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    v2->Add( Bccmin, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    v2->Add( tx, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    v2->Add( Bccmax, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    dritter->Add( v1, 0, wxALIGN_LEFT | wxALL, 5 );
    dritter->Add( v2, 0, wxALIGN_LEFT | wxALL, 5 );

    lpanel->SetSizer( dritter );

    return lpanel;
}

wxPanel * Preferences::CreateBNodePage( void )
{
    int size, r, g, b;
    DSettings * doptions = GetDisplayOptions();
    wxString mm;
    wxPanel * bpanel = new wxPanel( book );

    wxBoxSizer * zweiter = new wxBoxSizer( wxVERTICAL );
    wxBoxSizer * z2 = new wxBoxSizer( wxHORIZONTAL );
    wxBoxSizer * z1 = new wxBoxSizer( wxVERTICAL );
    wxBoxSizer * v3 = new wxBoxSizer( wxVERTICAL );
    wxBoxSizer * v1 = new wxBoxSizer( wxHORIZONTAL );
    wxBoxSizer * v2 = new wxBoxSizer( wxHORIZONTAL );

    doptions->GetDNodeColor( &r,&g,&b );
    wxColor col(r,g,b);
    size = doptions->GetNodeSize();
    ddotcolsize = new DotPane( bpanel,wxSize(50,50),size,col );
    z2->Add( ddotcolsize, 0, wxALIGN_LEFT | wxALL, 5 );
    doptions->GetTNodeColor( &r,&g,&b );
    col.Set(r,g,b);
    tdotcolsize = new DotPane( bpanel,wxSize(50,50),size,col );
    z2->Add( tdotcolsize, 0, wxALIGN_LEFT | wxALL, 5 );
    wxButton * dcolor = new wxButton( bpanel, ID_PDColor,
wxT("D-Node-Color"), wxDefaultPosition, wxDefaultSize );
    z1->Add( dcolor, 0, wxALIGN_LEFT | wxALL, 5 );
    wxButton * tcolor = new wxButton( bpanel, ID_PTColor,
wxT("T-Node-Color"), wxDefaultPosition, wxDefaultSize );
    z1->Add( tcolor, 0, wxALIGN_LEFT | wxALL, 5 );
    mm.Printf( "%d", nodesize );
    wxSpinCtrl * bnodespin = new wxSpinCtrl( bpanel, ID_Pbnodesize, mm, wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 1, 10, size );
    z1->Add( bnodespin, 0, wxALIGN_LEFT | wxALL, 5 );
    wxCheckBox * bnumshow = new wxCheckBox( bpanel, ID_Pnodeshownumbers,
wxT("Show Nodenumbers"), wxDefaultPosition, wxDefaultSize );
    z1->Add( bnumshow, 0, wxALIGN_LEFT | wxALL, 5 );
    pnodeshownumbers = false;
    bnumshow->SetValue( false );
    cbnode = new wxListBox( bpanel, ID_pbnodelistbox, wxDefaultPosition, wxDefaultSize, 0, NULL, wxLB_SINGLE);
    v3->Add( cbnode, 0, wxALIGN_LEFT | wxALL, 5 );
    ///////////////////////////////////////////////////////
    int x, count, * welche=0;
    //append attributes to list selector for bnodes
    ElleNodeAttributeList( & welche, & count );
    cbnode->Append( wxT("NONE") );
    cbnode->Append( wxT("NEIGHBOURS") );
    cbnode->Append( wxT("TRIPLES") );
    for ( x = 0; x < count; x++ )
    {
		cbnode->Append( GetAttributeName( welche[x],BNODES ) );
	}
	wxString name = GetAttributeName( usebnodea,BNODES );
    int number = cbnode->FindString( name );
    cbnode->SetSelection( number, true );
    if (welche) free( welche );
    ///////////////////////////////////////////////////////////
    wxCheckBox * ccolor =new wxCheckBox( bpanel, ID_NClampColor,
wxT("Clamp color between"), wxDefaultPosition, wxDefaultSize );
    ccolor->SetValue( nclampcolor );
    Nccmin = new wxTextCtrl( bpanel, ID_NCCMin, mm, wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
    mm.Printf("%lf",nccmin);
    Nccmin->SetValue( mm );
    Nccmax = new wxTextCtrl( bpanel, ID_NCCMax, mm, wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
    mm.Printf("%lf",nccmax);
    Nccmax->SetValue( mm );
    wxStaticText * tx = new wxStaticText( bpanel, -1, wxT("and") );

    v2->Add( ccolor, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    v2->Add( Nccmin, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    v2->Add( tx, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    v2->Add( Nccmax, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );

    v1->Add( z1, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    v1->Add( z2, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    zweiter->Add( v1, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    zweiter->Add( v3, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
    zweiter->Add( v2, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );

    bpanel->SetSizer( zweiter );

    return bpanel;
}

void Preferences::OnColorSingleFlynn( wxCommandEvent & event )
{
    wxTextEntryDialog * count = new wxTextEntryDialog( this, wxT("Please enter number of Flynn"), wxT("HEY THERE!") );
    wxColour col;
    count->ShowModal();
    GetUserColor(&col);
    if(col.Ok())
    {
        ElleFlynnSetColor( atoi( count->GetValue() ), true, col.Red(), col.Green(), col.Blue() );
        EllePlotRegions( 0 );
    }
}

void Preferences::OnResetSingleFlynn( wxCommandEvent & event )
{
    wxTextEntryDialog * count = new wxTextEntryDialog( this,
			wxT("Please enter number of Flynn"), wxT("HEY THERE!") );
    count->ShowModal();
    ElleFlynnSetColor( atoi( count->GetValue() ), false, 0, 0, 0 );
    EllePlotRegions( 0 );
}

void Preferences::OnRangeFlagUnodes( wxCommandEvent & event )
{
    //The user wants to see the flagged unodes or not
    prangeflagunodes = event.IsChecked();
}

void Preferences::OnNotRangeFlagUnodes( wxCommandEvent & event )
{
    //The user wants to see the flagged unodes or not
    pnotrangeflagunodes = event.IsChecked();
}

void Preferences::OnRangeFlagSegments( wxCommandEvent & event )
{
    //The user wants to see the flagged unodes or not
    prangeflagsegments = event.IsChecked();
}

void Preferences::OnNotRangeFlagSegments( wxCommandEvent & event )
{
    //The user wants to see the flagged unodes or not
    pnotrangeflagsegments = event.IsChecked();
}

void Preferences::OnFlynnShowNumbers( wxCommandEvent & event )
{
    //The user wants to see the numbers, or not
    pflynnshownumbers = event.IsChecked();
}

void Preferences::OnNodeShowNumbers( wxCommandEvent & event )
{
    //The user wants to see the numbers, or not
    pnodeshownumbers = event.IsChecked();
}

void Preferences::OnCancel( wxCommandEvent & event )
{
    GetParent()->Update();
    Show(false);
    Close(true);
}

void Preferences::OnLineSize( wxSpinEvent & event )
{
    //The user changed the thickness of the lines
    linecolsize->SetWidth( event.GetPosition() );
    linecolsize->paintNow();
}

void Preferences::OnLineColor( wxCommandEvent & event )
{
    wxColour col;
    //the color of the lines should be changed, let's open a color diolog
    GetUserColor(&col);
    if(col.Ok())
    {
        linecolsize->SetColor( col );
        linecolsize->paintNow();
    }
}

void Preferences::OnSegmentSize( wxSpinEvent & event )
{
    segmentsize = event.GetPosition();
}

void Preferences::OnUNodeSize( wxSpinEvent & event )
{
    unodesize = event.GetPosition();
}

void Preferences::OnRainChange( wxSpinEvent & event )
{
    rainchange = event.GetPosition();
}

void Preferences::OnBNodeSize( wxSpinEvent & event )
{
    //The user has changed the diameter of the nodes
    ddotcolsize-> SetSize(event.GetPosition());
    tdotcolsize-> SetSize(event.GetPosition());
    ddotcolsize->paintNow();
    tdotcolsize->paintNow();
}

void Preferences::OnApply( wxCommandEvent & event )
{
    DSettings * doptions = GetDisplayOptions();
    //get the value of the attribs
    wxColor col;
    int size;
    size = linecolsize->GetWidth();
    col = linecolsize->GetColor();
    doptions->SetLineColor( col.Red(), col.Green(), col.Blue() );
    doptions->SetLineSize( size );
    size = ddotcolsize->GetSize();
    doptions->SetNodeSize( size );
    doptions->SetUNodeSize( unodesize );
    doptions->SetSegmentSize( segmentsize );
    col = ddotcolsize->GetColor();
    doptions->SetDNodeColor( col.Red(), col.Green(), col.Blue() );
    col = tdotcolsize->GetColor();
    doptions->SetTNodeColor( col.Red(), col.Green(), col.Blue() );
    doptions->SetUnodesRangeFlagMaxValue( urangevalmax );
    doptions->SetUnodesRangeFlagMinValue( urangevalmin );
    doptions->SetUnodesRangeFlag( prangeflagunodes );
    doptions->SetUnodesNotRangeFlagMaxValue( unotrangevalmax );
    doptions->SetUnodesNotRangeFlagMinValue( unotrangevalmin );
    doptions->SetUnodesNotRangeFlag( pnotrangeflagunodes );
	useunodea = GetAttributeInt( cunode->GetStringSelection(),UNODES );
    doptions->SetSegmentsRangeFlagMaxValue( srangevalmax );
    doptions->SetSegmentsRangeFlagMinValue( srangevalmin );
    doptions->SetSegmentsRangeFlag( prangeflagsegments );
    doptions->SetSegmentsNotRangeFlagMaxValue( snotrangevalmax );
    doptions->SetSegmentsNotRangeFlagMinValue( snotrangevalmin );
    doptions->SetSegmentsNotRangeFlag( pnotrangeflagsegments );
    usesegmenta = GetAttributeInt( csegment->GetStringSelection(),SEGMENTS );
    usebnodea = GetAttributeInt( cbnode->GetStringSelection(),BNODES );
	attrib = GetAttributeInt( cflynn->GetStringSelection(),FLYNNS );

    Settings * user_options = GetUserOptions();
    if(triangulate)
        user_options->draw_nodes=1;
    else
        user_options->draw_nodes=0;

    if ( attrib != NONE )
        doptions->SetShowArgs( FLYNNS, true, attrib );
    else
        doptions->SetShowArgs( FLYNNS, false, attrib );

	battrib = GetAttributeInt( cline->GetStringSelection(),BOUNDARIES );
    if ( battrib != NONE )
    {
        doptions->SetShowArgs( BOUNDARIES, true, battrib );
        if(battrib==RAINBOW)
            doptions->SetRainStages(ElleCount());
    }
    else
        doptions->SetShowArgs( BOUNDARIES, false, attrib );
    if ( usebnodea != NONE )
        doptions->SetShowArgs( BNODES, true, usebnodea );
    else
        doptions->SetShowArgs( BNODES, false, usebnodea );

    if ( !ElleUnodesActive() ) useunodea = NONE;
    if ( useunodea != NONE )
    {
        if ( useunodea == LOCATION )
            useunodea = U_LOCATION;
        doptions->SetShowArgs( UNODES, true, useunodea );
    }
    else
        doptions->SetShowArgs( UNODES, false, useunodea );
    if ( unodes3d == true )
    {
        unodes3d = false;
        ( ( Canvas * ) GetParent() )->ResetScreen();
    }
    if ( !(ElleMaxSegments()>0) ) usesegmenta = NONE;
    if ( usesegmenta != NONE )
    {
        if ( usesegmenta == LOCATION )
            usesegmenta = S_LOCATION;
        doptions->SetShowArgs( SEGMENTS, true, usesegmenta );
    }
    else
        doptions->SetShowArgs( SEGMENTS, false, usesegmenta );
    doptions->ShowFlynnNumbers( pflynnshownumbers );
    doptions->ShowNodeNumbers( pnodeshownumbers );
    doptions->SetRainChange(rainchange);

    if(fclampcolor && attrib!=NONE)
    {
        doptions->SetClampColor(0,true,fccmin,fccmax);
        user_options->SetFlynnAttribOption(attrib,fccmin,fccmax);
    }
    else
        doptions->SetClampColor(0,false,fccmin,fccmax);

    if(bclampcolor && battrib!=NONE && battrib!=RAINBOW)
    {
        doptions->SetClampColor(1,true,bccmin,bccmax);
        user_options->SetBndAttribOption(battrib,bccmin,bccmax);
    }
    else
        doptions->SetClampColor(1,false,bccmin,bccmax);

    if(nclampcolor &&usebnodea!=NONE)
    {
        doptions->SetClampColor(2,true,nccmin,nccmax);
        user_options->SetNodeAttribOption(usebnodea,nccmin,nccmax);
    }
    else
        doptions->SetClampColor(2,false,nccmin,nccmax);

    if(uclampcolor && useunodea!=NONE && useunodea!=U_LOCATION && useunodea!=LOCATION)
    {
        doptions->SetClampColor(3,true,uccmin,uccmax);
        user_options->SetUnodeAttribOption(useunodea,uccmin,uccmax);
    }
    else
        doptions->SetClampColor(3,false,uccmin,uccmax);

    if(sclampcolor && usesegmenta!=NONE && usesegmenta!=S_LOCATION && usesegmenta!=LOCATION)
    {
        doptions->SetClampColor(4,true,sccmin,sccmax);
        user_options->SetSegmentAttribOption(usesegmenta,sccmin,sccmax);
    }
    else
        doptions->SetClampColor(4,false,sccmin,sccmax);

    EllePlotRegions( 0 );
}

void Preferences::OnOk( wxCommandEvent & event )
{
    DSettings * doptions = GetDisplayOptions();
    wxColor col;
    int size;
    size = linecolsize->GetWidth();
    col = linecolsize->GetColor();
    doptions->SetLineColor( col.Red(), col.Green(), col.Blue() );
    doptions->SetLineSize( size );
    size = ddotcolsize->GetSize();
    doptions->SetNodeSize( size );
    doptions->SetUNodeSize( unodesize );
    doptions->SetSegmentSize( segmentsize );
    col = ddotcolsize->GetColor();
    doptions->SetDNodeColor( col.Red(), col.Green(), col.Blue() );
    col = tdotcolsize->GetColor();
    doptions->SetTNodeColor( col.Red(), col.Green(), col.Blue() );

    doptions->SetUnodesRangeFlagMinValue( urangevalmin );
    doptions->SetUnodesRangeFlagMaxValue( urangevalmax );
    doptions->SetUnodesRangeFlag( prangeflagunodes );

    doptions->SetUnodesNotRangeFlagMaxValue( unotrangevalmax );
    doptions->SetUnodesNotRangeFlagMinValue( unotrangevalmin );
    doptions->SetUnodesNotRangeFlag( pnotrangeflagunodes );

    doptions->SetSegmentsRangeFlagMaxValue( srangevalmax );
    doptions->SetSegmentsRangeFlagMinValue( srangevalmin );
    doptions->SetSegmentsRangeFlag( prangeflagsegments );
    doptions->SetSegmentsNotRangeFlagMaxValue( snotrangevalmax );
    doptions->SetSegmentsNotRangeFlagMinValue( snotrangevalmin );
    doptions->SetSegmentsNotRangeFlag( pnotrangeflagsegments );
    usesegmenta = GetAttributeInt( csegment->GetStringSelection(),SEGMENTS );
    doptions->SetRainChange(rainchange);
    Settings * user_options = GetUserOptions();
    if(triangulate)
        user_options->draw_nodes=1;
    else
        user_options->draw_nodes=0;
	attrib = GetAttributeInt( cflynn->GetStringSelection(),FLYNNS );
	useunodea = GetAttributeInt( cunode->GetStringSelection(),UNODES );
	usebnodea = GetAttributeInt( cbnode->GetStringSelection(),BNODES );
    if ( attrib != NONE )
        doptions->SetShowArgs( FLYNNS, true, attrib );
    else
        doptions->SetShowArgs( FLYNNS, false, attrib );

	battrib = GetAttributeInt( cline->GetStringSelection(),BOUNDARIES );
    if ( battrib != NONE )
    {
        doptions->SetShowArgs( BOUNDARIES, true, battrib );
        if(battrib==RAINBOW)
            doptions->SetRainStages(ElleCount());
    }
    else
        doptions->SetShowArgs( BOUNDARIES, false, battrib );

    if ( usebnodea != NONE )
        doptions->SetShowArgs( BNODES, true, usebnodea );
    else
        doptions->SetShowArgs( BNODES, false, usebnodea );

    if ( useunodea != NONE )
    {
        if ( useunodea == LOCATION )
            useunodea = U_LOCATION;

        doptions->SetShowArgs( UNODES, true, useunodea );
    }
    else
        doptions->SetShowArgs( UNODES, false, useunodea );
    if ( unodes3d == true )
    {
        unodes3d = false;
        ( ( Canvas * ) GetParent() )->ResetScreen();
    }
    if ( !(ElleMaxSegments()>0) ) usesegmenta = NONE;
    if ( usesegmenta != NONE )
    {
        if ( usesegmenta == LOCATION )
            usesegmenta = S_LOCATION;
        doptions->SetShowArgs( SEGMENTS, true, usesegmenta );
    }
    else
        doptions->SetShowArgs( SEGMENTS, false, usesegmenta );
    doptions->ShowFlynnNumbers( pflynnshownumbers );
    doptions->ShowNodeNumbers( pnodeshownumbers );

    if(fclampcolor && attrib!=NONE)
    {
        doptions->SetClampColor(0,true,fccmin,fccmax);
        user_options->SetFlynnAttribOption(attrib,fccmin,fccmax);
    }
    else
        doptions->SetClampColor(0,false,fccmin,fccmax);

    if(bclampcolor && battrib!=NONE && battrib!=RAINBOW)
    {
        doptions->SetClampColor(1,true,bccmin,bccmax);
        user_options->SetBndAttribOption(battrib,bccmin,bccmax);
    }
    else
        doptions->SetClampColor(1,false,bccmin,bccmax);

    if(nclampcolor &&usebnodea!=NONE)
    {
        doptions->SetClampColor(2,true,nccmin,nccmax);
        user_options->SetNodeAttribOption(usebnodea,nccmin,nccmax);
    }
    else
        doptions->SetClampColor(2,false,nccmin,nccmax);

    if(uclampcolor && useunodea!=NONE && useunodea!=U_LOCATION && useunodea!=LOCATION)
    {
        doptions->SetClampColor(3,true,uccmin,uccmax);
        user_options->SetUnodeAttribOption(useunodea,uccmin,uccmax);
    }
    else
        doptions->SetClampColor(3,false,uccmin,uccmax);

    if(sclampcolor && usesegmenta!=NONE && usesegmenta!=S_LOCATION && usesegmenta!=LOCATION)
    {
        doptions->SetClampColor(4,true,sccmin,sccmax);
        user_options->SetSegmentAttribOption(usesegmenta,sccmin,sccmax);
    }
    else
        doptions->SetClampColor(4,false,sccmin,sccmax);

    EllePlotRegions( 0 );
    Show(false);
    Close(true);
}

void Preferences::OnDNodeColor( wxCommandEvent & event )
{
    wxColourDialog dia( this );
    wxColour col;
    GetUserColor(&col);
    if(col.Ok())
    {
        ddotcolsize->SetColor( col );
        ddotcolsize->paintNow();
    }
}

void Preferences::OnTNodeColor( wxCommandEvent & event )
{
    wxColourDialog dia( this );
    wxColour col;
    GetUserColor(&col);
    if(col.Ok())
    {
        tdotcolsize->SetColor( col );
        tdotcolsize->paintNow();
    }
}

wxString Preferences::GetAttributeName( int welche, int type )
{
    char *clabel = new char[MAX_OPTION_NAME];
    bool valid = false;
    clabel[0] = '\0';
    switch(type) {
    case FLYNNS: if (id_match(region_terms,welche,clabel))
                   valid = true;
                 break;
    case BOUNDARIES: if (id_match(boundary_terms,welche,clabel))
                   valid = true;
                 break;
    case BNODES: if (id_match(node_terms,welche,clabel))
                   valid = true;
                 break;
    case UNODES: if (id_match(unode_terms,welche,clabel))
                   valid = true;
                 break;
    case SEGMENTS: if (id_match(segment_terms,welche,clabel))
                   valid = true;
                 break;
    default:     break;
    }
    wxString label = clabel;
    delete clabel;
    return(label);
}

int Preferences::GetAttributeInt( wxString welche,int type )
{
	int id = -1;
    switch(type) {
    case FLYNNS:
	id = name_match( ( ( char * ) welche.c_str() ), region_terms );
		break;
    case BOUNDARIES:
	id = name_match( ( ( char * ) welche.c_str() ), boundary_terms );
		break;
    case BNODES:
	id = name_match( ( ( char * ) welche.c_str() ), node_terms );
		break;
    case UNODES:
	id = name_match( ( ( char * ) welche.c_str() ), unode_terms );
		break;
    case SEGMENTS:
	id = name_match( ( ( char * ) welche.c_str() ), segment_terms );
		break;
    default:     break;
    }
#if XY
	id = name_match( ( ( char * ) welche.c_str() ), FileKeys );
	if ( id == -1 )
		id = name_match( ( ( char * ) welche.c_str() ), BoundaryKeys );
	if ( id == -1 )
		id = name_match( ( ( char * ) welche.c_str() ), MineralKeys );
	if ( id == -1 )
		id = name_match( ( ( char * ) welche.c_str() ), VelocityKeys );
	if ( id == -1 )
		id = name_match( ( ( char * ) welche.c_str() ), StressKeys );
	if ( id == -1 )
		id = name_match( ( ( char * ) welche.c_str() ), StrainKeys );
	if ( id == -1 )
		id = name_match( ( ( char * ) welche.c_str() ), FiniteStrainKeys );
	if ( id == -1 )
		id = name_match( ( ( char * ) welche.c_str() ), FlynnAgeKeys );
	if ( id == -1 )
		id = name_match( ( ( char * ) welche.c_str() ), FlynnStrainKeys );
	if ( id == -1 )
		id = name_match( ( ( char * ) welche.c_str() ), Leftovers );
#endif
	return ( id );
}


void Preferences::ColormapOnRangeFlynn( wxCommandEvent & event )
{
    int attr;
    double max, min;
    wxString attrib;
    bselect->SetSelection( 0 );
    uselect->SetSelection( 0 );
    if ( ( attr = fselect->GetSelection() ) != 0 )
    {
        attrib = fselect->GetString( attr );
		attr = GetAttributeInt( attrib,FLYNNS );
        ElleFindFlynnAttributeRange( attr, & min, & max );
        attrib = "";
        attrib.Printf( "%lf", max );
        fmax->SetLabel( attrib );
        attrib = "";
        attrib.Printf( "%lf", min );
        fmin->SetLabel( attrib );
    }
    else
    {
        fmin->SetLabel( wxT("             ") );
        fmax->SetLabel( wxT("          ") );
    }
}

void Preferences::ColormapOnRangeBNode( wxCommandEvent & event )
{
    int attr;
    double max, min;
    wxString attrib;

    fselect->SetSelection( 0 );
    uselect->SetSelection( 0 );
    if ( ( attr = bselect->GetSelection() ) != 0 )
    {
        attrib = bselect->GetString( attr );
		attr = GetAttributeInt( attrib,BOUNDARIES );

        ElleFindBndAttributeRange( attr, & min, & max );
        attrib = wxT("");
        attrib.Printf( "%lf", max );
        fmax->SetLabel( attrib );
        attrib = wxT("");
        attrib.Printf( "%lf", min );
        fmin->SetLabel( attrib );
    }
    else
    {
        fmin->SetLabel( wxT("             ") );
        fmax->SetLabel( wxT("          ") );
    }

}

void Preferences::ColormapOnRangeUNode( wxCommandEvent & event )
{
    int attr;
    double max, min;
    wxString attrib;

    bselect->SetSelection( 0 );
    fselect->SetSelection( 0 );
    if ( ( attr = uselect->GetSelection() ) != 0 )
    {
        attrib = uselect->GetString( attr );
		attr = GetAttributeInt( attrib,UNODES );

        ElleFindUnodeAttributeRange( attr, & min, & max );
        attrib = wxT("");
        attrib.Printf( "%lf", max );
        fmax->SetLabel( attrib );
        attrib = wxT("");
        attrib.Printf( "%lf", min );
        fmin->SetLabel( attrib );
    }
    else
    {
        fmin->SetLabel( wxT("             ") );
        fmax->SetLabel( wxT("          ") );
    }

}

void Preferences::ColormapOnRangeSegment( wxCommandEvent & event )
{
    int attr;
    double max, min;
    wxString attrib;

    bselect->SetSelection( 0 );
    fselect->SetSelection( 0 );
    uselect->SetSelection( 0 );
    if ( ( attr = sselect->GetSelection() ) != 0 )
    {
        attrib = sselect->GetString( attr );
		attr = GetAttributeInt( attrib,SEGMENTS );

        ElleFindSegmentAttributeRange( attr, & min, & max );
        attrib = wxT("");
        attrib.Printf( "%lf", max );
        fmax->SetLabel( attrib );
        attrib = wxT("");
        attrib.Printf( "%lf", min );
        fmin->SetLabel( attrib );
    }
    else
    {
        fmin->SetLabel( wxT("             ") );
        fmax->SetLabel( wxT("          ") );
    }

}

void Preferences::ColormapOnLoad( wxCommandEvent & event )
{
    string filename;
    DSettings * doptions = GetDisplayOptions();
    filename = wxFileSelector( wxT("Choose Colormap-File"), "", "", "", "Colormap-Files  (*.cmap)|*.cmap|All Files (*)|*",wxOPEN );
    doptions->CmapLoad( filename );
    cbar->paintNow();
}

void Preferences::ColormapOnSave( wxCommandEvent & event )
{
    string filename;
    DSettings * doptions = GetDisplayOptions();
    filename = wxFileSelector( "Choose filename to save Colormap", "", "", "",
                               "Colormap-Files (*.cmap)|*.cmap|All Files (*)|*",wxSAVE );
    doptions->CmapSave( filename );
    Refresh( true, NULL );
}

void Preferences::ColormapUploadCmap()
{
    /*DSettings * doptions = GetDisplayOptions();
    int n, j, r, g, b, set;
    unsigned char * AUX;
    glcan->SetCurrent();
    glReadBuffer( GL_BACK );
    AUX = (unsigned char *)malloc( 3 * 256 * sizeof( unsigned char ) );
    glReadPixels( 0, 25, 255, 1, GL_RGB, GL_UNSIGNED_BYTE, AUX );
    glFinish();
    glcan->SwapBuffers();
    for ( n = 0, j = 0; j < 256; n = n + 3, j++ )
    {
    	doptions->CmapGetColor( j, & r, & g, & b, & set );
    	if ( set == 0 )
    			doptions->CmapChangeColor( j, AUX[n], AUX[n + 1], AUX[n + 2], 0 );
    }
    delete AUX ;
    */
}

void Preferences::ColormapResetToDef()
{
    DSettings * doptions = GetDisplayOptions();
    doptions->CmapResetToDefault();
    cbar->paintNow();
    first = true;
    Refresh(true,NULL);
}

wxPanel* Preferences::ColormapCreateControls(void)
{
    wxPanel * cpanel = new wxPanel( book);
    wxBoxSizer * item2 = new wxBoxSizer( wxVERTICAL );

    wxBoxSizer * item3 = new wxBoxSizer( wxVERTICAL );
    item2->Add( item3, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );

    wxBoxSizer * colsiz = new wxBoxSizer( wxHORIZONTAL );
    item3->Add( colsiz, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );

    wxStaticText * t1 = new wxStaticText( cpanel, -1, wxT("Change colour at") );
    colsiz->Add( t1, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );

    wxTextCtrl * item4 = new wxTextCtrl( cpanel, ID_TEXTCTRL, _( "" ), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
    colsiz->Add( item4, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );

    wxStaticText * t2 = new wxStaticText( cpanel, -1, wxT("%") );
    colsiz->Add( t2, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );

    cbar = new CBarPane(cpanel,wxSize(256,50));
    item3->Add( cbar, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );

    wxBoxSizer * item6 = new wxBoxSizer( wxHORIZONTAL );
    item2->Add( item6, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );

    wxButton * item8 = new wxButton( cpanel, ID_DEFAULT, _( "Default" ), wxDefaultPosition, wxDefaultSize, 0 );

    item6->Add( item8, 0, wxALIGN_CENTER_VERTICAL | wxALL, 5 );

    wxBoxSizer * last = new wxBoxSizer( wxVERTICAL );
    item2->Add( last, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );

    wxButton * save = new wxButton( cpanel, ID_CSAVE, _( "Save cmap" ), wxDefaultPosition, wxDefaultSize, 0 );
    item6->Add( save, 0, wxALIGN_CENTER_VERTICAL | wxALL, 5 );

    wxButton * load = new wxButton( cpanel, ID_CLOAD, _( "Load cmap" ), wxDefaultPosition, wxDefaultSize, 0 );
    item6->Add( load, 0, wxALIGN_CENTER_VERTICAL | wxALL, 5 );


    wxBoxSizer * last1 = new wxBoxSizer( wxHORIZONTAL );
    wxBoxSizer * last2 = new wxBoxSizer( wxHORIZONTAL );

    int x, count, * welche;
    wxString choices1[] =
        {
            _T( "Flynns" )
        };
    wxString choices2[] =
        {
            _T( "BNodes" )
        };
    wxString choices3[] =
        {
            _T( "UNodes" )
        };
    fselect = new wxChoice( cpanel, ID_sflynn, wxDefaultPosition, wxDefaultSize, 1, choices1 );
    bselect = new wxChoice( cpanel, ID_sbnode, wxDefaultPosition, wxDefaultSize, 1, choices2 );
    uselect = new wxChoice( cpanel, ID_sunode, wxDefaultPosition, wxDefaultSize, 1, choices3 );
    ElleFlynnDfltAttributeList( & welche, & count );
    for ( x = 0; x < count; x++ )
    {
		fselect->Append( GetAttributeName( welche[x],FLYNNS ) );
	}
    if (welche) { free(welche); welche=0; }

    ElleNodeAttributeList( & welche, & count );
    for ( x = 0; x < count; x++ )
    {
		bselect->Append( GetAttributeName( welche[x],BNODES ) );
    }
    if (welche) { free(welche); welche=0; }

    ElleUnodeAttributeList( & welche, & count );
    for ( x = 0; x < count; x++ )
    {
		uselect->Append( GetAttributeName( welche[x],UNODES ) );
    }
    if (welche) { free(welche); welche=0; }

    min = new wxStaticText( cpanel, -1, wxT("Min.:") );
    max = new wxStaticText( cpanel, -1, wxT("Max.:") );
    fmin = new wxStaticText( cpanel, -1, wxT("                      ") );
    fmax = new wxStaticText( cpanel, -1, wxT("            ") );

    last1->Add( fselect, 0, wxALIGN_CENTER_VERTICAL | wxALL, 5 );
    last1->Add( bselect, 0, wxALIGN_CENTER_VERTICAL | wxALL, 5 );
    last1->Add( uselect, 0, wxALIGN_CENTER_VERTICAL | wxALL, 5 );

    last2->Add( min, 0, wxALIGN_CENTER_VERTICAL | wxALL, 5 );
    last2->Add( fmin, 0, wxALIGN_CENTER_VERTICAL | wxALL, 5 );
    last2->Add( max, 0, wxALIGN_CENTER_VERTICAL | wxALL, 5 );
    last2->Add( fmax, 0, wxALIGN_CENTER_VERTICAL | wxALL, 5 );

    last->Add( last1, 0, wxALIGN_CENTER_VERTICAL | wxALL, 5 );
    last->Add( last2, 0, wxALIGN_CENTER_VERTICAL | wxALL, 5 );

    wxBoxSizer * vlast = new wxBoxSizer( wxHORIZONTAL );

    wxStaticText * c1 = new wxStaticText( cpanel, -1, wxT("Change colour at value:") );
    wxTextCtrl * c2 = new wxTextCtrl( cpanel, ID_CTEXT, _( "" ), wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );

    vlast->Add( c1, 0, wxALIGN_CENTER_VERTICAL | wxALL, 5 );
    vlast->Add( c2, 0, wxALIGN_CENTER_VERTICAL | wxALL, 5 );
    item2->Add( vlast, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );

    wxString mm;
    cpanel->SetSizer( item2 );
    return cpanel;
}

void Preferences::GetUserColor(wxColour *col)
{
    *col=wxGetColourFromUser();
}

void Preferences::ColormapOnChangeColorValue( wxCommandEvent & event )
{
    int in;
    int r, g, b;
    double tmp, vmin, vmax, dp;
    DSettings * doptions = GetDisplayOptions();
    wxColour col;
    ( ( ( wxTextCtrl * ) event.GetEventObject() )->GetValue() ).ToDouble( & tmp );
    ( fmin->GetLabel() ).ToDouble( & vmin );
    ( fmax->GetLabel() ).ToDouble( & vmax );
    if ( tmp > vmax || tmp < vmin )
        wxMessageBox( wxT("Value out of range!"), wxT("Attention"),
					wxOK, this );
    else
    {
        dp = ( vmax - vmin ) / 100;
        tmp = tmp / dp;
        in = ( int )( ceil( tmp * 2.55 ) ); //ohoh, that calls for trouble!
        if ( tmp < 0 )
            tmp = 0.0;
        if ( tmp > 255 )
            tmp = 255.0;
        GetUserColor(&col);
        if(col.Ok())
        {
            r = col.Red(); g = col.Green(); b = col.Blue();
            doptions->CmapChangeColor( in, r, g, b, 1 );
            cbar->paintNow();
            book->Refresh( true, NULL );
            ColormapUploadCmap();
        }
        in = ( ( wxTextCtrl * ) event.GetEventObject() )->GetLineLength( 0 );
        ( ( wxTextCtrl * ) event.GetEventObject() )->SetSelection( 0, in );
    }
}

void Preferences::ColormapOnChangeColorPercent( wxCommandEvent & event )
{
    int in;
    int r, g, b;
    double tmp;
    DSettings * doptions = GetDisplayOptions();
    wxColour col;
    ( ( ( wxTextCtrl * ) event.GetEventObject() )->GetValue() ).ToDouble( & tmp );
    in = ( int )( ceil( tmp * 2.55 ) ); //ohoh, that calls for trouble!
    if ( tmp < 0 || tmp > 255 )
        wxBell();
    else
    {
        GetUserColor(&col);
        if(col.Ok())
        {
            r = col.Red(); g = col.Green(); b = col.Blue();
            doptions->CmapChangeColor( in, r, g, b, 1 );
            cbar->paintNow();
            book->Refresh( true, NULL );
            ColormapUploadCmap();
        }
    }
    in = ( ( wxTextCtrl * ) event.GetEventObject() )->GetLineLength( 0 );
    ( ( wxTextCtrl * ) event.GetEventObject() )->SetSelection( 0, in );
}

void CBarPane::paintEvent(wxPaintEvent &evt)
{
    int n, set;
    int r, g, b;
    wxPen pen;
    wxPaintDC dc(this);
    if (dc.IsOk()) {
        DSettings * doptions = GetDisplayOptions();
        for ( n = 0; n < 256; n++ )
        {
            doptions->CmapGetColor( n, & r, & g, & b, & set );
            pen.SetColour(wxColour(r,g,b));
            dc.SetPen(pen);
            dc.DrawLine(n,0,n,50 );
        }
    }
}

void DotPane::paintEvent(wxPaintEvent &evt)
{
    wxClientDC dc(this);
    UpdateDotDisp(dc);
}

void LinePane::paintEvent(wxPaintEvent &evt)
{
    wxClientDC dc(this);
    UpdateLineDisp(dc);
}

void DotPane::paintNow()
{
    wxClientDC dc(this);
    UpdateDotDisp(dc);
}

void LinePane::paintNow()
{
    wxClientDC dc(this);
    UpdateLineDisp(dc);
}

/*
 * In most cases, this will not be needed at all; simply handling
 * paint events and calling Refresh() when a refresh is needed
 * will do the job.
 */
void CBarPane::paintNow()
{
    wxClientDC dc(this);
    ColormapUpdateColorDisp(dc);
}

void DotPane::UpdateDotDisp(wxDC& dotdc)
{
    wxBrush dotbrush;
    wxPen dotpen;
    if (dotdc.IsOk()) {
        dotdc.SetBackground( wxBrush( wxColour( 0, 0, 0 ), wxSOLID ) );
        dotdc.Clear();
    //get Node stored values
        dotpen.SetColour( GetColor() );
        dotpen.SetStyle( wxSOLID );
        dotbrush.SetColour( GetColor() );
        dotbrush.SetStyle( wxSOLID );
        int size =GetSize();
        dotdc.SetPen( dotpen );
        dotdc.SetBrush( dotbrush );
        dotdc.DrawCircle( 25, 25, size );
    }
}

void LinePane::UpdateLineDisp(wxDC& linedc)
{
    wxPen linepen;
    if (linedc.IsOk()) {
        linedc.SetBackground( wxBrush( wxColour( 0, 0, 0 ), wxSOLID ) );
        linedc.Clear();
        //doptions->GetLineColor( & r, & g, & b );
        linepen.SetColour( GetColor() );
        linepen.SetStyle( wxSOLID );
        linepen.SetWidth( GetWidth() );
        linedc.SetPen( linepen );
        linedc.DrawLine( 0, 25, 50, 25 );
    }
}

void CBarPane::ColormapUpdateColorDisp(wxDC& dc)
{
    int n, set;
    int r, g, b;
    wxPen pen;
    if (dc.IsOk()) {
        DSettings * doptions = GetDisplayOptions();
        for ( n = 0; n < 256; n++ )
        {
            doptions->CmapGetColor( n, & r, & g, & b, & set );
            pen.SetColour(wxColour(r,g,b));
            dc.SetPen(pen);
            dc.DrawLine(n,0,n,50 );
        }
    }
}

void Preferences::ColormapOnDefault( wxCommandEvent & event )
{
    ColormapResetToDef();
    Refresh( true, NULL );
}

void Preferences::OnFCCMin(wxCommandEvent &event)
{
    ( ( ( wxTextCtrl * ) event.GetEventObject() )->GetValue() ).ToDouble( & fccmin );
}
void Preferences::OnFCCMax(wxCommandEvent &event)
{
    ( ( ( wxTextCtrl * ) event.GetEventObject() )->GetValue() ).ToDouble( & fccmax );
}
void Preferences::OnBCCMin(wxCommandEvent &event)
{
    ( ( ( wxTextCtrl * ) event.GetEventObject() )->GetValue() ).ToDouble( & bccmin );
}
void Preferences::OnBCCMax(wxCommandEvent &event)
{
    ( ( ( wxTextCtrl * ) event.GetEventObject() )->GetValue() ).ToDouble( & bccmax );
}
void Preferences::OnNCCMin(wxCommandEvent &event)
{
    ( ( ( wxTextCtrl * ) event.GetEventObject() )->GetValue() ).ToDouble( & nccmin );
}
void Preferences::OnNCCMax(wxCommandEvent &event)
{
    ( ( ( wxTextCtrl * ) event.GetEventObject() )->GetValue() ).ToDouble( & nccmax );
}
void Preferences::OnUCCMin(wxCommandEvent &event)
{
    ( ( ( wxTextCtrl * ) event.GetEventObject() )->GetValue() ).ToDouble( & uccmin );
}
void Preferences::OnUCCMax(wxCommandEvent &event)
{
    ( ( ( wxTextCtrl * ) event.GetEventObject() )->GetValue() ).ToDouble( & uccmax );
}
void Preferences::OnSCCMin(wxCommandEvent &event)
{
    ( ( ( wxTextCtrl * ) event.GetEventObject() )->GetValue() ).ToDouble( & sccmin );
}
void Preferences::OnSCCMax(wxCommandEvent &event)
{
    ( ( ( wxTextCtrl * ) event.GetEventObject() )->GetValue() ).ToDouble( & sccmax );
}

void Preferences::OnFClampColor( wxCommandEvent & event )
{
    fclampcolor=event.IsChecked();
}

void Preferences::OnBClampColor( wxCommandEvent & event )
{
    bclampcolor=event.IsChecked();
}

void Preferences::OnNClampColor( wxCommandEvent & event )
{
    nclampcolor=event.IsChecked();
}

void Preferences::OnUClampColor( wxCommandEvent & event )
{
    uclampcolor=event.IsChecked();
}

void Preferences::OnSClampColor( wxCommandEvent & event )
{
    sclampcolor=event.IsChecked();
}
