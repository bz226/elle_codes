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

#include "rundia.h"
#include "runopts.h"
#include "settings.h"
#include "dsettings.h"

extern Settings * GetUserOptions();
extern DSettings * GetDisplayOptions();

IMPLEMENT_CLASS( RunDia, wxDialog )

BEGIN_EVENT_TABLE( RunDia, wxDialog ) EVT_BUTTON( ID_OK, RunDia::OnOk ) EVT_BUTTON( ID_RDEFAULT, RunDia::OnReset )
EVT_BUTTON( ID_Cancel, RunDia::OnCancel ) EVT_BUTTON( ID_SaPic, RunDia::OnSavePics )
EVT_CHECKBOX( ID_logfile, RunDia::OnLogFile ) EVT_CHECKBOX( ID_loglevel0, RunDia::OnLogLevel0 )
EVT_CHECKBOX( ID_loglevel1, RunDia::OnLogLevel1 )
EVT_GRID_CELL_LEFT_CLICK( RunDia::OnCellEdit )
EVT_CHECKBOX( ID_loglevel2, RunDia::OnLogLevel2 ) END_EVENT_TABLE()

RunDia::RunDia( wxWindow * parent, wxWindowID id, const wxString & caption, const wxPoint & pos,
                const wxSize & size, long style )
{
	wxDialog::Create( parent, id, caption, pos, wxSize( 285, 460 ), style );
	CreateControls();
	this->Fit();
}

void RunDia::OnLogFile( wxCommandEvent & event )
{
	if ( event.IsChecked() )
		logf->Enable( true );
	else
	{
		logf->SetValue( false );
		logf->Enable( false );
	}
}

void RunDia::OnCellEdit( wxGridEvent & event )
{
	UserData ud;
	int row = event.GetRow(), col = event.GetCol();
	wxString sval,oval;
	double dval;
	ElleUserData(ud);
	if(col==0)
		wxMessageBox( wxT("You can not change these values"),
					wxT("This is an important message!"),
					wxOK | wxICON_ERROR );
	else
	{
		sval = wxGetTextFromUser( wxT("Enter value"), wxT("Input text"), oval, this );
		if(sval!=oval)
		{
			dval=atof(sval.c_str());
			ud[row]=dval;
			ugrid->SetCellValue(row,col,sval);
		}
	}
}

void RunDia::CreateControls()
{
	RunDia * item1 = this;

	wxStaticBoxSizer * item6 = new wxStaticBoxSizer( wxVERTICAL,item1,wxT("Run option settings") );
	wxBoxSizer * item2 = new wxBoxSizer( wxHORIZONTAL );
	wxBoxSizer * item3 = new wxBoxSizer( wxVERTICAL );
	wxBoxSizer * item4 = new wxBoxSizer( wxVERTICAL );
	wxBoxSizer * item5 = new wxBoxSizer( wxHORIZONTAL );
	wxBoxSizer *main1=new wxBoxSizer(wxVERTICAL);
	wxBoxSizer *main=new wxBoxSizer(wxHORIZONTAL);
	wxBoxSizer *main2=new wxBoxSizer(wxHORIZONTAL);

	item2->Add( item3 );
	item2->Add( item4 );
	item6->Add( item2 );
	item6->Add( item5 );
	main->Add( item6 );
	main1->Add( main );
	main1->Add( main2 );

	item1->SetSizer( main1 );
	item1->SetAutoLayout( TRUE );

	wxStaticText * l1 = new wxStaticText( item1, -1, wxT("Stages"), wxDefaultPosition, wxSize( 100, 28 ) );
	item3->Add( l1, 0, wxALIGN_LEFT | wxALL, 5 );

	stages = new wxTextCtrl( item1, ID_stages, _( "" ), wxDefaultPosition, wxDefaultSize );
	item4->Add( stages, 0, wxALIGN_RIGHT | wxALL, 5 );

	wxStaticText * l2 = new wxStaticText( item1, -1, wxT("Save interval"), wxDefaultPosition, wxSize( 100, 25 ) );
	item3->Add( l2, 0, wxALIGN_LEFT | wxALL, 5 );

	sinterv = new wxTextCtrl( item1, ID_sinterv, _( "" ), wxDefaultPosition, wxDefaultSize );
	item4->Add( sinterv, 0, wxALIGN_RIGHT | wxALL, 5 );

	wxStaticText * l3 = new wxStaticText( item1, -1, wxT("Save file root"), wxDefaultPosition, wxSize( 100, 25 ) );
	item3->Add( l3, 0, wxALIGN_LEFT | wxALL, 5 );

	sfiler = new wxTextCtrl( item1, ID_sfiler, _( "" ), wxDefaultPosition, wxDefaultSize );
	item4->Add( sfiler, 0, wxALIGN_RIGHT | wxALL, 5 );

	wxStaticText * l14 = new wxStaticText( item1, -1, wxT("Save pics root"), wxDefaultPosition, wxSize( 130, 25 ) );
	item3->Add( l14, 0, wxALIGN_LEFT | wxALL, 5 );

	wxButton * spic = new wxButton( item1, ID_SaPic, _( "SavePics" ), wxDefaultPosition, wxDefaultSize, 0 );
	item4->Add( spic, 0, wxALIGN_CENTER_VERTICAL | wxALL, 5 );

	dfreq = new wxTextCtrl( item1, ID_dfreq, _( "" ), wxDefaultPosition, wxDefaultSize );
	item4->Add( dfreq, 0, wxALIGN_RIGHT | wxALL, 5 );

	wxStaticText * l15 = new wxStaticText( item1, -1, wxT("Display frequency"), wxDefaultPosition, wxSize( 130, 25 ) );
	item3->Add( l15, 0, wxALIGN_LEFT | wxALL, 5 );

	wxStaticText * l4 = new wxStaticText( item1, -1, wxT("Switch distance"), wxDefaultPosition, wxSize( 130, 25 ) );
	item3->Add( l4, 0, wxALIGN_LEFT | wxALL, 5 );

	switchd = new wxTextCtrl( item1, ID_switchd, _( "" ), wxDefaultPosition, wxDefaultSize );
	item4->Add( switchd, 0, wxALIGN_RIGHT | wxALL, 5 );

	wxStaticText * l5 = new wxStaticText( item1, -1, wxT("Speed up"), wxDefaultPosition, wxSize( 100, 25 ) );
	item3->Add( l5, 0, wxALIGN_LEFT | wxALL, 5 );

	sup = new wxTextCtrl( item1, ID_sup, _( "" ), wxDefaultPosition, wxDefaultSize );
	item4->Add( sup, 0, wxALIGN_RIGHT | wxALL, 5 );

	logfile = new wxCheckBox( item1, ID_logfile, wxT("Save Logfile"), wxDefaultPosition, wxDefaultSize );
	logf = new wxCheckBox( item1, ID_logfileappend, wxT("Append to old Logfile"), wxDefaultPosition, wxDefaultSize );

	loglevel0 = new wxCheckBox( item1, ID_loglevel0, wxT("Loglevel: quiet"), wxDefaultPosition, wxDefaultSize );
	loglevel1 = new wxCheckBox( item1, ID_loglevel1, wxT("Loglevel: medium"), wxDefaultPosition, wxDefaultSize );
	loglevel2 = new wxCheckBox( item1, ID_loglevel2, wxT("Loglevel: full"), wxDefaultPosition, wxDefaultSize );
	loglevel0->SetValue( true );
	loglevel1->SetValue( false );
	loglevel2->SetValue( false );
	item3->Add( logfile, 0, wxALIGN_LEFT | wxALL, 5 );
	item3->Add( logf, 0, wxALIGN_LEFT | wxALL, 5 );
	item3->Add( loglevel0, 0, wxALIGN_LEFT | wxALL, 5 );
	item3->Add( loglevel1, 0, wxALIGN_LEFT | wxALL, 5 );
	item3->Add( loglevel2, 0, wxALIGN_LEFT | wxALL, 5 );

	wxButton * bdef = new wxButton( item1, ID_RDEFAULT, _( "Restore Run options" ), wxDefaultPosition, wxDefaultSize, 0 );
	main2->Add( bdef, 0, wxALIGN_CENTER_VERTICAL | wxALL, 5 );

	wxButton * bok = new wxButton( item1, ID_OK, _( "OK" ), wxDefaultPosition, wxDefaultSize, 0 );
	main2->Add( bok, 0, wxALIGN_CENTER_VERTICAL | wxALL, 5 );

	wxButton * bcan = new wxButton( item1, ID_Cancel, _( "Cancel" ), wxDefaultPosition, wxDefaultSize, 0 );
	main2->Add( bcan, 0, wxALIGN_CENTER_VERTICAL | wxALL, 5 );

	UserData ud;
	wxString dat,dat2;
	int i;
    char* names[9];
    ElleGetOptNames(names);
	wxStaticBoxSizer *vsizer=new wxStaticBoxSizer(wxVERTICAL,item1,wxT("UserData"));
	ElleUserData(ud);
	ugrid=new wxGrid(item1, -1, wxDefaultPosition,wxSize(280,300));
	ugrid->EnableEditing( true );
	ugrid->CreateGrid( 0, 2 );
	//ugrid->SetColumnWidth(1, 120);
	ugrid->SetColLabelValue( 0, wxT("Number") );
	ugrid->SetColLabelValue( 1, wxT("Data") );
	for (i=0;i<MAX_U_DATA;i++)
	{
		ugrid->AppendRows( 1 );
		dat.Printf( "%E", ud[i] );
		dat2.Printf("%s",names[i]);
		//row,col
		ugrid->SetCellValue( i, 0, dat2 );
		ugrid->SetCellValue( i, 1, dat );
	}
	ugrid->AutoSizeColumns(true);
	wxStaticText * tx = new wxStaticText( item1, -1, wxT("Not all of the displayed values are necessarily used.") );
	wxStaticText * tx1 = new wxStaticText( item1, -1, wxT("This depends on the process. Therefore, undefined") );
	wxStaticText * tx2 = new wxStaticText( item1, -1, wxT("values (usually very high or very low) might show up"));
	wxStaticText * tx3 = new wxStaticText( item1, -1, wxT("in the table.  Just pretend they are not there!") );
	vsizer->Add( ugrid, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
	vsizer->Add( tx, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
	vsizer->Add( tx1, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
	vsizer->Add( tx2, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
	vsizer->Add( tx3, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
	main->Add( vsizer );
	Centre();
	Init();
}

void RunDia::Init()
{
	wxString val;
	ElleGetRunOptionValue( RO_SVFREQ, & vsfreq );
	val.Printf( "%d", vsfreq );
	sinterv->SetValue( val );

	ElleGetRunOptionValue( RO_STAGES, & vstages );
	val.Printf( "%d", vstages );
	stages->SetValue( val );

	ElleGetRunOptionValue( RO_SWITCHDIST, ( int * ) & vsdist );
	val.Printf( "%lf", vsdist );
	switchd->SetValue( val );

	ElleGetRunOptionValue( RO_SPEEDUP, ( int * ) & vsup );
	val.Printf( "%lf", vsup );
	sup->SetValue( val );

	Settings * options = GetUserOptions();
	DSettings *display_options=GetDisplayOptions();
	options->GetUpdFreqOption(&displayfrequency);
	val.Printf( "%d", displayfrequency );
	dfreq->SetValue( val );

	char * name;
	name = ElleSaveFileRoot();
	sfiler->SetValue( name );

	loglevel = display_options->GetLogLevel();

	Settings * user_options = GetUserOptions();
	if ( user_options->GetLogFile() )
	{
		logfile->SetValue( true );
		logf->Enable( true );
		if ( user_options->GetLogAppend() )
			logf->SetValue( true );
		else
			logf->SetValue( false );
	}
	else
	{
		logf->Enable( false );
		logfile->SetValue( false );
	}
	if ( loglevel == 0 )
	{
		loglevel0->SetValue( true );
		loglevel1->SetValue( false );
		loglevel2->SetValue( false );
	}
	else if ( loglevel == 1 )
	{
		loglevel0->SetValue( false );
		loglevel1->SetValue( true );
		loglevel2->SetValue( false );
	}
	else if ( loglevel == 2 )
	{
		loglevel0->SetValue( false );
		loglevel1->SetValue( false );
		loglevel2->SetValue( true );
	}
	ElleCopyRunOptions();
}

void RunDia::OnOk( wxCommandEvent & event )
{
	Settings * user_options = GetUserOptions();
	DSettings * display_options = GetDisplayOptions();
	vstages = atoi( stages->GetValue() );
	ElleSetStages( vstages );

	displayfrequency=atoi(dfreq->GetValue());
	user_options->SetOptionValue(O_UPDFREQ,(double*)&displayfrequency);

	vsfreq = atoi( sinterv->GetValue() );
	ElleSetSaveFrequency( vsfreq );

	( switchd->GetValue() ).ToDouble( & vsdist );
	ElleSetSwitchdistance( vsdist );

	( sup->GetValue() ).ToDouble( & vsup );
	ElleSetSpeedup( vsup );

	vsfiler = ( sfiler->GetValue() );
	ElleSetSaveFileRoot( ( char * ) ( vsfiler.c_str() ) );

	user_options->SetLogFile( logfile->GetValue() );
	if ( !logfile->GetValue() )
		logf->SetValue( false );
	user_options->SetLogAppend( logf->GetValue() );

	display_options->SetLogLevel( loglevel );

	//UserData
	UserData ud;
	int row;
	wxString sval;
	double dval;
	ElleUserData(ud);
	for(row=0;row<9;row++)
	{
		sval=ugrid->GetCellValue(row,1);
		dval=atof(sval.c_str());
		ud[row]=dval;
	}
	ElleSetUserData(ud);
	EndModal( wxID_OK );
}

void RunDia::OnLogLevel0( wxCommandEvent & event )
{
	loglevel0->SetValue( true );
	loglevel1->SetValue( false );
	loglevel2->SetValue( false );
	loglevel = 0;
}

void RunDia::OnLogLevel1( wxCommandEvent & event )
{
	loglevel0->SetValue( false );
	loglevel1->SetValue( true );
	loglevel2->SetValue( false );
	loglevel = 1;
}

void RunDia::OnLogLevel2( wxCommandEvent & event )
{
	loglevel0->SetValue( false );
	loglevel1->SetValue( false );
	loglevel2->SetValue( true );
	loglevel = 2;
}

void RunDia::OnReset( wxCommandEvent & event )
{
	ElleRestoreRunOptions();
	Init();
	Refresh( false, NULL );
}

void RunDia::OnCancel( wxCommandEvent & event )
{
	EndModal( wxID_CANCEL );
}

void RunDia::OnSavePics( wxCommandEvent & event )
{
	DSettings * display_options = GetDisplayOptions();
	display_options->SetSavePic( true );
	display_options->SetSavePicFName( wxFileSelector( 
                    wxT("Choose filename to save the pic"), 
                    wxT(""), wxT(""), wxT(""),
	               wxT("PNG files (*.png)|*.png|JPEG files (*.jpg)|*.jpg|PCX files (*.pcx)|*.pcx|BMP files (*.bmp)|*.bmp|All Files (*)|*") ,wxSAVE) );
}

