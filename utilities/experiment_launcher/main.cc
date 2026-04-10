/*!\file main.cpp*/
#include "main.h"
#include <string>
#include "file_utils.h"

BEGIN_EVENT_TABLE( mainwinFrame,wxFrame )
EVT_MENU( Menu_File_Quit, mainwinFrame::OnQuit )
EVT_MENU( Menu_File_About, mainwinFrame::OnAbout )
EVT_BUTTON( ID_BUTTONGO, mainwinFrame::OnButtonGo)
//EVT_BUTTON( ID_BUTTONSTOP, mainwinFrame::OnButtonStop)
EVT_BUTTON( ID_BUTTONGO2, mainwinFrame::OnButtonGo2)
EVT_LISTBOX( ID_ListBox, mainwinFrame::OnListBox )
EVT_LISTBOX( ID_ListBox2, mainwinFrame::OnListBox2 )
END_EVENT_TABLE()

IMPLEMENT_APP( mainwinapp )


bool mainwinapp::OnInit()
{
	int hSize,vSize;
    wxString arg0 = argv[0];
	mainwinFrame * frame = new mainwinFrame( wxT( "Elle-book examples"), wxPoint( 50, 50 ), wxSize( 300, 300 ), arg0 );
	SetTopWindow( frame );
	frame->Init();
	frame->Show(true);
	frame->GetClientSize(&hSize,&vSize);
	frame->Fit();
	return TRUE;
}

mainwinFrame::mainwinFrame( const wxString & title, const wxPoint & pos,
const wxSize & size, const wxString & arg0 ) :
		wxFrame( ( wxFrame * ) NULL, 11, title, pos, size )
{
	wxString path = wxGetenv("ELLEPATH");
	if (path.empty())
		path = arg0.BeforeLast(E_DIR_SEPARATOR);
	else {
		path.Append(E_DIR_SEPARATOR);
		path.Append("binwx");
	}
	path.Append(E_DIR_SEPARATOR);
	LoadFileMulti( path + "examples.txt");
	LoadFileSingle(path + "single.txt");
	wxString description;
	wxMenu * menuFile = new wxMenu;

	menuFile->Append( Menu_File_About, wxT( "&About..." ) );
	menuFile->AppendSeparator();
	menuFile->Append( Menu_File_Quit, wxT( "E&xit" ) );

	wxMenuBar * menuBar = new wxMenuBar;
	menuBar->Append( menuFile, wxT( "&File" ) );

	SetMenuBar( menuBar );
	CreateStatusBar( 2 );
	SetStatusText( wxT( "Welcome to the Elle Experiment Launcher" ), 0 );
	/*Starting the button-stuff
	 * 
	 * First, we need a sizer to make things look nice.
	 */
	wxBoxSizer * base = new wxBoxSizer( wxVERTICAL );
	this->SetSizer( base );
	this->SetAutoLayout( TRUE );

	wxBoxSizer * erster = new wxBoxSizer( wxHORIZONTAL );
	wxBoxSizer * zweiter = new wxBoxSizer( wxHORIZONTAL );

	wxBoxSizer *e1= new wxBoxSizer( wxVERTICAL );
	wxBoxSizer *e2= new wxBoxSizer( wxVERTICAL );
	wxStaticText *l1=new wxStaticText(this, -1, "Examples", wxDefaultPosition,wxDefaultSize, wxALIGN_LEFT);
	wxStaticText *l2=new wxStaticText(this, -1, "Description", wxDefaultPosition,wxDefaultSize, wxALIGN_LEFT);
	lb=new wxListBox(this, ID_ListBox, wxDefaultPosition, wxDefaultSize, 0, NULL, wxLB_SINGLE);
	for(int n=0;n<tfile.size();n++)
		lb->Append(tfile.at(n).name);
	//now the textfield. First, we put the description of the textfield into the variable description
	description=wxT("Welcome to the experiment launcher.\n\nPlease choose an example from the list on the left, then click Go");
	text=new wxTextCtrl(this, wxID_ANY, description, wxDefaultPosition, wxSize(250,100),wxTE_MULTILINE|wxTE_READONLY);

	erster->Add( e1, 0, wxALIGN_LEFT | wxALL, 5 );
	erster->Add( e2, 0, wxALIGN_LEFT | wxALL, 5 );
	e1->Add( l1, 0, wxALIGN_LEFT | wxALL, 5 );
	e1->Add( lb, 0, wxALIGN_LEFT | wxALL, 5 );
	e2->Add( l2, 0, wxALIGN_LEFT | wxALL, 5 );
	e2->Add( text, 0, wxALIGN_LEFT | wxALL, 5 );

	//everything is sorted left to right in this sizer, so the button first
	wxButton * button1 = new wxButton( this, ID_BUTTONGO, "Go", wxDefaultPosition, wxDefaultSize);
	//wxButton * button3 = new wxButton( this, ID_BUTTONSTOP, "Stop", wxDefaultPosition, wxDefaultSize);
	//and now we add it to the base sizer
	zweiter->Add( button1, 0, wxALIGN_LEFT | wxALL, 5 );
	//zweiter->Add( button3, 0, wxALIGN_LEFT | wxALL, 5 );



	wxBoxSizer * dritter = new wxBoxSizer( wxHORIZONTAL );
	wxBoxSizer *vierter=new wxBoxSizer(wxHORIZONTAL);
	wxBoxSizer *d1= new wxBoxSizer( wxVERTICAL );
	wxBoxSizer *d2= new wxBoxSizer( wxVERTICAL );
	wxStaticText *ld1=new wxStaticText(this, -1, "Processes/utilities", wxDefaultPosition,wxDefaultSize, wxALIGN_LEFT);
	wxStaticText *ld2=new wxStaticText(this, -1, "Description", wxDefaultPosition,wxDefaultSize, wxALIGN_LEFT);
	lbd=new wxListBox(this, ID_ListBox2, wxDefaultPosition, wxDefaultSize, 0, NULL, wxLB_SINGLE);
for(int n=0;n<dfile.size();n++)
        lbd->Append(dfile.at(n).name);
	//now the textfield. First, we put the description of the textfield into the variable description
	description=wxT("Start utilities from here.\n\nNo input file will be loaded on startup.");
	textd=new wxTextCtrl(this, wxID_ANY, description, wxDefaultPosition, wxSize(250,100),wxTE_MULTILINE|wxTE_READONLY);

	dritter->Add( d1, 0, wxALIGN_LEFT | wxALL, 5 );
	dritter->Add( d2, 0, wxALIGN_LEFT | wxALL, 5 );
	d1->Add( ld1, 0, wxALIGN_LEFT | wxALL, 5 );
	d1->Add( lbd, 0, wxALIGN_LEFT | wxALL, 5 );
	d2->Add( ld2, 0, wxALIGN_LEFT | wxALL, 5 );
	d2->Add( textd, 0, wxALIGN_LEFT | wxALL, 5 );


	//everything is sorted left to right in this sizer, so the button first
	wxButton * button2 = new wxButton( this, ID_BUTTONGO2, "Go", wxDefaultPosition, wxDefaultSize);
	//and now we add it to the base sizer
	vierter->Add( button2, 0, wxALIGN_LEFT | wxALL, 5 );
	wxStaticBoxSizer *up=new wxStaticBoxSizer(wxVERTICAL,this,"Experiments");
	wxStaticBoxSizer *down=new wxStaticBoxSizer(wxVERTICAL,this,"Utilities");
	//Finally, we add our new sizer to the base sizer so that it shows.
	up->Add( erster, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
	up->Add( zweiter, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
	down->Add( dritter, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
	down->Add( vierter, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
	base->Add( up, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
	base->Add( down, 0, wxALIGN_CENTER_HORIZONTAL | wxALL, 5 );
}

void mainwinFrame::OnQuit( wxCommandEvent & WXUNUSED( event ) )
{
	Close( TRUE );
}

void mainwinFrame::OnAbout( wxCommandEvent & WXUNUSED( event ) )
{
	wxMessageBox( wxT( "To Run an Experiment, select it from the list and click on Go.\nThis thing was written by\n\nDr. J.K. Becker\nbecker@jkbecker.de" ), wxT( "About" ),
	              wxOK | wxICON_INFORMATION, this );
}

void mainwinFrame::Init()
{

	Refresh( true, NULL );
}


/*!This sets the name of the script and executes it by calling the OnExecute function.
 */
long pid = 0;
void mainwinFrame::OnButtonGo(wxCommandEvent &event)
{
    if(!escript.Contains("null")) // obvious hack to prevent top level experiments giving an error, but could blow up if process or path has null in it!
	// permission denied to kill pid
	pid = wxExecute(escript, wxEXEC_ASYNC);
}

/*
void mainwinFrame::OnButtonStop(wxCommandEvent &event)
{
	wxKillError rc;
	extern long pid;
	wxKill(pid, wxSIGKILL, &rc);//this kills sh, need to kill children
	printf("killing %ld rc=%d\n",pid,rc);
}
*/

void mainwinFrame::OnButtonGo2(wxCommandEvent &event)
{
    if(!escript2.Contains("null")) // obvious hack to prevent top level experiments giving an error, but could blow up if process or path has null in it!
	wxExecute(escript2, wxEXEC_ASYNC|wxEXEC_NOHIDE);
}

void mainwinFrame::OnListBox(wxCommandEvent &event)
{
	wxString sel;
	int in;
	in = lb->GetSelection();
	sel = lb->GetString( in );
    in=FindPosMulti(sel);
    SetMulti(tfile.at(in).description,tfile.at(in).command);
}
void mainwinFrame::OnListBox2(wxCommandEvent &event)
{
	wxString sel;
	int in;
	in = lbd->GetSelection();
	sel = lbd->GetString( in );
    in=FindPosSingle(sel);
    SetSingle(dfile.at(in).description,dfile.at(in).command);
}

void mainwinFrame::SetMulti(wxString desc,wxString script)
{
#ifdef __WIN32__
	wxString E_SCRIPT_EXT = ".bat";
	script.Replace("/","\\");
#else
	wxString E_SCRIPT_EXT = ".shelle";
	script.Replace("\\","/");
#endif
	text->SetValue(desc);
	script.Append(E_SCRIPT_EXT);
	if (!wxFileExists(script))
	{
	  wxString path = wxGetenv("ELLEPATH");
	  if (!path.empty())
	  {
		path+=E_DIR_SEPARATOR;
		path+="..";
		path+=E_DIR_SEPARATOR;
		path+="experiments";
		path+=E_DIR_SEPARATOR;
		script = path+script;
	  }
	}
#ifdef __WIN32__
	//script.Append(" /MIN /WAIT");
/*
  this call requires starting dir and title to be set in the bat file
 */
	script.Append(" /WAIT");
/*
   the following lines allow starting dir and title to be
   passed to the experiment cmd win but the original cmd window
   is also created
	wxString workdir= script.BeforeLast(E_DIR_SEPARATOR);
	wxString title = script.AfterLast(E_DIR_SEPARATOR);
	script = title;
	script.Prepend(" /D"+workdir+" /WAIT ");
	script.Prepend("cmd /C start \""+title+'"');
 */
#else
	script.Prepend( _T("xterm -e "));
#endif
	escript=script;
}

void mainwinFrame::SetSingle(wxString desc,wxString script)
{
	textd->SetValue(desc);
#ifdef __WIN32__
	std::string E_BROWSER_CMD = "rundll32 url.dll,FileProtocolHandler";
	wxString E_SCRIPT_EXT = ".bat";
	script.Replace("/","\\");
#else
	std::string E_BROWSER_CMD = "mozilla";
	wxString E_SCRIPT_EXT = "";
	script.Replace("\\","/");
#endif
	wxString path = wxGetenv("ELLEPATH");
	if (!path.empty())
	{
		path+=E_DIR_SEPARATOR;
		path+="binwx";
		path+=E_DIR_SEPARATOR;
	}
	if (!wxFileExists(script)) script = path+script;

	std::string extstr = script.c_str();
	if (extstr.rfind(".htm")!=std::string::npos) //check for "htm"  too
	{
	  wxString cmd(E_BROWSER_CMD.c_str());
	  // browser needs full path
          if (script.Find(path)==-1) script = path+script;
	  script.Prepend(cmd + " ");
	//wxLogError("Do not know how to open files of type html");
	}
	else if (extstr.rfind("sybil")!=std::string::npos) //sybil needs .bat ext
	{
		script.Append(E_SCRIPT_EXT);
	}
	escript2=script;
}

bool mainwinFrame::LoadFileMulti(wxString filename)
{
	Data tmp;
	wxString t1,t2,t3;
	wxTextFile file;
	if(file.Open(filename))
	{
		t1=file.GetFirstLine();
		while(t1=="")
			t1=file.GetNextLine();
		t2=file.GetNextLine();
		t3=file.GetNextLine();
		while(!file.Eof())
		{
			tmp.name=t1;
			tmp.description=t2;
			tmp.command=t3;
			tfile.push_back(tmp);
			t1=file.GetNextLine();
			while(!file.Eof() && t1=="")
				t1=file.GetNextLine();
			if (!file.Eof()) t2=file.GetNextLine();
			if (!file.Eof()) t3=file.GetNextLine();
		}
		tmp.name=t1;
		tmp.description=t2;
		tmp.command=t3;
		tfile.push_back(tmp);
		return true;
	}
	else
	{
		printf("Could not open file, sorry for that.");
		return false;
	}
}

bool mainwinFrame::LoadFileSingle(wxString filename)
{
    Data tmp;
    wxString t1,t2,t3;
    wxTextFile file;
    if(file.Open(filename))
    {
        t1=file.GetFirstLine();
        while(t1=="")
            t1=file.GetNextLine();
        t2=file.GetNextLine();
        t3=file.GetNextLine();
        while(!file.Eof())
        {
            tmp.name=t1;
            tmp.description=t2;
            tmp.command=t3;
            dfile.push_back(tmp);
            if (!file.Eof()) t1=file.GetNextLine();
            while(!file.Eof() && t1=="")
                t1=file.GetNextLine();
            if (!file.Eof()) t2=file.GetNextLine();
            if (!file.Eof()) t3=file.GetNextLine();
        }
	tmp.name=t1;
	tmp.description=t2;
	tmp.command=t3;
	dfile.push_back(tmp);
	return true;
    }
    else
    {
        printf("Could not open file, sorry for that.");
        return false;
    }
}

int mainwinFrame::FindPosMulti(wxString name)
{
	int n=0,pos=0;
	while(name!=tfile.at(n).name && n<tfile.size())
		n++;
	return n;
}

int mainwinFrame::FindPosSingle(wxString name)
{
    int n=0,pos=0;
    while(name!=dfile.at(n).name && n<dfile.size())
        n++;
    return n;
}
