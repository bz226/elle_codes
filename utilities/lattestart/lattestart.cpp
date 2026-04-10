//-----------------------------------------------------------------------------
// GCC implementation
//-----------------------------------------------------------------------------

#ifdef __GNUG__
 #pragma implementation "lattestart.h"
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

#include <locale.h>

//-----------------------------------------------------------------------------
// Header of this .cpp file
//-----------------------------------------------------------------------------

#include "lattestart.h"

//-----------------------------------------------------------------------------
// Remaining headers: Needed wx headers, then wx/contrib headers, then application headers
//-----------------------------------------------------------------------------

#include "wx/image.h"               // wxImage

//-----------------------------------------------------------------------------

#include "wx/xrc/xmlres.h"          // XRC XML resouces

//-----------------------------------------------------------------------------

#include "myframe.h"

#include "wx/msgdlg.h"

IMPLEMENT_APP( MyApp )

//-----------------------------------------------------------------------------
// Public methods
//-----------------------------------------------------------------------------

// 'Main program' equivalent: the program execution "starts" here
bool MyApp::OnInit()
{

	wxString lattestartxrcloc;

	setlocale( LC_ALL, "en_US" );

	wxImage::AddHandler( new wxXPMHandler );

	wxXmlResource::Get() ->InitAllHandlers();

	wxString loca = argv[0];

	if (wxIsAbsolutePath(loca))
		wxSetWorkingDirectory(wxPathOnly(loca));
	else
	{
		loca.Prepend(wxGetCwd());
		wxSetWorkingDirectory(wxPathOnly(loca));
	}

	lattestartxrcloc = loca+wxT(".xrc");
	//cout << lattestartxrcloc.ToAscii() << endl;

	if ( !wxFileExists(lattestartxrcloc) )
	{

		wxPathList loc;
		loc.AddEnvList( wxT( "PATH" ) );
		lattestartxrcloc = loc.FindAbsoluteValidPath( wxT( "lattestart.xrc" ) );

		if ( lattestartxrcloc == wxEmptyString )
		{
			cout << "lattestart.xrc not found!" << endl;
			exit(1);
		}
	}

	//dialogs
	wxXmlResource::Get() ->Load( lattestartxrcloc );
	//frame
	//wxXmlResource::Get()->Load(wxT("frame.xrc"));

	wxToolTip::Enable( 1 );

	MyFrame *frame = new MyFrame();

	// Show the frame.
	frame->Show( true );

	// Return true to tell program to continue (false would terminate).
	return true;
}

