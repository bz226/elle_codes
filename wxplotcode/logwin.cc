// Written in 2003
// Author: Dr. J.K. Becker 
// Copyright: Dr. J.K. Becker (becker@jkbecker.de)

#include "logwin.h"

IMPLEMENT_CLASS(LogWin, wxThread)

LogWin::LogWin(wxWindow parent)
{
	Create();
	logframe=new wxLogWindow(parent,"Logging window",false,false);
	Run();
	logframe->Show(true);
}
