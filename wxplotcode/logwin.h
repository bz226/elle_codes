// Written in 2003
// Author: Dr. J.K. Becker 
// Copyright: Dr. J.K. Becker (becker@jkbecker.de)

#include "wx/thread.h"
#include "wx/wx.h"

class LogWin: public wxThread
{
	DECLARE_CLASS(LogWin)
	public:
		LogWin(wxWindow parent);
		wxLogWindow *logframe;
}
