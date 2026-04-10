// Written in 2003
// Author: Dr. J.K. Becker
// Copyright: Dr. J.K. Becker (becker@jkbecker.de)

#ifndef _E_showelle_h
  #define _E_showelle_h


  #include "wx/wx.h"
  #include "wx/log.h"
  #include "wx/dir.h"
  #include "wx/utils.h"
  #include "canvas.h"
  #include "wx/image.h"
  #include "runopts.h"
  #include "preferences.h"
  #include "psprint.h"
  #include "icons.h"
  #include "file.h"
  #include "simwiz.h"
  #include "rundia.h"
  #include "wx/html/htmlwin.h"
  #include "display.h"
  #include "settings.h"
  #include "basetable.h"
  #include "interface.h"
  #include "file_utils.h"
  #include "dsettings.h"
  #include "zlib.h"
  #include "cppfile.h"
  #include "defs.h"
  #include "file_utils.h"
#include "wx/zipstrm.h"
#include "wx/wfstream.h"
#include "wx/txtstrm.h"
#include "plotaxes.elle.h"
   
/*#ifndef __WIN32__
//For offscreen rendering
  #include "GL/gl.h"
#include "GL/glx.h"
#include "GL/glu.h"
#endif*/

enum
{
	Tool_zoomin, Tool_zoomout, Tool_stop, Tool_left, Tool_right, Tool_up, Tool_down, Tool_getbnode, Tool_getunode,
    Tool_label, ID_Table_New_Unode, ID_Table_New_Bnode, ID_Table_New_Flynn, ID_Wizard, ID_Quit, ID_About, ID_OnlineHelp,
    ID_Open, ID_Make, ID_Graphics, ID_Prefs, ID_SPic, ID_GSaveRun, ID_GSaveSingle, ID_Overlay, ID_Zoom, ID_Run, ID_ReRun,
    ID_Save, ID_SaveAs, ID_RunDia, ID_Log, ID_Redraw, ID_PSPrint,ID_Overlay_always,ID_SaveStereo,
};




class ShowelleFrame : public wxFrame
{
    DECLARE_CLASS( ShowelleFrame )
public :
    ShowelleFrame( const wxString & title, const wxPoint & pos, const wxSize & size );
    Canvas * panel;
    void GetInfoBnode( double, double );
    void GetInfoUnode( double, double );
    void RepaintCanvas( bool );
void LoadArchive(wxString filename);
    wxBoxSizer * base;
    wxLogWindow * logframe;
private:
wxMenu * menuGraphic;
Preferences * prefs ;
    wxBitmap * toolbaricon[11];
    wxFrame * tool;
    wxTextCtrl * tooltext, * logtext;
    TableData * table;
    void OnPaint( wxPaintEvent & event );
    void OnRunDia( wxCommandEvent & event );
    void OnZoom(wxCommandEvent & event);
    void LoadFile( wxString filename );
    void CreateToolbarZoom( void );
    void OnShowLog( wxCommandEvent & event );
    void OnQuit( wxCommandEvent & event );
    void OnCloseWindow( wxCloseEvent & event );
    void OnAbout( wxCommandEvent & event );
    void OnPSPrint( wxCommandEvent & event );
    void OnOnline( wxCommandEvent & event );
    void OnOpen( wxCommandEvent & event );
    void OnSaveRun( wxCommandEvent & event );
    void OnPrefs( wxCommandEvent & event );
    void OnSaveSingle( wxCommandEvent & event );
    void OnOverlayAlways( wxCommandEvent& event);
    void OnNewTableUnode( wxCommandEvent & event );
    void OnNewTableBnode( wxCommandEvent & event );
    void OnNewTableFlynn( wxCommandEvent & event );
    void OnToolIn( wxCommandEvent & event );
    void OnToolOut( wxCommandEvent & event );
    void OnToolStop( wxCommandEvent & event );
    void OnGetInfoBnode( wxCommandEvent & event );
    void OnGetInfoUnode( wxCommandEvent & event );
    void OnLabel( wxCommandEvent & event );
    void OnSimWiz( wxCommandEvent & event );
    void OnRun(wxCommandEvent & event);
    void OnReRun(wxCommandEvent & event);
    void OnSave(wxCommandEvent & event);
    void OnSaveAs(wxCommandEvent & event);
    void OnSaveStereo( wxCommandEvent & event );
    void OnRedraw( wxCommandEvent & event );
    void OnSize( wxSizeEvent & event );
    void OnEraseBackground( wxEraseEvent & event );
    void SaveArchive(wxString filename);
	void PSPrint(void);
DECLARE_EVENT_TABLE()
};


class PrintPS:public wxPrintout
{
	public:
	PrintPS(Canvas *im, wxChar *title = _T("PrintMe")):wxPrintout(title){ pscan=im;}
	bool OnPrintPage(int page);
	void GetPageInfo(int *minPage, int *maxPage,int *pageFrom, int *pageTo);
	bool HasPage(int pageNum);
	private:
			Canvas *pscan;
};
#endif
