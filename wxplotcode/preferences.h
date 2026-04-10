// Written in 2003
// Author: Dr. J.K. Becker
// Copyright: Dr. J.K. Becker (becker@jkbecker.de)
#ifndef _E_preferences_h
  #define _E_preferences_h

  #include "wx/wx.h"
  #include "wx/colordlg.h"
  #include "wx/font.h"
  #include "wx/spinctrl.h"
  #include "wx/scrolwin.h"
  #include "wx/image.h"
  #include "wx/notebook.h"
  #include <math.h>
  //#include "wx/glcanvas.h"
  //#include <GL/gl.h>
  //#include <GL/glu.h>
  #include <stdlib.h>
  #include "wx/grid.h"

//////////Elle-Zeug
  #include "nodes.h"
  #include "interface.h"
  #include "init.h"
  #include "file.h"
  #include "unodes.h"
  #include "general.h"

enum
{
    ID_PCancel, ID_POk, ID_PApply, ID_PDColor, ID_PTColor, ID_PLColor, ID_Pbnodesize, ID_Plinesize, ID_Pbnodeshow,
    ID_Pbitmapsize, ID_Pflynnshownumbers, ID_Pcolorsingleflynn, ID_Pnodeshownumbers, ID_Unodes3d, ID_PResetSingleFlynnColor,
    ID_Prangeflagunodes, ID_punoderangeflagmaxval, ID_punoderangeflagminval, ID_Pnotrangeflagunodes,
    ID_punodenotrangeflagmaxval, ID_punodenotrangeflagminval, ID_PSave,
    ID_Prangeflagsegments, ID_psegmentrangeflagmaxval, ID_psegmentrangeflagminval, ID_Pnotrangeflagsegments,
    ID_psegmentnotrangeflagmaxval, ID_psegmentnotrangeflagminval,
ID_PLoad, ID_Punodetriangles,ID_FClampColor,ID_FCCMin,ID_FCCMax,ID_BClampColor,ID_BCCMin,ID_BCCMax,ID_NClampColor,ID_NCCMin,ID_NCCMax,ID_UClampColor,ID_UCCMin,ID_UCCMax,
ID_SClampColor,ID_SCCMin,ID_SCCMax,
    ID_punodelistbox,ID_pbnodelistbox,ID_pflynnlistbox,ID_plinelistbox,
    ID_psegmentlistbox,ID_makeunodes,ID_Punodesize,
    ID_makesegments,ID_Psegmentsize,ID_Prainchange,ID_DIALOG,ID_TEXTCTRL,ID_PANEL,ID_DEFAULT,
    ID_CSAVE ,ID_CLOAD,ID_sflynn,ID_sbnode,ID_sunode,ID_ssegment,ID_CTEXT,
};

#define CLAMP1 0.0039215686

class Canvas;

class CBarPane : public wxPanel
{
    DECLARE_CLASS( CBarPane )

public:
    CBarPane( wxWindow *parent,
                   const wxSize=wxDefaultSize);

    void ColormapUpdateColorDisp(wxDC &dc);
    void paintNow();
    void paintEvent( wxPaintEvent & evt );
    DECLARE_EVENT_TABLE()
};

class LinePane : public wxPanel
{
    DECLARE_CLASS( LinePane )

private:
    int linewdth;
    wxColor linecol;
public:
    LinePane( wxWindow *parent,
              const wxSize=wxDefaultSize,
              int wdth=1,wxColor col= *wxWHITE );

    void SetWidth(int wdth) { linewdth = wdth; }
    int GetWidth() { return (linewdth); }
    void SetColor(wxColor col) { linecol = col; }
    wxColor GetColor() { return (linecol); }
    void UpdateLineDisp(wxDC &dc);
    void paintNow();
    void paintEvent( wxPaintEvent & evt );
    DECLARE_EVENT_TABLE()
};

class DotPane : public wxPanel
{
    DECLARE_CLASS( DotPane )

private:
    int dotsize;
    wxColor dotcol;
public:
    DotPane( wxWindow *parent,
             const wxSize=wxDefaultSize,
             int size=1, wxColor col = *wxBLUE);

    void SetSize(int size) { dotsize = size; }
    int GetSize() { return (dotsize); }
    void SetColor(wxColor col) { dotcol = col; }
    wxColor GetColor() { return (dotcol); }
    void UpdateDotDisp(wxDC &dc);
    void paintNow();
    void paintEvent( wxPaintEvent & evt );
    DECLARE_EVENT_TABLE()
};


class Preferences : public wxDialog
{
    DECLARE_CLASS( Preferences )

public :
    Preferences( wxWindow * parent );
    //lewxBitmap bnodebit, linebit, tnodebit;
    wxPen pbnodepen, plinepen, ptnodepen;
    wxBrush pbnodebrush, ptnodebrush;
    wxString attribute;
    int nodesize, useunodea, attrib,battrib, usebnodea,unodesize,rainchange;
    int usesegmenta,segmentsize;
    wxListBox * cflynn, * cunode, * cbnode, * cline, * csegment;
    wxStaticText * bitmaplabel;
    bool plineshow, pflynnshownumbers, prangeflagunodes, pnotrangeflagunodes;
    bool prangeflagsegments, pnotrangeflagsegments;
    bool pnodeshownumbers, unodes3d;
private:
    wxToolBar * toolbar;
    wxNotebook * book;
    wxBitmap * toolbarbitmap[2];
    int GetAttributeInt( wxString welche,int type );
    wxTextCtrl * uvalmin,*uvalmax,*Fccmin,*Fccmax,*Bccmin,*Bccmax,*Nccmin,*Nccmax,*Uccmin,*Uccmax;
    wxTextCtrl * svalmin,*svalmax,*Sccmin,*Sccmax;
    double urangevalmax, urangevalmin, unotrangevalmax, unotrangevalmin;;
    double srangevalmax, srangevalmin, snotrangevalmax, snotrangevalmin;;
    wxString GetAttributeName( int welche,int type );
    void GetUserColor(wxColour *col);
    void OnClose( wxCloseEvent & event );
    void OnListBox(wxCommandEvent & event);
    void OnLoadPrefs( wxCommandEvent & event );
    void OnSavePrefs( wxCommandEvent & event );
    void OnDNodeColor( wxCommandEvent & event );
    void OnTNodeColor( wxCommandEvent & event );
    void OnApply( wxCommandEvent & event );
    void OnCancel( wxCommandEvent & event );
    void OnOk( wxCommandEvent & event );
    void OnLineColor( wxCommandEvent & event );
    void OnBNodeSize( wxSpinEvent & event );
    void OnLineSize( wxSpinEvent & event );
    void OnFlynnShowNumbers( wxCommandEvent & event );
    void OnNodeShowNumbers( wxCommandEvent & event );
    void OnColorSingleFlynn( wxCommandEvent & event );
    void OnResetSingleFlynn( wxCommandEvent & event );
    void OnUnodes3d(wxCommandEvent & event);
    void OnRangeFlagUnodes( wxCommandEvent & event );
    void OnRangeFlagSegments( wxCommandEvent & event );
    void OnUnodesRangeFlagTxtMax( wxCommandEvent & event );
    void OnUnodesRangeFlagTxtMin( wxCommandEvent & event );
    void OnNotRangeFlagUnodes( wxCommandEvent & event );
    void OnUnodesNotRangeFlagTxtMax( wxCommandEvent & event );
    void OnUnodesNotRangeFlagTxtMin( wxCommandEvent & event );
    void OnUNodesTriangulate(wxCommandEvent &event);
    void OnUNodeSize( wxSpinEvent & event );
    void OnSegmentsRangeFlagTxtMax( wxCommandEvent & event );
    void OnSegmentsRangeFlagTxtMin( wxCommandEvent & event );
    void OnNotRangeFlagSegments( wxCommandEvent & event );
    void OnSegmentsNotRangeFlagTxtMax( wxCommandEvent & event );
    void OnSegmentsNotRangeFlagTxtMin( wxCommandEvent & event );
    void OnSegmentSize( wxSpinEvent & event );
    void OnRainChange( wxSpinEvent & event );
    void OnPaint( wxPaintEvent & event );
    wxPanel * CreateSegmentPage( void );
    wxPanel * CreateUNodePage( void );
    wxPanel * CreateBNodePage( void );
    wxPanel * CreateLinePage( void );
    wxPanel * CreateFlynnPage( void );
    wxPanel * CreateCmapPage( void );
    wxPanel* ColormapCreateControls();
    void ColormapOnChangeColorPercent( wxCommandEvent & event );
    void ColormapOnDefault( wxCommandEvent & event );
    //void ColormapUpdateColorDisp(wxDC &dc);
    void ColormapUploadCmap();
    void ColormapOnLoad( wxCommandEvent & event );
    void ColormapOnSave( wxCommandEvent & event );
    void ColormapResetToDef();
    void ColormapOnRangeFlynn(wxCommandEvent & event);
    void ColormapOnRangeBNode(wxCommandEvent & event);
    void ColormapOnRangeUNode(wxCommandEvent & event);
    void ColormapOnRangeSegment(wxCommandEvent & event);
    void ColormapOnChangeColorValue(wxCommandEvent &event);
    void OnFCCMin( wxCommandEvent & event );
    void OnFCCMax( wxCommandEvent & event );
    void OnBCCMin( wxCommandEvent & event );
    void OnBCCMax( wxCommandEvent & event );
    void OnNCCMin( wxCommandEvent & event );
    void OnNCCMax( wxCommandEvent & event );
    void OnUCCMin( wxCommandEvent & event );
    void OnUCCMax( wxCommandEvent & event );
    void OnSCCMin( wxCommandEvent & event );
    void OnSCCMax( wxCommandEvent & event );
    void OnFClampColor( wxCommandEvent & event );
    void OnBClampColor( wxCommandEvent & event );
    void OnNClampColor( wxCommandEvent & event );
    void OnUClampColor( wxCommandEvent & event );
    void OnSClampColor( wxCommandEvent & event );
    CBarPane *cbar;
    LinePane *linecolsize;
    DotPane *tdotcolsize, *ddotcolsize;
    bool first, triangulate,fclampcolor,bclampcolor,nclampcolor,uclampcolor;
    bool sclampcolor;
    wxStaticText * fmin,*fmax;
    wxChoice * fselect,*bselect,*uselect,*sselect;
    wxStaticText * min,*max;
    double fccmin,fccmax,bccmin,bccmax,nccmin,nccmax,uccmin,uccmax;
    double sccmin,sccmax;
    DECLARE_EVENT_TABLE()
};


#endif //_E_preferences_h
