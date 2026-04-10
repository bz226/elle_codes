// Written in 2003
// Author: Dr. J.K. Becker
// Copyright: Dr. J.K. Becker (becker@jkbecker.de)

#ifndef _E_canbit_h
  #define _E_canbit_h

#include "wx/tipwin.h"
  #include "wx/wx.h"
  #include "wx/colordlg.h"
  #include "wx/font.h"
  #include "wx/spinctrl.h"
  #include "wx/scrolwin.h"
  #include "wx/image.h"
  #include <math.h>
  #include <cstring>
  #include <stdlib.h>
  #include "preferences.h"
  #include "nodes.h"
  #include "interface.h"
  #include "init.h"
  #include "file.h"
  #include "unodes.h"
  #include "segments.h"
  #include "general.h"
  #include "check.h"
  #include "polygon.h"
/*! Make class preferences known */
class Preferences;



/*! Class Canvas */
class Canvas : public wxScrolledWindow {
    /*!Declare class to wxWidgets */
    DECLARE_CLASS( Canvas )
public : /*!Constructor for new Canvas. The panel has OpenGL capabilities. ::Canvas */
    Canvas( wxWindow * parent, int vsize, int hsize );
wxDC *psDC;    
/*!Bool variables to store lots of settings */
    bool enablezoom,mzoom, bnodeshow, lineshow, flynnshow, flynnshownumbers, getinfobnode, getinfounode;
    bool init3d, nodeshownumbers, unodeshow, unodes3d, markpoly;
    /*!Variables to store pens (here for bnodes and lines) */
    wxPen bnodepen, linepen;
    /*!Variable to store the bnode brush. Used for the color of bnodes */
    wxBrush bnodebrush;
    /*!Stores the attribute (e.g. of flynns etc.). */
    int attribute;
    /*!Used to store the content of the screen in. Usefull for saving out pictures */
    //wxBitmap bitmap;
    int bnodesize, useunodeattrib, usesegmentattrib, GLBaseFont;
    void OnPaint( wxPaintEvent & event );
    /*!Draws all Bnodes */
    void ElleDrawBNodes();
    /*!Draws all unodes */
    void ElleDrawUNodes( int attrib );
    /*!Draws all segments */
    void ElleDrawSegments( int attrib );
    /*!Draws the flynn numbers. */
    void ElleShowFlynnNumbers( int index, float * x, float * y, int num );
    void ElleShowFlynnNumbers();
    /*!Draws all the node numbers. Only usefull if in zooming. See the code Canvas.cc */
    void ElleShowNodeNumbers();
    void OnOverlay();
    void OnSaveSingle( wxString filename );
    void ChangeView( int );
    void OnMouse( wxMouseEvent & event );
    void Init();
    void ClearScreen();
    void GLDrawPolygon( double coords[] [3], int num, int r, int g, int b );
    void GLDrawLines( double coords[] [3], int num, int r, int g, int b );
    void GLDrawLinesBound( double coords[] [3], int num, int r, int g, int b );
    void GLDrawSingleBNode( double x, double y, int r, int g, int b );
    void ResetScreen();
    void UnZoom();
    void OnSize( wxSizeEvent & event );
    void CheckUnitCell();
	wxBitmap* GetImage();
	void DrawToPS(bool t);
wxBitmap *image;
private:
	bool isinit,printps;
    void SetClip(wxSize s, wxDC * dc);
    double psc[1000][3];
    double zoomfactor;
    int pcount;
    bool leftclick;
    void IMDrawUNodes( int attrib );
void PSDrawUNodes( int attrib );
    void IMDrawSegments( int attrib );
void PSDrawSegments( int attrib );
wxMemoryDC temp_dc;
	int viewstartx,viewstarty;
	int orig_width,orig_height;
    void SetScrollbarsTicks(void);
    void GLDrawTransparentPolygon( double coords[] [3], int num, int r, int g, int b );
    void OnKey( wxKeyEvent & event );
    void MarkPolygon( double, double );
    void EndSession( wxCloseEvent & event );
void IMDrawLines( double coords[] [3], int num, int r, int g, int b );
void PSDrawLines( double coords[] [3], int num, int r, int g, int b);
void IMDrawSingleBNode( double x, double y, int r, int g, int b );
void PSDrawSingleBNode( double x, double y, int r, int g, int b );
 void IMDrawPolygon( double coords[] [3], int num, int r, int g, int b );
 void PSDrawPolygon( double coords[] [3], int num, int r, int g, int b );
void IMDrawLinesBound( double coords[] [3], int num, int r, int g, int b );
void PSDrawLinesBound( double coords[] [3], int num, int r, int g, int b );
 void IMShowFlynnNumbers( int index, float * x, float * y, int num );
  void PSShowFlynnNumbers( int index, float * x, float * y, int num );
  void IMShowNodeNumbers();
  void PSShowNodeNumbers();
DECLARE_EVENT_TABLE()
};

#endif //_E_canbit_h
