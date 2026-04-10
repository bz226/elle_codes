// Written in 2004
// Author: Dr. J.K. Becker
// Copyright: Dr. J.K. Becker (becker@jkbecker.de)
#ifndef _E_table_h
  #define _E_table_h

  #include "wx/wx.h"
  #include "wx/grid.h"
  #include "canvas.h"
  #include "icons.h"
  #include "display.h"
  #include "defs.h"


enum
{
  Tsave, Trow, Tmpoly, Tedit, Tdselect,TSaveSelected,
};

using namespace std;

class TableData : public wxFrame
{
  DECLARE_CLASS( TableData )
public :
  TableData( wxWindow * parent, int id,wxString title );
  void SetValueD( int, int, double );
  double GetValueD( int, int );
  void SetValueI( int, int, int );
  int GetValueI( int, int );
  void SetValueS( int, int, wxString );
  wxString GetValueS( int, int );
  void SetColHeader( int, wxString );
  void AddCol();
  void AddRow();
  bool DecideMark( double coords[1000] [3], int count );
  void CloseTable(wxCommandEvent & event);
void AutoSize(void);
private:
  wxBitmap * toolbaricon[5];;
  wxGrid * tgrid;
  bool editselcell;
  void OnSaveData( wxCommandEvent & event );
  void MarkBNodes( double coords[1000] [3], int count );
  void MarkUNodes( double coords[1000] [3], int count );
  void MarkFlynns( double coords[1000] [3], int count );
  void OnClickLabel( wxGridEvent & event );
  int GetFlynnIndex( double coords[1000] [3], int count );
  void OnEdit(wxCommandEvent & event);
  void OnMarkPolygon( wxCommandEvent & event );
  void MarkRows( int index[], int count );
  void OnCellEdit( wxGridEvent & event );
  int GetPolIndexBNodes( double coords[1000] [3], int count);
  int GetPolIndexUNodes( double coords[1000] [3], int count );
  void OnDeselectRows(wxCommandEvent & event);
  void EndSession(wxCloseEvent &event);
  int GetColHeaderInt(int col);
  int GetRowHeaderInt(int row);
  void EditSelCells(int col);
  void OnSaveSelected(wxCommandEvent &event);
  DECLARE_EVENT_TABLE()
};

#endif
