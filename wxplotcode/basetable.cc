// Written in 2004
// Author: Dr. J.K. Becker
// Copyright: Dr. J.K. Becker (becker@jkbecker.de)

#include "basetable.h"


IMPLEMENT_CLASS( TableData, wxFrame )


BEGIN_EVENT_TABLE( TableData, wxFrame ) EVT_TOOL( Tsave, TableData::OnSaveData ) EVT_TOOL( Tmpoly, TableData::OnMarkPolygon )
     EVT_TOOL( TSaveSelected, TableData::OnSaveSelected ) EVT_TOOL( Tedit, TableData::OnEdit )
     EVT_TOOL( Tdselect, TableData::OnDeselectRows ) EVT_GRID_LABEL_LEFT_CLICK( TableData::OnClickLabel )
     EVT_GRID_CELL_LEFT_CLICK( TableData::OnCellEdit ) EVT_CLOSE( TableData::EndSession ) END_EVENT_TABLE()


     TableData::TableData( wxWindow * parent, int id, wxString title ) : wxFrame( parent, id, title, wxPoint( 100, 100 ), wxSize( 430, 380 ) )
     {
       tgrid = new wxGrid( this, -1, wxDefaultPosition, wxDefaultSize );
       tgrid->EnableEditing( false );
       tgrid->CreateGrid( 1, 1 );
       tgrid->SetColLabelValue( 0, "Number" );
	     tgrid->AutoSizeColumns(false);
       wxToolBar * toolbar = this->CreateToolBar( wxNO_BORDER, -1, "ToolBar" );
       toolbaricon[0] = new wxBitmap( save_xpm );
       toolbaricon[1] = new wxBitmap( saveselected );
       toolbaricon[2] = new wxBitmap( markflynn );
       toolbaricon[3] = new wxBitmap( editor_xpm );
       toolbaricon[4] = new wxBitmap( unmarkrow );
       toolbar->AddTool( Tsave, "Save data", * ( toolbaricon[0] ), "Save data to seperate file", wxITEM_NORMAL );
       toolbar->AddTool( TSaveSelected, "Save selected data", * ( toolbaricon[1] ), "Save only selected data", wxITEM_NORMAL );
       toolbar->AddTool( Tmpoly, "Mark polygon", * ( toolbaricon[2] ), "Mark polygon", wxITEM_NORMAL );
       toolbar->AddTool( Tedit, "Edit selected", * ( toolbaricon[3] ), "Edit selected", wxITEM_NORMAL );
       toolbar->AddTool( Tdselect, "Deselect all", * ( toolbaricon[4] ), "Deselect all", wxITEM_NORMAL );
       toolbar->Realize();
}

void TableData::EndSession( wxCloseEvent & event )
{
  this->Destroy();
}

void TableData::OnEdit( wxCommandEvent & event )
{
  wxMessageBox( "You will change all marked values.\nClick on the colum you want to change", "Message", wxOK | wxICON_ERROR );
  editselcell = true;
}

void TableData::EditSelCells( int col )
{
  wxString sval;
  double dval;
  int row = 0, a, z, id, ival;
  bool allowed = false, integer = false;
  wxArrayInt srows;
  if ( col < 3 && GetTitle() != "Flynn-Data" )
    wxMessageBox( "You can not change these values", "Message", wxOK | wxICON_ERROR );
  else if ( col < 1 && GetTitle() == "Flynn-Data" )
    wxMessageBox( "You can not change these values", "Message", wxOK | wxICON_ERROR );
  else
    allowed = true;
  if ( allowed )
  {
    sval = wxGetTextFromUser( "Enter value", "Input text", "", this );
    if ( sval != "" )
    {
      // this is bad - should use attrib id to determine val type
      if ( sval.Find( "." ) != -1 )
      {
        sval.ToDouble( & dval );
      }
      else
      {
        ival = atoi( sval.c_str() );
        dval = ival;
        integer = true;
      }
      srows = tgrid->GetSelectedRows();
      a = srows.GetCount();
      id = GetColHeaderInt( col );
      if ( a == 0 )
      {
        if ( integer )
          SetValueI( row, col, ival );
        else
          SetValueD( row, col, dval );
        if ( GetTitle() == "BNode-Data" )
          ElleSetNodeAttribute( GetRowHeaderInt( row ), dval, id );
        if ( GetTitle() == "UNode-Data" )
          ElleSetUnodeAttribute( GetRowHeaderInt( row ), dval, id );
        if ( GetTitle() == "Flynn-Data" )
          ElleSetFlynnRealAttribute( GetRowHeaderInt( row ), dval, id );
      }
      else
      {
        for ( z = 0; z < a; z++ )
        {
          if ( integer )
            SetValueI( srows[z], col, ival );
          else
            SetValueD( srows[z], col, dval );
          if ( GetTitle() == "BNode-Data" )
            ElleSetNodeAttribute( GetRowHeaderInt( srows[z] ), dval, id );
          if ( GetTitle() == "UNode-Data" )
            ElleSetUnodeAttribute( GetRowHeaderInt( srows[z] ), dval, id );
          if ( GetTitle() == "Flynn-Data" )
            ElleSetFlynnRealAttribute( GetRowHeaderInt( srows[z] ), dval, id );
        }
      }
    }
  }
}

void TableData::OnCellEdit( wxGridEvent & event )
{
  wxString sval,oval;
  double dval;
  int row = event.GetRow(), col = event.GetCol(), a, z, id, ival;
  bool integer=false;
  wxArrayInt srows;
  if ( GetTitle() == "BNode-Data" || GetTitle() == "UNode-Data" )
  {
    if ( col <= 2 )
      wxMessageBox( "You can not change these values", "Message", wxOK | wxICON_ERROR );
    else
    {
      oval= tgrid->GetCellValue( row, col );
      sval = wxGetTextFromUser( "Enter value", "Input text", oval, this );
      if ( sval != oval )
      {
      // this is bad - should use attrib id to determine val type
        if ( sval.Find( "." ) != -1 )
        {
          sval.ToDouble( & dval );
        }
        else
        {
          ival = atoi( sval.c_str() );
          dval = ival;
          integer = true;
        }
        srows = tgrid->GetSelectedRows();
        a = srows.GetCount();
        id = GetColHeaderInt( col );
        if ( a == 0 )
          if ( GetTitle() == "BNode-Data" )
          {
            if ( integer )
              SetValueI( row, col, ival );
            else
              SetValueD( row, col, dval );
            ElleSetNodeAttribute( GetRowHeaderInt( row ), dval, id );
          }
          else
          {
            if ( integer )
              SetValueI( row, col, ival );
            else
              SetValueD( row, col, dval );

            ElleSetUnodeAttribute( GetRowHeaderInt( row ), dval, id );
          }
        else
        {
          for ( z = 0; z < a; z++ )
            if ( GetTitle() == "BNode-Data" )
            {
              if ( integer )
                SetValueI( row+z, col, ival );
              else
                SetValueD( row+z, col, dval );

              ElleSetNodeAttribute( GetRowHeaderInt( srows[z] ), dval, id );
            }
            else
            {
              if ( integer )
                SetValueI( row+z, col, ival );
              else
                SetValueD( row+z, col, dval );

              ElleSetUnodeAttribute( GetRowHeaderInt( srows[z] ), dval, id );
            }
        }
      }
    }
  }
  if ( GetTitle() == "Flynn-Data" )
  {
    if ( col <= 0 )
      wxMessageBox( "You can not change these values", "Message", wxOK | wxICON_ERROR );
    else
    {
        oval= tgrid->GetCellValue( row, col );
      sval = wxGetTextFromUser( "Enter new value", "Input", oval, this );
      if ( sval != "" )
      {
        if ( sval.Find( "." ) != -1 )
        {
          sval.ToDouble( & dval );
        }
        else
        {
          ival = atoi( sval.c_str() );
          dval = ival;
          integer = true;
        }
        srows = tgrid->GetSelectedRows();
        a = srows.GetCount();
        id = GetColHeaderInt( col );
        if ( a == 0 )
        {
          if ( integer )
            SetValueI( row, col, ival );
          else
            SetValueD( row, col, dval );
          ElleSetFlynnRealAttribute( GetRowHeaderInt( row ), dval, id );
        }
        else
          for ( z = 0; z < a; z++ )
          {
            if ( integer )
              SetValueI( row+z, col, ival );
            else
              SetValueD( row+z, col, dval );

            ElleSetFlynnRealAttribute( GetRowHeaderInt( srows[z] ), dval, id );
          }
      }
    }
  }
}

void TableData::OnMarkPolygon( wxCommandEvent & event )
{
  ( ( Canvas * ) FindWindowById( CANVASWIN ) )->markpoly = true;
}

void TableData::SetValueD( int row, int col, double val )
{
  wxString sval;
  sval.Printf( "%lE", val );
  tgrid->SetCellValue( row, col, sval );
}

double TableData::GetValueD( int row, int col )
{
  wxString sval;
  double val;
  sval = tgrid->GetCellValue( row, col );
  return sval.ToDouble( & val );
}

void TableData::SetValueI( int row, int col, int val )
{
  wxString sval;
  sval.Printf( "%d", val );
  tgrid->SetCellValue( row, col, sval );
}

int TableData::GetValueI( int row, int col )
{
  int val;
  val = atoi( ( tgrid->GetCellValue( row, col ) ).c_str() );
  return val;
}

void TableData::SetValueS( int row, int col, wxString val )
{
  tgrid->SetCellValue( row, col, val );
}

wxString TableData::GetValueS( int row, int col )
{
  return tgrid->GetCellValue( row, col );
}

void TableData::SetColHeader( int col, wxString val )
{
  tgrid->SetColLabelValue( col, val );
}

int TableData::GetColHeaderInt( int col )
{
  wxString welche;
  int id = -1;
  welche = tgrid->GetColLabelValue( col );
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
  return ( id );

}

int TableData::GetRowHeaderInt( int row )
{
  wxString val;
  int id;
  val = tgrid->GetCellValue( row, 0 );
  id = atoi( val.c_str() );
  return ( id );
}

void TableData::AddRow( void )
{
  tgrid->AppendRows( 1 );
}

void TableData::AddCol( void )
{
  tgrid->AppendCols( 1 );
}

void TableData::OnSaveData( wxCommandEvent & event )
{
  wxString filename, val;
  int row, col;
  filename = wxFileSelector( "Choose File", "", "", "", "CSV files (*.csv)|Text files (*.txt)|*.txt|All Files (*)|*",wxSAVE );
  if ( !filename.IsEmpty() )
  {
    FILE * fp;
    fp = fopen( filename, "w" );
    for ( col = 0; col < tgrid->GetNumberCols(); col++ )
    {
      val = tgrid->GetColLabelValue( col );
      fprintf( fp, "%s", val.c_str() );
      fprintf( fp, "," );
    }
    fprintf( fp, "\n" );
    for ( row = 0; row < tgrid->GetNumberRows(); row++ )
    {
      for ( col = 0; col < tgrid->GetNumberCols(); col++ )
      {
        val = tgrid->GetCellValue( row, col );
        fprintf( fp, "%s", val.c_str() );
        fprintf( fp, "," );
      }
      fprintf( fp, "\n" );
    }
    fclose( fp );
  }
}

void TableData::OnSaveSelected( wxCommandEvent & event )
{
  wxString filename, val;
  int row, col;
  filename = wxFileSelector( "Choose File", "", "", "", "CSV files (*.csv)|Text files (*.txt)|*.txt|All Files (*)|*" ,wxSAVE);
  if ( !filename.IsEmpty() )
  {
    FILE * fp;
    fp = fopen( filename, "w" );
    for ( col = 0; col < tgrid->GetNumberCols(); col++ )
    {
      val = tgrid->GetColLabelValue( col );
      fprintf( fp, "%s", val.c_str() );
      fprintf( fp, "," );
    }
    fprintf( fp, "\n" );
    for ( row = 0; row < tgrid->GetNumberRows(); row++ )
    {
      if ( tgrid->IsInSelection( row, 1 ) )
      {
        for ( col = 0; col < tgrid->GetNumberCols(); col++ )
        {
          val = tgrid->GetCellValue( row, col );
          fprintf( fp, "%s", val.c_str() );
          fprintf( fp, "," );
        }
        fprintf( fp, "\n" );
      }
    }
    fclose( fp );
  }
}

void TableData::OnClickLabel( wxGridEvent & event )
{
  int n = 0;
  Coords xy;
  if ( editselcell ) EditSelCells( event.GetCol() );
  else {
  if ( event.GetRow() != -1 )
  {
    if ( GetTitle() == "BNode-Data" )
    {
      ElleNodePosition( GetValueI( event.GetRow(), 0 ), & xy );
      //( ( Canvas * ) FindWindowById( CANVASWIN ) )->SetCurrent();
     /* glPointSize( 7 );
      glColor3d( 1, 0, 0 );
      glBegin( GL_POINTS );
      glVertex3f( xy.x * 2 - 1, xy.y * 2 - 1, 0 );
      glEnd();
      glPointSize( 1 );
      glFlush();*/
      ( ( Canvas * ) FindWindowById( CANVASWIN ) )->Refresh( false, NULL );
    }
    if ( GetTitle() == "UNode-Data" )
    {
      ElleGetUnodePosition( GetValueI( event.GetRow(), 0 ), & xy );
      //( ( Canvas * ) FindWindowById( CANVASWIN ) )->SetCurrent();
      /*glPointSize( 7 );
      glColor3d( 1, 0, 0 );
      glBegin( GL_POINTS );
      glVertex3f( xy.x * 2 - 1, xy.y * 2 - 1, 0 );
      glEnd();
      glPointSize( 1 );
      glFlush();*/
      ( ( Canvas * ) FindWindowById( CANVASWIN ) )->Refresh( false, NULL );
    }
    if ( GetTitle() == "Flynn-Data" )
    {
      bool c;
      int r, g, b;
      ElleFlynnGetColor( GetValueI( event.GetRow(), 0 ), & c, & r, & g, & b );
      if ( c )
        ElleFlynnSetColor( GetValueI( event.GetRow(), 0 ), false, 0, 0, 0 );
      else
        ElleFlynnSetColor( GetValueI( event.GetRow(), 0 ), true, 200, 100, 0 );
      EllePlotRegions( 0 );
      // ( ( Canvas * ) FindWindow( CANVASWIN ) )->Refresh( false, NULL );
    }
    }
  }
}

//Test which node is inside the polygon. if it is, the index of it will be stored
int TableData::GetPolIndexBNodes( double coords[1000] [3], int count )
{
  int a = 0, s, ip = 0, zin[100000];
  double * pgon=0, point[2];
  pgon = ( double * ) malloc( sizeof( double ) * 2 * count );
  Coords xy;
  for ( a = 0, s = 0; a < count; a++ )
  {
    pgon[s] = coords[a] [0];
    s++;
    pgon[s] = coords[a] [1];
    s++;
  }
  for ( a = 0; a < ElleMaxNodes(); a++ )
  {
    if ( ElleNodeIsActive( a ) )
    {
      ElleNodePosition( a, & xy );
      point[0] = xy.x;
      point[1] = xy.y;
      if ( CrossingsTest( pgon, count, point ) )
      {
        zin[ip] = a;
        ip++;
      }
    }
  }
  if (pgon) free( pgon );
  MarkRows( zin, ip );
  return ip;
}

//Test which unode is inside the polygon. if it is, the index of it will be stored
int TableData::GetPolIndexUNodes( double coords[1000] [3], int count )
{
  int qa = 0, qs, qip = 0, zin[100000];
  double * pgon=0, point[2];
  Coords xy;
  pgon = ( double * ) malloc( sizeof( double ) * 2 * count );
  for ( qa = 0, qs = 0; qa < count; qa++ )
  {
    pgon[qs] = coords[qa] [0];
    qs++;
    pgon[qs] = coords[qa] [1];
    qs++;
  }
  for ( qa = 0; qa < ElleMaxUnodes(); qa++ )
  {
    ElleGetUnodePosition( qa, & xy );
    point[0] = xy.x;
    point[1] = xy.y;
    if ( CrossingsTest( pgon, count, point ) && qip < 100000 )
    {
      zin[qip] = qa;
      qip += 1;
    }
  }
  MarkRows( zin, qip );
  if (pgon) free( pgon );
  return qip;
}

void TableData::MarkRows( int index[], int count )
{
  int a, s, x;
  for ( a = 0; a < tgrid->GetNumberRows(); a++ )
  {
    for ( s = 0; s < count; s++ )
    {
      x = GetValueI( a, 0 );
      if ( x == index[s] )
        if ( s == 0 )
          tgrid->SelectRow( a );
        else
          tgrid->SelectRow( a, true );
    }
  }
}

void TableData::MarkBNodes( double coords[1000] [3], int count )
{
  int num;
  num = GetPolIndexBNodes( coords, count );
}

void TableData::MarkUNodes( double coords[1000] [3], int count )
{
  int num;
  num = GetPolIndexUNodes( coords, count );
}

bool TableData::DecideMark( double coords[1000] [3], int count )
{
  if ( this->GetId() == BNODETABLE )
    MarkBNodes( coords, count );
  if ( this->GetId() == UNODETABLE )
    MarkUNodes( coords, count );
  if ( this->GetId() == FLYNNTABLE )
    MarkFlynns( coords, count );
  return ( true );
}

void TableData::MarkFlynns( double coords[1000] [3], int count )
{
  int num;
  num = GetFlynnIndex( coords, count );
}

//Test if each node of a flynn is inside the polygon, if so it's index will be stored
//if not it is dismissed. returns the number of flynns in the index
int TableData::GetFlynnIndex( double coords[1000] [3], int count )
{
  int a = 0, s, d, ip = 0, zin[100000], * fnodes, num_nodes;
  double * pgon=0, point[2];
  pgon = ( double * ) malloc( sizeof( double ) * 2 * count );
  Coords xy;
  bool all = true;
  for ( a = 0, s = 0; a < count; a++ )
  {
    pgon[s] = coords[a] [0];
    s++;
    pgon[s] = coords[a] [1];
    s++;
  }
  for ( a = 0; a < ElleMaxFlynns(); a++ )
  {
    if ( ElleFlynnIsActive( a ) )
    {
      ElleFlynnNodes( a, & fnodes, & num_nodes );
      d = 0;
      all = true;
      while ( d < num_nodes && all == true )
      {
        if ( ElleNodeIsActive( fnodes[d] ) )
        {
          ElleNodePosition( fnodes[d], & xy );
          point[0] = xy.x;
          point[1] = xy.y;
          if ( CrossingsTest( pgon, count, point ) )
            d++;
          else
            all = false;
        }
      }
      if ( all )
      {
        zin[ip] = a;
        ip++;
      }
    }
  }
  MarkRows( zin, ip );
  if (pgon) free( pgon );
  return ip;
}

void TableData::OnDeselectRows(wxCommandEvent & event)
{
  tgrid->ClearSelection();
}

void TableData::CloseTable( wxCommandEvent & event )
{
  Close( true );
}

void TableData::AutoSize(void)
{
	tgrid->AutoSizeColumns(true);
}
