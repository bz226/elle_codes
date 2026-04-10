// Written in 2003
// Author: Dr. J.K. Becker
// Copyright: Dr. J.K. Becker (becker@jkbecker.de)

#include "psprint.h"
#include <cmath>
#include "attrib.h"
#include "bflynns.h"
#include "nodes.h"
#include "display.h"
#include "runopts.h"
#include "general.h"
#include "file.h"
#include "interface.h"
#include "error.h"
#include "convert.h"
#include "check.h"
#include "misorient.h"
#include "unodes.h"
#include "setup.h"
#include "settings.h"


#define PAPERSIZE 500
extern DSettings * GetDisplayOptions();
extern Settings * GetUserOptions();

IMPLEMENT_CLASS( PSPrint, wxPrintout )

PSPrint::PSPrint( wxWindow * parent ) : wxPrintout( "Preview" )
{
  settings = GetUserOptions();
}

bool PSPrint::OnPrintPage( int n )
{
  wxDC * dc = GetDC();
  if ( dc )
  {
    //CalcPolygon();
    DrawLines();
    //DrawBNodes();

    //Draw bounding box
    dc->SetBrush( wxBrush( wxColour( 0, 0, 0 ), wxTRANSPARENT ) );
    dc->SetPen( wxPen( wxColour( 0, 0, 0 ), 1, wxSOLID ) );
    dc->DrawRectangle( 0, 0, PAPERSIZE + 2, PAPERSIZE + 2 );
    return ( true );
  }
  else
    return ( false );
}

void PSPrint::DrawBNodes()
//Draws all bnodes.
{
  wxDC * dc = GetDC();
  int n, count, rwx, rwy;
  count = ElleMaxNodes();
  Coords xy;
  for ( n = 0; n < count; n++ )
  {
    if ( ElleNodeIsActive( n ) )
    {
      ElleNodePosition( n, & xy );
      RWCoords( xy.x, xy.y, & rwx, & rwy );
      if ( ElleNodeIsDouble( n ) )
      {
        dc->SetBrush( wxBrush( wxColour( 0, 0, 255 ), wxSOLID ) );
        dc->SetPen( wxPen( wxColour( 0, 0, 255 ), 1, wxSOLID ) );
      }
      if ( ElleNodeIsTriple( n ) )
      {
        dc->SetBrush( wxBrush( wxColour( 0, 255, 0 ), wxSOLID ) );
        dc->SetPen( wxPen( wxColour( 0, 255, 0 ), 1, wxSOLID ) );
      }
      dc->DrawCircle( rwx, rwy, 1 );
    }
  }
}

void PSPrint::DrawLines()
{
  wxDC * bmp = GetDC();
  int x1, x2, y1, y2, n, count, z, minval = PAPERSIZE / 2, * id, num_nodes, dcwidth = PAPERSIZE;
  double dx1, dy1, dx2, dy2;
  wxPoint intersection, node1, node2, node3, node4;
  Coords xy1, xy2;
  count = ElleMaxFlynns();
  for ( n = 0; n < count; n++ )
  {
    if ( ElleFlynnIsActive( n ) )
    {
      ElleFlynnNodes( n, & id, & num_nodes );
      for ( z = 0; z + 1 < num_nodes; z++ )
      {
        ElleNodePosition( id[z], & xy1 );
        ElleNodePosition( id[z + 1], & xy2 );
        RWCoords( xy1.x, xy1.y, & x1, & y1 );
        RWCoords( xy2.x, xy2.y, & x2, & y2 );
        dx1 = ( ( x1 - x2 ) * ( x1 - x2 ) ) + ( ( y1 - y2 ) * ( y1 - y2 ) );
        dx2 = sqrt( dx1 );
        if ( dx2 <= 2*ElleSwitchdistance() ) // minval )
        {
          bmp->DrawLine( x1, y1, x2, y2 );
        }
      }
    }
  }
}

void PSPrint::CalcPolygon()
{
  int n, z, count, * id, num_nodes, num, comp;
  //int (*coords)[3]=new int [num][3] ;;
  int coords[1000] [3];
  Coords xy;
  wxDC * ps = GetDC();
  bool sort = true, y = false, x = false;
  int tmp[3];
  char cc[300];
  count = ElleMaxFlynns();
  for ( n = 0; n < count; n++ )
  {
    y = false; x = false;
    if ( ElleFlynnIsActive( n ) )
    {
      ElleFlynnNodes( n, & id, & num_nodes );
      for ( z = 0; z < num_nodes; z++ )
      {
        ElleNodePosition( id[z], & xy );
        RWCoords( xy.x, xy.y, & coords[z] [0], & coords[z] [1] );
        coords[z] [2] = z;
      }
      //bubble-sorting y-coordinates, can be faster I guess
      while ( sort == false )
      {
        sort = true;
        for ( z = 1; z < num_nodes; z++ )
        {
          if ( coords[z - 1] [1] > coords[z] [1] )
          {
            tmp[0] = coords[z - 1] [0];
            tmp[1] = coords[z - 1] [1];
            tmp[2] = coords[z - 1] [2];

            coords[z - 1] [0] = coords[z] [0];
            coords[z - 1] [1] = coords[z] [1];
            coords[z - 1] [2] = coords[z] [2];

            coords[z] [0] = tmp[0];
            coords[z] [1] = tmp[1];
            coords[z] [2] = tmp[2];

            sort = false;
          }
        }
      }
      for ( z = 1; z < num_nodes; z++ )
      {
        if ( ( coords[z] [1] - coords[z - 1] [1] ) > ( PAPERSIZE / 3 ) )
        {
          y = true;
          coords[z] [1] -= PAPERSIZE;
        }
      }
      sort = false;
      //bubble-sorting x-coordinates, can be faster I guess
      while ( sort == false )
      {
        sort = true;
        for ( z = 1; z < num_nodes; z++ )
        {
          if ( coords[z - 1] [0] > coords[z] [0] )
          {
            tmp[0] = coords[z - 1] [0];
            tmp[1] = coords[z - 1] [1];
            tmp[2] = coords[z - 1] [2];

            coords[z - 1] [0] = coords[z] [0];
            coords[z - 1] [1] = coords[z] [1];
            coords[z - 1] [2] = coords[z] [2];

            coords[z] [0] = tmp[0];
            coords[z] [1] = tmp[1];
            coords[z] [2] = tmp[2];

            sort = false;
          }
        }
      }
      for ( z = 1; z < num_nodes; z++ )
      {
        //sprintf(cc,"%d--%d--%d",coords[z][0],coords[z][1],coords[z][2]);
        //LogMessage(cc);
        if ( ( coords[z] [0] - coords[z - 1] [0] ) > ( PAPERSIZE / 3 ) )
        {
          x = true;
          coords[z] [0] -= PAPERSIZE;
        }
      }
      sort = false;
      //bubble-sorting number-values, can be faster I guess
      while ( sort == false )
      {
        sort = true;
        for ( z = 1; z < num_nodes; z++ )
        {
          if ( coords[z - 1] [2] > coords[z] [2] )
          {
            tmp[0] = coords[z - 1] [0];
            tmp[1] = coords[z - 1] [1];
            tmp[2] = coords[z - 1] [2];

            coords[z - 1] [0] = coords[z] [0];
            coords[z - 1] [1] = coords[z] [1];
            coords[z - 1] [2] = coords[z] [2];

            coords[z] [0] = tmp[0];
            coords[z] [1] = tmp[1];
            coords[z] [2] = tmp[2];

            sort = false;
          }
        }
      }
      wxPoint cc[num_nodes];
      for ( z = 0; z < num_nodes; z++ )
      {
        cc[z].x = coords[z] [0];
        cc[z].y = coords[z] [1];
      }
      ps->SetBrush( wxBrush( wxColour( 255, 0, 0 ), wxSOLID ) );
      ps->DrawPolygon( num_nodes, cc );
      if ( y )
      {
        for ( z = 0; z < num_nodes; z++ )
        {
          coords[z] [1] += PAPERSIZE;
        }
        ps->SetBrush( wxBrush( wxColour( 0, 255, 0 ), wxSOLID ) );
        ps->DrawPolygon( num_nodes, cc );
      }
      if ( x )
      {
        for ( z = 0; z < num_nodes; z++ )
        {
          coords[z] [0] += PAPERSIZE;
        }
        ps->SetBrush( wxBrush( wxColour( 0, 255, 255 ), wxSOLID ) );
        ps->DrawPolygon( num_nodes, cc );
        if ( y )
        {
          for ( z = 0; z < num_nodes; z++ )
          {
            coords[z] [1] -= PAPERSIZE;
          }
          ps->DrawPolygon( num_nodes, cc );
        }
      }
    }
    //delete[]coords;
  }
}


void PSPrint::RWCoords( double x, double y, int * rwx, int * rwy )
     /* All coordinates are stored in the unit-cell as double values. These can not
     be used to draw on the screen. This function calculates screen-coordinates
     from unit-cell coordinates. There is a (small) rounding error in these calculations!*/
     {
       x = x * PAPERSIZE;
       y = PAPERSIZE - ( y * PAPERSIZE );
       * rwx = ( int )ceil( x );
       * rwy = ( int )ceil( y );
}
