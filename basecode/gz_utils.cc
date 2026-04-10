 /*****************************************************
 * Copyright: (c) Dr J.K. Becker
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#include "gz_utils.h"

/*****************************************************

static const char rcsid[] =
       "$Id$";

******************************************************/
/*!
	\brief		Read a line in a zipped file
	\par		Description:
	Read chars until \n append to a std::string variable
	\remarks
		This function uses zlib routines
*/
                                                                                
//What about eof?

std::string gzReadLineSTD( gzFile in )
{
  char c;
  std::string line;
  c = gzgetc( in );
  while ( c != '\n' )
  {
    line.append( 1, c );
    c = gzgetc( in );
  }
  return ( line );
}
                                                                                
/*!
	\brief		Read a word in a zipped file
	\par		Description:
	Read chars until space char, append to a std::string variable
	\remarks
		This function uses zlib routines
		Words are recognized if single space delimited
*/
std::string gzReadSingleString( gzFile in )
{
  //Strings have to be seperated by a single space
  char c;
  std::string line;
  c = gzgetc( in );
  //get rid of leading spaces
  while ( c == ' ' )
  {
    c = gzgetc( in );
  }
  //this is the string we want
  while ( c != ' ' && c != '\n' )
  {
    line.append( 1, c );
    c = gzgetc( in );
  }
  return ( line );
}

