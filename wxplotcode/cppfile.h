// Written in 2004
// Author: Dr. J.K. Becker
// Copyright: Dr. J.K. Becker (becker@jkbecker.de)

#ifndef _E_cppfile
  #define _E_cppfile

  #include "wx/wx.h"
  #include "zlib.h"
  #include "dsettings.h"
  #include "settings.h"
  #include "runopts.h"
  #include <cstdio>
  #include "bflynns.h"
  #include "version.h"
  #include <string>
  #include "nodes.h"
  #include "string_utils.h"
  #include "file_utils.h"
  #include "runopts.h"
  #include "lut.h"
  #include "regions.h"
  #include "general.h"
  #include "interface.h"
  #include "unodes.h"
  #include "file.h"
  #include "stats.h"
  #include "error.h"

  #include "display.h"


  #define E_OK 0
  #define E_ERROR 1

class cppFile
{
public:
  cppFile();
  int SetFilename( wxString name );
  int GetFilename();
  int SaveZipFile();
  int LoadZipFile();
private:
  //general read-functions
 wxString gzReadLine( gzFile in );
  string gzReadLineSTD( gzFile in );
  string gzReadSingleString( gzFile in );
  int LoadZIPRunTimeOpts( gzFile in );
  int LoadZIPDisplayOptions( gzFile in );
  int LoadZIPColormap( gzFile in );
  int SaveZIPDSettings( gzFile out );
  int SaveZIPColormap( gzFile out );
  int SaveZIPUserOptions( gzFile out );
  int SaveZIPDataFile( gzFile out );
  int SaveZIPRunTimeOpts( gzFile out );
  int LoadZIPData(gzFile in);
  wxString filename, filedir, fileend;
};
#endif
