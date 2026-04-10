// Written in 2004
// Author: Dr. J.K. Becker
// Copyright: Dr. J.K. Becker (becker@jkbecker.de)

#include "cppfile.h"

extern DSettings * GetDisplayOptions();

cppFile::cppFile()
{

}

int cppFile::SaveZipFile()
{
  gzFile out;
  out = gzopen( filename.c_str(), "wb9" );
  SaveZIPColormap( out );
  SaveZIPDSettings( out );
  SaveZIPUserOptions( out );
  SaveZIPRunTimeOpts( out );
  SaveZIPDataFile( out );
  gzclose( out );
}

int cppFile::LoadZipFile()
{
  gzFile in;
  wxString line;
  in = gzopen( filename.c_str(), "r" );
  while ( line != "COLORMAP" )
    line = gzReadLine( in );
  LoadZIPColormap( in );
  while ( line != "DISPLAY OPTIONS" )
    line = gzReadLine( in );
  LoadZIPDisplayOptions( in );
  while ( line != "RUNTIME OPTIONS" )
    line = gzReadLine( in );
  LoadZIPRunTimeOpts( in );
  while ( line != "DATA FILE" )
    line = gzReadLine( in );
  LoadZIPData( in );
}

int cppFile::LoadZIPData( gzFile in )
{
  const int MAX_KEYS = 20;
  char *str, * ptr, * start;
  int num, key, keys[MAX_KEYS], finished = 0, err = 0;
  int i;
  gzReadLine(in);
  str = ( (char*)(gzReadSingleString( in )).c_str());
  while ( !gzeof( in ) && !finished )
  {
    /* * find keywords */
    validate( str, & key, FileKeys );
    switch ( key )
    {
      case E_OPTIONS:
        //if ( err = LoadZIPOptionData( in, str ) )
          finished = 1;
      break;
      case REGIONS:
      case FLYNNS:
        if ( err = LoadZIPFlynnData( in, str ) )
          finished = 1;
      break;
      case PARENTS:
        if ( err = LoadZIPParentData( in, str ) )
          finished = 1;
      break;
      case UNODES:
        /* * read the order of attribute vals * old code (pre 2.3.4) had attributes on * same line as position data */
        i = 0;
        if ( gzgets( in, str, 80 ) != NULL && !gzeof( in ) )
        {
          start = str;
          while ( ( ptr = strtok( start, " \t\r\n" ) ) && i < MAX_KEYS && !finished )
          {
            validate( ptr, & keys[i], FileKeys );
            if ( keys[i] < 0 ) finished = 1;
            else
            {
              start = 0;
              i++;
            }
          }
        }
        if ( err = LoadZIPUnodeData( in, str, keys, i ) )
          finished = 1;
      break;
      case LOCATION:
        if ( err = LoadZIPLocationData( in, str ) )
          finished = 1;
      break;
      case AGE:
        /* * read the order of age vals */
        i = 0;
        if ( gzgets( in, str, 80 ) != NULL && !gzeof( in ) )
        {
          start = str;
          while ( ( ptr = strtok( start, " \t\r\n" ) ) && i < NUM_FLYNN_AGE_VALS && !finished )
          {
            validate( ptr, & keys[i], FlynnAgeKeys );
            if ( keys[i] < 0 ) finished = 1;
            else
            {
              start = 0;
              i++;
            }
          }
          if ( i < 1 ) finished = 1;
        }
        if ( finished ) err = READ_ERR;
        else if ( err = LoadZIPAgeData( in, str, keys, i ) )
          finished = 1;
      break;
      case ENERGY:
      case VISCOSITY:
      case DISLOCDEN:
      case F_ATTRIB_A:
      case F_ATTRIB_B:
      case F_ATTRIB_C:
      case F_ATTRIB_D:
      case F_ATTRIB_E:
      case F_ATTRIB_F:
      case F_ATTRIB_I:
      case F_ATTRIB_J:
      case F_ATTRIB_K:
        if ( err = LoadZIPFlynnRealAttribData( in, str, key ) )
          finished = 1;
      break;
      case EXPAND:
      case COLOUR:
      case SPLIT:
      case GRAIN:
        if ( err = LoadZIPFlynnIntAttribData( in, str, key ) )
          finished = 1;
      break;
      case VELOCITY:
        gzReadLine( in );
        /* read VEL_X VEL_Y */
        if ( err = LoadZIPVelocityData( in, str ) )
          finished = 1;
      break;
      case CONC_A:
      case N_ATTRIB_A:
      case N_ATTRIB_B:
      case N_ATTRIB_C:
      case N_ATTRIB_D:
      case N_ATTRIB_E:
      case N_ATTRIB_F:
        if ( err = LoadZIPNodeRealAttribData( in, str, key ) )
          finished = 1;
      break;
      case N_MCONC_A:
        if ( err = LoadZIPNodeMarginAttribData( in, str, key ) )
          finished = 1;
      break;
      case STRESS:
        /* * read the order of stress vals */
        i = 0;
        if ( gzgets( in, str, 80 ) != NULL && !gzeof( in ) )
        {
          start = str;
          while ( ( ptr = strtok( start, " \t\r\n" ) ) && i < NUM_STRESS_VALS && !finished )
          {
            validate( ptr, & keys[i], StressKeys );
            if ( keys[i] < 0 ) finished = 1;
            else
            {
              start = 0;
              i++;
            }
          }
          if ( i < 1 ) finished = 1;
        }
        if ( finished ) err = READ_ERR;
        else if ( err = LoadZIPStressData( in, str, keys, i ) )
          finished = 1;
      break;
      case STRAIN:
        /* * read the order of strain vals */
        i = 0;
        if ( gzgets( in, str, 80 ) != NULL && !gzeof( in ) )
        {
          start = str;
          while ( ( ptr = strtok( start, " \t\r\n" ) ) && i < NUM_STRAIN_VALS && !finished )
          {
            validate( ptr, & keys[i], StrainKeys );
            if ( keys[i] < 0 ) finished = 1;
            else
            {
              start = 0;
              i++;
            }
          }
          if ( i < 1 ) finished = 1;
        }
        if ( finished ) err = READ_ERR;
        else if ( err = LoadZIPStrainData( in, str, keys, i ) )
          finished = 1;
      break;
      case CAXIS:
        if ( err = LoadZIPCAxisData( in, str ) )
          finished = 1;
        /* ElleCheckFlynnDefaults(CAXIS); */
      break;
      case EULER_3:
        if ( err = LoadZIPEuler3Data( in, str ) )
          finished = 1;
      break;
      case GBE_LUT:
        if ( err = LoadZIPGBEnergyLUT( in, str ) )
          finished = 1;
      break;
      case MINERAL:
        if ( err = LoadZIPMineralData( in, str ) )
          finished = 1;
        /* ElleCheckFlynnDefaults(MINERAL); */
      break;
      case FLYNN_STRAIN:
        /* * read the order of strain vals */
        i = 0;
        if ( gzgets( in, str, 80 ) != NULL && !gzeof( in ) )
        {
          start = str;
          ptr = strtok( start, " \t\r\n" );
          while ( ptr && i < NUM_FLYNN_STRAIN_VALS && !finished )
          {
            validate( ptr, & keys[i], FlynnStrainKeys );
            if ( keys[i] < 0 ) finished = 1;
            else
            {
              start = 0;
              ptr = strtok( start, " \t\r\n" );
              i++;
            }
          }
          if ( i < 1 ) finished = 1;
        }
        if ( finished ) err = READ_ERR;
        else if ( err = LoadZIPFlynnStrainData( in, str, keys, i ) )
          finished = 1;
      break;
      case U_CONC_A:
      case U_ATTRIB_A:
      case U_ATTRIB_B:
      case U_ATTRIB_C:
      case U_ATTRIB_D:
      case U_ATTRIB_E:
      case U_ATTRIB_F:
        if ( key == U_CONC_A ) key = CONC_A;
        if ( err = LoadZIPUnodeRealAttribData( in, str, & key, 1 ) )
          finished = 1;
      break;
      case U_STRAIN:
        /* * read the order of strain vals */
        i = 0;
        if ( gzgets( in, str, 80 ) != NULL && !gzeof( in ) )
        {
          start = str;
          while ( ( ptr = strtok( start, " \t\r\n" ) ) && i < NUM_FINITE_STRAIN_VALS && !finished )
          {
            validate( ptr, & keys[i], FiniteStrainKeys );
            if ( keys[i] < 0 ) finished = 1;
            else
            {
              start = 0;
              i++;
            }
          }
          if ( i < 1 ) finished = 1;
        }
        if ( finished ) err = READ_ERR;
        else if ( err = LoadZIPUnodeRealAttribData( in, str, keys, i ) )
          finished = 1;
      break;
      case U_EULER_3:
        /* * read the 3 euler angle values */
        keys[0] = E3_ALPHA;
        keys[1] = E3_BETA;
        keys[2] = E3_GAMMA;
        if ( err = LoadZIPUnodeRealAttribData( in, str, keys, 3 ) )
          finished = 1;
      break;
#if XY
#endif
      case COMMENT:
        gzReadLineSTD( in );
        gzReadLineSTD( in );
        gzReadLineSTD( in );
      break;
      default:
        err = KEY_ERR;
        finished = 1;
      break;
    }
  }
#ifndef NO_DSP
  //if ( ElleDisplay() ) ElleUpdateSettings();
#endif
  ElleSetCount( 0 );
  return ( err );
}

wxString cppFile::gzReadLine( gzFile in )
{
  wxString line;
  char c;
  c = gzgetc( in );
  while ( c != '\n' )
  {
    line.Append( c, 1 );
    c = gzgetc( in );
  }
  return ( line );
}

string cppFile::gzReadLineSTD( gzFile in )
{
  char c;
  string line;
  c = gzgetc( in );
  while ( c != '\n' )
  {
    line.append( 1, c );
    c = gzgetc( in );
  }
  return ( line );
}

string cppFile::gzReadSingleString( gzFile in )
{
  //Strings have to be seperated by a single or more spaces
  string line;
  char c;
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

int cppFile::LoadZIPColormap( gzFile in )
{
  DSettings * doptions = GetDisplayOptions();
  int n, set, pos, pos1, r, g, b;
  string line;
  line = gzReadLineSTD( in );
  for ( n = 0; n < 256; n++ ) //there are 256 colors in the file
  {
    line.empty();
    line = gzReadLineSTD( in );
    pos = 0; pos1 = 0;
    pos = line.find( ',' );
    r = atoi( ( line.substr( 0, pos ) ).c_str() );
    pos++;
    pos1 = line.find( ',', pos );
    g = atoi( ( line.substr( pos, pos1 ) ).c_str() );
    pos1++;
    pos = line.find( ',', pos1 );
    b = atoi( ( line.substr( pos1, pos - pos1 ) ).c_str() );
    pos++;
    set = atoi( ( line.substr( pos, 1 ) ).c_str() );
    doptions->CmapChangeColor( n, r, g, b, set );
  }
  return ( E_OK );
}

int cppFile::SetFilename( wxString name )
{
  DSettings * doptions = GetDisplayOptions();
  if ( !name.IsEmpty() )
  {
    filename = name;
    filedir = name.BeforeLast( E_DIR_SEPARATOR );
    doptions->SetLoadFileDir( filedir );
    fileend = name.AfterLast( '.' );
    return ( E_OK );
  }
  return ( E_ERROR );

}

int cppFile::GetFilename()
{
  DSettings * doptions = GetDisplayOptions();
  filedir = doptions->GetLoadFileDir();
  filename = wxFileSelector( "Choose File", filedir, "", "",
       "ELLE files (*.elle)|*.elle|ELLE-Archive files (*.zip)|*.zip|All Files (*)|*" );
  if ( !filename.empty() )
  {
    filedir = filename.BeforeLast( E_DIR_SEPARATOR );
    doptions->SetLoadFileDir( filedir );
    fileend = filename.AfterLast( '.' );
    return ( E_OK );
  }
  return ( E_ERROR );
}

int cppFile::SaveZIPColormap( gzFile out )
{
  int r, g, b, set, n;
  DSettings * doptions = GetDisplayOptions();
  gzprintf( out, "-----------------------------------\n" );
  gzprintf( out, "COLORMAP\n" );
  gzprintf( out, "-----------------------------------\n" );
  for ( n = 0; n < 256; n++ )
  {
    doptions->CmapGetColor( n, & r, & g, & b, & set );
    gzprintf( out, "%d,%d,%d,%d\n", r, g, b, set );
  }
  return ( E_OK );
}

int cppFile::LoadZIPDisplayOptions( gzFile in )
{
  DSettings * doptions = GetDisplayOptions();
  string line;
  wxString wxline;
  int val[3], n;
  line = gzReadLine( in );
  //LineColor
  line = gzReadSingleString( in );
  for ( n = 0; n < 3; n++ )
    val[n] = atoi( (gzReadSingleString( in )).c_str());
  doptions->SetLineColor( val[0], val[1], val[2] );
  //LineSize
  line = gzReadSingleString( in );
  val[0] = atoi(  (gzReadSingleString( in )  ).c_str());
  doptions->SetLineSize( val[0] );
  //LineArgs
  line = gzReadSingleString( in );
  line = gzReadSingleString( in );
  doptions->SetShowArgs( BOUNDARIES, true, atoi( line.c_str() ) );
  //NodeSize
  line = gzReadSingleString( in );
  line = gzReadSingleString( in );
  doptions->SetNodeSize( atoi( line.c_str() ) );
  //BNodeColor
  line = gzReadSingleString( in );
  for ( n = 0; n < 3; n++ )
    val[n] = atoi( ( gzReadSingleString( in )).c_str());
  doptions->SetDNodeColor( val[0], val[1], val[2] );
  //TNodeColor
  line = gzReadSingleString( in );
  for ( n = 0; n < 3; n++ )
   val[n] = atoi(  (gzReadSingleString( in )  ).c_str());
  doptions->SetTNodeColor( val[0], val[1], val[2] );
  //NodeArgs
  line = gzReadSingleString( in );
  line = gzReadSingleString( in );
  doptions->SetShowArgs( BNODES, true, atoi( line.c_str() ) );
  //FlynnArgs
  line = gzReadSingleString( in );
  line = gzReadSingleString( in );
  doptions->SetShowArgs( FLYNNS, true, atoi( line.c_str() ) );
  //UnodesNotRangeFlag
  line = gzReadSingleString( in );
  line = gzReadSingleString( in );
  if ( line == "TRUE" )
  {
    doptions->SetUnodesNotRangeFlag( true );
    line = gzReadSingleString( in );
    line = gzReadSingleString( in );
    doptions->SetUnodesNotRangeFlagMaxValue( atof( line.c_str() ) );
    line = gzReadSingleString( in );
    line = gzReadSingleString( in );
    doptions->SetUnodesNotRangeFlagMinValue( atof( line.c_str() ) );
  }
  else
    doptions->SetUnodesNotRangeFlag( false );
  //UnodesRangeFlag
  line = gzReadSingleString( in );
  line = gzReadSingleString( in );
  if ( line == "TRUE" )
  {
    doptions->SetUnodesRangeFlag( true );
    line = gzReadSingleString( in );
    line = gzReadSingleString( in );
    doptions->SetUnodesRangeFlagMaxValue( atof( line.c_str() ) );
    line = gzReadSingleString( in );
    line = gzReadSingleString( in );
    doptions->SetUnodesRangeFlagMinValue( atof( line.c_str() ) );
  }
  else
    doptions->SetUnodesRangeFlag( false );
  //Set LoadFileDir
  line = gzReadSingleString( in );
  line = gzReadSingleString( in );
  wxline = line.c_str();
  doptions->SetLoadFileDir( wxline );
  //Set SaveFileDir
  line = gzReadSingleString( in );
  line = gzReadSingleString( in );
  wxline = line.c_str();
  doptions->SetSaveFileDir( wxline );
}

int cppFile::SaveZIPDSettings( gzFile out )
{
  int r, g, b;
  bool test;
  wxString name;
  DSettings * doptions = GetDisplayOptions();
  gzprintf( out, "-----------------------------------\n" );
  gzprintf( out, "DISPLAY OPTIONS\n" );
  gzprintf( out, "-----------------------------------\n" );
  doptions->GetLineColor( & r, & g, & b );
  gzprintf( out, "Linecolor: %d %d %d\n", r, g, b );
  gzprintf( out, "Linesize: %d\n", doptions->GetLineSize() );
  gzprintf( out, "Lineargs: %d\n", doptions->GetShowArgs( BOUNDARIES ) );
  gzprintf( out, "Nodesize: %d\n", doptions->GetNodeSize() );
  doptions->GetDNodeColor( & r, & g, & b );
  gzprintf( out, "BNodecolor: %d %d %d\n", r, g, b );
  doptions->GetTNodeColor( & r, & g, & b );
  gzprintf( out, "TNodecolor: %d %d %d\n", r, g, b );
  gzprintf( out, "Nodeargs: %d\n", doptions->GetShowArgs( BNODES ) );
  gzprintf( out, "Flynnargs: %d\n", doptions->GetShowArgs( FLYNNS ) );
  test = doptions->GetUnodesNotRangeFlag();
  if ( test == true )
  {
    gzprintf( out, "UNodesNotRangeFlag: TRUE\n" );
    gzprintf( out, "UNodesNotRangeFlagMax: %lf\n", doptions->GetUnodesNotRangeFlagMaxValue() );
    gzprintf( out, "UNodesNotRangeFlagMin: %lf\n", doptions->GetUnodesNotRangeFlagMinValue() );
  }
  else
    gzprintf( out, "UNodesNotRange: FALSE\n" );
  test = doptions->GetUnodesRangeFlag();
  if ( test == true )
  {
    gzprintf( out, "UNodesRangeFlag: TRUE\n" );
    gzprintf( out, "UNodesRangeFlagMax: %lf\n", doptions->GetUnodesRangeFlagMaxValue() );
    gzprintf( out, "UNodesRangeFlagMin: %lf\n", doptions->GetUnodesRangeFlagMinValue() );
  }
  else
    gzprintf( out, "UNodesRangeFlag: FALSE\n" );

  name = doptions->GetLoadFileDir();
  gzprintf( out, "LoadFileDir: %s\n", name.c_str() );
  name = doptions->GetSaveFileDir();
  gzprintf( out, "SaveFileDir: %s\n", name.c_str() );
  return ( E_OK );
}

int cppFile::SaveZIPUserOptions( gzFile out )
{
  gzprintf( out, "-----------------------------------\n" );
  gzprintf( out, "USER OPTIONS\n" );
  gzprintf( out, "-----------------------------------\n" );

}

int cppFile::SaveZIPDataFile( gzFile out )
{
  int err = 0;
  gzprintf( out, "-----------------------------------\n" );
  gzprintf( out, "DATA FILE\n" );
  gzprintf( out, "-----------------------------------\n" );
  if ( !( err = SaveZIPRegionData( out ) ) )
    err = SaveZIPNodeData( out );
  if ( !err ) err = SaveZIPUnodeData( out );
}

int cppFile::LoadZIPRunTimeOpts( gzFile in )
{
  string line;
  line = gzReadSingleString( in );
  line = gzReadSingleString( in );
  line = gzReadSingleString( in );
  ElleSetSaveFrequency( atoi( line.c_str() ) );
  line = gzReadSingleString( in );
  line = gzReadSingleString( in );
  ElleSetStages( atoi( line.c_str() ) );
  line = gzReadSingleString( in );
  line = gzReadSingleString( in );
  ElleSetSwitchdistance( atof( line.c_str() ) );
  line = gzReadSingleString( in );
  line = gzReadSingleString( in );
  ElleSetSpeedup( atof( line.c_str() ) );
  line = gzReadSingleString( in );
  line = gzReadSingleString( in );
  ElleSetSaveFileRoot( ( char * ) ( line.c_str() ) );
}

int cppFile::SaveZIPRunTimeOpts( gzFile out )
{
  int vstages, vsfreq;
  double vsdist, vsup;
  DSettings * display_options = GetDisplayOptions();

  gzprintf( out, "-----------------------------------\n" );
  gzprintf( out, "RUNTIME OPTIONS\n" );
  gzprintf( out, "-----------------------------------\n" );
  wxString val;
  ElleGetRunOptionValue( RO_SVFREQ, & vsfreq );
  gzprintf( out, "RO_SVFREQ: %d\n", vsfreq );

  ElleGetRunOptionValue( RO_STAGES, & vstages );
  gzprintf( out, "RO_STAGES: %d\n", vstages );

  ElleGetRunOptionValue( RO_SWITCHDIST, ( int * ) & vsdist );
  gzprintf( out, "RO_SWITCHDIST: %lf\n", vsdist );

  ElleGetRunOptionValue( RO_SPEEDUP, ( int * ) & vsup );
  gzprintf( out, "RO_SPEEDUP: %lf\n", vsup );

  gzprintf( out, "ELLSAVEFILEROOT: %s\n", ElleSaveFileRoot() );
}


