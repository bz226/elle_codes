/////////////////////////////////////////////////////////////////////////////
// Name:        dsettings.h
// Purpose:
// Author:      Dr. J.K. Becker
// Modified by:
// Created:
// RCS-ID:
// Copyright: Dr. J.K. Becker
// Licence: GPL
/////////////////////////////////////////////////////////////////////////////

#ifndef _E_dsettings_h_
  #define _E_dsettings_h_
  #include <string>
  #include "wx/wx.h"
  #include "wx/textfile.h"
#include "wx/zipstrm.h"
#include "wx/wfstream.h"
#include "wx/txtstrm.h"
#include "runopts.h"
#include "wx/filesys.h"
#include "wx/fs_zip.h"
#include "file_utils.h"
#include <iostream>


using namespace std;



class DSettings
{
public:
	DSettings( void );
	void SetSavePic(bool t);
	bool GetSavePic();
	void CmapChangeColor( int in, int r, int g, int b, int set );
	void CmapGetColor( int in, int * r, int * g, int * b, int * set );
	void CmapResetToDefault( void );
	int CmapSave( string filename );
	int CmapLoad( string filename );
	void SetShowArgs( int key, bool state, int attrib );
	int GetShowArgs( int key );
	bool GetFlagUnodes();
	bool GetFlagSegments();
	void SetNodeSize( int size );
	int GetNodeSize();
	void SetUNodeSize( int size );
	int GetUNodeSize();
	void SetSegmentSize(int size);
	int GetSegmentSize();
	void SetLineSize( int size );
	int GetLineSize();
	void GetDNodeColor( int * r, int * g, int * b );
	void SetDNodeColor( int r, int g, int b );
	void GetTNodeColor( int * r, int * g, int * b );
	void SetTNodeColor( int r, int g, int b );
	void SetLineColor( int r, int g, int b );
	void GetLineColor( int * r, int * g, int * b );
	void SetUnodesRangeFlag( bool );
	bool GetUnodesRangeFlag();
	void SetUnodesRangeFlagMaxValue( double );
	void SetUnodesRangeFlagMinValue( double );
	double GetUnodesRangeFlagMaxValue();
	double GetUnodesRangeFlagMinValue();
	void SetUnodesNotRangeFlag( bool );
	bool GetUnodesNotRangeFlag();
	void SetUnodesNotRangeFlagMaxValue( double );
	void SetUnodesNotRangeFlagMinValue( double );
	double GetUnodesNotRangeFlagMaxValue();
	double GetUnodesNotRangeFlagMinValue();
	void SetSegmentsRangeFlag( bool );
	bool GetSegmentsRangeFlag();
	void SetSegmentsRangeFlagMaxValue( double );
	void SetSegmentsRangeFlagMinValue( double );
	double GetSegmentsRangeFlagMaxValue();
	double GetSegmentsRangeFlagMinValue();
	void SetSegmentsNotRangeFlag( bool );
	bool GetSegmentsNotRangeFlag();
	void SetSegmentsNotRangeFlagMaxValue( double );
	void SetSegmentsNotRangeFlagMinValue( double );
	double GetSegmentsNotRangeFlagMaxValue();
	double GetSegmentsNotRangeFlagMinValue();
	void SetLoadFileDir( wxString loadfiledir );
	wxString GetLoadFileDir();
	void SetSaveFileDir( wxString savefiledir );
	wxString GetSaveFileDir();
	void SaveDSettings( void );
	bool LoadDSettings( void );
	void ShowNodeNumbers( bool );
	void ShowFlynnNumbers( bool );
	bool ShowNodeNumbers();
	bool ShowFlynnNumbers();
	void SetLogLevel(int level);
	char* GetFilename();
	int SetFilename(char *filen);
	int GetLogLevel();
	void SetSavePicFName(wxString s);
	void SetRainStages(int s);
	int GetRainStages();
	void IncrementRainColor();
	void SetRainColor(int i);
	int GetRainColor();
	void SetRainChange(int s);
	int GetRainChange();
	wxString GetSavePicFName();
	void SaveArchive(wxString filename);
	void LoadArchive(wxString filename,bool run);
	void LoadArchiveDSettings(wxTextInputStream *text);
	void LoadArchiveCmap(wxTextInputStream *text);
	void LoadArchiveRunOpts(wxTextInputStream *text);
	void SaveArchiveDSettings(wxTextOutputStream &text);
	void SaveArchiveCmap(wxTextOutputStream &text);
	void SaveArchiveRunOpts(wxTextOutputStream &text);
    void SaveArchiveUserData(wxTextOutputStream &text);
    void LoadArchiveUserData(wxTextInputStream *text);
	void SetClampColor(int t,bool f,double min,double max);
	bool GetClampColor(int t,double *min,double *max);
    void SetClampColor(int t,bool f);
    bool GetClampColor(int t);
    void SetUCmdArgs(bool t);
private:
	bool ucmdargs,rangeflagunodes, notrangeflagunodes, showflynnnumbers, shownodenumbers,savepics,rainbow,fclampcolor,bclampcolor,nclampcolor,uclampcolor;
	bool rangeflagsegments, notrangeflagsegments,sclampcolor;
    double unodesrangeflagmaxvalue, unodesrangeflagminvalue, unodesnotrangeflagmaxvalue, unodesnotrangeflagminvalue;
    double segmentsrangeflagmaxvalue, segmentsrangeflagminvalue, segmentsnotrangeflagmaxvalue, segmentsnotrangeflagminvalue;
	int nodesize,unodesize, segmentsize,linesize, tnodecolor[3], dnodecolor[3], linecolor[3],loglevel,cstages,raincolor,rainchange;
	int cmap[264] [4];
	double fccmin,fccmax,bccmin,bccmax,nccmin,nccmax,uccmin,uccmax;
	double sccmin,sccmax;
	wxString loadfiledir, savefiledir,savepicfname;
	//char *filename;
};

#endif //_E_settings_h_
