/*! \mainpage Welcome to OpenGL
*
*\section legal Legal Stuff
* This software is given to you under the GPL. If you do not know what that is, please go to the <a href="http://www.gnu.org/copyleft/gpl.html"> GPL-Site</A>
* and read the license agreements. Copyright 2005 by Dr. J.K. Becker.
* */


/*!\file main.h
 * \brief Main programm, including the wxWidgets entry class and the main class.
 *
 * The class mainapp is just the entry point for wxWidgets. The class mainwinFrame is the
 * main class making all the windows and other needed tings.
 * */
#ifndef _MAIN_H_
#define _MAIN_H_
#include "wx/textfile.h"
#include "wx/wx.h"
#include <cstdlib>
#include <vector>

using std::vector;

/*!\brief Enumerations of ID's for buttons and menues.
 */
enum
{
    Menu_File_Quit,
Menu_File_About,ID_BUTTONGO,ID_ListBox,ID_ListBox2,ID_BUTTONGO2,
};


class Data
{
public:
	wxString name,description,command;
};

/*!\class mainwinapp
 \brief This class is the entry point for wxWidgets.*/
class mainwinapp : public wxApp
{
public:
	virtual bool OnInit();
	//wxImage paint;
};


/*!\class mainwinFrame
 \brief This class is the main class that is used to do things.

  * This is the central class that controls everything.*/
class mainwinFrame : public wxFrame
{
public:
	mainwinFrame( const wxString & title, const wxPoint & pos, const wxSize & po1s, const wxString & arg0 );
	void Init();
private:
	bool LoadFileMulti(wxString filename);
        bool LoadFileSingle(wxString filename);
	void OnAbout(wxCommandEvent &event);
	/*!\brief End programm.*/
	void OnQuit(wxCommandEvent &event);
	void OnButtonGo(wxCommandEvent &event);
	void OnButtonGo2(wxCommandEvent &event);
	//void OnButtonStop(wxCommandEvent &event);
	void OnListBox(wxCommandEvent &event);
	void OnListBox2(wxCommandEvent &event);
	void SetSingle(wxString desc,wxString script);
	void SetMulti(wxString desc,wxString script);
	int FindPosMulti(wxString name);
        int FindPosSingle(wxString name);
	wxString escript,escript2;
	wxListBox *lb,*lbd;
	wxTextCtrl *text,*textd;
	vector<Data> tfile,dfile;
	DECLARE_EVENT_TABLE()
};

#endif //_MAIN_H_

