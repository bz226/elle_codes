/*----------------------------------------------------------------
 *    Elle:   menus.c  1.1  1 October 1998
 *
 *    Copyright (c) 1997 by L.A. Evans & T.D. Barr
 *----------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <Xm/Xm.h>
#include <X11/cursorfont.h>
#if XmVersion<1002
#include <Xt/StringDefs.h>
#endif

/*
 * Public include files for widgets
 */
#include <Xm/MainW.h>
#include <Xm/RowColumn.h>
#include <Xm/PushBG.h>
#include <Xm/PushB.h>
#include <Xm/CascadeBG.h>
#include <Xm/CascadeB.h>
#include <Xm/Label.h>
#include <Xm/SeparatoG.h>
#include <Xm/SelectioB.h>
#include <Xm/MessageB.h>
#include <Xm/TextF.h>

#include "types.h"
#include "csettings.h"
#include "plot.h"
#include "menus.h"
#include "globals.h"
#include "error.h"
#include "string_utils.h"
#include "file_utils.h"
#include "runopts.h"

#define XY 0
#define NO_VALUE        -10

#define XYPLOT    2
#define GRAINS    12
#define SUBGRAINS 13
#define UNITS     14
#define TRIANGLES 15

#define OPEN              0
#define SAVE              1
#define SEP               2
#define EXIT              3

#define SOLN              0
#define LOG               1
#define STTGS             2
#define PS                3

#define LABEL_INDX        0
#define COLOUR_INDX       1
#define LINE_INDX         2
#define DEL_INDX          3

#define EDIT_INDX         0
#define FONT_INDX         1
#define LRGE_FONT         5

#define CURR_CELL         11
#define ALL_CELLS         12

#define NEXT              0
#define PREV              1
#define SPEC              2
#define LAST              3
#define FIRST             4

/*
#define SOLID            -1
#define DASH              1
*/

#define VIEW              0
#define SOLNFILESDIR    "FD.sols"

Cursor Waiting;
Widget main_Menu, mainWindow, topLevel;
Widget plotArea, currvalsArea;
Widget LabelPrompt;
XmFontList Curr_fontlist;
XtAppContext app_context;

/* These values are only used if the app_defaults file is not found */
String default_resources[] = {
        /* Plot resources*/
    "*cellWidthInPixels:      500",
    "*cellHeightInPixels:     200",
/*
    "*pixmapWidthInCells:       2",
    "*pixmapHeightInCells:      2",
*/
        /* Appearance Resources*/
    "*main_Menu*Foreground:     black",
    "*main_Menu*Background:     grey75",
    "*plotArea*Foreground:   white",
    "*plotArea*Background:   black",
    "currvalsArea*Foreground:   black",
    "currvalsArea*Background:   grey75",
    "*Help*Foreground:     white",
    "*Help*Background:   grey75",
    NULL
};
void ExitDialog(),Exit_App(),Exit_Proc();
void SaveDialog();
void RunChosen(),RerunChosen(),BndBoxChosen();
void create_working_message(), ForceUpdate();
void ResizePlotWin();
void clear_menus();
/*int DisplayElleErrorMsg(char *msg,int err_num);*/
static void FileSolnSelectionOK(),FileLogSelectionOK(),FileOpenChosen();
static void FileSaveSelectionOK(),FileSaveChosen();
extern void create_error_message(), CleanUp();
extern void OptsDialog(), RunOptsDialog(), StatsOptsDialog();
extern void ChangeCell(), clear_X(), DeleteAllCells(), DeleteCell();
extern void ChangeCellConfig(), Resize();

extern int ElleWriteData();
extern char *ElleFile(), *ElleEXtraFile();

MenuItemData filesaveopts[] = {
    { "Data", &xmPushButtonGadgetClass, NO_VALUE, 0, FileSaveChosen,
            (XtPointer)SOLN, NULL, NO_VALUE },
    { "Postscript", &xmPushButtonGadgetClass, NO_VALUE, 0, FileSaveChosen,
            (XtPointer)PS, NULL, NO_VALUE },
    NULL,
};

MenuItemData runopts[] = {
    { "Run", &xmPushButtonGadgetClass, NO_VALUE, 0, RunChosen,
            (XtPointer)GRAINS, NULL, NO_VALUE },
    { "Rerun", &xmPushButtonGadgetClass, NO_VALUE, 0, RerunChosen,
            (XtPointer)GRAINS, NULL, NO_VALUE },
    { "Set Options...", &xmPushButtonGadgetClass, NO_VALUE, 0, RunOptsDialog,
            (XtPointer)&main_Menu, NULL, NO_VALUE },
    NULL,
};

MenuItemData statsopts[] = {
    { "Set Options...", &xmPushButtonGadgetClass, NO_VALUE, 0,
             StatsOptsDialog, (XtPointer)&main_Menu, NULL, NO_VALUE },
    NULL,
};

MenuItemData optionopts[] = {
    { "Set Options...", &xmPushButtonGadgetClass, NO_VALUE, 0, OptsDialog,
            (XtPointer)&main_Menu, NULL, NO_VALUE },
    NULL,
};
        /* MENUS */
MenuItemData fileopts[] = {
    { "Open...", &xmPushButtonGadgetClass, NO_VALUE, 0, FileOpenChosen,
            (XtPointer)OPEN, NULL, NO_VALUE },
    { "Save...", &xmPushButtonGadgetClass, NO_VALUE, 0, NULL,
            (XtPointer)SAVE, filesaveopts, NO_VALUE },
    { "Sep", &xmSeparatorGadgetClass, NO_VALUE, 0, NULL,
            (XtPointer)SEP, NULL, NO_VALUE },
    { "Exit", &xmPushButtonGadgetClass, NO_VALUE, 0, Exit_Proc, 
            (XtPointer)EXIT, NULL, NO_VALUE },
/*
    This gives exit confirm dialog box 
    { "Exit", &xmPushButtonGadgetClass, NO_VALUE, 0, ExitDialog, 
            (XtPointer)EXIT, NULL, NO_VALUE },
*/
    NULL,
};


MenuItemData mainopts[] = {
        /* Help button must be the last in the list */
    { "File", &xmCascadeButtonGadgetClass, NO_VALUE, 0, NULL, 
            (XtPointer)0, fileopts, NO_VALUE },
    { "Run", &xmCascadeButtonGadgetClass, NO_VALUE, 0, NULL,
            (XtPointer)GRAINS, runopts, NO_VALUE },
    { "Display", &xmCascadeButtonGadgetClass, NO_VALUE, 0, NULL,
            (XtPointer)0, optionopts, NO_VALUE },
    { "Stats", &xmCascadeButtonGadgetClass, NO_VALUE, 0, NULL,
            (XtPointer)0, statsopts, NO_VALUE },
    NULL,
};

static void Syntax(argc, argv)
int argc;
char * argv[];
{
    int i;
    static int errs = False;

#if XY
    /* first argument is program name - skip that */
    for (i = 1; i < argc; i++) {
        if (!errs++) /* do first time through */
            fprintf(stderr, "sybil: command line option not understood:\n");
        fprintf(stderr, "option: %s\n", argv[i]);
    }
#endif

    fprintf(stderr, "Elle understands Xt command line options as well as\n");
    fprintf(stderr, "these additional options:\n");
    fprintf(stderr, "Option   Valid Range\n");
    fprintf(stderr, "-i       name of elle file \n");
    fprintf(stderr, "-s       no. of stages to run each time\n");
    fprintf(stderr, "-n       run without the display window\n");
    fprintf(stderr, "-h       print this message\n");
}
 
/*
 * Working Messages
 */
WorkingMsg( display )
unsigned char display;
{
    char message[80];
    static Widget workingdialog;
    Arg args[5];
    int i=0;
    Dimension wdth;
    Position x,y;
    XmString xmstr;

    if (display==0 && workingdialog) {
        XtUnmanageChild( workingdialog );
        return;
    }
    create_working_message(message);
#if XmVersion<1002
    xmstr=XmStringCreateSimple(message);
#else
    xmstr=XmStringCreateLocalized(message);
#endif
    if (!workingdialog) {
        XtSetArg(args[i],XmNmessageString,xmstr); i++;
        XtSetArg(args[i], XmNtitle, "STATUS"); i++;
        XtSetArg(args[i], XmNdefaultPosition, False); i++;
        XtVaGetValues(main_Menu, XmNx, &x, XmNwidth, &wdth, NULL);
        x += (Position)(wdth/2);
        XtSetArg(args[i], XmNx, x); i++;
        y = HeightOfScreen(XtScreen(main_Menu))/3;
        XtSetArg(args[i], XmNy, y); i++;
        workingdialog = XmCreateWorkingDialog(main_Menu,"working",args,i);
        XtUnmanageChild((Widget) XmMessageBoxGetChild(workingdialog,
                    XmDIALOG_OK_BUTTON));
        XtUnmanageChild((Widget) XmMessageBoxGetChild(workingdialog,
                    XmDIALOG_HELP_BUTTON));
        XtUnmanageChild((Widget) XmMessageBoxGetChild(workingdialog,
                    XmDIALOG_CANCEL_BUTTON));
    }
    else XtVaSetValues(workingdialog,XmNmessageString,xmstr,NULL);
    XmStringFree(xmstr);
    XtManageChild( workingdialog );
    XtPopup( XtParent( workingdialog ), XtGrabNone );
    ForceUpdate(workingdialog);
    XFlush( XtDisplay(topLevel));
}

void ForceUpdate(w)
Widget w;
{
    Widget shell, topshell;
    Window win, topwin;
    Display *dsply;
    XtAppContext cntxt = XtWidgetToApplicationContext(w);
    XWindowAttributes winattr;
    XEvent event;

    /* Find the shell */
    for (shell=w; !XtIsShell(shell); shell=XtParent(shell)) ;
    for (topshell=shell; !XtIsTopLevelShell(topshell);
                           topshell=XtParent(topshell)) ;
    /* Don't do anything unless both shells are realized */
    if (XtIsRealized(shell) && XtIsRealized(topshell)) {
        dsply = XtDisplay(topshell);
        win = XtWindow(shell);
        topwin = XtWindow(topshell);
        /* Wait for dialog to be mapped */
        while (XGetWindowAttributes(dsply,win,&winattr) &&
                    winattr.map_state != IsViewable ) {
            if (XGetWindowAttributes( dsply,topwin,&winattr) &&
                        winattr.map_state != IsViewable ) break;
            XtAppNextEvent( cntxt, &event );
            XtDispatchEvent(&event);
        }
    }
    /* next XSync() or XFlush() will get an expose event */
    XmUpdateDisplay(topshell);
}

void create_working_message( message )
char *message;
{
    switch(Plot_info.plot_type) {
    case XYPLOT: strcpy(message,"Plotting ");
                 strcat(message,Plot_info.variable);
                 break;
    }
}

/*
 * Error Messages
 */
int DisplayElleErrorMsg(char *msg, int err_num)
{
    ErrorMsg(msg,main_Menu,err_num);
}

ErrorMsg( msg, w, err_num )
char *msg;
Widget w;
int err_num;
{
    static Widget errdialog;
    Arg args[5];
    char message[80],buf[20];
    int i=0;
    XmString xmstr;

    strcpy(message,"");
    strncpy(message,msg,80-21);
    create_error_message(message,err_num);
    xmstr=XmStringCreateSimple(message);
    if (!errdialog) {
        XtSetArg(args[0],XmNmessageString,xmstr); i++;
           XtSetArg(args[i], XmNtitle, "ERROR MESSAGE"); i++;
        errdialog = XmCreateErrorDialog(w,"err",args,i);
        XtUnmanageChild((Widget) XmMessageBoxGetChild(errdialog,
                    XmDIALOG_HELP_BUTTON));
        XtUnmanageChild((Widget) XmMessageBoxGetChild(errdialog,
                    XmDIALOG_CANCEL_BUTTON));
    }
    else XtVaSetValues(errdialog,XmNmessageString,xmstr,NULL);
       XmStringFree(xmstr);
    XtManageChild( errdialog );
    XtPopup( XtParent( errdialog ), XtGrabNone );
}

/**********
 * FILE_OPT
 **********/

#if XY
#endif
static void FileOpenSelectionOK(w, client_data, cbs)
Widget w;
XtPointer client_data;
XmFileSelectionBoxCallbackStruct *cbs;/*always third param*/
{
    char *filename,*msg,*fname;
	/*Dimension wdth,hgt,w1,h1;*/
	/*float  oldwdth,oldhgt;*/
    FILE *fp;
    Arg args[5];
    int i,err=0,type=0;
    /*int i,err=0,type=0,chng=0,chng_plot=0;*/
    XmString xmstr;
    input_options tmp;
    ElleRunFunc init;

    if (!XmStringGetLtoR(cbs->value, "", &filename))
                return;
    XtVaGetValues(w, XmNuserData, &type, NULL);
    if (filename[strlen(filename)-1]!=E_DIR_SEPARATOR) {
        switch(type) {
        case SOLN: ElleSetFile(filename);
				   /*oldwdth=Pwindo[XCMAX]-Pwindo[XCMIN];*/
				   /*oldhgt=Pwindo[YCMAX]-Pwindo[YCMIN];*/
                   if ((init = ElleInitFunction())) (*init)();
#if XY
                   init_box( Pwindo,Settings.rescale );
 				    if (chng_plot) {
                     Settings.rescale = 1;
                     ChangeCellConfig(plotArea);
				 	 ChangeCellSize(plotArea,wdth,hgt);
					}
                    XtVaGetValues(main_Menu,XtNheight,&h1,NULL);
					hgt+=h1;
                    XtVaGetValues(currvalsArea,XtNheight,&h1,NULL);
					hgt+=h1;
					XtUnmanageChild(main_Menu);
					XtUnmanageChild(plotArea);
					XtUnmanageChild(currvalsArea);
                    XtMakeResizeRequest(topLevel,wdth,hgt,&w1,&h1);
					XtManageChild(main_Menu);
					XtManageChild(plotArea);
					XtManageChild(currvalsArea);
#endif
#if XY
#endif
                   if (!err) {
                       UpdateMsgArea();
                       XtUnmanageChild(w);
                       EllePlotRegions(1);
                   }
                   else ErrorMsg(filename,w,err);
                   break;
        default:   break;
        }
    }
    else ErrorMsg("The file selected is a directory",w,0);
    XtFree( filename );
}       /* FileOpenSelectionOK */

/*
 * FileSelectionCancel() simply unmaps the file selection box.
 */
static void FileSelectionCancel(w, client_data, call_data)
Widget w;
XtPointer client_data;
XtPointer call_data;
{
    XtUnmanageChild(w);
}       /* FileSelectionCancel */

/*
 * Sets the string for the file mask to all files in
 * current directory
 */
int GetDirSpec( path,file_type )
char *path;
int file_type;
{
    char ext[30];
    int err;

    if (!(err = getcwd(path,FILENAME_MAX))) {
        fprintf(stderr,path);
        return(0);
    }
    switch(file_type) {
    case SOLN: strcpy(ext,"/*.elle");
               strcat(path,ext);
               break;
    case PS  : strcpy(ext,"/*.ps");
               strcat(path,ext);
               break;
    default  : break;
    }
    return(1);
}

/* 
 * maps a file selection box
 */
void OpenDialog(title,file_type)
char *title;
int file_type;
{
    static Widget fileSelection;

    char buf[FILENAME_MAX];
    XmString tmp1, tmp2;
    Arg args[5];
    int i=0;

    if (!fileSelection) {
    tmp1 = XmStringCreateSimple(title);
    XtSetArg(args[i], XmNdialogTitle, tmp1); i++;
    if (GetDirSpec( buf,file_type )) {
#if XmVersion<1002
        tmp2 = XmStringCreateSimple(buf);
#else
        tmp2 = XmStringCreateLocalized(buf);
#endif
        XtSetArg(args[i], XmNdirMask, tmp2); i++;
    }
    XtSetArg(args[i], XmNuserData, file_type); i++;
    fileSelection= (Widget) XmCreateFileSelectionDialog(
            main_Menu,  /* parent widget */
            "fileSelection",  /* widget name */
            args,   /* argument list*/
            i   /* arglist size */
            );
    XtUnmanageChild((Widget) XmFileSelectionBoxGetChild(fileSelection,
                    XmDIALOG_HELP_BUTTON));
    XtAddCallback(fileSelection, XmNcancelCallback,
                    FileSelectionCancel, NULL);
    XtAddCallback(fileSelection, XmNokCallback, FileOpenSelectionOK, NULL);
    /*else XtSetValues(fileSelection,args,i);*/
    XmStringFree(tmp1); XmStringFree(tmp2);
    }
    XtManageChild(fileSelection);
    XtPopup(XtParent(fileSelection), XtGrabNone);
}
/*
 * File open button callback function
 */
static void FileOpenChosen(w, client_data, call_data)
Widget w;
XtPointer client_data;
XtPointer call_data;
{
    switch((int)client_data) {
    case SOLN: OpenDialog("Solution files",SOLN);
               break;
    default:   break;
    }
}

/*
 * File save button callback function
 */
static void FileSaveChosen(w, client_data, call_data)
Widget w;
XtPointer client_data;
XtPointer call_data;
{
    if (strlen(ElleFile())==0) {
        ErrorMsg("No data file open",main_Menu,0);
    }
    else 
        switch((int)client_data) {
        case SOLN: SaveDialog("",SOLN);
                   break;
        case PS:   SaveDialog("",PS);
                   break;
        default:   break;
        }
}


/*
 * maps a file selection box
 */
void SaveDialog(title,file_type)
char *title;
int file_type;
{
    static Widget fileSave;

    char buf[FILENAME_MAX+1];
    XmString tmp1, tmp2;
    Arg args[5];
    int i=0;

    tmp1 = XmStringCreateSimple(title);
    XtSetArg(args[i], XmNdialogTitle, tmp1); i++;
    if (GetDirSpec( buf,file_type )) {
#if XmVersion<1002
        tmp2 = XmStringCreateSimple(buf);
#else
        tmp2 = XmStringCreateLocalized(buf);
#endif
        XtSetArg(args[i], XmNdirMask, tmp2); i++;
    }
    XtSetArg(args[i], XmNuserData, file_type); i++;
    if (!fileSave) {
        fileSave= (Widget) XmCreateFileSelectionDialog(
                main_Menu,  /* parent widget */
                "fileSave",  /* widget name */
                args,   /* argument list*/
                i   /* arglist size */
                );
        XtUnmanageChild((Widget) XmFileSelectionBoxGetChild(fileSave,
                    XmDIALOG_HELP_BUTTON));
        XtAddCallback(fileSave, XmNcancelCallback,
                    FileSelectionCancel, NULL);
        XtAddCallback(fileSave, XmNokCallback, FileSaveSelectionOK, NULL);
    }
    else XtSetValues(fileSave,args,i);
    XmStringFree(tmp1); XmStringFree(tmp2);
    XtManageChild(fileSave);
    XtPopup(XtParent(fileSave), XtGrabNone);
}

static void FileSaveSelectionOK(w, client_data, cbs)
Widget w;
XtPointer client_data;
XmFileSelectionBoxCallbackStruct *cbs;/*always third param*/
{
    char *filename,infilename[SYB_FILENAME_MAX],msg[80];
    char command[SYB_FILENAME_MAX];
    FILE *fp;
    Arg args[5];
    int i,err=0,type=0, len=0;
    XmString xmstr;

    if (!XmStringGetLtoR(cbs->value, "", &filename))
                return;
#if XY
#endif
    XtVaGetValues(w, XmNuserData, &type, NULL);
    if (filename[strlen(filename)-1]!=E_DIR_SEPARATOR) {
        switch(type) {
        case SOLN: err=ElleWriteData(filename);
                    break;
        case PS:    len = strlen(DEFAULT_OPTIONS_EXT);
                    strncpy(infilename,"tmp",SYB_FILENAME_MAX-len-1);
                    strcat(infilename,DEFAULT_OPTIONS_EXT);
                    err=WriteSettingsFile(infilename);
                    sprintf(command,"%s -i %s -d %s -o %s &", "showelleps",
                                         ElleFile(),infilename,filename);
                    system(command);
                    break;
        default:    break;
        }
        if (!err) XtUnmanageChild(w);
        else ErrorMsg(filename,w,err);
    }
    else ErrorMsg("The file selected is a directory",w,0);
    XtFree( filename );
}       /* FileSaveSelectionOK */

void warning_msg(err_num,str)
int err_num;
char *str;
{
    ErrorMsg(str,main_Menu,err_num);
}

/*
 * function called when user confirms exit
 */
void Exit_Proc()
{
    Exit_App(0);
}

void Exit_App(err)
int err;
{
    if (Plot_info.inp_file!=NULL) {
        if (Plot_info.inp_file->fp!=NULL) fclose(Plot_info.inp_file->fp);
        free(Plot_info.inp_file);
    }
    if (plotArea) clear_X(plotArea);
    CleanUp();
    exit(0); 
}

/*
 * exit button callback function
 */
void ExitDialog()
{ 
    Widget warndialog;
    Arg args[5];
    int i=0;
    XmString msg,label1,label2;

    label1 = XmStringCreateSimple("Yes");
    label2 = XmStringCreateSimple("No");
       msg=XmStringCreateSimple("Do you really want to quit?");
    XtSetArg(args[i],XmNmessageString,msg); i++;
    XtSetArg(args[i],XmNokLabelString,label1); i++;
    XtSetArg(args[i],XmNcancelLabelString,label2); i++;
       XtSetArg(args[i], XmNtitle, "WARNING"); i++;
    warndialog = XmCreateWarningDialog(main_Menu,"warn",args,i);
    XtAddCallback( warndialog, XmNokCallback, Exit_Proc, NULL );
    XtUnmanageChild((Widget) XmMessageBoxGetChild(warndialog,
                    XmDIALOG_HELP_BUTTON));
       XmStringFree(msg);
       XmStringFree(label1);
       XmStringFree(label2);
    XtManageChild( warndialog );
    XtPopup( XtParent( warndialog ), XtGrabNone );
}

void RunChosen(w, client_data, call_data)
Widget w;
XtPointer client_data;
XtPointer call_data;
{
    Widget button,pulldown;
    char *var_name;
    XmString xmstr;
    int err=0, err_num=0;
    int len, hgt=8;
    int choice = (int) client_data;
    ElleRunFunc run;

    if (XtIsRealized(main_Menu)) {
        XDefineCursor(XtDisplay(topLevel),
                            XtWindow(XtParent(main_Menu)),Waiting);
        XFlush( XtDisplay(topLevel));
    }
    XtVaGetValues(w,XmNlabelString,&xmstr,NULL);
    XmStringGetLtoR(xmstr,"",&var_name);
    XmStringFree(xmstr);
    len = sizeof(Plot_info.variable) - 1 - strlen(var_name);
     /*
      * these should pull out parent button's label and be tied
      * to structures in log.h
      */
    sprintf(Plot_info.variable,"%s%s%s","Grains",".",var_name);
    strncpy(Plot_info.dflt_label1,var_name, MAXNAME-1);
    XtFree(var_name);
    Plot_info.var_num= choice;
    Plot_info.plot_type= XYPLOT;
    Plot_info.plot_description= 0;
    /*if (Plot_info.inp_file==NULL) { */
    if (strlen(ElleFile())==0) { /*****hack ************/
        XDefineCursor(XtDisplay(topLevel),
                          XtWindow(XtParent(main_Menu)),None);
        XFlush( XtDisplay(topLevel));
        ErrorMsg("No data file open",main_Menu,0);
    }
    else {
        /* WorkingMsg(1); */
        /*UpdateMsgArea();*/
        init_box( Pwindo,Settings.rescale );
        if ((run = ElleRunFunction())) (*run)();
        if (err) {
            XDefineCursor(XtDisplay(topLevel),
                          XtWindow(XtParent(main_Menu)),None);
            XFlush( XtDisplay(topLevel));
            ErrorMsg(Err_str,main_Menu,err);
        }
        else {
            UpdateCell( plotArea,Plot_info.curr_cell,DRAW );
            XDefineCursor(XtDisplay(topLevel),
                          XtWindow(XtParent(main_Menu)),None);
            XFlush( XtDisplay(topLevel));
        }
        /* WorkingMsg(0); */
    }
}

void RerunChosen(w, client_data, call_data)
Widget w;
XtPointer client_data;
XtPointer call_data;
{
    ElleRunFunc init, run;

    if (XtIsRealized(main_Menu)) {
        XDefineCursor(XtDisplay(topLevel),
                            XtWindow(XtParent(main_Menu)),Waiting);
        XFlush( XtDisplay(topLevel));
    }

    if ((init = ElleInitFunction())) (*init)();
    if (strlen(ElleFile())==0) { /*****hack ************/
        XDefineCursor(XtDisplay(topLevel),
                          XtWindow(XtParent(main_Menu)),None);
        XFlush( XtDisplay(topLevel));
        ErrorMsg("No data file open",main_Menu,0);
    }
    else {
        if ((run = ElleRunFunction())) (*run)();
    
        UpdateCell( plotArea,Plot_info.curr_cell,DRAW );
        XDefineCursor(XtDisplay(topLevel),
                          XtWindow(XtParent(main_Menu)),None);
        XFlush( XtDisplay(topLevel));
    }
}

/*
 * update the stage field of Plot_info and update msg area
 */
int SetStage(int stage)
{
    Plot_info.stage = stage;
    UpdateMsgArea();
    return(0);
}

/*
 * update the data displayed below the drawing area
 */
int UpdateMsgArea()
{
    int i;
    XmString str;
    char buf[MAX_LINE_LEN+1], *tmp=0, *start=0;
    XExposeEvent event;

    strcpy(buf,"");
    for (i=0;i<MAX_LINE_LEN;i++) buf[i] = ' ';
    tmp = ElleFile();
    start = strrchr(tmp,E_DIR_SEPARATOR);
    if (start==0) start=tmp;
    else start++;
    sprintf(buf, "stage=%d %s", Plot_info.stage,start);
#if XmVersion<1002
    str = XmStringCreateSimple(buf);
#else
    str = XmStringCreateLocalized(buf);
#endif
    XtVaSetValues(currvalsArea, XmNlabelString, str, NULL);
    XmStringFree(str);
    ForceUpdate(currvalsArea);
    XFlush( XtDisplay(topLevel));
}

/*
 * Create the menu, the cascade button
 * that owns the menu, and then the submenu items.
 */
Widget BuildMenu(parent, menu_title, spacing, data)
Widget parent;
char *menu_title;
int spacing;
MenuItemData *data;
{
    Widget pulldown, cascade, widget;
    char buf[10];
    static int owner;
    int i;
    XmString str;
    Arg args[5];
    Pixel bg, fg,top_shdw, bottom_shdw,select_col;
    extern XColor Colours[];
    extern Colormap cmap;

    i=0;
    if (spacing>0) {
        XtSetArg(args[i], XmNmarginWidth, spacing); i++;
        XtSetArg(args[i], XmNmarginHeight, spacing); i++;
        XtSetArg(args[i], XmNspacing, spacing); i++;
    }
    pulldown = XmCreatePulldownMenu(parent, "plldwn", args, i);

    str = XmStringCreateSimple(menu_title);
    cascade = XtVaCreateManagedWidget(menu_title,
        xmCascadeButtonGadgetClass, parent,
        XmNsubMenuId,   pulldown,
        XmNlabelString, str,
        NULL);

    XmStringFree(str);

    /* add the menu items */
    for (i = 0; data[i].label != NULL; i++) {
        /*
         * If submenu exists, create the pull-right menu
         * If owner is set let submenu items store their origin
         */
        if (data[i].submenu) {
            if (data[i].submenu_data!=NO_VALUE) owner=data[i].submenu_data;
            else owner=NO_VALUE;
            widget = BuildMenu(pulldown,
                            data[i].label, data[i].spacing, data[i].submenu);
        }
        else
           widget = XtVaCreateManagedWidget(data[i].label,
                *data[i].class, pulldown,
                NULL);
        if (data[i].bg!=NO_VALUE) {
            bg = Colours[data[i].bg].pixel;
            XmGetColors( XtScreen(pulldown),cmap,bg,
                            &fg, &top_shdw, &bottom_shdw, &select_col);
            XtVaSetValues(widget,
                    XmNbackground, bg,
                    XmNborderColor, fg,
                    XmNtopShadowColor, top_shdw,
                    XmNbottomShadowColor, bottom_shdw,
                    XmNarmColor, select_col,
                    NULL);
        }
        /*
         * note - activateCallback
         */
        if (data[i].callback)
            XtAddCallback(widget, XmNactivateCallback,
                data[i].callback, data[i].callback_data);
        if (owner!=NO_VALUE)
            XtVaSetValues(widget,
                    XmNuserData, owner, NULL);
    }
    return cascade;
}

int Create_widgets( top )
Widget top;
{
    Widget mainWindow, button, submenu;
    Pixel fg,bg;
    int i;

    /* create main window */
    mainWindow = XtVaCreateManagedWidget( "mainWindow",
            xmMainWindowWidgetClass, top,
			/*XmNresizable, True,*/
            NULL);

    main_Menu = XmCreateMenuBar(
        mainWindow,   /* parent widget*/
        "main_Menu",  /* widget name */
        NULL,
        0
        );
    XtManageChild( main_Menu );

    for ( i=0;mainopts[i].label!=NULL;i++ ) {
        button = BuildMenu( main_Menu,mainopts[i].label,
                                mainopts[i].spacing,
                                mainopts[i].submenu );
    }
        /*
         * help button must be the last one created.
         * Set the resource so that it is placed on
         * the far right of the menu bar
    XtVaSetValues( main_Menu,XmNmenuHelpWidget,button,NULL );
         */

    plotArea = XtVaCreateManagedWidget( "plotArea",
            plotWidgetClass, mainWindow,
            NULL);

    currvalsArea = XtVaCreateManagedWidget("current values",
    xmLabelWidgetClass, mainWindow,
    XmNalignment,   XmALIGNMENT_BEGINNING,
    NULL);
  
    XmMainWindowSetAreas(
        mainWindow,   /* widget */
        main_Menu,      /* menu */
        NULL,          /* command */
        NULL,          /* h scroll */
        NULL,          /* v scroll */
        plotArea      /* region */
        );

    XtVaSetValues(mainWindow,
        XmNmessageWindow, currvalsArea,
			/*XmNresizable, True,*/
        NULL);

    XtVaGetValues(main_Menu,
        XmNforeground, &fg,
        XmNbackground, &bg,
        NULL);

    XtVaSetValues(currvalsArea,
        XmNforeground, fg,
        XmNbackground, bg,
			/*XmNresizable, True,*/
        NULL);
        
    
    return(0);
}

void ResizePlotWin(int chng_plot)
{
	Dimension wdth,hgt,w1,h1;
	int chng=0;

    XtVaGetValues(plotArea,XtNwidth,&wdth,
                           XtNheight,&hgt,
                           NULL);
    XtVaGetValues(topLevel,XtNwidth,&w1,
                           XtNheight,&h1,
                           NULL);
/*   chng_plot=((fabs(oldwdth-(Pwindo[XCMAX]-Pwindo[XCMIN]))>0)||
				   (fabs(oldhgt-(Pwindo[YCMAX]-Pwindo[YCMIN]))>0)?1:0);*/
				   chng=(chng_plot || (wdth!=w1) || (hgt!=h1) ? 1:0);
    if (chng>0) {
        if (chng_plot) {
            XtUnrealizeWidget(topLevel);
            ChangeCellConfig(plotArea);
            XtVaGetValues(plotArea,XtNwidth,&wdth,
                                                   XtNheight,&hgt,
                                                   NULL);
            /*if (wdth<Minwidth) wdth = Minwidth;*/
            XtMakeResizeRequest(topLevel,wdth,hgt,&w1,&h1);
                                XtRealizeWidget(topLevel);
        }
    }
}
void update_()
{
    UpdateCell( plotArea,Plot_info.curr_cell,DRAW );
}

Init_App(argc, argv, optsfp)
int argc;
char **argv;
FILE **optsfp;
{
    int err=0, i;
    char optsfile[SYB_FILENAME_MAX],LogFile[80],PsFile[80];
    char input_str[MAX_LINE_LEN], *tmpstr, *dir;
    FILE *fp;
    extern Display *Dsply;

    static XrmOptionDescRec table[] = {
        {"-pw",            "*pixmapWidthInCells", XrmoptionSepArg, NULL},
        {"-ph",            "*pixmapHeightInCells", XrmoptionSepArg, NULL},
               /* just to make motif ignore -i option */
        {"-ilbl",            "*labelList", XrmoptionSepArg, NULL},
    };
  
    XtToolkitInitialize();
    app_context = XtCreateApplicationContext();
    XtAppSetFallbackResources(app_context,default_resources);
    if ((Dsply = XtOpenDisplay(app_context,NULL,NULL,"Xelle",
                           table,XtNumber(table),&argc,argv))==0)
        OnError("Could not open X display",0);

        /* read display settings */
    tmpstr=ElleDspOptsFile();
    if (strlen(tmpstr)!=0) {
        strncpy(optsfile,tmpstr,SYB_FILENAME_MAX);
        fp=fopen(optsfile,"r");
        if (fp==NULL) error_msg(OPEN_ERR,optsfile);
    }
    else {
        strcpy(optsfile,DEFAULT_OPTIONS_FILE);
        /* check in current dir for options file */
        if ((fp = fopen(optsfile,"r"))==NULL) {
            /* check in user's home dir for options file */
            if ((dir = getenv("HOME"))!=NULL) {
                strcpy(optsfile,dir);
                strcat(optsfile,"/");
                strcat(optsfile,DEFAULT_OPTIONS_FILE);
                fp = fopen(optsfile,"r");
            }
        }
    }
    if (fp!=NULL) {
        if (err=ReadSettings(fp,input_str,1))
            error_msg(err,input_str);
        fclose(fp);;
    }
    return( err );
}

Set_Colours(red,green,blue,irgbv,steps)
int *red,*green,*blue,*irgbv,steps;
{
    extern Display *Dsply;

    if ((Plot_info.max_colours=CreateColormap( Dsply,
                                               XDefaultScreenOfDisplay(Dsply),
                                               red,green,blue,irgbv,
                                               steps)) == 0) return( 1);
#if XY
    Colour_range[0] = USEFL_SIZE;
    Colour_range[1] = Plot_info.max_colours;
    Arrow_colours[0] = 5; /* cyan */
    Arrow_colours[1] = 7; /* magenta */
#endif
    return(0);
}
 
Run_App(logfp)
FILE *logfp;
{
    int dum=1;
    ElleRunFunc init;
    extern Display *Dsply;
    extern Colormap cmap;

    topLevel = XtVaAppCreateShell(NULL,"Xelle",
                                applicationShellWidgetClass,
                                Dsply,
                                XtNcolormap,cmap,
								/*XmNallowShellResize,True,*/
                                NULL);
    if (Create_widgets( topLevel )) return(1);

    setpencolor_(&Settings.fg);
    setlinewidth_(&Settings.linewidth);
    dashln_(&Settings.linestyle,&dum);

    Waiting = XCreateFontCursor( XtDisplay(topLevel),XC_watch);

    if ((init = ElleInitFunction())) (*init)();

    XtRealizeWidget(topLevel);

    XtAppMainLoop(app_context);

    return(0);
}
