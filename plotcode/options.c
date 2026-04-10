/*----------------------------------------------------------------
 *    Elle:   options.c  1.0  11 September 1997
 *
 *    Copyright (c) 1997 by L.A. Evans & T.D. Barr
 *----------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <Xm/DialogS.h>
#include <Xm/PushB.h>
#include <Xm/PushBG.h>
#include <Xm/PanedW.h>
#include <Xm/Form.h>
#include <Xm/RowColumn.h>
#include <Xm/LabelG.h>
#include <Xm/ToggleBG.h>
#include <Xm/SeparatoG.h>
#include <Xm/ArrowBG.h>
#include <Xm/TextF.h>
#include "types.h"
#include "csettings.h"
#include "globals.h"
#include "error.h"
#include "string_utils.h"
#include "runopts.h"
#include "options.h"
#include "opttypes.h"
#include "file.h"

#define COLS  8
#define NUMBARS 4
#define MAXBUTTONS MAX_RGN_OPTIONS
#define XY  0


typedef struct {
    Widget max;
    Widget min;
    int id;
    int min_id;
    int max_id;
} MaxMinData;

typedef struct {
    char *label;
    valid_terms *items;
    Widget *buttonlist;
    void (*callback)();
    int userdata;
    int userdata_min;
    int userdata_max;
} OptsBarItem;

/*
typedef struct {
    int num_ints;
    int num_toggles;
    int num_floats;
    Widget p_u_shell;
} OptionDataItem;

typedef struct {
    int id;
    int type;
} OptionItem;

typedef struct {
    int id;
} ToggleItem;

typedef struct {
    char *label;
    void (*callback)();
    OptionDataItem *data;
} OptsActionAreaItem;
*/

extern void ElleFlynnDfltAttributeList(int **active, int *maxa);
extern void ElleUnodeAttributeList(int **active, int *maxa);
extern int ElleMaxLevels();
extern int ElleUnodesActive();

Widget Txt_int_wdgts[8], Txt_fl_wdgts[8], RunOpts_wdgts[16];
Widget Buttons[NUMBARS][MAX_RGN_OPTIONS];
struct plot_params Tmp;
void OptsDialog();
void RunOptsDialog();
void BuildOptionBar(OptsBarItem *info, Widget parent, Widget *bar,
                    int *fl_cnt, MaxMinData *maxmin);
static void CreateActionArea();
static void OptsDlgDone();
static void RunOptsDlgDone();
static void UserCancel();
void toggled();
void toggled_int();
void change_bar(), change_level();

OptsBarItem Baroptions[] = {
{ "Region Attribute", region_terms, &(Buttons[0][0]),change_bar,
   O_FLYNNATTRIB, O_FLYNNATTRMIN, O_FLYNNATTRMAX },
{ "Node Attribute", node_terms, &(Buttons[1][0]),change_bar,
   O_NODEATTRIB, O_NODEATTRMIN, O_NODEATTRMAX },
{ "U_node Attribute", unode_terms, &(Buttons[2][0]),change_bar,
   O_UNODEATTRIB, O_UNODEATTRMIN, O_UNODEATTRMAX },
{ "Boundary Attribute", boundary_terms, &(Buttons[3][0]),change_bar,
   O_BNDATTRIB, O_BNDATTRMIN, O_BNDATTRMAX },
   NULL
  } ;

MaxMinData MaxMin[NUMBARS];  /* one for each Baroption */


OptionItem  Option_items_int[] = {
    { O_UPDFREQ, SYB_INT },
    };

ToggleItem Option_items_toggle[] = {
    { O_OVERLAY },
    { O_LBLREGION },
    { O_DRWUNODETYP },
    };

OptionItem  RunOption_items[] = {
    /*{ RO_DISPLAY, SYB_INT },*/
    { RO_STAGES, SYB_INT },
    { RO_SVFREQ, SYB_INT },
    { RO_SVFILEROOT, SYB_STRING },
    { RO_SWITCHDIST, SYB_DOUBLE },
    { RO_SPEEDUP, SYB_DOUBLE },
    };

void RunOptsDialog( w, client_data, call_data )
Widget w;
XtPointer client_data;
XtPointer call_data;
{
    XmString labels[3];
    static Widget run_opts_dialog;
    Widget *parent,rowcol1,rowcol2,pane,toggle_box;
    char buf[30],name[FILENAME_MAX], wdgt_label[MAX_OPTION_NAME+1];
    int i=0, j, val, state=0;
    float fval;
    double dval;
    Dimension wdth;
    Position x,y;
    Arg args[2];
    static OptionDataItem run_user_data;
    static OptsActionAreaItem button_items_run[] = {
        { "Ok",     RunOptsDlgDone, NULL          },
        { "Cancel", UserCancel, NULL          },
    };

    ElleCopyRunOptions();

    if (!run_opts_dialog) {
        run_user_data.num_ints = run_user_data.num_toggles =
             run_user_data.num_floats = 0;
        parent = (Widget *)client_data;
        run_opts_dialog = XtVaCreatePopupShell("Set_Runtime_Options",
            xmDialogShellWidgetClass, *parent,
            NULL);

        button_items_run[0].data = &run_user_data;
        button_items_run[1].data = &run_user_data;

        pane = XtVaCreateWidget("pane",
            xmPanedWindowWidgetClass, run_opts_dialog,
            XmNsashWidth,  1,
            XmNsashHeight, 1,
            NULL);

        rowcol1 = XtVaCreateWidget("rowcol1",
            xmRowColumnWidgetClass, pane,
            XmNpacking,            XmPACK_COLUMN,
            XmNnumColumns,        XtNumber(RunOption_items),
            XmNorientation,        XmHORIZONTAL,
            XmNisAligned,        True,
            XmNentryAlignment,    XmALIGNMENT_END,
            NULL);

        /*
         * create TextField Widgets with Labels
         */
        for (i=0,j=0;j<XtNumber(RunOption_items);j++) {
            id_match(run_option_terms,RunOption_items[j].id,wdgt_label);
            switch(RunOption_items[j].type) {
            case SYB_INT: ElleGetRunOptionValue(RunOption_items[j].id,
                                           &val);
                            sprintf(buf,"%4d",val);
                            break;
            case SYB_FLOAT: ElleGetRunOptionValue(RunOption_items[j].id,
                                         (int *)(&fval));
                            sprintf(buf,"%f",fval);
                            break;
            case SYB_DOUBLE:ElleGetRunOptionValue(RunOption_items[j].id,
                                         (int *)(&dval));
                            sprintf(buf,"%.3e",dval);
                            break;
            case SYB_STRING:
                            ElleGetRunOptionValue(RunOption_items[j].id,
                                         (int *)name);
                            strncpy(buf,name,29);
                            break;
            default:        break;
            }
            XtVaCreateManagedWidget(wdgt_label,
                xmLabelGadgetClass, rowcol1, NULL );

            RunOpts_wdgts[i] = XtVaCreateManagedWidget("txt_wdgt",
                xmTextFieldWidgetClass, rowcol1, 
                XmNuserData, (XtPointer)RunOption_items[j].id,
                XmNvalue, buf,
                XmNcolumns, 6, NULL );
            i++;
        }
/*
 * store the number of integer plot parameters
 */
        run_user_data.num_ints = i;

        XtManageChild (rowcol1);

        XtVaGetValues( rowcol1,XmNwidth,&wdth,NULL );

    /* Create the action area
     * *****change the width that is sent if more widgets are added
     */
        CreateActionArea(pane,button_items_run,
                              XtNumber(button_items_run),wdth*12);

        XtTranslateCoords( w,(Position)0,
                   (Position)0,&x,&y);
        XtVaSetValues( run_opts_dialog,XmNx,x,
                   XmNy,y,
                   NULL );
        XtManageChild (pane);
        run_user_data.p_u_shell = run_opts_dialog;
    }
    else {
        i=0;
        for (j=0;j<run_user_data.num_ints;j++,i++) {
            switch(RunOption_items[j].type) {
            case SYB_INT: ElleGetRunOptionValue(RunOption_items[j].id,
                                           &val);
                            sprintf(buf,"%4d",val);
                            break;
            case SYB_FLOAT: ElleGetRunOptionValue(RunOption_items[j].id,
                                         (int *)(&fval));
                            sprintf(buf,"%f",fval);
                            break;
            case SYB_DOUBLE:ElleGetRunOptionValue(RunOption_items[j].id,
                                         (int *)(&dval));
                            sprintf(buf,"%.3e",dval);
                            break;
            case SYB_STRING:
                            ElleGetRunOptionValue(RunOption_items[j].id,
                                         (int *)name);
                            strncpy(buf,name,29);
                            break;
            default:        break;
            }
            XtVaSetValues(RunOpts_wdgts[i], XmNvalue, buf, NULL);
        }
        XtTranslateCoords( w,(Position)0,
                   (Position)0,&x,&y);
        XtVaSetValues( run_opts_dialog,XmNx,x,
                   XmNy,y,
                   NULL );
    }

    XtPopup(run_opts_dialog, XtGrabNone);
}

int Max_level_items;
Widget Rgn_min=0, Rgn_max=0;
Widget Unode_min=0, Unode_max=0;

void OptsDialog( w, client_data, call_data )
Widget w;
XtPointer client_data;
XtPointer call_data;
{
    XmString bar, label;
    static Widget opts_dialog, toggle_box, bar_menus[NUMBARS],
                  level, level_menu, level_button[MAX_LEVEL_OPTIONS];
    Widget *parent,rowcol1,rowcol2,pane,menu,tmp_w;
    Widget *list;
    char buf[30], wdgt_label[MAX_OPTION_NAME+1];
    int i=0, j, k, fl_cnt=0, val, state=0, start, num_items;
    int *f_attr, num_f_attr, *n_attr, num_n_attr, found, n;
    double fval;
    Dimension wdth;
    Position x,y;
    Arg args[5];
    static OptionDataItem user_data;
    static OptsActionAreaItem button_items_opts[] = {
        { "Ok",     OptsDlgDone, NULL          },
        { "Cancel", UserCancel, NULL          },
    };

    /*
     * this check is in case a new file with <>levels has been opened
     */
    GetCBndLevelOption(&state);
    n=ElleMaxLevels();
    if (n<state || (!state && n))
        SetCBndLevelOption(n);
    Tmp = Settings.plot_opts;
    /* this is only needed because they both call UserCancel - fix */
    ElleCopyRunOptions();

    if (!opts_dialog) {
        for (i=0;i<NUMBARS;i++) bar_menus[i]=NULL;
        user_data.num_ints = user_data.num_toggles =
             user_data.num_floats = 0;
        parent = (Widget *)client_data;
        opts_dialog = XtVaCreatePopupShell("Set_Display_Options",
            xmDialogShellWidgetClass, *parent,
            NULL);

        button_items_opts[0].data = &user_data;
        button_items_opts[1].data = &user_data;

        pane = XtVaCreateWidget("pane",
            xmPanedWindowWidgetClass, opts_dialog,
            XmNsashWidth,  1,
            XmNsashHeight, 1,
            NULL);

        i=0;
#if XY
#endif
        rowcol1 = XtVaCreateWidget("rowcol1",
            xmRowColumnWidgetClass, pane,
            XmNpacking,            XmPACK_COLUMN,
            XmNnumColumns,        2,
            XmNorientation,        XmVERTICAL,
            XmNisAligned,        True,
            XmNentryAlignment,    XmALIGNMENT_END,
            NULL);

        /*
         * create TextField Widgets with Labels
         */
        for (j=0;j<XtNumber(Option_items_int);j++) {
            id_match(option_terms,Option_items_int[j].id,wdgt_label);
            XtVaCreateManagedWidget(wdgt_label,
                xmLabelGadgetClass, rowcol1, NULL );

            GetOptionValue(&Tmp,Option_items_int[j].id,
                                           (XtPointer)(&val));
            sprintf(buf,"%4d",val);
            Txt_int_wdgts[i] = XtVaCreateManagedWidget("txt_wdgt",
                xmTextFieldWidgetClass, rowcol1, 
                XmNuserData, (XtPointer)Option_items_int[j].id,
                XmNvalue, buf,
                XmNcolumns, 4, NULL );
            i++;
        }

/*
 * store the number of integer plot parameters
 */
        user_data.num_ints = i;

        toggle_box=XtVaCreateManagedWidget("toggle",
            xmRowColumnWidgetClass, pane,
            XmNisAligned, True, NULL );

        /*
         * create ToggleButton gadgets with Labels
         */
        for (j=0;j<XtNumber(Option_items_toggle);j++) {
            id_match(option_terms,Option_items_toggle[j].id,wdgt_label);
            Txt_int_wdgts[i] = XtVaCreateManagedWidget(wdgt_label,
                xmToggleButtonGadgetClass, toggle_box,
                XmNuserData, (XtPointer)Option_items_toggle[j].id,
                NULL );
            XtAddCallback( Txt_int_wdgts[i],XmNvalueChangedCallback,
                            toggled, (XtPointer)Option_items_toggle[j].id );
            GetOptionValue(&Tmp,Option_items_toggle[j].id,
                                           (XtPointer)(&val));
            XmToggleButtonGadgetSetState(Txt_int_wdgts[i], (val==1), 0);
            i++;
        }
  /*
   * store the number of boolean plot parameters
   */
        user_data.num_toggles = i-user_data.num_ints;

        XtVaCreateManagedWidget("sep1",
            xmSeparatorGadgetClass, toggle_box, NULL );

        for (j=0;Baroptions[j].label!=NULL;j++)
            BuildOptionBar( &(Baroptions[j]),toggle_box,&bar_menus[j],
                       &fl_cnt, &MaxMin[j] );

        /*
         * create the option menu for specifying boundary level
         */
        Max_level_items = MAX_LEVEL_OPTIONS;
#if XmVersion<1002
        bar = XmStringCreateSimple("Boundary Level");
#else
        bar = XmStringCreateLocalized("Boundary Level");
#endif
        level_menu = XmCreatePulldownMenu(toggle_box,"_pulldown",NULL,0);
        n=0;
        XtSetArg(args[n],XmNlabelString,bar); n++;
        XtSetArg(args[n],XmNsubMenuId,level_menu); n++;
        level = XmCreateOptionMenu(toggle_box, "level_menu",
                                          args,n);
        GetOptionValue(&Tmp,O_BNDLEVEL,(XtPointer)(&state));
        for (k=1,n=0;n<MAX_LEVEL_OPTIONS;n++,k++) {
            sprintf(buf,"%2d",k);
            level_button[n] = XtVaCreateManagedWidget(buf,
                               xmPushButtonGadgetClass,level_menu,
                               XmNuserData, O_BNDLEVEL,
                               NULL);
            XtAddCallback(level_button[n],XmNactivateCallback,change_level,
                           (XtPointer)k);
        }
        XmStringFree(bar);
        XtManageChild(level);

        XtManageChild (rowcol1);
        XtManageChild (toggle_box);

        XtVaGetValues( toggle_box,XmNwidth,&wdth,NULL );

    /* Create the action area */
        CreateActionArea(pane,button_items_opts,
                                  XtNumber(button_items_opts),wdth);

        /*user_data.data = &Tmp;*/
        user_data.num_floats = fl_cnt;
        user_data.p_u_shell = opts_dialog;

        /* position dialog box near menu item */
        XtTranslateCoords( w,(Position)0,
                   (Position)0,&x,&y);
        XtVaSetValues( opts_dialog,XmNx,x,
                   XmNy,y,
                   NULL );
        XtManageChild (pane);
    }
    else {
#if XY
#endif
        i=0;
        for (j=0;j<user_data.num_ints;j++,i++) {
            GetOptionValue(&Tmp,Option_items_int[j].id,
                                           (XtPointer)(&val));
            sprintf(buf,"%4d",val);
            XtVaSetValues(Txt_int_wdgts[i], XmNvalue, buf, NULL);
        }
        i=user_data.num_ints;
        for (j=0;j<user_data.num_toggles;j++,i++) {
            GetOptionValue(&Tmp,Option_items_toggle[j].id,
                                           (XtPointer)(&val));
            XmToggleButtonGadgetSetState(Txt_int_wdgts[i], (val==1), 0);
        }

#if XY
        if (id_match(boundary_terms,
                   Tmp.boundary_attrib,buf)) {
#if XmVersion<1002
            label = XmStringCreateSimple(buf);
#else
            label = XmStringCreateLocalized(buf);
#endif
            XtVaSetValues( XmOptionButtonGadget(bnd_bar_menu),
                           XmNlabelString,label,NULL );
            XmStringFree(label);
        }
#endif

        /* position dialog box near menu item */
        XtTranslateCoords( w,(Position)0,
                   (Position)0,&x,&y);
        XtVaSetValues( opts_dialog,XmNx,x,
                   XmNy,y,
                   NULL );
    }

    /*
     * grey out inactive attributes
     * first item (NONE) always sensitive 
     * (first two items (NONE, STANDARD) always sensitive for bnd
     * (first two items (NONE, LOCATION) always sensitive for unodes
     * set the choice to NONE if current attrib not active
     */
    for (j=0;Baroptions[j].label!=NULL;j++) {
        GetActiveAttributeList(Baroptions[j].userdata,
                               &f_attr,&num_f_attr,&start);
        GetOptionValue(&Tmp,Baroptions[j].userdata,
                                           (XtPointer)(&state));
        list = Baroptions[j].buttonlist;
        tmp_w=NULL;
        for (k=start;Baroptions[j].items[k].name!=NULL;k++) {
            for (n=0,found=0;n<num_f_attr && !found;n++)
                if (Baroptions[j].items[k].id==f_attr[n] ||
                     (Baroptions[j].items[k].id==MISORIENTATION &&
                       f_attr[n]==E3_ALPHA)) found=1;
            if (Baroptions[j].items[k].id==TRPL_ONLY) found=1;
            if (Baroptions[j].items[k].id==U_LOCATION &&
                                   ElleUnodesActive()) found=1;
            if (found) {
                XtSetSensitive(list[k],True);
                if (Baroptions[j].items[k].id==state) tmp_w=list[k];
            }
            else {
                XtSetSensitive(list[k],False);
            }
        }
        if (tmp_w!=NULL)
            XtVaSetValues(bar_menus[j],XmNmenuHistory,tmp_w,NULL);
        GetOptionValue(&Tmp,MaxMin[j].min_id, (XtPointer)(&fval));
        sprintf(buf,"%-.3e",fval);
        XmTextSetString( MaxMin[j].min,buf );
        GetOptionValue(&Tmp,MaxMin[j].max_id, (XtPointer)(&fval));
        sprintf(buf,"%-.3e",fval);
        XmTextSetString( MaxMin[j].max,buf );
        if (f_attr) free(f_attr);
    }
    /*
     * show the appropriate option menu items for boundary level
     */
    num_items = ElleMaxLevels();
    if (num_items==0) num_items=1;
    if (num_items<Max_level_items) {
        GetCBndLevelOption(&state);
        if (state>num_items) {
            state=num_items;
            SetOptionValue(&Tmp,O_BNDLEVEL,(XtPointer)(&state));
            SetCBndLevelOption(state);
        }
        for (k=num_items;k<Max_level_items;k++)
            XtUnmanageChild(level_button[k]);
    }
    else if (num_items>Max_level_items)
        for (k=Max_level_items;k<num_items;k++)
            XtManageChild(level_button[k]);
    Max_level_items = num_items;


    XtPopup(opts_dialog, XtGrabNone);
}

void BuildOptionBar(OptsBarItem *info, Widget parent, Widget *bar,
                    int *fl_cnt, MaxMinData *maxmin)
{
    char buf[30];
    Arg args[5];
    Widget rowcol2, menu, bar_menu, tmp_w, wdgt_label, button;
    XmString label;
    int num_items, state=0, n, k;

    for(k=0;k<MAXBUTTONS;k++) info->buttonlist[k]=NULL;
  /*
   * create the option menu for attribute
   */
        GetOptionValue(&Tmp,info->userdata,(XtPointer)(&state));
#if XmVersion<1002
        label = XmStringCreateSimple(info->label);
#else
        label = XmStringCreateLocalized(info->label);
#endif
        if (id_match(info->items,Tmp.region,buf)) {
          while (info->items[state].name!=NULL
                   && strcmp(buf,info->items[state].name))
              state++;
          if (info->items[state].name==NULL) state = 0;
        }

        menu = XmCreatePulldownMenu(parent,"_pulldown",NULL,0);
        n=0;
        XtSetArg(args[n],XmNlabelString,label); n++;
        XtSetArg(args[n],XmNsubMenuId,menu); n++;
        bar_menu = *bar = XmCreateOptionMenu(parent, "region_menu",
                                          args,n);
        for (k=0,tmp_w=NULL;info->items[k].name;k++) {
            info->buttonlist[k] = XtVaCreateManagedWidget(
                               info->items[k].name,
                               xmPushButtonGadgetClass,menu,
                               XmNuserData, maxmin,
                               NULL);
            if (info->items[k].id==state) tmp_w=info->buttonlist[k];
            XtAddCallback(info->buttonlist[k],XmNactivateCallback,
                           info->callback,
                           (XtPointer)info->items[k].id);
        }
        if (tmp_w!=NULL)
            XtVaSetValues(bar_menu,XmNmenuHistory,tmp_w,NULL);
        XmStringFree(label);
        XtManageChild(bar_menu);

        rowcol2 = XtVaCreateWidget("min_max",
            xmRowColumnWidgetClass, parent,
            XmNpacking,            XmPACK_COLUMN,
            XmNnumColumns,        4,
            XmNorientation,        XmVERTICAL,
            XmNisAligned,        True,
            XmNentryAlignment,    XmALIGNMENT_END,
            NULL);
        sprintf(buf,"%s","Min:");
        XtVaCreateManagedWidget(buf,
                xmLabelGadgetClass, rowcol2, NULL );

        maxmin->id = info->userdata;
        Txt_fl_wdgts[*fl_cnt]=maxmin->min =
            XtVaCreateManagedWidget("rgn_min",
                                    xmTextFieldWidgetClass, rowcol2, 
                                    XmNuserData, (XtPointer)info->userdata_min,
                                    XmNcolumns, 10, NULL );
        maxmin->min_id = info->userdata_min;
        (*fl_cnt)++;

        sprintf(buf,"%s","Max:");
        XtVaCreateManagedWidget(buf,
                xmLabelGadgetClass, rowcol2, NULL );

        Txt_fl_wdgts[*fl_cnt]=maxmin->max =
            XtVaCreateManagedWidget("rgn_max",
                                    xmTextFieldWidgetClass, rowcol2, 
                                    XmNuserData, (XtPointer)info->userdata_max,
                                    XmNcolumns, 10, NULL );
        maxmin->max_id = info->userdata_max;
        (*fl_cnt)++;
        XtManageChild(rowcol2);

        XtVaCreateManagedWidget("sep",
            xmSeparatorGadgetClass, parent, NULL );
}

/*
 * dialog ok button callback function for plotting options
 */
static void OptsDlgDone(w, client_data, call_data)
Widget w;
XtPointer client_data, call_data;
{
    unsigned char reset = 0;
    char *str;
    int items, i, val;
    double fval;
    OptionDataItem *user_data;
    XtPointer ptr;
    extern Widget plotArea;

    user_data = (OptionDataItem *)client_data;
    
    /*
     * get values of integer parameters
     */
    items = user_data->num_ints;
    for (i=0;i<items;i++) {
        XtVaGetValues(Txt_int_wdgts[i],
               XmNuserData, &ptr,
               NULL);
        str = XmTextFieldGetString(Txt_int_wdgts[i]);
        val = atoi(str);
        XtFree(str);
        SetOptionValue(&Tmp,(int)ptr,(XtPointer)(&val));
    }
    /*
     * get values of float parameters
     */
    items = user_data->num_floats;
    for (i=0;i<items;i++) {
        XtVaGetValues(Txt_fl_wdgts[i],
               XmNuserData, &ptr,
               NULL);
        str = XmTextFieldGetString(Txt_fl_wdgts[i]);
        fval = atof(str);
        XtFree(str);
        SetOptionValue(&Tmp,(int)ptr,(XtPointer)(&fval));
    }
    
    Settings.plot_opts = Tmp;

    XtPopdown(user_data->p_u_shell);
}

/*
 * dialog ok button callback function for runtime options
 */
static void RunOptsDlgDone(w, client_data, call_data)
Widget w;
XtPointer client_data, call_data;
{
    unsigned char reset = 0;
    char *str;
    int items, i, j, val, type, id;
    float fval;
    double dval;
    OptionDataItem *user_data;
    XtPointer ptr;
    extern Widget plotArea;

    user_data = (OptionDataItem *)client_data;
    
    /*
     * get values of integer parameters
     * save any changed values for log file
     */
    items = user_data->num_ints+user_data->num_toggles;
    for (i=0;i<items;i++) {
        XtVaGetValues(RunOpts_wdgts[i],
               XmNuserData, &ptr,
               NULL);
        id = (int)ptr;
        str = XmTextFieldGetString(RunOpts_wdgts[i]);
        j=0;
        while(j<XtNumber(RunOption_items) &&
             RunOption_items[j].id!=id) j++;
/*******do something if j==XtNumber(RunOption_items)******/
        switch(RunOption_items[j].type) {
        case SYB_INT:   val = atoi(str);
                        ElleSetRunOptionValue(id,&val);
                        break;
        case SYB_FLOAT: fval = (float)atof(str);
                        ElleSetRunOptionValue(id,(int *)&fval);
                        break;
        case SYB_DOUBLE:dval = atof(str);
                        ElleSetRunOptionValue(id,(int *)&dval);
                        break;
        case SYB_STRING:ElleSetRunOptionValue(id,(int *)str);
                        break;
        default:        break;
            }
        XtFree(str);
    }

    XtPopdown(user_data->p_u_shell);
}

/*
 * dialog cancel button callback function
 */
static void UserCancel(w, client_data, call_data)
Widget w;
XtPointer client_data, call_data;
{
    OptionDataItem *user_data;
    int flag = (int)call_data;

    /* reset to original state */
    if (flag) ElleRestoreRunOptions();

    user_data = (OptionDataItem *)client_data;
    XtPopdown(user_data->p_u_shell);
}

void toggled(w, client_data, call_data)
Widget w;
XtPointer client_data, call_data;
{
    int val;
    int id = (int)client_data;

    XmToggleButtonCallbackStruct *data = 
        (XmToggleButtonCallbackStruct *) call_data;
    if (data->set) val=1;
    else val = 0;
    SetOptionValue(&Tmp,id,(XtPointer)(&val));
}

void change_bar(w, client_data, call_data)
Widget w;
XtPointer client_data, call_data;
{
    char buf[10];
    int attr = (int)client_data;
    int id;
    double fval;
    MaxMinData *maxmin;

    XtVaGetValues(w,XmNuserData,&id,NULL);
    XtVaGetValues(w,XmNuserData,&maxmin,NULL);
    /*
     * setting the flynn attribute will reset the max, min
     */
    SetOptionValue(&Tmp,maxmin->id,(XtPointer)(&attr));
    strcpy(buf,"");
    GetOptionValue(&Tmp,maxmin->min_id, (XtPointer)(&fval));
    sprintf(buf,"%-.3e",fval);
    XmTextSetString( maxmin->min,buf );
    GetOptionValue(&Tmp,maxmin->max_id, (XtPointer)(&fval));
    sprintf(buf,"%-.3e",fval);
    XmTextSetString( maxmin->max,buf );
}

void change_level(w, client_data, call_data)
Widget w;
XtPointer client_data, call_data;
{
    char buf[10];
    int attr = (int)client_data;
    int id;

    XtVaGetValues(w,XmNuserData,&id,NULL);
    SetOptionValue(&Tmp,id,(XtPointer)(&attr));
}

int GetActiveAttributeList(int id,int **list,int *num,int *start)
{
    int err=0;
    *start = 1;
    switch(id) {
    case O_FLYNNATTRIB: ElleFlynnDfltAttributeList(list,num);
                        break;
    case O_NODEATTRIB:  ElleNodeAttributeList(list,num);
                        *start = 2;
                        break;
    case O_UNODEATTRIB: ElleUnodeAttributeList(list,num);
                        *start = 2;
                        break;
    case O_BNDATTRIB: ElleFlynnDfltAttributeList(list,num);
                        *start = 2;
                        break;
    }
    return(err);
}

static void CreateActionArea(parent, actions, num_actions, wdth)
Widget parent;
OptsActionAreaItem *actions;
int num_actions;
Dimension wdth;
{
    Widget action_area, widget;
    int i;

    action_area=XtVaCreateWidget("action_area", xmFormWidgetClass, parent,
        XmNfractionBase, num_actions*2 + 1,
        XmNwidth, wdth,
        NULL);

    for (i = 0; i < num_actions; i++) {
        widget = XtVaCreateManagedWidget(actions[i].label,
            xmPushButtonWidgetClass, action_area,
            XmNleftAttachment,       XmATTACH_POSITION,
            XmNleftPosition,         i*2+1,
            XmNtopAttachment,        XmATTACH_FORM,
            XmNbottomAttachment,     XmATTACH_FORM,
/*
            XmNrightAttachment,         XmATTACH_POSITION,
            XmNrightPosition,        i*2 + 2,
*/
            XmNshowAsDefault,        i == 0,
            XmNdefaultButtonShadowThickness, 1,
            NULL);
        if (actions[i].callback)
            XtAddCallback(widget, XmNactivateCallback,
                actions[i].callback, actions[i].data);
        if (i == 0) {
            /* Set the action_area's default button to the first widget
             * created (or, make the index a parameter to the function
             * or have it be part of the data structure). Also, set the
             * pane window constraint for max and min heights so this
             * particular pane in the PanedWindow is not resizable.
             */
            Dimension height, h;
            XtVaGetValues(action_area, XmNmarginHeight, &h, NULL);
            XtVaGetValues(widget, XmNheight, &height, NULL);
            height += 2 * h;
            XtVaSetValues(action_area,
                XmNdefaultButton, widget,
                XmNpaneMaximum,   height,
                XmNpaneMinimum,   height,
                NULL);
        }
    }

    XtManageChild(action_area);
}
#if XY
#endif
