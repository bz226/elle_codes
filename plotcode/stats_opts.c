/*----------------------------------------------------------------
 *    Elle:   options.c  1.0  11 September 1997
 *
 *    Copyright (c) 1997 by L.A. Evans & T.D. Barr
 *----------------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <Xm/DialogS.h>
#include <Xm/PushB.h>
#include <Xm/PanedW.h>
#include <Xm/Form.h>
#include <Xm/RowColumn.h>
#include <Xm/LabelG.h>
#include <Xm/ToggleBG.h>
#include <Xm/SeparatoG.h>
#include <Xm/TextF.h>
#include "types.h"
#include "globals.h"
#include "error.h"
#include "string_utils.h"
#include "stats.h"
#include "options.h"
#include "opttypes.h"

#define COLS  8
#define XY  0

#define SO_NODES      150
#define SO_GRAINS     151
#define SO_AV_AREA    152
#define SO_BNDLEN     153
#define SO_M2_SIDES   154
#define SO_M2_AREAS   155

Widget StatsOpts_wdgts[16];
void StatsOptsDialog();
static void CreateActionArea();
static void OptsDlgDone();
static void UserCancel();
void GetStatsOptionValue(int id,int *val);
void SetStatsOptionValue(int id,int val);

OptionItem  StatsOption_items_int[] = {
    { O_UPDFREQ, SYB_INT },
    };

ToggleItem StatsOption_items_toggle[] = {
    { SO_NODES },
    { SO_GRAINS },
    { SO_AV_AREA },
    { SO_BNDLEN },
    { SO_M2_SIDES },
    { SO_M2_AREAS },
    };

static valid_terms statsoption_terms[] = {
                                "Update Interval",O_UPDFREQ,
                                "Node count", SO_NODES,
                                "Grain count", SO_GRAINS,
                                "Boundary length", SO_BNDLEN,
                                "Average area", SO_AV_AREA,
                                "m2 Sides", SO_M2_SIDES,
                                "m2 Areas", SO_M2_AREAS,
                                 NULL };

void StatsOptsDialog( w, client_data, call_data )
Widget w;
XtPointer client_data;
XtPointer call_data;
{
    XmString bar,labels[3];
    static Widget opts_dialog,bar_menu;
    Widget *parent,rowcol1,rowcol2,pane,toggle_box;
    char buf[30], wdgt_label[MAX_OPTION_NAME+1];
    int i=0, j, val, state=0;
    Dimension wdth;
    Position x,y;
    Arg args[2];
    static OptionDataItem user_data;
    static OptsActionAreaItem button_items_opts[] = {
        { "Ok",     OptsDlgDone, NULL          },
        { "Cancel", UserCancel, NULL          },
    };

    if (!opts_dialog) {
        user_data.num_ints = user_data.num_toggles =
             user_data.num_floats = 0;
        parent = (Widget *)client_data;
        opts_dialog = XtVaCreatePopupShell("Set_Stats_Options",
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
        for (j=0;j<XtNumber(StatsOption_items_int);j++) {
            id_match(statsoption_terms,StatsOption_items_int[j].id,
                                                        wdgt_label);
            XtVaCreateManagedWidget(wdgt_label,
                xmLabelGadgetClass, rowcol1, NULL );

            GetStatsOptionValue(StatsOption_items_int[j].id,&val);
            sprintf(buf,"%4d",val);
            StatsOpts_wdgts[i] = XtVaCreateManagedWidget("txt_wdgt",
                xmTextFieldWidgetClass, rowcol1, 
                XmNuserData, (XtPointer)StatsOption_items_int[j].id,
                XmNvalue, buf,
                XmNcolumns, 4, NULL );
            i++;
        }

/*
 * store the number of integer plot parameters
 */
        user_data.num_ints = i;

        toggle_box=XtVaCreateManagedWidget("toggle",
            xmRowColumnWidgetClass, pane, NULL );

        /*
         * create ToggleButton gadgets with Labels
         */
        for (j=0;j<XtNumber(StatsOption_items_toggle);j++) {
            id_match(statsoption_terms,StatsOption_items_toggle[j].id,
                                                           wdgt_label);
            StatsOpts_wdgts[i] = XtVaCreateManagedWidget(wdgt_label,
                xmToggleButtonGadgetClass, toggle_box,
                XmNuserData, (XtPointer)StatsOption_items_toggle[j].id,
                NULL );
/*
            XtAddCallback( Opts_wdgts[i],XmNvalueChangedCallback,
                            toggled, StatsOption_items_toggle[j].id );
*/
            GetStatsOptionValue(StatsOption_items_toggle[j].id,&val);
            XmToggleButtonGadgetSetState(StatsOpts_wdgts[i], (val==1), 0);
            i++;
        }
/*
 * store the number of boolean plot parameters
 */
        user_data.num_toggles = i-user_data.num_ints;

        XtManageChild (rowcol1);
        XtManageChild (toggle_box);

        XtVaGetValues( toggle_box,XmNwidth,&wdth,NULL );

    /* Create the action area */
        CreateActionArea(pane,button_items_opts,
                                  XtNumber(button_items_opts),wdth);

    /* position dialog box near menu item */
        XtTranslateCoords( w,(Position)0,
                   (Position)0,&x,&y);
        XtVaSetValues( opts_dialog,XmNx,x,
                   XmNy,y,
                   NULL );

        XtManageChild (pane);
        /*user_data.data = &Tmp;*/
        user_data.p_u_shell = opts_dialog;
    }
    else {
#if XY
#endif
        i=0;
        for (j=0;j<user_data.num_ints;j++,i++) {
            GetStatsOptionValue(StatsOption_items_int[j].id,&val);
            sprintf(buf,"%4d",val);
            XtVaSetValues(StatsOpts_wdgts[i], XmNvalue, buf, NULL);
        }
        i=user_data.num_ints;
        for (j=0;j<user_data.num_toggles;j++,i++) {
            GetStatsOptionValue(StatsOption_items_toggle[j].id,&val);
            XmToggleButtonGadgetSetState(StatsOpts_wdgts[i], (val==1), 0);
        }

    /* position dialog box near menu item */
        XtTranslateCoords( w,(Position)0,
                   (Position)0,&x,&y);
        XtVaSetValues( opts_dialog,XmNx,x,
                   XmNy,y,
                   NULL );
    }

    XtPopup(opts_dialog, XtGrabNone);
}

/*
 * dialog ok button callback function for plotting options
 */
static void OptsDlgDone(w, client_data, call_data)
Widget w;
XtPointer client_data, call_data;
{
    unsigned char t_val;
    char *str;
    int items, i, j, val;
    OptionDataItem *user_data;
    XtPointer ptr;

    user_data = (OptionDataItem *)client_data;
    
    /*
     * get values of integer parameters
     * save any changed values for log file
     */
    items = user_data->num_ints;
    for (i=0;i<items;i++) {
        XtVaGetValues(StatsOpts_wdgts[i],
               XmNuserData, &ptr,
               NULL);
        str = XmTextFieldGetString(StatsOpts_wdgts[i]);
        val = atoi(str);
        XtFree(str);
        SetStatsOptionValue((int)ptr,val);
    }
    items = user_data->num_toggles;
    for (j=0;j<items;j++,i++) {
        XtVaGetValues(StatsOpts_wdgts[i],
               XmNuserData, &ptr,
               XmNset, &t_val,
               NULL);
        val = (t_val ? 1 : 0);
        SetStatsOptionValue((int)ptr,val);
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

    user_data = (OptionDataItem *)client_data;
    XtPopdown(user_data->p_u_shell);
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

/*
 * get a field value
 */
void GetStatsOptionValue(int id,int *val)
{
    switch(id) {
    case SO_NODES  : *val = ES_SaveNodes();
                     break;
    case SO_GRAINS : *val = ES_SaveGrains();
                     break;
    case SO_AV_AREA: *val = ES_SaveavArea();
                     break;
    case SO_BNDLEN : *val = ES_SaveBLength();
                     break;
    case SO_M2_SIDES:*val = ES_Savem2Sides();
                     break;
    case SO_M2_AREAS:*val = ES_Savem2Areas();
                     break;
    case O_UPDFREQ  :*val = ES_statsInterval();
                     break;
    default:         break;
    }
}

/*
 * set a stats value
 */
void SetStatsOptionValue(int id,int val)
{
    switch(id) {
    case SO_NODES  : ES_SetSaveNodes(val);
                     break;
    case SO_GRAINS : ES_SetSaveGrains(val);
                     break;
    case SO_AV_AREA: ES_SetSaveavArea(val);
                     break;
    case SO_BNDLEN : ES_SetSaveBLength(val);
                     break;
    case SO_M2_SIDES:ES_SetSavem2Sides(val);
                     break;
    case SO_M2_AREAS:ES_SetSavem2Areas(val);
                     break;
    case O_UPDFREQ  :ES_SetstatsInterval(val);
                     break;
    default:         break;
    }
}

