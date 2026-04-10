/*----------------------------------------------------------------
 *    Elle:   menus.h  1.0  11 September 1997
 *
 *    Copyright (c) 1997 by L.A. Evans & T.D. Barr
 *----------------------------------------------------------------*/
#ifndef _menus_h
#define _menus_h
typedef struct _menuitemdata{
    char        *label;         /* the label for the item */
    WidgetClass *class;         /* pushbutton, label, separator... */
    int		bg;		/* button background colour (-1 dflt) */
    int		spacing;	/* button separation (0 dflt) */
    void       (*callback)();   /* routine to call; NULL if none */
    XtPointer    callback_data; /* client_data for callback() */
    struct _menuitemdata *submenu; /* pullright menu items, if not NULL */
    int     submenu_data;
} MenuItemData;
#endif
 
