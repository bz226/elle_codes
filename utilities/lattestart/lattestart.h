
#ifndef _MIKESTART_H_
#define _MIKESTART_H_

//-----------------------------------------------------------------------------
// GCC interface
//-----------------------------------------------------------------------------

#if defined(__GNUG__) && !defined(__APPLE__)
    #pragma interface "lattestart.h"
#endif

//-----------------------------------------------------------------------------
// Headers
//-----------------------------------------------------------------------------

#include "wx/app.h"             // wxApp
#include "wx/tooltip.h"

class MyApp : public wxApp
{

public:

    virtual bool OnInit();

};

//-----------------------------------------------------------------------------
// End single inclusion of this .h file condition
//-----------------------------------------------------------------------------

#endif 
