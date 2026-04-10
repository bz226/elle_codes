// Written in 2003
// Author: Dr. J.K. Becker
// Copyright: Dr. J.K. Becker (becker@jkbecker.de)

/*! \mainpage The ELLE numerical simulation package
 *
 * <p>You have succesfully downloaded the Elle package and are now ready to install it. Well done so far!
 * Please read the information below to see how to proceed to install Elle on your computer.</p>
 * 
 * \section intro_sec Introduction
 *
 *<div style="text-align: justify;">The aim of this project is to investigate a class of problems in structural geology and metamorphicpetrology where the kinetics of the deformation processes and metamorphic reactions involve similar time scales. It has proved extremely difficult to make much progress in experiments of deforming and reacting rocks or analogues, because the kinetics of the deformation processes cannot generally be simultaneously scaled with the kinetics of the reaction processes.
 *The investigation of deformation microstructures often assumes the dominance of a single deformation process, and this is no doubt in part because the theoretical framework for multi-process systems is still in its infancy (eg Karato 1987, Jessell & Lister 1990, Zhang et al. 1995). The interpretation of metamorphic assemblages generally takes into account the impact of deformation to distinguish different generations of minerals, on the other hand the role of deformation on the textures and mineralogy of a single assemblage is often hard to determine.
 *We are examining the link between the metamorphic and deformation histories of rocks and their microstructures by building a two-dimensional numerical simulation of the interaction between the grain scale processes involved in deformation and metamorphism.</div>
 * 
 * \section install_sec Installation
 *  <ul>
 * <li><a href="#linux">Linux</a></li>
 * <li><a href="#windows">Windows</a></li>
 * <li><a href="#macos">MacOS</a></li>
 * 
 * <a name="linux"></a>
 * \subsection linux Installing Elle under linux
 * 
 * It has been done. Other people have done it already. Chances are very good you can do it too!<br>
 * \subsection prereq Prerequisites to install Elle under Linux
 * 
 * You will need the following packages:
 * <ul>
 * <li> X11 libraries, headers (should come with Unix or with XFree86 distributions)</li>
 * <li> Motif (or Lesstif on Linux/Mac OSX/Cygwin) libraries and headers.  Headers are usually in the developer packages or rpms.</li>
 * <li> gcc -GNU project C++ compiler</li>
 * <li> Fortran 77 compiler (usually included with g++)</li>
 * <li> Optional but recommended: xmkmf, imake & X11 configuration files which are contained in the "xdevel" distribution</li>
 * 
 * \subsection instprc Installation procedure
 * <div style="text-align: justify;">
 * <ul><li>Extract the elle tar file (e.g. "tar xvz elle.2.3.3.tar.gz")</li>
 * <li>Change to the resulting elle directory</li>
 * <li>Run the command "./install.sh >& out". 
 * <li>Include the elle/bin directory in your PATH
 *    <ul><li>for csh,tcsh shells use: SETENV PATH $PATH:/usr/local/elle/bin </li>
 *    <li>for bash,ksh shells use: export PATH=$PATH:/usr/local/elle/bin </li></ul>
 *
 * (if necessary, replace /usr/local with the path to your elle directory)</li>
 * </ul>
 * If anything goes wrong during the installation, please reread the above instructions again carefully and make sure you
 * followed all of them. If it still does not work, please visit the main Elle website and read the installation instructions there and look in the forums. It is very likely
 * that other people already had the same problem and we solved it. If none of the given information solves your problem, ask for help in one of 
 * the forums. 
 * </div>
 *
 * <a name="windows"></a>
 * \subsection windows Installing Elle under windows
 *
 * It has been done. Lot's have tried. Only the best have succeeded. Chances are medium you can do it too! <br>
 * To be written...
 * 
 * <a name="macos"></a>
 * \subsection macos Installing Elle under MacOS
 * 
 * It has been done. Some have tried. Only a few have succeeded. Chances are slim you can do it too! <br>
 * To be written...
 * 
 * \section additional_info Additional information
 * 
 * Additional information can be obtained from the following locations:<br>
 * <ul>
 * <li>
 * <a href="http://www.microstructure.uni-tuebingen.de/phoenix/index.php" target="_blank">Main Elle website</a><br><br>
 * <div style="text-align: justify;">This is the main Elle website. This has most of the information you will ever need, 
 * including forums, tutorials etc.. For any problems arising, this should be your first choice 
 * for help and questions.</div><br>
 * </li>
 * <li>
 * <a href="http://www.jkbecker.de" target="_blank">www.jkbecker.de</a><br><br>
 * This website has information about the GUI, how to use it and what still has to be fixed.
 * </li>
 * </ul>
 * </td>
 */


#ifdef WX_PRECOMP
  #include "wx/wxprec.h"
#endif
#ifndef WX_PRECOMP
  #include "wx/wx.h"
#endif
#ifndef wx_start_h
#define wx_start_h
#include "wx/wx.h"
#include "wx/log.h"
#include "wx/defs.h"
#include "showelle.h"
//#include "wx/glcanvas.h"
#include "setup.h"
#include "runopts.h"
#include "settings.h"
#include "dsettings.h"
#include "display.h"
#include "file_utils.h"
#include <stdio.h>
#include "wx/splash.h"


class Showelle : public wxApp
{
public:
    ShowelleFrame * mframe;
    virtual bool OnInit();
    wxLogWindow *logframe;
};



DECLARE_APP( Showelle )

void warning_msg(int, const char*);
void SavePic();
void DrawUnodes(int attrib);
void DrawSsegments(int attrib);
#endif //wx_start_h

