#ifndef FS_scalefile_h
#define FS_scalefile_h

/* Check the include list for missing or needless header files:*/
#include <string>
#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <iomanip>
#include <stdio.h>
#include <math.h>
#include "attrib.h"
#include "nodes.h"
#include "update.h"
#include "error.h"
#include "parseopts.h"
#include "runopts.h"
#include "file.h"
#include "interface.h"
#include "init.h"
#include "log.h"
#include "setup.h"
#include "triattrib.h"
#include "unodes.h"
#include "polygon.h"
#include "mat.h"
#include "file.h"

#define PI 3.141592654

/*
 * indices for User data values for this process
 */
const int UScaleX=0; // enter a very small number here (smaller than -999) to scale box back to 1x1
const int UScaleY=1; // is UScaleX < -999 this value will be used as upscale factor of the 1x1 box

/*
 * Function declaration
 */
int InitThisProcess();
int ProcessFunction();
void ScaleModelBox(double dScaleX,double dScaleY);
void Scale2OneByOne(double dUpscale);

#endif
