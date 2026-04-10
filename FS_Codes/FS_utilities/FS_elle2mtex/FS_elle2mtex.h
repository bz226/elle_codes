#ifndef FS_elle2mtex_h
#define FS_elle2mtex_h

/* Check the include list for missing or needless header files:*/
#include <string>
#include <iostream>
#include <fstream>
#include <list>
#include <vector>
#include <iomanip>
#include <stdio.h>
#include <math.h>
#include <time.h>
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
 * Function declaration
 */
int InitThisProcess();
int ProcessFunction();
bool FileExists(const char *filename);
void Wait(int iSeconds);

#endif
