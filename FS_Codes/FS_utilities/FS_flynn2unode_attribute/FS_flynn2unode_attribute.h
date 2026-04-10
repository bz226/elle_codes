#ifndef FS_flynn2unode_attribute_h
#define FS_flynn2unode_attribute_h

/* The include list of elle2fft.cc:
 * + file.h
 */
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
const int Viscosity =0; 
const int Euler_3   =1;           
const int Dislocden =2;           
const int F_Attrib_A=3;       

/*
 * Function declaration
 */
int InitF2U();
int Start_F2U();
    void CheckUnodesFlynn();
void TransferViscosity();
void TransferEuler3();
void TransferDislocden();
void TransferAttribA();

#endif
