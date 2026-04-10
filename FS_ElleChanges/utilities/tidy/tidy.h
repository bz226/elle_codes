#ifndef tidy_h
#define tidy_h

/*
 * indices for User data values for this process
 */
const int BnodeSpacing=0;   // change bnode spacing (0=interactive, > 0=spacing)
const int Ranorient=1;      // randomise EULER angles
const int MakeMica=2;       // set flynnsith this F_ATTRIB_A to MICA  
const int OrientMica=3; 	// set MICA axes wrt to grain long axis
const int AddUnodes=4;    	// add unodes to file (-1 = strip, >0 = number
							//  of rows, number of cells in SEMI_RAN)
const int UnodePattern=5;   // unode pattern if AddUnodes>0
                            // (0=hex, 1=square, 2=ran, 3=semi_ran)
const int UnodeCells=6;    	// no of unodes per cell for semi_ran
const int Rotation=7;    	// rotation of whole model in degrees  (clockwise is +ve)

int ReSpaceBnodes(double spacing);
int MicaQuartzFlynn(double flynncode);
int OrientMicaAxis();
int RanorientQuartz();
int Rotate();
double GetLongAxis(int flynn);
#endif
