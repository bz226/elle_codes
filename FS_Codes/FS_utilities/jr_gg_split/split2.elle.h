#ifndef SPLIT2_ELLE_H_
#define SPLIT2_ELLE_H_
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "flynnarray.h"
#include "attrib.h"
#include "nodes.h"
#include "file.h"
#include "display.h"
#include "check.h"
#include "error.h"
#include "runopts.h"
#include "init.h"
#include "general.h"
#include "stats.h"
#include "update.h"
#include "interface.h"
#include "polygon.h"

typedef struct
{
	int x, y;
	double error;
} DEVIATION;

int Init_Split2(void);
int intsplit2(void);
int directsplit2(int flynn, int start, int end, int *c1, int *c2);
int randomsplit2(int flynn, double mcs, int *c1, int *c2);
int directionsplit2(int flynn, double x, double y, double mcs, int *c1, int *c2);
int nodes2childs(int **id, int num_nodes, int start, int end, int **child1, int **child2, int *nchild1, int *nchild2);
double areacheck(int **nodes, int num_nodes);
int intersectioncheck(int **child, int nchild);
int flynnsplit2(int flynnindex, int start, int end, int **child1, int **child2, int *nchild1, int *nchild2, int **c1, int **c2);
int assignstruct(int **id, double dir, int num_nodes, int *possis);
void sortstruct(DEVIATION items[], int left, int right);

#endif /* SPLIT2_ELLE_H_ */


