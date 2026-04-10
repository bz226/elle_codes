/*	Created June 2009 by Jens Roessiger
 *
 * This is how my version of split works.
 * split2 contains 3 different split modes.
 *
 * **************************************************************************************************************************************
 *
 * the most important one is "directionsplit"
 *
 * int directionsplit2(int flynn, double x, double y, double mcs, int *child1, int *child2)
 *
 * In general some information as input
 * which flynn, which direction (it is supplied divided into x and y parts), how big the childs should be at least, two pointers for the childs.
 *
 * first of all split2 randomizes the startnode of the flynn
 * 2nd it calculates the min_area for the childes by multiplying the given value with the area of the flynn
 * 3rd it calculates the direction with the given x and y values
 *
 * 4th this direction is compared with every possible direction between double nodes and stored with their deviation to it
 * but none directions is stored twice since not 56 to 98 is the same as 98 to 56 only the first one will be calculated.
 * The struct contains: start node, end node, deviation
 *
 * --> if the flynn contains less than 2 dnodes split is aborted and returns 2 for too small flynn.
 *
 * 5th this struc is sorted by the deviation. first array in the struct is the one with the least deviation.
 * using quicksort
 *
 * 6th beginning with the first array the flynn nodes are devided into two child arrays with the node information from the deviation array
 *
 * 7th area checks for child 1 and 2, if both are equal or less than the min_child_size (mcs) calculated in step 2 split continues
 * if not it starts with step 6 again with the next node information from the deviation array.
 *
 * 8th after the area checks split2 checks for intersections with the flynn boundary along the split path.
 * to accomplish that it calculates the x position of the new boundary for every y position of the nodes of the old flynn boundary
 * and substracts that from the x value of the node. For both childs the sign of the result has to stay the same for every node of the new childs.
 *
 * 9th if all 4 checks haven't returned errors, the flynn is split, the two new childs are stored in the pointers
 * BUT the childs are NOT promoted and the old flynn ist NOT deleted. If this is preferred one has to use
 * EllePromoteFlynn(flynn)
 * for both childs and
 * ElleRemoveFlynn(flynn)
 * for the old flynn.
 *
 * returns 1 for a successful split
 *
 * 10th IF all possible directions from the deviation struct have been checked and there is still no successful split than the mcs is halfed
 * and split2 starts a 2nd attempt to slit the flynn. Basically the same as above but with halved mcs.
 *
 * if this is successful split2 returns 3 for a successful split with the 2nd try.
 *
 * 11th if 10th fails as well, 0 is returned for an error.
 *
 *******************************************************************************************************************************
 *
 * probably the 2nd most important one is "randomsplit"
 *
 * int randomsplit2(int flynn, double mcs, int *child1, int *child2)
 *
 * this is the same as directionsplit, but you won't need to supply a direction. The direction is random for this splittype.
 * you still need to supply mcs as above. The rest is the same.
 *
 *******************************************************************************************************************************
 *
 * the least important one is "directsplit"
 *
 * int directsplit2(int flynn, int start, int end, int *child1, int *child2)
 *
 * for this split mode it is IMPORTANT to know that there are NO topology checks. You have to KNOW what you do before using this split mode.
 *
 * instead of direction and mcs you only need to supply start and end node of the flynn and it splits between those two nodes regardless of what
 * would happen.
 *
 ********************************************************************************************************************************
 *
 * last notes
 *
 * you can also use split from command line with all the options discussed above using userdata.
 *
 * 5 option for userdata
 * -u 1 2 (3 4 5)
 *
 * 	1: flynn number – always required
 * 	2: mcs (min child size)
 * 		required if another than the global or MINERAL value should be used
 *		if set to 1 (auto value) the global or MINERAL value is used
 *		ranging from 0.5 (theoretically - expect many errors for that value or use many many nodes) recommended is 0.45 or something similar to 0.0 (which disables this check at all)
 *	3: split type – if not supplied 1 as auto value is taken which means random split
 *		1: random split
 *		2: direction split
 *		3: direct split (without checks, the split nodes have to be supplied)
 *	4: required for split mode 2 and 3
 *		in split mode 2: x-value of the direction of the split
 *		in split mode 3: start node of the split
 *	5: required for split mode 2 and 3
 *		in split mode 2: y-value of the direction of the split
 *		in split mode 3: end node of the split.
 *
 * IMPORTANT NOTE
 *
 * if you use the commandline option with userdata the childs are always promoted and the old flynn is always deleted.
 *
 * *******************************************************************************************************************************
 * that's it
 *
 *	ah well, since this file has changed so much during development there can be outdated comments in it.
 */
#include "split2.elle.h"

using std::vector;

// this is IMPORTANT. A flynn has to have a minimum of defined double nodes
// otherwise it won't split. Dependent on switch distance it influences the
// min flynn size.
#define MINDNODES 2
// this defines whether the 2nd try approach (step 10 for direction and randomsplit) is used or not
// set to 0 if you don't want to use it.
#define SECONDTRY 1

UserData userdata;

FILE *split;

DEVIATION *dev;

int Init_Split2(void)
{
    int err=0;
    int max, maxf, n;
    char *infile;

    ElleReinit();
    ElleSetRunFunction(intsplit2);

    infile = ElleFile();
    if (strlen(infile)>0) {
        if (err=ElleReadData(infile)) OnError(infile,err);

        ElleAddDoubles();
    }
}

int intsplit2(void)
{
	int splittype, flynn, start, end, c1, c2, check=0;
	double dx, dy, mcs; //mcs=min_child_size

    ElleUserData(userdata);
    splittype = (int)userdata[2];
    flynn = (int)userdata[0];
    //get the min_child_size as fraction of the parent grain.
    //If userdata returns 1 than the global or MINERAL specific data is retrieved.
    mcs = (double)userdata[1];
    if (mcs==1)
    	mcs=ElleFindFlynnMinArea(flynn);

    //if splittype is out of range, define random split as default
    if (splittype<1 || splittype>3)
    	splittype=1;

    //for random split define random dx and dy and split
    if (splittype==1) {
		check = randomsplit2(flynn, mcs, &c1, &c2);
	}
    //for direction split get the dx and dy directions from userdata and split
    if (splittype==2) {
    	dx = (double)userdata[3];
    	dy = (double)userdata[4];
    	check = directionsplit2(flynn, dx, dy, mcs, &c1, &c2);
    }
    //for direct split get the start and end nodes from userdata and split
    if (splittype==3) {
		start=(int)userdata[3];
		end=(int)userdata[4];
		directsplit2(flynn, start, end, &c1, &c2);
		check=1;
	}

    //printf("Childs: %d & %d\n", c1, c2);
    if (check==1 || check==3) {
		EllePromoteFlynn(c1);
		EllePromoteFlynn(c2);
		ElleRemoveFlynn(flynn);
		ElleAddDoubles();
    }
    ElleUpdate();

}

int directsplit2(int flynn, int start, int end, int *c1, int *c2)
{
	int *nodes=0, num_nodes;
	int *child1=0, *child2=0, nchild1=0, nchild2=0;

	ElleFlynnNodes(flynn, &nodes, &num_nodes);
	nodes2childs(&nodes, num_nodes, start, end, &child1, &child2, &nchild1, &nchild2);
	flynnsplit2(flynn, start, end, &child1, &child2, &nchild1, &nchild2, &c1, &c2);

	//EllePromoteFlynn(c1);
	//EllePromoteFlynn(c2);
	//ElleRemoveFlynn(flynn);

	free(nodes);
	free(child1);
	free(child2);
}

int randomsplit2(int flynn, double mcs, int *cc1, int *cc2)
{
	int i, c1, c2;
	double x, y;

	x=ElleRandomD();
	y=ElleRandomD();
	for (;x==0.0 && y==0.0;) {
		x = ElleRandomD();
		y = ElleRandomD();
	}
	x*=2;
	x-=1;
	y*=2;
	y-=1;
	i = directionsplit2(flynn, x, y, mcs, &c1, &c2);
	*cc1=c1;
	*cc2=c2;
	return i;
}

//#define MINAREA 0.0002 // flynn has to be larger than that to actually be able to split.

int directionsplit2(int flynn, double x, double y, double mcs, int *c1, int *c2)
{
	int i, j=0, check=0, *id=0, maxnint, num_nodes, start=0, end=0, starti=0, endi=0, *child1=0, *child2=0, nchild1=0, nchild2=0, possis, nd=0;
	double dir, min_area, test_area, maxn;
	vector<int> seq;

	// find all the Nodes of a specified flynn
	ElleFlynnNodes(flynn, &id, &num_nodes);

	nd = SECONDTRY;

	// don't know if this is really good, but it helps to randomize the split origins....
	maxn = ElleRandomD();
	maxn *= num_nodes;
	maxnint = (int)maxn;
	ElleSetFlynnFirstNode(flynn, *(id+maxnint));
	free(id);
	ElleFlynnNodes(flynn, &id, &num_nodes);


	// set minimum area
	min_area = areacheck(&id, num_nodes);
	min_area *= mcs;

	//calculate direction relative to x-axis
	dir=atan(y/x);

			// assignt the struct of arrays explained in step 4
			if ((check=assignstruct(&id, dir, num_nodes, &possis))==1) {
				// step 5 use quicksort to sort the struct
				sortstruct(dev, 0, possis-1);
				//if ((test_area=areacheck(&id, num_nodes))>MINAREA) {
					// start from the first to the last entry in the deviation struct
					for (j=0,i=0;j<possis && i==0;j++) {
						start=dev[j].x;
						end=dev[j].y;
						if ((check=nodes2childs(&id, num_nodes, start, end, &child1, &child2, &nchild1, &nchild2))==1)
							//check min area of child 1
							if ((test_area=areacheck(&child1, nchild1))>=min_area)
								//if ok, check min area of child 2
								if ((test_area=areacheck(&child2, nchild2))>=min_area)
									//if ok, check intersections of child 1 with split direction
									if (intersectioncheck(&child1, nchild1))
										//if ok, check intersections of child 2
										if (intersectioncheck(&child2, nchild2)) {
											flynnsplit2(flynn, start, end, &child1, &child2, &nchild1, &nchild2, &c1, &c2);
											//printf("Successfully split flynn %d\n", flynn);
											i=1;
										}
					}
					if (j==possis && i==0 && nd == 1) {
						min_area /= 2;
						//nd=1; // marker for the 2nd try
						//split=fopen("split.txt", "a");
						//fprintf(split,"nd-try startet\n");
						//fclose(split);
						for (j=0,i=0;j<possis && i==0;j++) {
							start=dev[j].x;
							end=dev[j].y;
							if ((check=nodes2childs(&id, num_nodes, start, end, &child1, &child2, &nchild1, &nchild2))==1)
								//check min area of child 1
								if ((test_area=areacheck(&child1, nchild1))>=min_area)
									//if ok, check min area of child 2
									if ((test_area=areacheck(&child2, nchild2))>=min_area)
										//if ok, check intersections of child 1 with split direction
										if (intersectioncheck(&child1, nchild1))
											//if ok, check intersections of child 2
											if (intersectioncheck(&child2, nchild2)) {
												flynnsplit2(flynn, start, end, &child1, &child2, &nchild1, &nchild2, &c1, &c2);
												//printf("Successfully split flynn %d\n", flynn);
												i=3;
											}
						}
					}
	//				for(i=0; i<possis; i++)
	//					printf("DEV%d: %d - %d : %f\n", i, dev[i].x, dev[i].y, dev[i].error);
	//				printf("Possies: %d\n", possis);
					free(dev);
					free(child1);
					free(child2);
				//} else {
					//printf("ERROR: flynn too small to split\n");
				//}
			} else {
				if (check==2) // this is to not count split attempts of too small grains as errors
					i=2;
				else
					printf("ERROR: split2 completely failed: assignstruct error\n");
			}

	free(id);

	if (i==1)		// successful split
		return 1;
	else if (i==2)  // too small grain
		return 2;
	else if (i==3)	// successful split after 2nd try with half min_area
		return 3;
	else
		return 0;	// error
}

int nodes2childs(int **id, int num_nodes, int start, int end, int **child1, int **child2, int *nchild1, int *nchild2)
{
	int i, j, starti, endi, temp, *iptr;
	/* if a matching direction is found
	 * the nodes are are written into two possible child arrays which can be
	 * further investigated.
	 * start node of the possible split is always element 0
	 * and end node always the last element
	 * all the other nodes are arranged between those two
	 *
	 * returns 1 if successful and 0 if not successful.
	 */

	// find the position of the start and end nodes within the array
	for (i=0; i<num_nodes;i++) {
		if (start==*(*id+i))
			starti=i;
	}
	for (i=0; i<num_nodes;i++) {
		if (end==*(*id+i))
			endi=i;
	}
	//printf("%d, %d\n", start, end);

	// Exchanges starti and endi if starti is bigger than endi
	// this is needed because the first child doesn't check for start/end overstep
	if (starti>endi) {
		temp=starti;
		starti=endi;
		endi=temp;
	}

	// child1
	// count elements for child 1 and sets *nchild1 accordingly
	for (i=starti,j=0;i<=endi;i++,j++) {
		;
	}
	*nchild1=j;

	// writes all the elements of child1 into the child1 array
	if ((*child1 = (int *)malloc(*nchild1 * sizeof(int)))==0) {
		printf("ERROR: nodes2childs: Malloc_Err: child1\n");
		return 0;
	}
	for (i=starti,j=0,iptr=*child1;i<=endi;i++,j++)
		iptr[j] = *(*id+i);


	// child2
	// do the same for child2
	// except it starts at endi to the end and continues at 0 to stari
	for (i=endi,j=0;i<num_nodes;i++,j++) {
		;
	}
	for (i=0; i<=starti;i++,j++) {
		;
	}
	*nchild2=j;

	if ((*child2 = (int *)malloc(*nchild2 * sizeof(int)))==0) {
		printf("ERROR: nodes2childs: Malloc_Err: child2\n");
		return 0;
	}
	for (i=endi,j=0,iptr=*child2;i<num_nodes;i++,j++)
		iptr[j] = *(*id+i);
	for (i=0;i<=starti;i++,j++)
		iptr[j] = *(*id+i);

	return 1;
}

double areacheck(int **nodes, int num_nodes)
{
	/* This one is copied from elsewhere except
	 * I have commented out the ElleFlynnNodes function and pass the these
	 * values as function arguments instead.
	 * It returns the area between the nodes passed.
	 */
    int j; //*id=0;
    double area, *coordsx=0, *coordsy=0, *ptrx, *ptry;
    Coords xy,prev;

    //ElleFlynnNodes(poly,&id,&num_nodes);
    if ((coordsx = (double *)malloc(num_nodes*sizeof(double)))== 0)
    	printf("ERROR: areacheck: Malloc_Err: coordsx\n"); //OnError("ElleRegionArea",MALLOC_ERR);
    if ((coordsy = (double *)malloc(num_nodes*sizeof(double)))== 0)
    	printf("ERROR: areacheck: Malloc_Err: coordsy\n");  //OnError("ElleRegionArea",MALLOC_ERR);
    ElleNodePosition(*(*nodes),&prev);
    for (j=0,ptrx=coordsx,ptry=coordsy;j<num_nodes;j++) {
        ElleNodePlotXY(*(*nodes+j),&xy,&prev);
        *ptrx = xy.x; ptrx++;
        *ptry = xy.y; ptry++;
        prev = xy;
    }
    area = polyArea(coordsx,coordsy,num_nodes);
    free(coordsx);
    free(coordsy);
    //if (id) free(id);
    return(area);
}

int intersectioncheck(int **child, int nchild)
{
	int i, j, check, test, horizontal=0, wrap;
	double dir, dir_test, l, l_test;
	Coords start, end, temp;

	// First get the node position of the start node which is the first one in the array
	// Then get the position of the last node to determine the split direction against which
	// all the other directions are tested.
	//wrap=wrapcheck(&child, nchild);

	ElleNodePosition(**child, &start);
	ElleNodePlotXY(*(*child+(nchild-1)), &end, &start);
	end.x = end.x - start.x;
	end.y = end.y - start.y;
	if (end.y==0)
		horizontal=1;


	// for the dir test the equation x2 = (x1/y1) * y2 is used

	// if the split direction is horizontal y2 = (y1/x1) * x2 is used instead.

	// this is the part between the brackets

	if (horizontal==0)
		dir=(end.x/end.y);
	else if (horizontal==1)
		dir=(end.y/end.x);

	//determine the length of the split for the second part of the test (without root)
	l=(end.y*end.y)+(end.x*end.x);

	// get the node position of the 2nd node to determine whether this child
	// is above or below the split.
	ElleNodePlotXY(*(*child+1), &temp, &start);
	//ElleNodePosition(*(*child+1), &temp);
	temp.x = temp.x - start.x;
	temp.y = temp.y - start.y;
	// if there is no difference between the split direction and the test direction
	// print out an error and quit.
	if (temp.x==0 && temp.y==0) {
		printf("Error: intersection check: first node check is the same as split direction\n");
		return 0;
	}
	// otherwise determine the test direction

	if (horizontal==0)
		dir_test=dir*temp.y;
	else if (horizontal==1)
		dir_test=dir*temp.x;
	/*  for x2 = (x1/y1) * y2 the part between the brackets has already been calculated
	 *  above. Now this party is multiplied with the y-part of the test location
	 *  to see whether this point is above or below the split boundary.
	 *
	 *  The result will be the virtual x location of the split boundary for the y value
	 *  of the test location. Afterwards the two locations are compared and depending
	 *  on if the result is larger or smaller then the test location a value is stored
	 *  that is needed for comparison of the other points afterwards.
	 *
	 *  If the virual location for every point is the same (in terms of smaller or larger) than the
	 *  test location there is no intersection.
	 */
	if (temp.x>dir_test)
		check=0;
	else if (temp.x<dir_test)
		check=1;
	else
		printf("ERROR: intersection check: check determination\n");

	// this is needed for the second part of the test.
	test=check;

	// now the loop for all the other nodes starting from the 3rd to the penultimate node
	for (i=2; i<(nchild-1) && test==check; i++) {
		ElleNodePlotXY(*(*child+i), &temp, &start);
		//ElleNodePosition(*(*child+i), &temp);
		temp.x = temp.x - start.x;
		temp.y = temp.y - start.y;

		if (horizontal==0)
			dir_test=dir*temp.y;
		else if (horizontal==1)
			dir_test=dir*temp.x;

		if (temp.x>dir_test)
			test=0;
		else if (temp.x<dir_test)
			test=1;
		else if (temp.x==dir_test)
			test=2;
		else
			printf("ERROR: intersection check: check determination 2\n");

		// Second part of the test.
		// If the check and the test differs, the length of the split against the check is tested
		// If the test length is longer than the split length then the split is still ok -- no intersection.
		if (test!=check) {
			if ((l_test=(temp.y*temp.y)+(temp.x*temp.x))<l) {
				//printf("Error: intersection check: length: INTERSECTION\n");
				break;
			}
			else if (l_test==l) {
				//printf("Error: intersection check: length: possible Intersection\n");
				break;
			}
			else if (l_test>l)
				// test is set equal to check again, because split is possible if
				// l_test is longer than l
				test=check;
			else
				printf("Error: intersection check: length: undefined error\n");
		}

	}
	// if test was equal to check for the whole loop, return that the split is possible without intersection.
	if (test==check)
		return 1;
	else
		return 0;
}

int flynnsplit2(int flynnindex, int start, int end, int **child1, int **child2, int *nchild1, int *nchild2, int **c1, int **c2)
{
	int i, j;
	ERegion rgn1, rgn2;

    // create 2 new children
    // search for 2 spare flynns and set them as childs of the parent grain
	// assign an ERegion to them.
    **c1 = ElleFindSpareFlynn(); // first -> end
    **c2 = ElleFindSpareFlynn(); // end -> first
    ElleAddFlynnChild(flynnindex, **c1);
    ElleAddFlynnChild(flynnindex, **c2);
    rgn1 = **c1;
    rgn2 = **c2;

    // First Child
    // for all the nodes in the array of child 1
    for (i=0;i<*nchild1;i++) {
    	// for the last node, connect it with the first node
    	if (i==*nchild1-1) {
    		// find the NO_NB entry in the neighbours of the last node
    		j = ElleFindNbIndex(*(*child1),*(*child1+i));
    		// assign this NO_NB entry in the neighbours of the last node to the first node --> new triple node
			ElleSetNeighbour(*(*child1+i), j, *(*child1), &rgn1);
			//printf("End of Child1: %d - %d: %d\n", *(*child1+i), *(*child1), rgn1);
    	} else {
    		// find out which of the 3 neighbours of a node the next node is
			j = ElleFindNbIndex(*(*child1+(i+1)),*(*child1+i));
			// set the next node as this neighbour for the node
			ElleSetNeighbour(*(*child1+i), j, *(*child1+(i+1)), &rgn1);
			//ElleSetRegionEntry(*child1+i,j,rgn1);
			//printf("child1: %d - %d: %d\n", *(*child1+i), *(*child1+(i+1)), rgn1);
    	}
    }
    // set the first node as first node of child 1
    ElleSetFlynnFirstNode(**c1, start);

    // Second Child
    // do the same for child 2 except that start and end nodes are exchanged.
    for (i=0;i<*nchild2;i++) {
    	if (i==*nchild2-1) {
    		j = ElleFindNbIndex(*(*child2),*(*child2+i));
			ElleSetNeighbour(*(*child2+i), j, *(*child2), &rgn2);
			//printf("End of Child2\n");
    	} else {
			j = ElleFindNbIndex(*(*child2+(i+1)),*(*child2+i));
			ElleSetNeighbour(*(*child2+i), j, *(*child2+(i+1)), &rgn2);
			//ElleSetRegionEntry(*child2+i,j,rgn2);
			//printf("child2\n");
    	}
	}
    // set the end node as first node of child 1
    ElleSetFlynnFirstNode(**c2, end);
    // add new double nodes to the newly created boundary.
    ElleAddDoubles();
}

// not from me, a common quicksort alogarithm
void sortstruct(DEVIATION items[], int left, int right)
{

  register int i, j;
  double x;
  DEVIATION temp;

  i = left; j = right;
  x = items[(left+right)/2].error;

  do {
    while(items[i].error < x && (i < right)) i++;
    while(items[j].error > x && (j > left)) j--;
    if(i <= j) {
      temp = items[i];
      items[i] = items[j];
      items[j] = temp;
      i++; j--;
    }
  } while(i <= j);

  if(left < j) sortstruct(items, left, j);
  if(i < right) sortstruct(items, i, right);
}

int assignstruct(int **id, double dir, int num_nodes, int *possis)
{
	int i, j, k=0;
	double dir_test;
	Coords n1, n2;

	// count the maximum connections that need to be tested num_nodes!
	// without the triple nodes because they aren't used
	for (j=0,i=0; j<num_nodes; j++)
		if (ElleNodeIsDouble(*(*id+j)))
			i++;

	// only do that if there are at least MINDNODES double nodes.
	if (i>=MINDNODES) {
		// substract 1 from the amount of double nodes because that one is used as starting
		// node and therefore isn't used in this calculation. Then calculate factorial of the nodes
		for (--i,j=0;i>0;i--)
			j+=i;

		//printf("Possis: %d\n", j);


		if ((dev = (DEVIATION *)malloc(j * sizeof *dev))==0) {
			printf("ERROR: assignstruct: Malloc_Err: deviation struct\n");
			return 0;
		}

		//find a pair of nodes fitting the direction
		/* compares the direction from every node to every other node
		 * only accepts double nodes as possible split nodes
		 */
		for(i=0, k=0; i<num_nodes; i++) {
			if (ElleNodeIsDouble(*(*id+i))) {
				ElleNodePosition(*(*id+i), &n1);
				for(j=i; j<num_nodes ;j++) {
					if (i!=j && ElleNodeIsDouble(*(*id+j))) {
						//ElleNodePosition(*(*id+j), &n2);
						ElleNodePlotXY(*(*id+j), &n2, &n1);
						dir_test=atan((n2.y-n1.y)/(n2.x-n1.x));
						dev[k].x = *(*id+i);
						dev[k].y = *(*id+j);
						dev[k++].error = fabs(dir-dir_test);
					}
				}
			}
		}

		*possis=k;
		return 1;
	} else {
		//printf("ERROR: assignsstruct: too less d-nodes: no possibility to split\n");
		return 2;
	}
}
