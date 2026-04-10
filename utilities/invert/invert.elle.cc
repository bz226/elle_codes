#include <stdio.h>
#include <math.h>
#include <string.h>
#include "error.h"
#include "runopts.h"
#include "nodes.h"
#include "version.h"
#include "file.h"
#include "interface.h"
#include "init.h"
#include "polygon.h"
#include "regions.h"
#include "bflynns.h"

int DoSomethingToFlynn(int flynn);
int InitThisProcess(), ProcessFunction();
int ElleWriteReverseFlynnData(FILE *fp);

/*
 * this function will be run when the application starts,
 * when an elle file is opened or
 * if the user chooses the "Rerun" option
 */
int InitThisProcess()
{
    char *infile;
    int err=0;
    
    srand48(50);

    ElleSetRunFunction(ProcessFunction);
    /*
     * read the data
     */
    infile = ElleFile();
    if (strlen(infile)>0) {
        if (err=ElleReadData(infile)) OnError(infile,err);
    }

    /*
     * this process just runs once - no display
    ProcessFunction();
     */
}

#define HEADER_MAX 100
int ProcessFunction()
{
    int i, j, k, l,debug_flag=1;
    int interval=0,err=0,max,mintype;

    char header[HEADER_MAX], filename[FILENAME_MAX];
    FILE *fp;

    sprintf(filename,"%s.%s", ElleSaveFileRoot(),"elle");
    if ((fp=fopen(filename,"w"))==NULL) return(OPEN_ERR);

    /* write version */
    sprintf(header,"%s", ElleGetCreationCString());
    if (fprintf(fp,"%s\n",header)<0) return(WRITE_ERR);
    if (!(err=ElleWriteOptionData(fp)))
        if (!(err = ElleWriteReverseFlynnData(fp)))
            err = ElleWriteRegionAttributeData(fp);
    if (!err)
        err=ElleWriteNodeData(fp);
    fclose(fp);
    if (err) OnError("",err);
    return(0);
} 

int ElleWriteReverseFlynnData(FILE *fp)
{
    char label[20];
    int i,j,max,num,err=0, parents=0;
    int *node_ids,num_nodes;
    int count=0, inv_cnt=0;
    ERegion full_id;

    max = ElleMaxFlynns();
    for (i=0;i<max && !err;i++) {
        if (ElleFlynnIsActive(i) && !isParent(i)) {
            count++;
            if (ElleRegionArea(i) < 0) inv_cnt++;
        }
    }
    fprintf(stderr,"flynns %d, inverted %d\n",count,inv_cnt);
    if (count!=inv_cnt) OnError("not completely inverted",0);

    /*
     * write the lowest level flynns (defined by nodes)
     */
    if (!id_match(FileKeys,FLYNNS,label)) return(KEY_ERR);
    fprintf(fp,"%s\n",label);
    for (i=0;i<max && !err;i++) {
        if (ElleFlynnIsActive(i) && !isParent(i)) {
            ElleFlynnNodes(i, &node_ids, &num_nodes);
            if ((num=fprintf(fp,"%d %d",i,num_nodes))<0)
                   err==WRITE_ERR;
            else {
                j=num_nodes-1;
                do {
                    if ((num=fprintf(fp," %d",node_ids[j]))<0)
                        err=WRITE_ERR;
                    j--;
                } while (j>=0 && !err);
                fprintf(fp,"\n");
            }
            if (node_ids) free(node_ids);
        }
    }
    for (i=0;i<max && !parents;i++)
        if (ElleFlynnIsActive(i) && isParent(i)) parents=1;
    /*
     * write the higher level flynns (defined by other flynns)
     */
    if (!err && parents) err = ElleWriteParentData(fp);
    return(err);
}
