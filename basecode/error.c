 /*****************************************************
 * Copyright: (c) L. A. Evans
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "runopts.h"
#include "error.h"
#include "log.h"

/*****************************************************

static const char rcsid[] =
       "$Id$";

******************************************************/
#ifdef __cplusplus
extern "C" {
#endif
void append_error_message(const char *message,int err_num);
void OptsSyntax(int argc,char **argv);
void ElleRemoveTriAttributes();
/*extern int DisplayElleErrorMsg(char *msg,int err_num);*/
#ifdef __cplusplus
}
#endif

char ErrMessage[FILENAME_MAX];

char logbuf[E_LOGBUFSIZ];

void OnError(const char *message,int err_num)
{
    /*if (ElleDisplay())*/
        /*DisplayElleErrorMsg(message,err_num);*/
    /*else {*/
        logbuf[0] = '\0';
        append_error_message(message,err_num);
        strcat(logbuf,ErrMessage);
        Log( -1,logbuf );
        if (!ElleDisplay()) {
            CleanUp();
            exit(1);
        }
    /*}*/
}

void append_error_message(const char *message,int err_num)
{
    int len = 0;
    strcpy(ErrMessage,"");
    strncpy(ErrMessage,ElleAppName(),FILENAME_MAX-1);
    len = strlen(ErrMessage);
    strncat(ErrMessage,": ",FILENAME_MAX-len-1);
    len = strlen(ErrMessage);
    strncat(ErrMessage,message,FILENAME_MAX-len-1);
    len = strlen(ErrMessage);
    switch (err_num) {
    case MALLOC_ERR:strncat(ErrMessage," - Memory error",FILENAME_MAX-len-1);
                break;
    case READ_ERR:strncat(ErrMessage," - Error reading file",FILENAME_MAX-len-1);
                break;
    case OPEN_ERR:strncat(ErrMessage," - Error opening file",FILENAME_MAX-len-1);
                break;
    case EOF_ERR:strncat(ErrMessage," - End of file",FILENAME_MAX-len-1);
                break;
    case NOFILE_ERR:strncat(ErrMessage," - No elle file open",FILENAME_MAX-len-1);
                break;
    case NODENUM_ERR:strncat(ErrMessage," - inactive node",FILENAME_MAX-len-1);
                break;
    case NONB_ERR:strncat(ErrMessage," - neighbour node not found",FILENAME_MAX-len-1);
                break;
    case GRNNUM_ERR:strncat(ErrMessage," - inactive grain",FILENAME_MAX-len-1);
                break;
    case TYPE_ERR:strncat(ErrMessage," - unknown region type",FILENAME_MAX-len-1);
                break;
    case ID_ERR:strncat(ErrMessage," - could not match id",FILENAME_MAX-len-1);
                break;
    case MAXATTRIB_ERR:strncat(ErrMessage," - attribute limit reached",FILENAME_MAX-len-1);
                break;
    case ATTRIBID_ERR:strncat(ErrMessage," - invalid attribute",FILENAME_MAX-len-1);
                break;
    case RGNWRP_ERR:strncat(ErrMessage," - region wrap",FILENAME_MAX-len-1);
                break;
    case LIMIT_ERR:strncat(ErrMessage," - array limit reached",FILENAME_MAX-len-1);
                break;
    case DATA_ERR:strncat(ErrMessage," - data not found",FILENAME_MAX-len-1);
                break;
    case INDEX_ERR:strncat(ErrMessage," - index already used",FILENAME_MAX-len-1);
                break;
    case MAXINDX_ERR:strncat(ErrMessage," - max array index exceeded",FILENAME_MAX-len-1);
                break;
    case ORD_ERR:strncat(ErrMessage," - order array not found",FILENAME_MAX-len-1);
                break;
    case KEY_ERR:strncat(ErrMessage," - Unknown keyword",FILENAME_MAX-len-1);
                break;
    case OLDKEY_ERR:strncat(ErrMessage," - Keyword no longer accepted",FILENAME_MAX-len-1);
                break;
    case INTERSECT_ERR:strncat(ErrMessage," - Intersection not found",FILENAME_MAX-len-1);
                break;
    case NODECNT_ERR:strncat(ErrMessage," - Node count differs",FILENAME_MAX-len-1);
                break;
    case SEGCNT_ERR:strncat(ErrMessage," - Segment count differs",FILENAME_MAX-len-1);
                break;
    case RANGE_ERR:strncat(ErrMessage," - Range error",FILENAME_MAX-len-1);
                break;
    case INVALIDF_ERR:strncat(ErrMessage," - File does not have necessary attributes for process",FILENAME_MAX-len-1);
                break;
    case SYNTAX_ERR: OptsSyntax(0,0);
                strcat(ErrMessage," - Syntax error");
                break;
    case HELP_ERR: OptsSyntax(0,0);
                break;
    default:    break;
    }
}

void CleanUp(void)
{
    ElleRunFunc exitfn;

    ElleRemoveArrays();
    ElleRemoveTriAttributes();
    ElleRemoveEnergyLUT();
    /*if ((exitfn=ElleExitFunction())!=0) (*exitfn)();*/
}

void OptsSyntax(int argc,char **argv)
{
/* PRINT THE VERSION */
    char *names[9];
    char *logptr;
    int i, j, len, lenstr;

    static char *optionsMsg[8]={
    "   -i   Elle file to open",
    "   -e   optional extra data file",
    "   -u   ",
    "   -s   number of stages to be run",
    "   -f   how often to save .elle files",
    "   -n   no display (command line mode)",
    "   -h   Print this message\n",
    "Example:  elle_gg -i growth20.elle -s 50 -f 10 -n\n"
    };

    ElleGetOptNames(names);

    len=0; logptr=logbuf;
    for (j=0;j<8 && strlen(logbuf)<E_LOGBUFSIZ-1;j++) {
      if (!strcmp(optionsMsg[j],"   -u   ")) {
        len = strlen(optionsMsg[j]);
        if (strcmp(names[0],"unused")) {
          lenstr = strlen(logbuf);
          if ((len+lenstr)<E_LOGBUFSIZ-1)
              len = sprintf(&logbuf[lenstr],"%s", optionsMsg[j]);
          for (i=0;i<9 && strlen(logbuf)<E_LOGBUFSIZ-1;i++) {
            if (strcmp(names[i],"unused")) {
                len = strlen(names[i]);
                lenstr = strlen(logbuf);
                if ((len+lenstr)<E_LOGBUFSIZ-1)
                  len = sprintf(&logbuf[lenstr],"%s ",names[i]);
            }
          }
        }
      }
      else {
        len = strlen(optionsMsg[j]);
        lenstr = strlen(logbuf);
        if ((strlen(logbuf)+len)<E_LOGBUFSIZ-1)
           sprintf(&logbuf[lenstr],"%s", optionsMsg[j]);
      }
      lenstr = strlen(logbuf);
      sprintf(&logbuf[lenstr], "\n" );
    }
    //fprintf(stderr,"%s",logbuf);
    //printf("logbuf len = %d\n",strlen(logbuf));
}
