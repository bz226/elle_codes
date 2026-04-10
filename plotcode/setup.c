/*----------------------------------------------------------------
 *    Elle:   setup.c  1.0  11 September 1997
 *
 *    Copyright (c) 1997 by L.A. Evans & T.D. Barr
 *----------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef XSYB
#include <X11/Xlib.h>  /* for type XFontStruct */
#endif

#include "types.h"
#include "error.h"
#include "colours.h"
#include "string_utils.h"
#include "options.h"
#include "runopts.h"

char Err_str[80];
plot_data Plot_info;
/* These are not used by elle */
input_options Settings, Initial_Settings;
int Mesh[MESH_ENTRIES];
float Pwindo[PWINDO_ENTRIES];

void init_arrays(),clear_arrays(),MeshParams();
void set_box_dimensions(),init_box();
void create_error_message(), Print_Comments();
extern void Exit_App(), GetColourmapOption(int *);
#ifdef XSYB
extern void ResizePlotWin();
#endif

SetupApp(argc, argv)
int argc;
char **argv;
{
    int err=0, i;
    int *red,*green,*blue,*irgbv;

    if (ElleDisplay()) {
    if (Init_App(argc,argv,&Plot_info.log_file_ptr)) exit(1);


    GetColourmapOption(&i);
    switch(i) {
    case GREY_MAP: red=Red0;
                   green=Green0;
                   blue=Blue0;
                   irgbv=Rgb0;
                   break;
    case STD_MAP:  red=Red1;
                   green=Green1;
                   blue=Blue1;
                   irgbv=Rgb1;
                   break;
    default:       fprintf(stderr,"Invalid colour map/n");
                   break;
    }
    if (err = Set_Colours(red,green,blue,irgbv,STEPS)) return(1);
    }

    /* save the initial options settings */
    /*Initial_Settings = Settings;*/
    return(0);
} 

int RescaleBox(float xmin, float xmax,
               float ymin, float ymax)
{
	int chng_plot=0;
    float  oldwdth,oldhgt;
	
    oldwdth=Pwindo[XCMAX]-Pwindo[XCMIN];
    oldhgt=Pwindo[YCMAX]-Pwindo[YCMIN];
    Pwindo[XCMIN] = xmin;
    Pwindo[XCMAX] = xmax;
    Pwindo[YCMIN] = ymin;
    Pwindo[YCMAX] = ymax;
#ifdef XSYB
	chng_plot=((fabs(oldwdth-(Pwindo[XCMAX]-Pwindo[XCMIN]))>0)||
	                (fabs(oldhgt-(Pwindo[YCMAX]-Pwindo[YCMIN]))>0)?1:0);
	ResizePlotWin(chng_plot);
#endif
    init_box(Pwindo,1);
    return(0);
}

void Init_Options()
{
    int i;

    Plot_info.curr_cell = 0;
    Plot_info.max_col = DEFAULT_COLUMNS-1;
    Plot_info.max_row = DEFAULT_ROWS-1;
    Plot_info.curr_x = Plot_info.curr_y = 0.0;
    Plot_info.origin_x = Plot_info.origin_y = 0.0;
    Plot_info.title_offset = 0;
    Plot_info.plot_type = -1;
    Plot_info.plot_description = 0;
    Plot_info.var_num = 0;
    Plot_info.title = NULL;
    strcpy(Plot_info.fontname,"Helvetica");
    strcpy(Plot_info.variable,"");
    strcpy(Plot_info.dflt_label1,"");
    strcpy(Plot_info.dflt_label2,"");
    strcpy(Plot_info.curr_label,"");
    Plot_info.display_log = 0;
    Plot_info.log_file_ptr = NULL;
    Plot_info.inp_file = NULL;
    Plot_info.stage = 0;

    InitOptions(&Settings);

    for (i=0;i<PWINDO_ENTRIES;i++) Pwindo[i]=0.0;
    Pwindo[XCMIN] =  Pwindo[YCMIN] = 0.0;
    Pwindo[XCMAX] =  Pwindo[YCMAX] = 1.0;
    Pwindo[XMIN] =  Pwindo[YMIN] = 0.0;
    Pwindo[XMAX] =  Pwindo[YMAX] = 1.0;

}


/*
 * if a file has been opened, clear the mesh arrays and reset
 * the interpolation mesh dimensions
 */
void NewMesh()
{
    if (Plot_info.inp_file!=NULL) {
        Mesh[NNY3] = 0;
        MeshParams( Pwindo[YCMAX]-Pwindo[YCMIN],Pwindo[XCMAX]-Pwindo[XCMIN],
                                                                 Mesh );
    }
}

void MeshParams( hgt,wdth,mesh )
float hgt, wdth;
int *mesh;
{
    int prev_nny3;

    /*
     * set array size for work arrays using value in pl_params.h
    mesh[NNX3] = Settings.plot_opts.nx3;
     */
    prev_nny3 = mesh[NNY3];
    mesh[NNY3] = (int)((mesh[NNX3]-3)*hgt/wdth)+3;
    if (mesh[NNY3]!=prev_nny3) {
/*
        clear_mesh_arrays();
*/
        mesh[NP3] = mesh[NNX3]*mesh[NNY3];
        /*if (Settings.plot_opts.dble) mesh[NP3] *= 2;*/
        mesh[ICON] = mesh[NP3]/7;
        mesh[NX3] = mesh[NNX3];
        mesh[NY3] = mesh[NNY3];
        mesh[NX2]=mesh[NX3]-1;
        mesh[NX1]=mesh[NX2]-1;
        mesh[NX0]=mesh[NX1]-1;
        mesh[NY2]=mesh[NY3]-1;
        mesh[NY1]=mesh[NY2]-1;
        mesh[NY0]=mesh[NY1]-1;
        mesh[NXY]=mesh[NX3]*mesh[NY3];
    }
}

void set_box_dimensions( pwin,x,y,width,height )
float *pwin,x,y,width,height;
{

    /* Pwindo values for plotting routines */
    pwin[XMIN]= x + width*Settings.xmargin;
    pwin[YMIN]= y + height*Settings.ymargin;
    /*pwin[YMIN]= y - height*(1.0-Settings.ymargin);*/
    pwin[XMAX] = pwin[XMIN] + width*(1.0-Settings.xmargin-Settings.xmargin);
    pwin[YMAX] = pwin[YMIN] + height*(1.0-2*Settings.ymargin);
}
 
void init_box(pwin,rescale)
unsigned char rescale;
float *pwin;
{
    int pen_state;
    float plotwdth, plothgt, wdth, hgt, rangex, rangey, centrex, centrey;

    if (rescale) {
        wdth = pwin[XMAX] - pwin[XMIN];
        hgt = pwin[YMAX] - pwin[YMIN];
        rangex = pwin[XCMAX]-pwin[XCMIN];
        rangey = pwin[YCMAX]-pwin[YCMIN];
        if (Settings.zoom!=1.0) {
            centrex = rangex/2;
            centrey = rangey/2;
            rangex /= Settings.zoom;
            rangey /= Settings.zoom;
            pwin[XCMIN] = pwin[XCMIN]+centrex-rangex/2;
            pwin[XCMAX] = pwin[XCMIN]+rangex;
            pwin[YCMIN] = pwin[YCMIN]+centrey-rangey/2;
            pwin[YCMAX] = pwin[YCMIN]+rangey;
        }
        plotwdth = wdth;
        plothgt = plotwdth*rangey/rangex;
        if (plothgt > hgt) {
            plothgt = hgt;
            plotwdth = plothgt*rangex/rangey;
        }
        pwin[SCLX]=rangex/plotwdth;
        pwin[SCLY]=rangey/plothgt;

        pwin[ULEFT] = pwin[XCMIN];
        pwin[URGHT] = pwin[ULEFT] + wdth*pwin[SCLX];
        pwin[UTOP] = pwin[YCMAX];
        pwin[UBASE] = pwin[UTOP] - hgt*pwin[SCLY];
    }
    scale_(&pwin[ULEFT],&pwin[URGHT],
                        &pwin[XMIN],&pwin[XMAX],
                        &pwin[UBASE],&pwin[UTOP],
                        &pwin[YMIN],&pwin[YMAX]);
    pen_state=PEN_UP;
    plotu_(&pwin[XCMIN],&pwin[YCMIN],&pen_state);
}

void create_error_message( message,err_num )
char *message;
int err_num;
{
    char buf[20];

    switch (err_num) {
    case MALLOC_ERR:strcat(message," - Memory error");
                break;
    case READ_ERR:strcat(message," - Error reading file");
                break;
    case OPEN_ERR:strcat(message," - Error opening file");
                break;
    case EOF_ERR:strcat(message," - End of file");
                if (Plot_info.inp_file!=NULL)
                sprintf(buf," %2d records",Plot_info.inp_file->rec_max);
                strcat(message,buf);
                break;
    case VHB_ERR:strcat(message," - No VHB array");
                break;
    case SSQ_ERR:strcat(message," - No SSQ array");
                break;
    case ELFX_ERR:strcat(message," - No IELFIX array");
                break;
    case REF_ERR:strcat(message," - Reference error");
                break;
    case LAG_ERR:strcat(message," - No Lagrangian arrays");
                break;
    case STR_ERR:strcat(message," - Error reading string");
                break;
    case NUM_ERR:strcat(message," - Error reading number");
                break;
    case VAR_ERR:strcat(message," - Unknown variable");
                break;
    case KEY_ERR:strcat(message," - Unknown keyword");
                break;
    case INIT_ERR:strcat(message," - Initial variable must be Options");
                break;
    case RANGE_ERR:strcat(message," - Value out of range");
                break;
    case STGS_ERR:strcat(message," - row, col or orientation mismatch");
                break;
    default:    break;
    }
}

error_msg(err_num,str)
int err_num;
char *str;
{
    char message[FILENAME_MAX],buf[20];

    strncpy(message,str,sizeof(message)-1);
    create_error_message(message,err_num);
    fprintf(stderr,"%s\n",message);
    Exit_App(err_num);
}
 
format_number_(str,num,format)
char *str;
float *num;
int *format;
{
    char form[7];
    int total,dec_places,i,len;

    total = *format/10;
    dec_places = *format%10;
    sprintf(form,"%%%d.%df",total,dec_places);
    sprintf(str,form,*num);
    /* strip leading and trailing spaces */
    while (str[0]==' ') {
        len = strlen(str)+1;
        for (i=0;i<len;i++) str[i] = str[i+1];
    }
    len = strlen(str);
    while (str[len]==' ') {
        str[len] = '\0';
        len--;
    }
}
