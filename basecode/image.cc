#include <cstdio>
#include <cstring>
#include <ctype.h>
#include <vector>
#include <math.h>
#include "file.h"
#include "general.h"
#include "error.h"
#include "log.h"
#include "mat.h"
#include "convert.h"
#include "image.h"

/*********************************************************
                                                                                
  ReadImage reads in a pixmap image format file
  and decimates image before stroring in array
  Currently, only ppm P3 P6 (ascii raw )

A PPM file consists of two parts, a header and the image data. The
header consists of at least three parts normally delinineated by
carriage returns and/or linefeeds but the PPM specification only
requires white space. The first "line" is a magic PPM identifier, it can
be "P3" or "P6" (not including the double quotes!). The next line
consists of the width and height of the image as ascii numbers. The last
part of the header gives the maximum value of the colour components for
the pixels, this allows the format to describe more than single byte
(0..255) colour values. In addition to the above required lines, a
comment can be placed anywhere with a "#" character, the comment extends
to the end of the line.

The following are all valid PPM headers.
Header example 1

P6 1024 788 255

Header example 2

P6 
1024 788 
# A comment
255

Header example 3

P3
1024 # the image width
788 # the image height
# A comment
1023

The format of the image data itself depends on the magic PPM identifier.
If it is "P3" then the image is given as ascii text, the numerical value
of each pixel ranges from 0 to the maximum value given in the header.
The lines should not be longer than 70 characters. 
                                                                                
LE -GIMP2.4.2 cannot correctly read images if there is a comment between the
number of colours and the image block

Keywords recognised on comment lines: (see image.h)

caxis
This is followed by a table which associates rgb colors with C-axis data
For each line the expected format is '#' followed by 3 integers (rgb values)
and 2 doubles (orientation, dip). Each element is separated by comma,
space or tab.
example input
#,caxis
#,255,255,255,152.10,38.18
#,255,0,255,81.29,45.49
#,255,0,0,128.08,47.48
#,0,255,0,315.68,55.58
#,0,255,255,29.84,35.49
#,0,0,255,15.06,44.16
#,100,100,100,304.08,31.36

dimension
The following line consists of a '#' and a double. Each element is separated by comma, space or tab.
example input
#,dimension
#,0.01

**********************************************************/
int  ReadPNMFile(char *fname,int *width, int *height, 
                 unsigned long *max, unsigned char **img,
                 std::vector<int> &rgbcols, std::vector<Coords_3D> &orient,
                 double *dim);
int Read_Comments(FILE *fp, std::vector<int> &rgbcols,
                            std::vector<Coords_3D> &orient,
                            double *dim);
int Read_Caxis(FILE *fp, std::vector<int> &rgbcols,
                            std::vector<Coords_3D> &orient);
int Read_Dimension(FILE *fp, double *dim );
                                                                                
int ElleReadImage(char *fname, int ***image, int *rows, int *cols,
                  int decimate,
                  std::vector<int> &rgbcols, std::vector<Coords_3D> &orient,
                  double *dim)
{
    int err=0;
    int i,j;
    unsigned long colr; // 8-bit colour
    int colsP,rowsP; // rows,cols in image
    unsigned long maxvalP; // number of colours

    unsigned char *img=0, *ptr;

    if ((err=ReadPNMFile(fname,&rowsP, &colsP, &maxvalP, &img,
                         rgbcols, orient, dim))!=0) {
        sprintf (logbuf, "\nError reading %s\n", fname);
        Log(2,logbuf);
        return(err);
    }

    *image=imatrix(0L,(long)rowsP/decimate,0L,(long)colsP/decimate);
    if (*image==0) return(MALLOC_ERR);
    // if maxvalP>255 then 2 bytes per pixel
    for(ptr=img,i=0;i<rowsP;i++)
    {
        for(j=0;j<colsP;j++)
        {
            //printf("rgb:%d %d
            //%d\n",pixmap[i][j].r,pixmap[i][j].g,pixmap[i][j].b);
            colr=*ptr +((*(ptr+1))*maxvalP)+((*(ptr+2)) *maxvalP*maxvalP);
            ptr += 3;
                                                                                
            if(i%decimate==0 && j%decimate==0)
                (*image)[i/decimate][j/decimate]=(int)colr;
        }
    }
    // convert the rgb to a single value and store in the red value
    for (i=0;i<rgbcols.size();i+=3)
        rgbcols[i] = rgbcols[i] + rgbcols[i+1]*maxvalP +
                     rgbcols[i+2]*maxvalP*maxvalP;
    delete [] img;
                                                                                
    for(i=0;i<rowsP/decimate;i++)
        (*image)[i][colsP/decimate]=(*image)[i][0];
                                                                                
    for(j=0;j<colsP/decimate;j++)
        (*image)[rowsP/decimate][j]=(*image)[0][j];
                                                                                
    (*image)[rowsP/decimate][colsP/decimate]=(*image)[0][0];
                                                                                
                                                                                
    *rows=(rowsP/decimate);
    *cols=(colsP/decimate);
    return(err);
}

int  ReadPNMFile(char *fname,int *width, int *height, 
                 unsigned long *max, unsigned char **img,
                 std::vector<int> &rgbcols, std::vector<Coords_3D> &orient,
                 double *dim)
{
    unsigned char *ptr=0;
    int err = 0;
    int i, c, val[3], dum[3], size=0, type=0, bpp=1;
    FILE *fp;

    *height=*width=*max=0;
    if ((fp=fopen(fname,"r"))==NULL) return(READ_ERR);
    c=getc(fp);
    while (c!= EOF && type==0) {
      switch(c) {
        case 'P': type = getc(fp);
                  break;
        case '#': dump_comments(fp);
                  break;
        default:  break;
      }
      c=getc(fp);
    }
    if (c==EOF) return(READ_ERR);
    switch(type) {
        case '3': c = getc(fp);
                  i=0;
                  while (c!=EOF && i<3) {
                      if (c=='#') Read_Comments(fp,rgbcols,orient,dim);
                      else if (isdigit(c)) {
                          ungetc(c,fp);
                          if (fscanf(fp,"%d",&val[i])==1) i++;
                      }
                      c=getc(fp);
                  }
                  if (isdigit(c))  ungetc(c,fp);
                  bpp = val[2]/255;
                  if (c==EOF) err = READ_ERR;
                  else size = 3*val[0]*val[1];
                  if (size>0 && (*img=new unsigned char[size*bpp])==0)
                      err = MALLOC_ERR;
                  ptr = *img; i=0;
                  while (!err && i<size/3)
                     if (fscanf(fp,"%d %d %d",&dum[0],&dum[1],&dum[2])!=3)
                         err = READ_ERR;
                     else {
                         *ptr++=(unsigned char)dum[0];
                         if (bpp>1) ptr += bpp-1;
                         *ptr++=(unsigned char)dum[1];
                         if (bpp>1) ptr += bpp-1;
                         *ptr++=(unsigned char)dum[2];
                         if (bpp>1) ptr += bpp-1;
                         i++;
                     }
                  break;
        case '6': c = getc(fp);
                  i=0;
                  while (c!=EOF && i<3) {
                      if (c=='#') {
                          ungetc(c,fp);
                          Read_Comments(fp,rgbcols,orient,dim);
                      }
                      else if (isdigit(c)) {
                          ungetc(c,fp);
                          if (fscanf(fp,"%d",&val[i])==1) i++;
                      }
                      c=getc(fp);
                  }
                  if (isdigit(c))  ungetc(c,fp);
                  bpp = val[2]/255;
                  if (c==EOF) err = READ_ERR;
                  else size = 3*val[0]*val[1];
                  if (size>0 && (*img=new unsigned char[size*bpp])==0)
                      err = MALLOC_ERR;
                  if (!err && fread(*img,bpp,size,fp)!=size)
                      err = READ_ERR;
                  break;
        case '2':
        case '5': Log(0,"Greyscale PNM not implemented");
                  err = READ_ERR;
                  break;
        default:  err = READ_ERR; break;
    }
    if (!err) {
        *width = val[0];
        *height = val[1];
        *max = val[2];
    }
    return(err);
}

int Read_Comments(FILE *fp, std::vector<int> &rgbcols,
                            std::vector<Coords_3D> &orient,
                            double *dim)
{
    bool done=false;
    char str[256]="";
    int err=0;
    int key=0;
    int c;

    c = getc(fp); //'#'
    while (!done) {
        c = getc(fp); //separator or end-of-line
        if (c==EOF || c=='\n' ||c=='\r') {
          done = true;
          if (c=='\r') dump_comments(fp);
        }
        else {
          c = getc(fp);
          if (isalpha(c)) {
            ungetc(c,fp);
            fscanf(fp,"%s",str);
            validate(str,&key,ppm_option_terms);
            switch(key) {
            case PPM_CAXIS:dump_comments(fp);
                            err = Read_Caxis(fp,rgbcols,orient);
                            break;
            case PPM_DIM:   dump_comments(fp);
                            err = Read_Dimension(fp,dim);
                            break;
            default: dump_comments(fp);
                             break;
            }
            c = getc(fp);
            if (c!='#') {
              done = true;
              ungetc(c,fp);
            }
          }
          else {
            dump_comments(fp);
          }
        }
    }
    return(err);
}

int Read_Caxis(FILE *fp, std::vector<int> &rgbcols,
                            std::vector<Coords_3D> &orient)
{
    bool done=false;
    char str[256]= "";
    char sep[10] = ", \t\r\n";
    char *start, *ptr;
    int c, i, nextc, num=0;
    int err=0;
    int intval[3];
    double eps=1.0e-10;
    double tmpori[3], ZXZori[3];
    Coords_3D ori;
    
    while (!done) {
      c = getc(fp);
      if (c!='#') {
          ungetc(c,fp);
          done = true;
      }
      else if (fgets(str,255,fp)!=NULL && !feof(fp)) {
        if (strlen(str)<2 || !isdigit(str[1])) {
          for (i=strlen(str)-1;i>=0;i--) ungetc((int)(str[i]),fp);
          ungetc(c,fp);
          done = true;
        }
        else {
              if (str[0]==',') {
                if ((num=sscanf(str, ",%d,%d,%d,%lf,%lf\n",
                              &intval[0], &intval[1],&intval[2],
                              &ori.x,&ori.y))!=5)
                  done = true;
              }
              else {
                if ((num=sscanf(str, "%d%d%d%lf%lf\n",
                              &intval[0], &intval[1],&intval[2],
                              &ori.x,&ori.y))!=5)
                  done = true;
              }
              if (!done) {
                for (i=0;i<3;i++) rgbcols.push_back(intval[i]);
                //convert geo convention to math
                GeoToMath(&ori.x,&ori.y);
                //convert caxis to euler
                PolarToCartesian(&tmpori[0],&tmpori[1],&tmpori[2],
                                 (ori.x)*DTOR,(ori.y)*DTOR);
                ZXZori[0] = ZXZori[1] = ZXZori[2] = 0.0;
                ZXZori[1] = acos(tmpori[2]);
                if  (fabs(ZXZori[1]) > eps && fabs(ZXZori[1]-PI) > eps) {
                  ZXZori[0] = atan2(tmpori[0],-tmpori[1]);
                }
                for (i=0;i<3;i++) tmpori[i] = ZXZori[i];
                eulerRange(tmpori,ZXZori);
                ori.x = ZXZori[0]*RTOD;
                ori.y = ZXZori[1]*RTOD;
                ori.z = ZXZori[2]*RTOD;
                orient.push_back(ori);
              }
          }
      }
    }
    return(err);
}

int Read_Dimension(FILE *fp, double *dim )
{
    bool done=false;
    char sep = ',';
    int c, i, nextc, num=0;
    int err=0;
    int intval=0;
    float fval=0;
    
    while (!done) {
      c = getc(fp);
      if (c!='#') {
          ungetc(c,fp);
          done = true;
      }
      else {
          c = getc(fp);
          if (c==',' || c==' ' || c=='\t') {
            nextc = getc(fp);
            ungetc(nextc,fp);
            ungetc(c,fp);
            if (!isdigit(nextc))  {
              done=true;
            }
            else {
              if ((num=fscanf(fp,"%c%f",&sep,&fval))!=2) err=READ_ERR;
              else *dim = fval;
            }
          }
          else {
            ungetc(c,fp);
            done = true;
          }
          if (!done) {
            dump_comments(fp);
          }
      //    else {
       //     ungetc('#',fp);
        //    ungetc(c,fp);
         // }
      }
    }
    return(err);
}
