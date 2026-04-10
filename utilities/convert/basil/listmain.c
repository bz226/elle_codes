#include <stdio.h>
#include "bndlist.h"

main()
{
    float yvals[5] = {0.0, 0.13, 0.25, 0.2, 0.09};
    float xvals[5] = {0.0, 0.1, 0.2, 0.3, 0.4};
    int i;

    for (i=0;i<5;i++) AddToList(xvals[i],yvals[i],i+1);
}
