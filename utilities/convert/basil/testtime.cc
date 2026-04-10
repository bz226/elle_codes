#include <iostream.h>
#include <string>
#include "timefn.h"

int main(int argc, char **argv)
{
    char *t;

    string x = "time is ";
    x += GetLocalTime();
    cout << x;
}
