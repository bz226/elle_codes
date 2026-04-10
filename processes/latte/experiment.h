
#ifndef _E_experiment_h
#define _E_experiment_h

#include <stdio.h>
#include <stdlib.h>

#include <math.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <sstream>

#include <wx/wx.h>

#include "phase_lattice.h"
#include "graingrowth.h"
//#include "melt.h"
#include "min_trans.h"


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


using namespace std;

extern int Init_Experiment(void);
extern int Run_Experiment(void);

struct procdata
  {
    bool exec[50]; // executed: yes/no
    float parameters[50][10]; // parameters given to the above defined function
    int i,k;

    procdata()
    {
      for (i=0; i<50; i++)
        exec[i] = false;

      for (i=0; i<50; i++)
        {
          for (k=0; k<10; k++)
            {
              parameters[i][k] = 0;
            }
        }
    }
  };

class Experiment : public GrainGrowth, public Min_Trans_Lattice
  {
  public:

    Experiment();                           // Constructor
    ~Experiment()
    {}
    ;                        // Destructor

    void Init();
    void Run();

  private:

    procdata rundata, setdata, statdata, externdata;
    char* file;
    double strain;

    void SetUpFromFile();
    void RunFromFile(int i);
    void read_custom_file();
    void read_from_string(wxString);
	 int experiment_time;

  };

#endif
