

#ifndef _USDAT_H_
#define _USDAT_H_

#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <fstream>
#include <string.h>
#include <sstream>

#include "wx/string.h"

#include "wx/variant.h"

#include "wx/txtstrm.h"
#include "wx/wfstream.h"
#include "wx/zipstrm.h"
#include "wx/sstream.h"

#include "wx/dir.h"

using namespace std;

class usdat
{

  public:

    //constructor
    usdat();

    wxString archive_out_path;
    wxString archive_path;

    //external processes
    bool graingrowth_proc;
    bool melt_proc;

    double graingrowth[10];
    double melt[10];
    double dis_plot[10];

    //setfunc-save
    bool gauss_strength_proc;
    bool linear_strength_proc;
    bool gauss_rate_proc;
    bool gauss_spring_proc;
    bool gauss_young_proc;

    bool multiply_relax_proc;
    bool timestep_proc;

    bool walls_proc;
    bool weaken_hor_proc;
    bool weaken_all_proc;
    bool make_gb_proc;

    bool dissolve_x_proc;
    bool dissolve_y_proc;
    bool plot_frac_proc;
    bool set_box_size_proc;

    bool set_min;
    bool heat_flow_example_proc;
    bool weaken_grain_proc;

    bool transition_proc;
    bool transition_heat_proc;
    bool heat_conduction_proc;

    bool extra_hor_layer_proc;

    bool lattice_fluid_gas_percent_proc;
    bool set_walls_lattice_gas_proc;
    bool fluid_lattice_gas_proc;
    bool fluid_lattice_gas_random_proc;
    bool set_fluid_lattice_gas_grain_proc;
    bool fluid_flow;

    bool add_doubles_proc;
    bool sin_anisotropy_proc;

    double transition[10];  // [1] == activation_energy, [2] == pressure_barrier
    double sin_anisotropy[10];
    double lattice_fluid_gas_percent[10];
    double set_walls_lattice_gas[10];
    double fluid_lattice_gas[10];
    double fluid_lattice_gas_random[10];
    double set_fluid_lattice_gas_grain[10];

    double heat_conduction[10];

    double gauss_young[10];
    double extra_hor_layer[10];
    double mineral[10];

    double weaken_grain[10];

    double gauss_strength[10];
    double linear_strength[10];
    double gauss_rate[10];
    double gauss_spring[10];

    double multiply_relax[10];
    double timestep[10];

    double walls[10];
    double weaken_hor[10];
    double weaken_all[10];
    double make_gb[10];
    double dissolve_x[10];
    double dissolve_y[10];
    double plot_frac[10];
    double set_box_size[10];

    //runfunction-save
    bool def_unilateral_proc;
    bool def_bilateral_proc;
    bool def_pureshear_proc;
    bool extra_deform;
    bool extra_def_unilateral_proc;
    bool extra_def_bilateral_proc;
    bool extra_def_pureshear_proc;
    bool viscosity_proc;
    bool stepsize_proc;
    bool stylolithe_dissolution;
    bool dissolution_strain;
    bool shrink_grain_proc;
    bool shrink_box;

    double box_shrinkage[10];
    double shrink_grain[10];
    double stylolithe_stress_shrink[10];
    double viscosity[10];
    double extra_deformation[10];
    double stepsize[10];

    //name-save
    //wxString name;

    //statistic-save
    bool stress_box_proc;
    bool stress_grain_proc;
    bool stress_two_grains_proc;
    bool surface_time_proc;
    bool xstress_proc;
    bool ystress_proc;
    bool surface_proc;

    double surface[10];
    double stress_box[10];
    double stress_grain[10];
    double stress_two_grains[10];
    double surface_time[10];
    double xstress[10];
    double ystress[10];

    //functions
    void SaveData();
    void OpenData();
    void read_from_string(wxString data_line);
    void CleanData();

    void SetBoudins();
    void SetInclusions();
    void SetGrooves();
    void SetShrinkage();
    void SetStylolithes();
    void SetFractures();
    void SetPhaseTransition();
    void SetGrainGrowth();
    void SetHeatFlow();
    void SetDiffusion();
    void SetFluidFlow();
    void SetViscousRelax();
    void SetGrainGrowthFracture();

    template<class INPUT, class OUTPUT>
    void swap( INPUT &inval, OUTPUT &outval);
  };

#endif
