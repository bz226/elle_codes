#include "userdat.h"
#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <fstream>
#include <string.h>
#include <sstream>

#include <ctype.h>

template <class INPUT, class OUTPUT>
void usdat::swap( INPUT &inval, OUTPUT &outval )
{
	stringstream ss;
	ss << inval; // insert value into stream
	ss >> outval; // get value from stream
}

usdat::usdat()
{
	CleanData();
}

void usdat::CleanData()
{
	int i;

	heat_flow_example_proc = false;

	stepsize_proc = true; //important! stepsize wont be read otherwise

	weaken_grain_proc = false;

	add_doubles_proc = false;
	transition_proc = false;
	transition_heat_proc = false;
	heat_conduction_proc = false;

	melt_proc = false;
	graingrowth_proc = false;

	def_unilateral_proc = false;

	gauss_strength_proc = false;
	linear_strength_proc = false;
	gauss_rate_proc = false;
	gauss_spring_proc = false;
	gauss_young_proc = false;

	multiply_relax_proc = false;
	timestep_proc = false;

	walls_proc = false;
	weaken_hor_proc = false;
	weaken_all_proc = false;
	make_gb_proc = false;

	dissolve_x_proc = false;
	dissolve_y_proc = false;
	plot_frac_proc = false;
	set_box_size_proc = false;

	set_min = false;

	def_unilateral_proc = false;
	def_bilateral_proc = false;
	def_pureshear_proc = false;
	viscosity_proc = false;
	stepsize_proc = false;
	stylolithe_dissolution = false;
	dissolution_strain = false;
	shrink_grain_proc = false;
	shrink_box = false;

	extra_def_unilateral_proc = false;
	extra_def_bilateral_proc = false;
	extra_def_pureshear_proc = false;
	extra_deform = false;

	stress_box_proc = false;
	stress_grain_proc = false;
	stress_two_grains_proc = false;
	surface_time_proc = false;
	xstress_proc = false;
	ystress_proc = false;

	surface_proc = false;
	extra_hor_layer_proc = false;

	lattice_fluid_gas_percent_proc = false;
	set_walls_lattice_gas_proc = false;
	fluid_lattice_gas_proc = false;
	fluid_lattice_gas_random_proc = false;
	set_fluid_lattice_gas_grain_proc = false;
	fluid_lattice_gas_random_proc = true;
	set_fluid_lattice_gas_grain_proc = false;
	set_walls_lattice_gas_proc = false;
	fluid_flow = false;

	sin_anisotropy_proc = false;
	for ( i = 0; i < 10; i++ )
	{
		transition[ i ] = 0.0;

		sin_anisotropy[ i ] = 0.0;

		extra_hor_layer[ i ] = 0.0;
		graingrowth[ i ] = 0.0;
		melt[ i ] = 0.0;

		dis_plot[ i ] = 0.0;

		lattice_fluid_gas_percent[ i ] = 0.0;
		set_walls_lattice_gas[ i ] = 0.0;
		fluid_lattice_gas[ i ] = 0.0;
		fluid_lattice_gas_random[ i ] = 0.0;
		set_fluid_lattice_gas_grain[ i ] = 0.0;

		heat_conduction[ i ] = 0.0;

		gauss_strength[ i ] = 0.0;
		linear_strength[ i ] = 0.0;
		gauss_rate[ i ] = 0.0;
		gauss_spring[ i ] = 0.0;
		gauss_young[ i ] = 0.0;

		multiply_relax[ i ] = 0.0;
		timestep[ i ] = 0.0;

		walls[ i ] = 0.0;
		weaken_hor[ i ] = 0.0;
		weaken_all[ i ] = 0.0;
		weaken_grain[ i ] = 0.0;
		make_gb[ i ] = 0.0;
		dissolve_x[ i ] = 0.0;
		dissolve_y[ i ] = 0.0;
		plot_frac[ i ] = 0.0;
		set_box_size[ i ] = 0.0;
		mineral[ i ] = 0.0;

		viscosity[ i ] = 0.0;
		stepsize[ i ] = 0.0;
		extra_deformation[ i ] = 0.0;
		stylolithe_stress_shrink[ i ] = 0.0;
		shrink_grain[ i ] = 0.0;
		box_shrinkage[ i ] = 0.0;

		stress_box[ i ] = 0.0;
		stress_grain[ i ] = 0.0;
		stress_two_grains[ i ] = 0.0;
		surface_time[ i ] = 0.0;
		xstress[ i ] = 0.0;
		ystress[ i ] = 0.0;

		surface[ i ] = 0.0;
	}
}

// help-function to read custom-files
void usdat::read_from_string( wxString data_line )
{
	float argument;
	int exec;
	int fctnb, i;

	//const char* line = (const char*)l.mb_str(wxConvUTF8);

	wxStringInputStream sis( data_line );
	wxTextInputStream ist( sis );

	ist >> exec;

	ist >> fctnb;

	//cout << line << endl;

	if ( fctnb < 50 && fctnb >= 0 )
	{

		i = 0;
		switch ( fctnb )
		{
				case 0:
				if ( exec )
					linear_strength_proc = true;
				while ( i < 10 )
				{
					ist >> argument;
					linear_strength[ i ] = argument;
					i++;
				}
				break;
				case 1:
				if ( exec )
					gauss_spring_proc = true;
				while ( i < 10 )
				{
					ist >> argument;
					gauss_spring[ i ] = argument;
					i++;
				}
				break;
				case 2:
				if ( exec )
					gauss_rate_proc = true;
				while ( i < 10 )
				{
					ist >> argument;
					gauss_rate[ i ] = argument;
					i++;
				}
				break;
				case 3:
				if ( exec )
					gauss_strength_proc = true;
				while ( i < 10 )
				{
					ist >> argument;
					gauss_strength[ i ] = argument;
					i++;
				}
				break;
				case 4:
				if ( exec )
					make_gb_proc = true;
				while ( i < 10 )
				{
					ist >> argument;
					make_gb[ i ] = argument;
					i++;
				}
				break;
				case 5:
				if ( exec )
					plot_frac_proc = true;
				while ( i < 10 )
				{
					ist >> argument;
					plot_frac[ i ] = argument;
					i++;
				}
				break;
				case 6:
				if ( exec )
					walls_proc = true;
				while ( i < 10 )
				{
					ist >> argument;
					walls[ i ] = argument;
					i++;
				}
				break;
				case 7:
				if ( exec )
					weaken_all_proc = true;
				while ( i < 10 )
				{
					ist >> argument;
					weaken_all[ i ] = argument;
					i++;
				}
				break;
				case 8:
				if ( exec )
					weaken_hor_proc = true;
				while ( i < 10 )
				{
					ist >> argument;
					weaken_hor[ i ] = argument;
					i++;
				}
				break;
				case 9:
				if ( exec )
					multiply_relax_proc = true;
				while ( i < 10 )
				{
					ist >> argument;
					multiply_relax[ i ] = argument;
					i++;
				}
				break;
				case 10:
				if ( exec )
					set_min = true;
				while ( i < 10 )
				{
					ist >> argument;
					mineral[ i ] = argument;
					i++;
				}
				break;
				case 11:
				if ( exec )
					set_box_size_proc = true;
				while ( i < 10 )
				{
					ist >> argument;
					set_box_size[ i ] = argument;
					i++;
				}
				break;
				case 12:
				if ( exec )
					timestep_proc = true;
				while ( i < 10 )
				{
					ist >> argument;
					timestep[ i ] = argument;
					i++;
				}
				break;
				case 13:
				if ( exec )
					dissolve_x_proc = true;
				while ( i < 10 )
				{
					ist >> argument;
					dissolve_x[ i ] = argument;
					i++;
				}
				break;
				case 14:
				if ( exec )
					dissolve_y_proc = true;
				while ( i < 10 )
				{
					ist >> argument;
					dissolve_y[ i ] = argument;
					i++;
				}
				break;
				case 16:
				if ( exec )
					extra_hor_layer_proc = true;
				while ( i < 10 )
				{
					ist >> argument;
					extra_hor_layer[ i ] = argument;
					i++;
				}
				break;
				case 17:
				if ( exec )
					gauss_young_proc = true;
				while ( i < 10 )
				{
					ist >> argument;
					gauss_young[ i ] = argument;
					i++;
				}
				break;
				case 18:
				if ( exec )
					weaken_grain_proc = true;
				while ( i < 10 )
				{
					ist >> argument;
					weaken_grain[ i ] = argument;
					i++;
				}
				break;
				case 25:
				if ( exec )
					heat_conduction_proc = true;
				while ( i < 10 )
				{
					ist >> argument;
					heat_conduction[ i ] = argument;
					i++;
				}
				break;
				case 20:
				if ( exec )
					fluid_lattice_gas_random_proc = true;
				while ( i < 10 )
				{
					ist >> argument;
					fluid_lattice_gas_random[ i ] = argument;
					i++;
				}
				break;
				case 21:
				if ( exec )
					set_fluid_lattice_gas_grain_proc = true;
				while ( i < 10 )
				{
					ist >> argument;
					set_fluid_lattice_gas_grain[ i ] = argument;
					i++;
				}
				break;
				case 30:
				if ( exec )
					sin_anisotropy_proc = true;
				while ( i < 10 )
				{
					ist >> argument;
					sin_anisotropy[ i ] = argument;
					i++;
				}
				break;
				case 31:
				while ( i < 10 )
				{
					ist >> argument;
					transition[ i ] = argument;
					i++;
				}
				break;

		}
	}
	// the runfunctions: enter here
	else if ( fctnb < 100 && fctnb >= 0 )
	{
		i = 0;
		switch ( fctnb )
		{
				case 50:
				if ( exec )
					def_unilateral_proc = true;
				break;
				case 51:
				if ( exec )
					def_pureshear_proc = true;
				break;
				case 56:
				if ( exec )
					def_bilateral_proc = true;
				break;
				case 59:
				if ( exec )
					extra_deform = true;
				while ( i < 10 )
				{
					ist >> argument;
					extra_deformation[ i ] = argument;
					i++;
				}
				break;
				case 60:
				if ( exec )
					extra_def_unilateral_proc = true;
				break;
				case 61:
				if ( exec )
					extra_def_pureshear_proc = true;
				break;
				case 62:
				if ( exec )
					extra_def_bilateral_proc = true;
				break;
				case 54:
				if ( exec )
					viscosity_proc = true;
				while ( i < 10 )
				{
					ist >> argument;
					viscosity[ i ] = argument;
					i++;
				}
				break;
				case 58:
				if ( exec )
					stepsize_proc = true;
				while ( i < 10 )
				{
					ist >> argument;
					stepsize[ i ] = argument;
					i++;
				}
				break;
				case 57:
				if ( exec )
					stylolithe_dissolution = true;
				while ( i < 10 )
				{
					ist >> argument;
					stylolithe_stress_shrink[ i ] = argument;
					i++;
				}
				break;
				case 55:
				if ( exec )
					dissolution_strain = true;
				break;
				case 52:
				if ( exec )
					shrink_grain_proc = true;
				while ( i < 10 )
				{
					ist >> argument;
					shrink_grain[ i ] = argument;
					i++;
				}
				break;
				case 53:
				if ( exec )
					shrink_box = true;
				while ( i < 10 )
				{
					ist >> argument;
					box_shrinkage[ i ] = argument;
					i++;
				}
				break;
				case 65:
				if ( exec )
					transition_proc = true;
				break;
				case 66:
				if ( exec )
					transition_heat_proc = true;
				break;
				case 71:
				if ( exec )
					set_walls_lattice_gas_proc = true;
				while ( i < 10 )
				{
					ist >> argument;
					set_walls_lattice_gas[ i ] = argument;
					i++;
				}
				break;
				case 70:
				if ( exec )
					lattice_fluid_gas_percent_proc = true;
				case 78:
				while ( i < 10 )
				{
					ist >> argument;
					dis_plot[ i ] = argument;
					i++;
				}
		}
	}
	//statistic functions
	else if ( fctnb >= 100 && fctnb < 150 )
	{
		i = 0;
		switch ( fctnb )
		{
				case 100:
				if ( exec )
					stress_box_proc = true;
				while ( i < 10 )
				{
					ist >> argument;
					stress_box[ i ] = argument;
					i++;
				}
				break;
				case 101:
				if ( exec )
					stress_grain_proc = true;
				while ( i < 10 )
				{
					ist >> argument;
					stress_grain[ i ] = argument;
					i++;
				}
				break;
				case 102:
				if ( exec )
					stress_two_grains_proc = true;
				while ( i < 10 )
				{
					ist >> argument;
					stress_two_grains[ i ] = argument;
					i++;
				}
				break;
				case 103:
				if ( exec )
					surface_time_proc = true;
				while ( i < 10 )
				{
					ist >> argument;
					surface_time[ i ] = argument;
					i++;
				}
				break;
				case 104:
				if ( exec )
					xstress_proc = true;
				while ( i < 10 )
				{
					ist >> argument;
					xstress[ i ] = argument;
					i++;
				}
				break;
				case 105:
				if ( exec )
					ystress_proc = true;
				while ( i < 10 )
				{
					ist >> argument;
					ystress[ i ] = argument;
					i++;
				}
				break;
				case 106:
				if ( exec )
					surface_proc = true;
				while ( i < 10 )
				{
					ist >> argument;
					surface[ i ] = argument;
					i++;
				}
				break;
		}
	}
	//external processes
	else if ( fctnb >= 150 && fctnb < 200 )
	{
		i = 0;
		switch ( fctnb )
		{
				case 150:
				if ( exec )
					graingrowth_proc = true;
				while ( i < 10 )
				{
					ist >> argument;
					graingrowth[ i ] = argument;
					i++;
				}
				break;
				case 151:
				if ( exec )
					melt_proc = true;
				while ( i < 10 )
				{
					ist >> argument;
					melt[ i ] = argument;
					i++;
				}
				break;
		}
	}
}

void usdat::OpenData()
{

	if ( wxFileExists( archive_path ) )
	{
		wxZipEntry * entry;

		//check if zip-file exists
		wxDir dir( wxGetCwd() );
		wxString filename;

		filename = archive_path;
		//bool cont = dir.GetFirst(&filename, wxT("*.zip"), wxDIR_FILES);

		// if it does, continue
		bool cont = true; // this is currently not necessary

		if ( cont )
		{
			//necessary declarations
			wxZipEntry * entry;
			wxFFileInputStream in( filename );
			wxZipInputStream zip( in );
			wxTextInputStream text( zip );
			wxString data_line;

			while ( entry = zip.GetNextEntry(), entry != NULL )
			{
				cont = false;
				if ( entry->GetName() == wxT( "latte_pref" ) )
					if ( zip.OpenEntry( *entry ) )
					{
						cont = true;
						break;
					}
			}

			// if file exists
			if ( cont )
			{
				while ( !zip.Eof() )
				{
					data_line = text.ReadLine();
					if ( !data_line.IsEmpty() )
					{
						//cout << data_line.ToAscii() << endl;
						read_from_string( data_line );
					}
				}
			}
		}
	}
}

void usdat::SaveData()
{
	double s;
	int i;
	//ofstream pref("latte_pref");

	wxZipEntry *entry;
	wxFFileOutputStream out( _T( "tmp.zip" ) );
	//wxTempFileOutputStream out(archive_path);
	wxZipOutputStream zip_out( out );
	wxTextOutputStream txt_out( zip_out );

	string filename( archive_out_path.ToAscii() );
	//cout << filename << endl;

	zip_out.PutNextEntry( _T( "latte_pref" ) );

	txt_out << gauss_spring_proc << wxT( " " ) << 1 << wxT( " " );
	for ( i = 0; i < 10; i++ )
	{
		txt_out << gauss_spring[ i ] << wxT( " " );
	}
	txt_out << endl;

	txt_out << fluid_lattice_gas_random_proc << wxT( " " ) << 20 << wxT( " " );
	for ( i = 0; i < 10; i++ )
	{
		txt_out << fluid_lattice_gas_random[ i ] << wxT( " " );
	}
	txt_out << endl;

	txt_out << set_fluid_lattice_gas_grain_proc << wxT( " " ) << 21 << wxT( " " );

	for ( i = 0; i < 10; i++ )
	{
		txt_out << set_fluid_lattice_gas_grain[ i ] << wxT( " " );
	}
	txt_out << endl;

	txt_out << set_walls_lattice_gas_proc << wxT( " " ) << 22 << wxT( " " ) << endl;
	;

	txt_out << lattice_fluid_gas_percent_proc << wxT( " " ) << 70 << wxT( " " ) << endl;

	txt_out << set_walls_lattice_gas_proc << wxT( " " ) << 71 << wxT( " " );
	for ( i = 0; i < 10; i++ )
	{
		txt_out << set_walls_lattice_gas[ i ] << wxT( " " );
	}
	txt_out << endl;

	txt_out << sin_anisotropy_proc << wxT( " " ) << 30 << wxT( " " );
	for ( i = 0; i < 10; i++ )
	{
		txt_out << sin_anisotropy[ i ] << wxT( " " );
	}
	txt_out << endl;

	txt_out << add_doubles_proc << wxT( " " ) << 23 << wxT( " " ) << endl;

	txt_out << graingrowth_proc << wxT( " " ) << 150 << wxT( " " );
	for ( i = 0; i < 10; i++ )
	{
		txt_out << graingrowth[ i ] << wxT( " " );
	}
	txt_out << endl;

	txt_out << melt_proc << wxT( " " ) << 151 << wxT( " " );
	for ( i = 0; i < 10; i++ )
	{
		txt_out << melt[ i ] << wxT( " " );
	}
	txt_out << endl;

	txt_out << plot_frac_proc << wxT( " " ) << 5 << wxT( " " );
	for ( i = 0; i < 10; i++ )
	{
		txt_out << plot_frac[ i ] << wxT( " " );
	}
	txt_out << endl;

	txt_out << gauss_strength_proc << wxT( " " ) << 3 << wxT( " " );
	for ( i = 0; i < 10; i++ )
	{
		txt_out << gauss_strength[ i ] << wxT( " " );
	}
	txt_out << endl;

	txt_out << linear_strength_proc << wxT( " " ) << 0 << wxT( " " );
	for ( i = 0; i < 10; i++ )
	{
		txt_out << linear_strength[ i ] << wxT( " " );
	}
	txt_out << endl;

	txt_out << gauss_rate_proc << wxT( " " ) << 2 << wxT( " " );
	for ( i = 0; i < 10; i++ )
	{
		txt_out << gauss_rate[ i ] << wxT( " " );
	}
	txt_out << endl;

	txt_out << gauss_young_proc << wxT( " " ) << 17 << wxT( " " );
	for ( i = 0; i < 10; i++ )
	{
		txt_out << gauss_young[ i ] << wxT( " " );
	}
	txt_out << endl;

	txt_out << multiply_relax_proc << wxT( " " ) << 9 << wxT( " " );
	for ( i = 0; i < 10; i++ )
	{
		txt_out << multiply_relax[ i ] << wxT( " " );
	}
	txt_out << endl;

	txt_out << timestep_proc << wxT( " " ) << 12 << wxT( " " );
	for ( i = 0; i < 10; i++ )
	{
		txt_out << timestep[ i ] << wxT( " " );
	}
	txt_out << endl;

	txt_out << walls_proc << wxT( " " ) << 6 << wxT( " " );
	for ( i = 0; i < 10; i++ )
	{
		txt_out << walls[ i ] << wxT( " " );
	}
	txt_out << endl;

	txt_out << weaken_hor_proc << wxT( " " ) << 8 << wxT( " " );
	for ( i = 0; i < 10; i++ )
	{
		txt_out << weaken_hor[ i ] << wxT( " " );
	}
	txt_out << endl;

	txt_out << weaken_all_proc << wxT( " " ) << 7 << wxT( " " );
	for ( i = 0; i < 10; i++ )
	{
		txt_out << weaken_all[ i ] << wxT( " " );
	}
	txt_out << endl;

	txt_out << make_gb_proc << wxT( " " ) << 4 << wxT( " " );
	for ( i = 0; i < 10; i++ )
	{
		txt_out << make_gb[ i ] << wxT( " " );
	}
	txt_out << endl;

	txt_out << dissolve_x_proc << wxT( " " ) << 13 << wxT( " " );
	for ( i = 0; i < 10; i++ )
	{
		txt_out << dissolve_x[ i ] << wxT( " " );
	}
	txt_out << endl;

	txt_out << dissolve_y_proc << wxT( " " ) << 14 << wxT( " " );
	for ( i = 0; i < 10; i++ )
	{
		txt_out << dissolve_y[ i ] << wxT( " " );
	}
	txt_out << endl;

	txt_out << plot_frac_proc << wxT( " " ) << 5 << wxT( " " );
	for ( i = 0; i < 10; i++ )
	{
		txt_out << plot_frac[ i ] << wxT( " " );
	}
	txt_out << endl;

	txt_out << plot_frac_proc << wxT( " " ) << 10 << wxT( " " );
	for ( i = 0; i < 10; i++ )
	{
		txt_out << mineral[ i ] << wxT( " " );
	}
	txt_out << endl;

	txt_out << set_box_size_proc << wxT( " " ) << 11 << wxT( " " );
	for ( i = 0; i < 10; i++ )
	{
		txt_out << set_box_size[ i ] << wxT( " " );
	}
	txt_out << endl;

	txt_out << extra_hor_layer_proc << wxT( " " ) << 16 << wxT( " " );
	for ( i = 0; i < 10; i++ )
	{
		txt_out << extra_hor_layer[ i ] << wxT( " " );
	}
	txt_out << endl;

	txt_out << weaken_grain_proc << wxT( " " ) << 18 << wxT( " " );
	for ( i = 0; i < 10; i++ )
	{
		txt_out << weaken_grain[ i ] << wxT( " " );
	}
	txt_out << endl;

	txt_out << heat_conduction_proc << wxT( " " ) << 25 << wxT( " " );
	for ( i = 0; i < 10; i++ )
	{
		txt_out << heat_conduction[ i ] << wxT( " " );
	}
	txt_out << endl;
	txt_out << heat_conduction_proc << wxT( " " ) << 75 << wxT( " " ) << endl;

	txt_out << 1 << wxT( " " ) << 78 << wxT( " " );
	for ( i = 0; i < 10; i++ )
	{
		txt_out << dis_plot[ i ] << wxT( " " );
	}
	txt_out << endl;

	txt_out << viscosity_proc << wxT( " " ) << 54 << wxT( " " );
	for ( i = 0; i < 10; i++ )
	{
		txt_out << viscosity[ i ] << wxT( " " );
	}
	txt_out << endl;

	txt_out << def_unilateral_proc << wxT( " " ) << 50 << wxT( " " ) << endl;

	txt_out << def_pureshear_proc << wxT( " " ) << 51 << wxT( " " ) << endl;

	txt_out << def_bilateral_proc << wxT( " " ) << 56 << wxT( " " ) << endl;

	//txt_out << stepsize_proc << wxT(" ") << 58 << wxT(" ");
	txt_out << 1 << wxT( " " ) << 58 << wxT( " " );
	for ( i = 0; i < 10; i++ )
	{
		txt_out << stepsize[ i ] << wxT( " " );
	}
	txt_out << endl;

	txt_out << stylolithe_dissolution << wxT( " " ) << 57 << wxT( " " );
	for ( i = 0; i < 10; i++ )
	{
		txt_out << stylolithe_stress_shrink[ i ] << wxT( " " );
	}
	txt_out << endl;

	txt_out << dissolution_strain << wxT( " " ) << 55 << wxT( " " ) << endl;

	txt_out << shrink_grain_proc << wxT( " " ) << 52 << wxT( " " );
	for ( i = 0; i < 10; i++ )
	{
		txt_out << shrink_grain[ i ] << wxT( " " );
	}
	txt_out << endl;

	txt_out << shrink_box << wxT( " " ) << 53 << wxT( " " );
	for ( i = 0; i < 10; i++ )
	{
		txt_out << box_shrinkage[ i ] << wxT( " " );
	}
	txt_out << endl;

	txt_out << extra_deform << wxT( " " ) << 59 << wxT( " " );
	for ( i = 0; i < 10; i++ )
	{
		txt_out << extra_deformation[ i ] << wxT( " " );
	}
	txt_out << endl;

	txt_out << extra_def_unilateral_proc << wxT( " " ) << 60 << wxT( " " ) << endl;

	txt_out << extra_def_pureshear_proc << wxT( " " ) << 61 << wxT( " " ) << endl;

	txt_out << extra_def_bilateral_proc << wxT( " " ) << 62 << wxT( " " ) << endl;

	// need both in order to set run- and setfunctions
	txt_out << transition_proc << wxT( " " ) << 65 << endl;
	txt_out << transition_proc << wxT( " " ) << 31 << wxT( " " );
	for ( i = 0; i < 10; i++ )
	{
		txt_out << transition[ i ] << wxT( " " );
	}
	txt_out << endl;

	txt_out << transition_heat_proc << wxT( " " ) << 66 << endl;

	txt_out << stress_box_proc << wxT( " " ) << 100 << wxT( " " );
	for ( i = 0; i < 10; i++ )
	{
		txt_out << stress_box[ i ] << wxT( " " );
	}
	txt_out << endl;

	txt_out << stress_grain_proc << wxT( " " ) << 101 << wxT( " " );
	for ( i = 0; i < 10; i++ )
	{
		txt_out << stress_grain[ i ] << wxT( " " );
	}
	txt_out << endl;

	txt_out << stress_two_grains_proc << wxT( " " ) << 102 << wxT( " " );
	for ( i = 0; i < 10; i++ )
	{
		txt_out << stress_two_grains[ i ] << wxT( " " );
	}
	txt_out << endl;

	txt_out << surface_time_proc << wxT( " " ) << 103 << wxT( " " );
	for ( i = 0; i < 10; i++ )
	{
		txt_out << surface_time[ i ] << wxT( " " );
	}
	txt_out << endl;

	txt_out << xstress_proc << wxT( " " ) << 104 << wxT( " " );
	for ( i = 0; i < 10; i++ )
	{
		txt_out << xstress[ i ] << wxT( " " );
	}
	txt_out << endl;

	txt_out << ystress_proc << wxT( " " ) << 105 << wxT( " " );
	for ( i = 0; i < 10; i++ )
	{
		txt_out << ystress[ i ] << wxT( " " );
	}
	txt_out << endl;

	txt_out << surface_proc << wxT( " " ) << 106 << wxT( " " );
	for ( i = 0; i < 10; i++ )
	{
		txt_out << surface[ i ] << wxT( " " );
	}
	txt_out << endl;

	//copy other files in zip-archive
	if ( wxFileExists( archive_out_path ) )
	{
		wxFFileInputStream in( archive_out_path );
		wxZipInputStream zip_in( in );
		
		while ( entry = zip_in.GetNextEntry(), entry != NULL )
		{
			if ( entry->GetName() != wxT( "latte_pref" ) )
			{
				zip_out.CopyEntry( entry, zip_in );
			}
		}
	}

	//close the streams
	zip_out.CloseEntry();
	zip_out.Close();
	out.Close();

	//finally: rename the temporary file to the real archive name
	wxRenameFile( wxT( "tmp.zip" ), archive_out_path );
}


void usdat::SetShrinkage()
{
	CleanData();

	linear_strength_proc = true;
	weaken_all_proc = true;
	plot_frac_proc = true;
	shrink_box = true;
	sin_anisotropy_proc = true;

	sin_anisotropy[ 0 ] = 20.0;
	sin_anisotropy[ 1 ] = 1.2;

	linear_strength[ 0 ] = 0.0;
	linear_strength[ 1 ] = 0.0;
	linear_strength[ 2 ] = 0.1;
	linear_strength[ 3 ] = 1.0;

	weaken_all[ 0 ] = 2.0;
	weaken_all[ 1 ] = 1.0;
	weaken_all[ 2 ] = 1.0;

	plot_frac[ 0 ] = 5.0;
	box_shrinkage[ 0 ] = 0.001;
}

void usdat::SetPhaseTransition()
{
	CleanData();

	int i;

	def_unilateral_proc = true;

	stepsize_proc = true;
	stepsize[ 0 ] = 0.002;

	transition[ 0 ] = 400000.0;   	// activation energy
	transition[ 1 ] = 12000000000.0; 	// pressrue-barrier for the grainboundary migration

	linear_strength_proc = false;
	linear_strength[ 0 ] = 0.0;
	linear_strength[ 1 ] = 0.0;
	linear_strength[ 2 ] = 5000.0;
	linear_strength[ 3 ] = 0.8;

	gauss_spring_proc = false;
	gauss_spring[ 0 ] = 20.0;
	gauss_spring[ 1 ] = 0.6;

	make_gb_proc = true;
	make_gb[ 0 ] = 1.0;
	make_gb[ 1 ] = 0.8;

	gauss_young_proc = false;
	gauss_young[ 0 ] = 1.0;
	gauss_young[ 1 ] = 0.1;

	set_min = true;
	mineral[ 0 ] = 3;

	set_box_size_proc = true;
	set_box_size[ 0 ] = 0.005;

	timestep_proc = true;
	timestep[ 0 ] = 311040000.0;

	transition_proc = true;

	transition_heat_proc = true;

	weaken_all_proc = true;
	weaken_all[ 0 ] = 20.0;
	weaken_all[ 1 ] = 0.0;
	weaken_all[ 2 ] = 5000.0;

}

void usdat::SetHeatFlow()
{
	CleanData();

	int i;

	set_box_size_proc = true;
	set_box_size[ 0 ] = 0.01;

	linear_strength_proc = true;
	linear_strength[ 0 ] = 0.0;
	linear_strength[ 1 ] = 0.0;
	linear_strength[ 2 ] = 5000.0;
	linear_strength[ 3 ] = 0.8;

	timestep_proc = true;
	timestep[ 0 ] = 10.0;

	def_pureshear_proc = true;
	stepsize_proc = true;
	stepsize[ 0 ] = 0.005;

	heat_conduction_proc = true;
	heat_conduction[ 0 ] = 5;
	heat_conduction[ 1 ] = 1500;

}

void usdat::SetGrainGrowth()
{
	CleanData();

	int i;

	add_doubles_proc = true;
	graingrowth_proc = true;

}

void usdat::SetDiffusion()
{
	CleanData();

	int i;
	fluid_lattice_gas_random_proc = true;
	fluid_lattice_gas_random[ 0 ] = 0.01;

	set_fluid_lattice_gas_grain_proc = true;
	set_walls_lattice_gas_proc = false;
	set_fluid_lattice_gas_grain[ 0 ] = 0.7;
	set_fluid_lattice_gas_grain[ 1 ] = 10.0;

	lattice_fluid_gas_percent_proc = true;

}

void usdat::SetFluidFlow()
{
	CleanData();

	int i;

	fluid_lattice_gas_random_proc = true;
	fluid_lattice_gas_random[ 0 ] = 0.03;

	set_walls_lattice_gas_proc = true;
	set_walls_lattice_gas[ 0 ] = 0.9;

	//set_fluid_lattice_gas_grain_proc = true;
}

void usdat::SetFractures()
{
	CleanData();

	linear_strength_proc = true;
	gauss_spring_proc = true;
	make_gb_proc = true;
	plot_frac_proc = true;
	def_pureshear_proc = true;
	extra_def_unilateral_proc = true;
	extra_deform = true;

	linear_strength[ 0 ] = 0.0;
	linear_strength[ 1 ] = 0.0;
	linear_strength[ 2 ] = 2.0;
	linear_strength[ 3 ] = 1.2;

	make_gb[ 0 ] = 1.0;
	make_gb[ 1 ] = 0.5;

	gauss_spring[ 0 ] = 1.0;
	gauss_spring[ 1 ] = 0.5;

	plot_frac[ 0 ] = 50.0;

	stepsize[ 0 ] = 0.001;

	extra_deformation[ 0 ] = 0.001;
	extra_deformation[ 1 ] = 5.0;
}

void usdat::SetInclusions()
{
	CleanData();

	linear_strength_proc = true;
	plot_frac_proc = true;
	shrink_grain_proc = true;

	set_min = false;

	linear_strength[ 0 ] = 0.0;
	linear_strength[ 1 ] = 0.0;
	linear_strength[ 2 ] = 0.8;
	linear_strength[ 3 ] = 1.2;

	plot_frac[ 0 ] = 20.0;

	shrink_grain[ 0 ] = 5.0;
	shrink_grain[ 1 ] = -0.002;
	shrink_grain[ 2 ] = 1.0;
}

void usdat::SetStylolithes()
{
	CleanData();

	linear_strength_proc = true;
	dissolve_y_proc = true;
	def_bilateral_proc = true;
	stylolithe_dissolution = true;
	dis_plot[ 0 ] = 100.0;

	gauss_rate_proc = true;
	gauss_rate[ 0 ] = 2.0;
	gauss_rate[ 1 ] = 0.5;

	timestep_proc = true;
	timestep[ 0 ] = 189216000000.0;

	weaken_all_proc = true;
	weaken_all[ 0 ] = 4.0;
	weaken_all[ 1 ] = 1.0;
	weaken_all[ 2 ] = 1.0;

	dissolve_y[ 0 ] = 0.49;
	dissolve_y[ 1 ] = 0.5;

	set_box_size_proc = true;
	set_box_size[ 0 ] = 0.1;

	set_min = true;
	mineral[ 0 ] = 1.0;
	stepsize[ 0 ] = 0.00005;

	stepsize_proc = true;

	dis_plot[ 0 ] = 100.0;
	stylolithe_stress_shrink[ 0 ] = 0.0;
	stylolithe_stress_shrink[ 1 ] = 0.001;

	linear_strength[ 0 ] = 0.0;
	linear_strength[ 1 ] = 0.0;
	linear_strength[ 2 ] = 500.0;
	linear_strength[ 3 ] = 0.8;

}

void usdat::SetBoudins()
{
	CleanData();

	gauss_strength_proc = true;
	weaken_hor_proc = true;
	weaken_all_proc = true;
	extra_hor_layer_proc = true;

	weaken_hor[ 0 ] = 0.2;
	weaken_hor[ 1 ] = 0.6;
	weaken_hor[ 2 ] = 10.0;
	weaken_hor[ 3 ] = 1.0;
	weaken_hor[ 4 ] = 1.0;

	extra_hor_layer[ 0 ] = 0.9;
	extra_hor_layer[ 1 ] = 0.92;
	extra_hor_layer[ 2 ] = 10.0;
	extra_hor_layer[ 3 ] = 1.0;
	extra_hor_layer[ 4 ] = 1.0;

	def_pureshear_proc = true;

	//   weaken_grain_proc = true;
	//   weaken_grain[0] = 10.0;
	//   weaken_grain[1] = 10.0;
	//   weaken_grain[2] = 1.0;
	//   weaken_grain[3] = 7.0;

	gauss_strength[ 0 ] = 2.0;
	gauss_strength[ 1 ] = 0.8;

	timestep_proc = true;
	viscosity_proc = true;
	timestep[ 0 ] = 1e11;
	viscosity[ 0 ] = timestep[ 0 ];

	weaken_all_proc = true;
	weaken_all[ 0 ] = 0.1;
	weaken_all[ 1 ] = 1.0;
	weaken_all[ 2 ] = 1.0;

	plot_frac_proc = true;
	plot_frac[ 0 ] = 50.0;

	stepsize[ 0 ] = 0.001;
}

void usdat::SetGrooves()
{
	CleanData();

	def_unilateral_proc = true;
	gauss_rate_proc = true;
	weaken_all_proc = true;
	dissolve_x_proc = true;
	set_box_size_proc = true;
	set_min = true;
	linear_strength_proc = true;
	timestep_proc = true;

	dissolution_strain = true;
	dis_plot[ 0 ] = 20.0;

	stepsize[ 0 ] = 0.002;

	linear_strength[ 0 ] = 0.0;
	linear_strength[ 1 ] = 0.0;
	linear_strength[ 2 ] = 500.0;
	linear_strength[ 3 ] = 0.8;

	gauss_rate[ 0 ] = 2.0;
	gauss_rate[ 1 ] = 0.3;

	weaken_all[ 0 ] = 8.0;
	weaken_all[ 1 ] = 1.0;
	weaken_all[ 2 ] = 1.0;

	mineral[ 0 ] = 1.0;

	set_box_size[ 0 ] = 0.0001;

	timestep[ 0 ] = 189216000000.0;

	dissolve_x[ 0 ] = 0.95;
	dissolve_x[ 1 ] = 1.1;
}

void usdat::SetViscousRelax()
{
	CleanData();

	def_pureshear_proc = true;
	stepsize[ 0 ] = 0.001;

	weaken_all_proc = true;
	weaken_all[ 0 ] = 0.8;
	weaken_all[ 1 ] = 1.0;
	weaken_all[ 2 ] = 1.0;

	weaken_grain_proc = true;
	weaken_grain[ 3 ] = 7.0;
	weaken_grain[ 0 ] = 10.0;
	weaken_grain[ 1 ] = 10.0;
	weaken_grain[ 2 ] = 1.0;

	linear_strength_proc = true;
	linear_strength[ 0 ] = 0.0;
	linear_strength[ 1 ] = 0.0;
	linear_strength[ 2 ] = 0.2;
	linear_strength[ 3 ] = 1.2;

	viscosity_proc = true;
	viscosity[ 0 ] = 1e11;

	make_gb_proc = true;
	make_gb[ 0 ] = 1.0;
	make_gb[ 0 ] = 1.0;
}

void usdat::SetGrainGrowthFracture()
{
	CleanData();

	linear_strength_proc = true;
	gauss_spring_proc = true;
	make_gb_proc = true;
	plot_frac_proc = true;
	def_pureshear_proc = true;

	add_doubles_proc = true;
	graingrowth_proc = true;

	linear_strength[ 0 ] = 0.0;
	linear_strength[ 1 ] = 0.0;
	linear_strength[ 2 ] = 200.0;
	linear_strength[ 3 ] = 1.0;

	make_gb[ 0 ] = 1.0;
	make_gb[ 1 ] = 0.5;

	gauss_spring[ 0 ] = 0.5;
	gauss_spring[ 1 ] = 0.5;

	plot_frac[ 0 ] = 1.0;

	stepsize[ 0 ] = 0.001;
}
