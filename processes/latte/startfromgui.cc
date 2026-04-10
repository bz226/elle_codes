
#include "experiment.h"

#include "wx/wx.h"

#include "wx/string.h"

#include "wx/variant.h"

#include "wx/txtstrm.h"
#include "wx/wfstream.h"
#include "wx/zipstrm.h"
#include "wx/sstream.h"

#include "wx/dir.h"

// called from read_custom_file(), puts data into related structs
void Experiment::read_from_string( wxString data_line )
{
	float argument;
	int exec;
	int fctnb, i;

	//const char* line = (const char*)l.mb_str(wxConvUTF8);

	//cout << data_line.ToAscii() << endl;

	wxStringInputStream sis( data_line );
	wxTextInputStream ist( sis );

	ist >> exec;

	if ( exec )
	{

		ist >> fctnb;

		if ( fctnb < 50 && fctnb >= 0 )
		{
			setdata.exec[ fctnb ] = true;

			i = 0;

			while ( i < 10 )
			{
				ist >> argument;
				setdata.parameters[ fctnb ][ i ] = argument;
				//cout << argument << endl;
				i++;
			}
		}
		else if ( fctnb < 100 && fctnb >= 50 )
		{

			rundata.exec[ fctnb - 50 ] = true;

			i = 0;

			while ( i < 10 )
			{
				ist >> argument;
				rundata.parameters[ fctnb - 50 ][ i ] = argument;
				i++;

			}
		}
		else if ( fctnb < 150 && fctnb >= 100 )
		{
			ist >> argument;
			statdata.exec[ fctnb - 100 ] = true;

			i = 0;

			while ( i < 10 )
			{
				ist >> argument;
				statdata.parameters[ fctnb - 100 ][ i ] = argument;
				i++;

			}
		}
		else if ( fctnb < 200 && fctnb >= 150 )
		{

			externdata.exec[ fctnb - 150 ] = true;

			i = 0;

			while ( i < 10 )
			{
				ist >> argument;
				externdata.parameters[ fctnb - 150 ][ i ] = argument;
				i++;

			}
		}
	}
}

// read custom-file (mike_prefs)

void Experiment::read_custom_file()
{

	wxZipEntry * entry;

	//check if zip-file exists
	wxDir dir( wxGetCwd() );
	wxString filename( file, wxConvUTF8 );

	//filename.Printf(file);

	filename = filename.BeforeLast( '.' ) + wxT( ".zip" );

	cout << filename.ToAscii() << endl;
	if ( filename == wxEmptyString )
		cout << "empty" << endl;

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


	//   char line[50],sign;
	//   string test;
	//   ifstream inputfile("pref");
	//
	//   if (!inputfile.is_open())
	//     {
	//       cout << " There is no mikeprefs-file to be read!" << endl;
	//     }
	//   else
	//     {
	//       while (!inputfile.eof())
	//         {
	//           inputfile.getline(line,50);
	//           test = line[0];
	//
	//           if (test != "#")
	//             {
	//               test = line;
	//               read_from_string(test);
	//             }
	//         }
	//     }
	//
	//   inputfile.close();
}

// set-up function

void Experiment::
SetUpFromFile()
{
	int i, j;

	cout << "You are using a custom-file" << endl;
	cout << "Lattice Version 2.0, 2004/5" << endl;

	read_custom_file();

	Activate_MinTrans();

	//just in case, now its defined
	heat_distribution.SetHeatFlowParameters( 4.2, 1005.0, 0.000001, 1000.0 );

	for ( i = 0; i < 50; i++ )
	{
		if ( setdata.exec[ i ] )
		{
			switch ( i )
			{
					case 0:
					SetPhase( setdata.parameters[ 0 ][ 0 ], setdata.parameters[ 0 ][ 1 ], setdata.parameters[ 0 ][ 2 ], setdata.parameters[ 0 ][ 3 ] );
					break;

					case 1:
					SetGaussianSpringDistribution( setdata.parameters[ 1 ][ 0 ], setdata.parameters[ 1 ][ 1 ] );
					break;

					case 2:
					SetGaussianRateDistribution( setdata.parameters[ 2 ][ 0 ], setdata.parameters[ 2 ][ 1 ] );
					break;

					case 3:
					SetGaussianStrengthDistribution( setdata.parameters[ 3 ][ 0 ], setdata.parameters[ 3 ][ 1 ] );
					break;

					case 4:
					AdjustConstantGrainBoundaries();
					MakeGrainBoundaries( setdata.parameters[ 4 ][ 0 ], setdata.parameters[ 4 ][ 1 ] );
					break;

					case 5:
					SetFracturePlot( int( setdata.parameters[ 5 ][ 0 ] ), 0 );
					break;

					case 6:
					SetWallBoundaries( int( setdata.parameters[ 6 ][ 0 ] ), setdata.parameters[ 6 ][ 1 ] );
					break;

					case 7:
					WeakenAll( setdata.parameters[ 7 ][ 0 ], int( setdata.parameters[ 7 ][ 1 ] ), setdata.parameters[ 7 ][ 2 ] );
					break;

					case 8:
					WeakenHorizontalParticleLayer( setdata.parameters[ 8 ][ 0 ], setdata.parameters[ 8 ][ 1 ], setdata.parameters[ 8 ][ 2 ], setdata.parameters[ 8 ][ 3 ], setdata.parameters[ 8 ][ 4 ] );

					// do it twice, if necessary
					if ( setdata.parameters[ 8 ][ 5 ] != 0.0 )
						WeakenHorizontalParticleLayer( setdata.parameters[ 8 ][ 5 ], setdata.parameters[ 8 ][ 6 ], setdata.parameters[ 8 ][ 7 ], setdata.parameters[ 8 ][ 8 ], setdata.parameters[ 9 ][ 9 ] );
					break;

					case 9:
					ChangeRelaxThreshold( setdata.parameters[ 9 ][ 0 ] );
					break;

					case 10:
					Set_Mineral_Parameters( int( setdata.parameters[ 10 ][ 0 ] ) );
					break;

					case 11:
					Set_Absolute_Box_Size( setdata.parameters[ 11 ][ 0 ] );
					break;

					case 12:
					Set_Time( setdata.parameters[ 12 ][ 0 ], 0 );
					break;

					case 13:
					DissolveXRow( setdata.parameters[ 13 ][ 0 ], setdata.parameters[ 13 ][ 1 ] );
					break;

					case 14:
					DissolveYRow( setdata.parameters[ 14 ][ 0 ], setdata.parameters[ 14 ][ 1 ], true );
					break;

					case 16:
					WeakenHorizontalParticleLayer( setdata.parameters[ 16 ][ 0 ], setdata.parameters[ 16 ][ 1 ], setdata.parameters[ 16 ][ 2 ], setdata.parameters[ 16 ][ 3 ], setdata.parameters[ 16 ][ 4 ] );
					break;

					case 17:
					if ( !rundata.exec[ 15 ] )
						// if not transition-process. unnecessary "if", the
						// other one would do as well.
						SetGaussianYoungDistribution( setdata.parameters[ 17 ][ 0 ], setdata.parameters[ 17 ][ 1 ] );
					else
						// if transition is activated. see above.
						SetGaussianYoungDistribution_2( setdata.parameters[ 17 ][ 0 ], setdata.parameters[ 17 ][ 1 ] );
					break;

					case 18:
					WeakenGrain( int( setdata.parameters[ 18 ][ 3 ] ), setdata.parameters[ 18 ][ 0 ], setdata.parameters[ 18 ][ 1 ], setdata.parameters[ 18 ][ 2 ] );
					break;

					case 20:
					SetFluidLatticeGasRandom( setdata.parameters[ 20 ][ 0 ] );
					break;
					case 21:
					for ( j = 0; j < 100; j++ )
					{
						SetFluidLatticeGasRandomGrain( setdata.parameters[ 21 ][ 0 ], j * int( setdata.parameters[ 21 ][ 1 ] ) );
					}
					break;
					case 22:
					for ( j = 0; j < 500; j++ )
					{
						SetWallsLatticeGas( j * 4 );
					}
					break;
					case 23:
					ElleAddDoubles();
					ElleUpdate();
					break;
					case 25:
					//heat-flow example, workaround: in full
					HeatGrain( int( setdata.parameters[ 25 ][ 1 ] ), int( setdata.parameters[ 25 ][ 0 ] ) );
					heat_distribution.SetHeatFlowParameters( 4.2, 1005.0, 0.000001, 1000.0 ); // rho, c, diffusivity,
					SetHeatLatticeHeatFlowExample();
					Set_Absolute_Box_Size( setdata.parameters[ 11 ][ 0 ] );
					break;
					case 30:
					//SetSinAnisotropy( int( setdata.parameters[ 30 ][ 0 ] ), setdata.parameters[ 30 ][ 1 ] );
					;
					break;
					case 31:
					SetReactions( setdata.parameters[ 31 ][ 0 ], setdata.parameters[ 31 ][ 1 ] );
					break;
			}
		}
	}

	UpdateElle();

}

void Experiment::RunFromFile( int experiment_time )  // i == timestep
{
	int k;

	// at first deform
	for ( k = 0; k < 50; k++ )
	{
		if ( rundata.exec[ k ] )
		{
			switch ( k )
			{
					case 0:
					if ( experiment_time >= rundata.parameters[ 9 ][ 1 ] )
					{
						strain = rundata.parameters[ 8 ][ 0 ];
						DeformLattice( strain, 1 );
					}
					break;
					case 1:
					if ( experiment_time >= rundata.parameters[ 9 ][ 1 ] )
					{
						strain = rundata.parameters[ 8 ][ 0 ];
						DeformLatticePureShear( strain, 1 );
					}
					break;
					case 10:
					if ( experiment_time < rundata.parameters[ 9 ][ 1 ] )
					{
						strain = rundata.parameters[ 9 ][ 0 ];
						DeformLattice( strain, 1 );
					}
					break;
					case 11:
					if ( experiment_time < rundata.parameters[ 9 ][ 1 ] )
					{
						strain = rundata.parameters[ 9 ][ 0 ];
						DeformLatticePureShear( strain, 1 );
					}
					break;
					case 12:
					if ( experiment_time < rundata.parameters[ 9 ][ 1 ] )
					{
						strain = rundata.parameters[ 9 ][ 0 ];
						DeformLatticeNewAverage2side( strain, 1 );
					}
					break;
					case 6:
					if ( experiment_time >= rundata.parameters[ 9 ][ 1 ] )
					{
						strain = rundata.parameters[ 8 ][ 0 ];
						DeformLatticeNewAverage2side( strain, 1 );
					}
					break;
			}
		}
	}

	//now do the things after the deformation
	for ( k = 0; k < 50; k++ )
	{
		if ( rundata.exec[ k ] )
		{
			switch ( k )
			{
					case 2:
					ShrinkGrain( int( rundata.parameters[ 2 ][ 0 ] ), rundata.parameters[ 2 ][ 1 ], 1 );
					break;
					case 3:
					ShrinkBox( rundata.parameters[ 3 ][ 0 ], 1, 0 );
					break;
					case 4:
					ViscousRelax( 1, rundata.parameters[ 4 ][ 0 ] );
					break;
					case 5:
					Dissolution_Strain( int( rundata.parameters[ 28 ][ 0 ] ) );
					break;
					case 7:
					Dissolution_StylosII( int( rundata.parameters[ 28 ][ 0 ] ), 0,int( rundata.parameters[ 7 ][ 0 ] ), int( rundata.parameters[ 7 ][ 1 ] ),1 );
					break;
					case 16:
					heat_distribution.SetHeatFlowEnabled( 1 );
					heat_distribution.SetHeatFlowParameters( 4.2, 1005.0, 0.000001, 1000.0 );
					break;
					case 15:
					Start_Reactions();
					break;
					case 20:
					UpdateFluidLatticeGas();
					break;
					case 21:
					InsertFluidLatticeGas( 0.03, rundata.parameters[ 21 ][ 0 ] );
					UpdateFluidLatticeGas();
					RemoveFluidLatticeGas( 0.97 );
					break;
					case 25:
					//heat-flow example, workaround: in full
					Heat_Flow( setdata.parameters[ 12 ][ 0 ] );
					Read_Heat_Lattice();
					break;
			}
		}
	}

	// statistic functions
	for ( k = 0; k < 50; k++ )
	{
		if ( statdata.exec[ k ] )
		{
			switch ( k )
			{
					case 0:
					DumpStatisticStressBox( statdata.parameters[ 0 ][ 0 ], statdata.parameters[ 0 ][ 1 ], statdata.parameters[ 0 ][ 2 ], statdata.parameters[ 0 ][ 3 ], strain );
					break;
					case 1:
					DumpStatisticStressGrain( strain, int( statdata.parameters[ 1 ][ 0 ] ) );
					break;
					case 2:
					DumpStatisticStressTwoGrains( strain, int( statdata.parameters[ 2 ][ 0 ] ), int( statdata.parameters[ 2 ][ 1 ] ) );
					break;
					case 3:
					DumpTimeStatisticSurface( strain, ( statdata.parameters[ 3 ][ 0 ] ), ( statdata.parameters[ 3 ][ 1 ] ) );
					break;
					case 4:
					DumpStressRow( statdata.parameters[ 0 ][ 0 ], statdata.parameters[ 0 ][ 1 ], statdata.parameters[ 0 ][ 2 ], statdata.parameters[ 0 ][ 3 ], int( statdata.parameters[ 0 ][ 4 ] ) );
					break;
					case 5:
					DumpYStressRow( statdata.parameters[ 0 ][ 0 ], statdata.parameters[ 0 ][ 1 ], statdata.parameters[ 0 ][ 2 ], statdata.parameters[ 0 ][ 3 ], int( statdata.parameters[ 0 ][ 4 ] ) );
					break;
					case 6:
					DumpStatisticSurface( strain );
					break;
			}
		}
	}

	// external functions
	for ( k = 0; k < 50; k++ )
	{
		if ( externdata.exec[ k ] )
		{
			switch ( k )
			{
					case 0:
					DoGrowth( experiment_time );
					GetNewElleStructure();
					break;
			}
		}
	}

	//UpdateElle();
}
