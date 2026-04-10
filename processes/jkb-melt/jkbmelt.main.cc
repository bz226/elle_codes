#include "jkbmelt.h"

/*!Usage: -u mobility, magic number, melt equilibrium, xlxl-energy,
 * xlliq-energy,liqliq-energy,melt increment
 * if the magic number (fudge factor) is negative, it will not be altered during the simulations,
 * otherwise it will be adjusted to keep the melt fraction stable.
 * 
* if melt increment is negative, the melt fraction decreases.*/
main( int argc, char * * argv )
{
  int err = 0;
  char buf[4096];
  UserData userdata;
  extern int InitMelt( void );
  /* * initialise */
  ElleInit();
  userdata[Mobility] = 1e-11; // mobility
  userdata[Magic] = 0.01; // magic number (area fudge factor)
  userdata[EquilMeltFrac] = 0.1; // equilibrium melt fraction (not currently used)
  userdata[XlXlEnergy] = 1; // xl-xl interface energy
  userdata[LiqXlEnergy] = 1; // liq-xl interface energy
  userdata[LiqLiqEnergy] = 0; // liq-liq interface energy
  userdata[IncMeltfraction] = 0; // incremental melt fraction change
  ElleSetUserData( userdata );
  if ( err = ParseOptions( argc, argv ) )
    OnError( "", err );
  /* * set the function to the one in your process file */
  ElleUserData( userdata );
  sprintf( buf, "mob\t\tmagic\t\tpercent\t\txl-xl\t\tliq-xl\t\tliq-liq\tinc\tchange\n%le\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n",
  userdata[Mobility], userdata[Magic], userdata[EquilMeltFrac], userdata[XlXlEnergy], userdata[LiqXlEnergy],
       userdata[LiqLiqEnergy], userdata[IncMeltfraction] );
  //Log( 0, buf );
  ElleSetInitFunction( InitMelt );
  /* * set up the X window */
  ElleSetSaveFileRoot( "melt" );
  if ( ElleDisplay() ) SetupApp( argc, argv );
  StartApp();
  return ( 0 );
}
