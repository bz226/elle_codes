#include "FS_scalefile.h"
using namespace std;

int main(int argc, char **argv)
{
    int err=0;
    UserData userdata;
    /*
     * Initialise and set option names
     */
    ElleInit();
    ElleSetOptNames("ScaleX","ScaleY","unused","unused","unused","unused","unused","unused","unused");
    /* 
     * Give default values for userdata:
     */
    ElleUserData(userdata);
    userdata[UScaleX]=1; // Default: 1
    userdata[UScaleY]=1; // Default: 1
    ElleSetUserData(userdata);
    
    if (err=ParseOptions(argc,argv)) OnError("",err);
    /*
     * Eventually set the interval for writing file, stages etc. if e.g.
	 * this utility should only run once etc.
     */
    //ElleSetSaveFrequency(1);
    //ElleSetStages(1);
    /*
     * Set the function to the one in your process file
     */
    ElleSetInitFunction(InitThisProcess);	
	/*
     * Set the default output name for elle files
     */
    char cFileroot[] = "scaled_file.elle";
    ElleSetSaveFileRoot(cFileroot);
    /*
     * Set up the X window
     */
    if (ElleDisplay()) SetupApp(argc,argv);
    /*
     * Run your initialisation function and start the application
     */
    StartApp();

    CleanUp();

    return(0);
} 


/*
 * this function will be run when the application starts,
 * when an elle file is opened or
 * if the user chooses the "Rerun" option
 */
int InitThisProcess()
{
    char *infile;
    int err=0;
    /*
     * Clear the data structures
     */
    ElleSetRunFunction(ProcessFunction);
    /*
     * Read the data
     */
    infile = ElleFile();
    if (strlen(infile)>0) 
    {
        if (err=ElleReadData(infile)) OnError(infile,err);
    }
    /*
     * Now that the Elle file is loaded, the user code could potentially check 
     * if e.g. unodes are in the file in case they are necessary or check for
     * attributes in flynns, bnodes or unodes that are needed in this code
     */
    /*! EXAMPLE:
    if(!ElleUnodesActive()) 
    {
        printf("Error: No unodes in file\n\n");  
        return 0;
    } 
    */
     
}

/* 
 * Anything can now be in the ProcessFunction itself:
 */ 
int ProcessFunction()
{
    int err=0;
    /*
     * Read the input data and store it in the array of type "Userdata" called
     * "userdata":
     */
    UserData userdata;              // Initialize the "userdata" array
    ElleUserData(userdata);         // Load the input data
    double dScaleX = userdata[UScaleX];  
    double dScaleY = userdata[UScaleY];  
    
    if (dScaleX>-999)
        ScaleModelBox(dScaleX,dScaleY);
    else
        Scale2OneByOne(dScaleY);

    err=ElleWriteData(ElleSaveFileRoot());
    if(err) OnError("",err);
    
    return 0;
}

void ScaleModelBox(double dScaleX,double dScaleY)
{
    printf("Scaling the box by a factors X,Y: %f,%f\n",dScaleX,dScaleY);
    // Upate unit cell:
    CellData unitcell;
    ElleCellBBox(&unitcell);
    Coords  corners[4];
    
    corners[BASELEFT].x = unitcell.cellBBox[BASELEFT].x*dScaleX;
    corners[BASELEFT].y = unitcell.cellBBox[BASELEFT].y*dScaleY;
    corners[BASERIGHT].x = unitcell.cellBBox[BASERIGHT].x*dScaleX;
    corners[BASERIGHT].y = unitcell.cellBBox[BASERIGHT].y*dScaleY;
    corners[TOPRIGHT].x = unitcell.cellBBox[TOPRIGHT].x*dScaleX;
    corners[TOPRIGHT].y = unitcell.cellBBox[TOPRIGHT].y*dScaleY;
    corners[TOPLEFT].x = unitcell.cellBBox[TOPLEFT].x*dScaleX;
    corners[TOPLEFT].y = unitcell.cellBBox[TOPRIGHT].y*dScaleY;
    
    ElleSetCellBBox(&corners[BASELEFT], &corners[BASERIGHT],&corners[TOPRIGHT], &corners[TOPLEFT]);
    
    // Update bnode positions
    Coords cBnodeXY;
    
    for (int i=0;i<ElleMaxNodes();i++)
    {
        if (ElleNodeIsActive(i))
        {
            ElleNodePosition(i,&cBnodeXY);
            cBnodeXY.x *= dScaleX;    
            cBnodeXY.y *= dScaleY; 
            ElleSetPosition(i,&cBnodeXY);        
        }
    }
    
    // Update unode positions
    if (ElleUnodesActive())
    {
        Coords cUnodeXY;
        
        for (int i=0;i<ElleMaxUnodes();i++)
        {
            ElleGetUnodePosition(i,&cUnodeXY);
            cUnodeXY.x *= dScaleX;
            cUnodeXY.y *= dScaleY; 
            ElleSetUnodePosition(i,&cUnodeXY);        
        }
    }
}

void Scale2OneByOne(double dUpscale)
{
    /* 
     * Scaling the model box back to 1x1 if user input 0 is < -999
     * In this case user input 1 is used as upscale factor, so box will
     * actually be dUpscale X dUpscale
     */
    CellData unitcell;
    ElleCellBBox(&unitcell);
    Coords  corners[4];
    
    double dWidth = 0.0,dHeight = 0.0;
    // Assuming a square or rectangular box, no parallelogram etc...
    dHeight = unitcell.cellBBox[TOPLEFT].y - unitcell.cellBBox[BASELEFT].y;
    dWidth = unitcell.cellBBox[BASERIGHT].x - unitcell.cellBBox[BASELEFT].x;
    
    double dScaleX = dUpscale / dWidth;
    double dScaleY = dUpscale / dHeight;
    
    printf("Scaling box back to 1x1:\n");
    printf("Upscaling by factor %e\n",dUpscale);
    printf("Using scale factors x,y = %e %e\n",dScaleX,dScaleY);
    
    corners[BASELEFT].x = unitcell.cellBBox[BASELEFT].x*dScaleX;
    corners[BASELEFT].y = unitcell.cellBBox[BASELEFT].y*dScaleY;
    corners[BASERIGHT].x = unitcell.cellBBox[BASERIGHT].x*dScaleX;
    corners[BASERIGHT].y = unitcell.cellBBox[BASERIGHT].y*dScaleY;
    corners[TOPRIGHT].x = unitcell.cellBBox[TOPRIGHT].x*dScaleX;
    corners[TOPRIGHT].y = unitcell.cellBBox[TOPRIGHT].y*dScaleY;
    corners[TOPLEFT].x = unitcell.cellBBox[TOPLEFT].x*dScaleX;
    corners[TOPLEFT].y = unitcell.cellBBox[TOPRIGHT].y*dScaleY;
    
    ElleSetCellBBox(&corners[BASELEFT], &corners[BASERIGHT],&corners[TOPRIGHT], &corners[TOPLEFT]);
    
    // Update bnode positions
    Coords cBnodeXY;
    
    for (int i=0;i<ElleMaxNodes();i++)
    {
        if (ElleNodeIsActive(i))
        {
            ElleNodePosition(i,&cBnodeXY);
            cBnodeXY.x *= dScaleX;    
            cBnodeXY.y *= dScaleY; 
            ElleSetPosition(i,&cBnodeXY);        
        }
    }
    
    // Update unode positions
    if (ElleUnodesActive())
    {
        Coords cUnodeXY;
        
        for (int i=0;i<ElleMaxUnodes();i++)
        {
            ElleGetUnodePosition(i,&cUnodeXY);
            cUnodeXY.x *= dScaleX;
            cUnodeXY.y *= dScaleY; 
            ElleSetUnodePosition(i,&cUnodeXY);        
        }
    }
}
