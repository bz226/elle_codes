#define ELLE_KEY "ELLE"
#define AREA_KEY "AREA"
#define QUALITY_KEY "QUALITY"
#define VISCOSITY_KEY "VISCOSITY"
#define FLYNN_REAL_ATTRIB_KEY "F_ATTRIB_A"
#define DEFAULT "Default"

/*
 * arbitrary minimum area for triangulation
 * smaller values can cause basil to fail array allocation
 */
#define AREA_MIN 0.00008

#ifdef __cplusplus
extern "C" {
#endif
int attribute_set(char *infile,const char *key,int max,
                  double *rgnattribs,double *dflt);
#ifdef __cplusplus
}
#endif
