 /*****************************************************
 * Copyright: (c) Dr J.K. Becker
 * File:      $RCSfile$
 * Revision:  $Revision$
 * Date:      $Date$
 * Author:    $Author$
 *
 ******************************************************/
/*!
	\file		gz_utils.h
	\brief		header for functions used in r/w of zip files
	\par		Description:
                function declarations
*/
#if !defined(_E_gz_utils_h)
#define _E_gz_utils_h
/*************************************************************
 *	INCLUDE FILES
 */
#include <zlib.h>
#include <string>
/*************************************************************
 *	CONSTANT DEFINITIONS
 */
/*************************************************************
 *	MACRO DEFINITIONS
 */
/************************************************************
 *	ENUMERATED DATA TYPES
 */
/*************************************************************
 *	STRUCTURE DEFINITIONS
 */
/*************************************************************
 *	IN-LINE FUNCTION DEFINITIONS
 */
/*************************************************************
 *	CLASS DECLARATIONS
 */
/*************************************************************
 *	EXTERNAL DATA DECLARATIONS
 */
/*************************************************************
 *	EXTERNAL FUNCTION PROTOTYPES
 */
std::string gzReadLineSTD(gzFile in);
std::string gzReadSingleString(gzFile in);
#endif	// _E_gz_utils_h
