/*----------------------------------------------------------------
 *    Elle:   string_utils.h  1.0  11 September 1997
 *
 *    Copyright (c) 1997 by L.A. Evans & T.D. Barr
 *----------------------------------------------------------------*/
#ifndef _E_string_utils_h
#define _E_string_utils_h
typedef struct { char *name; int id; } valid_terms;

#ifdef __cplusplus
extern "C" {
#endif

#define NumElements(arr)  ((int) (sizeof(arr)/sizeof(arr[0])))

/*int validate(char *str,int *key,valid_terms keywords[]);*/
int validate(char *str,int *key,valid_terms *keywords);
int id_match(valid_terms set[],int id,char *name);
int name_match( char *str,valid_terms *set );
#ifdef __cplusplus
}
#endif
#endif
