
#ifndef _cnvt_flags_h
#define _cnvt_flags_h

#ifdef __cplusplus
extern "C" {
#endif
void CnvtInitBoundaryFlagArray();
void CnvtInitCheckFlagArray();
void CnvtSetBoundaryFlag(int index,int val);
void CnvtSetCheckFlag(int index);
void CnvtClearBoundaryFlag(int index);
void CnvtClearCheckFlag(int node);
int CnvtNodeOnBoundary(int node);
void CnvtClearCheckFlagArray();
void CnvtClearBoundaryFlagArray();
int CnvtMaxCheckFlags();
int CnvtMaxBoundaryFlags();
int CnvtReallocateCheckFlagArray(int size);
int CnvtReallocateBoundaryFlagArray(int size);
#ifdef __cplusplus
}
#endif

#endif
