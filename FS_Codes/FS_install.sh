#!/bin/bash
#
# Script to install FFT codes and utilities. Remember to put this file and all the directories in elle/elle/ to make it work
#
DEBUG=
#DEBUG=echo

# Elle based codes
$DEBUG rm Makefile *~ 
$DEBUG xmkmf
$DEBUG make install_wx

$DEBUG make clean

# FFT codes
$DEBUG cd fft/version128
$DEBUG gfortran -O3 FS_PPC15E-9.05.FOR -o FFT_vs128
$DEBUG cp FFT_vs128 ../../../binwx/
$DEBUG cd ../version256
$DEBUG gfortran -O3 FS_PPC15E-9.05.FOR -o FFT_vs256
$DEBUG cp FFT_vs256 ../../../binwx/

$DEBUG cd ../..

$DEBUG make clean

echo " "
echo "Finished ..."
echo " "
