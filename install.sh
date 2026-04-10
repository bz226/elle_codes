#!/bin/sh

INVALID=0
BATCH=0
DSPWX=0
DSPX=0
HELP=0

if [ $# -eq 0 ]
then
	HELP=1
elif [ $# -gt 2 ]
then
	INVALID=1; OPT=$3 ;
elif [ $# -ge 1 ]
then
	if [ $1 != "help" -a $1 = "-h" -a $1 = "--help" -a $1 != "batch" -a $1 != "all"  -a $1 != "x" -a $1 != "X" -a $1 != "wx" -a $1 != "WX" ]
	then
		INVALID=1
		OPT=$1
	elif [ $1 = "help" -o $1 = "-h" -o $1 = "--help" ]
	then
		HELP=1
	elif [  $1 = "batch" ]
	then
		BATCH=1
		SETBIN="binb"
	elif [  $1 = "x" -o $1 = "X" ]
	then
		DSPX=1
		SETBIN="binx"
	elif [  $1 = "wx" -o $1 = "WX" ]
	then
		DSPWX=1
		SETBIN="binwx"
	elif [  $1 = "all" ]
	then
		BATCH=1
		DSPWX=1
		DSPX=1
		SETBIN="binx"
	fi
fi

if [ $# -eq 2 ]
then
	if [ $2 != "batch" -a $2 != "x" -a $2 != "X" -a $2 != "wx" -a $2 != "WX" ]
	then
		INVALID=1
		OPT=$2
	elif [ $2 = "help" -o $2 = "-h" -o $2 = "--help" ]
	then
		HELP=1
	elif [  $2 = "batch" ]
	then
		BATCH=1
		SETBIN="binb"
	elif [  $2 = "x" -o $2 = "X" ]
	then
		DSPX=1
		SETBIN="binx"
	elif [  $2 = "wx" -o $2 = "WX" ]
	then
		DSPWX=1
		SETBIN="binwx"
	fi
fi

if [ $HELP -eq 1 -o $INVALID -eq 1 ]
then
	echo
	echo 'Usage: '$0' OPTION1 [OPTION2]' 1>&2;
	echo
	echo $0 'with no options prints a list of the allowed options' 1>&2;
	echo
    echo 'OPTION1   batch | WX | X | all | help' 1>&2;
    echo 'OPTION2   batch | WX | X' 1>&2;
    echo
    echo 'help      prints this message' 1>&2;
    echo
    echo 'batch     build only the batch versions of the executables ' 1>&2;
    echo '            which will not display to the screen' 1>&2;
    echo
    echo 'WX        build the executables which display to the' 1>&2;
    echo '            screen using wx and gl libraries' 1>&2;
    echo
    echo 'X         build the executables which display to the' 1>&2;
    echo '            screen using X11 and Motif libraries' 1>&2;
    echo
    echo 'all       build all three versions' 1>&2;
    echo
    echo 'elle/bin will be linked to the last option or the' 1>&2;
    echo '	X11/Motif executables if the option is "all"' 1>&2;
	exit 2;
fi

#
# link to latest gpc
#
if ! [ -e utilities/gpc ]
then
# ln doesn't work in Win32 so have just copied the latest gpc
# to gpc/
#	if [ -d utilities/gpc231 ]
#	then
#        ( cd utilities; ln -sv gpc231 gpc )
#        if [ -e utilities/gpc ]
#        then
#        	echo "linked gpc to gpc231"
#		else
#            echo "linking gpc to gpc231 failed"
#            exit 1
#        fi
#    elif [ -d utilities/gpc222 ]
#    then
#        ( cd utilities; ln -sv gpc222 gpc )
#        if [ -e utilities/gpc ]
#        then
#            echo "linked gpc to gpc222"
#        else
#            echo "linking gpc to gpc222 failed"
#            exit 1
#        fi
#	else
#		echo "no gpc code found"
#  		exit 1
#	fi
	echo "no gpc code found"
  	exit 1
fi

#
# build the makefiles
#
xmkmf
if [ $? != 0 ]
then
  exit 1
fi
make Makefiles
if [ $? != 0 ]
then
  exit 1
fi

#
# install base libraries used by all versions
#
make install_base
if [ $? != 0 ]
then
  exit 1
fi
#
# install batch versions if flagged
#
if [  $BATCH -eq 1 ]
then
  make install_b
fi

#
# install graphics versions
#

if [  $DSPWX -eq 1 ]
then
  make install_wx
  if [ $? != 0 ]
  then
    exit 1
  fi
fi

if [  $DSPX -eq 1 ]
then
  make install_x
  if [ $? != 0 ]
  then
    exit 1
  fi
fi

if [ $? != 0 ]
then
  exit 1
fi

if [ "$SETBIN" -a -d $SETBIN ]
then
    if [ -f bin ]
    then
        /bin/rm  -f bin
    fi
    ln -s $SETBIN bin
fi

make depend

make clean
