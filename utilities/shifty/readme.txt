This utility adds and subtracts a random vertical displacement (bewteen 0 & 1) to an elle file.

The point is to simulate horizontal periodic boundaries for deformation experiments (vertical boudnaries 
can be periodic in BASIL and OOF in any case).

shifty creates two files "shift_state" and "shift_hist". "shift_state" stores the last shift (0.0 if a "deshift" 
was the last event, non zero otherwise). This file also stores the cumulative simple shear offset. shift_hist just stores
a list of ll the displacements, in case we want to know.

after each call to shifty we also need to call reposition, as in the example shelle script.

