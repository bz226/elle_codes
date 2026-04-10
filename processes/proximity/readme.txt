This process calculates a proximity map for each flynn with an expand attribute set, and creates an elle file storing this info in CONC_A for each node, and also creates an output elle file with the N_ATTRIB_A node value giving the sum of all the maps. At the moment the maps are calculated by setting the boundary nodes adjacent to each expand grain in turn to 1 each time step, and calculating grain boundary diffusion until every node is non-zero. ANother approach wouldd be to set the last non-zero node to be a sink and run until steady state is reached. At the moment no normalisation of the proximiaty maps is made wrt to original grain size and this should probably be done.

This code works in conjunction with the elle_nucleation code to produce anti-clustered "garnets"

mwj 18/4/2004
