elle_exchange: This code for elle_exchange is a code for calculating lattice diffusion and lattice grain boundary chemical exchange in garnet-biotite systems, and was developed by Dal Park, Youngdo Park & Jin-Han Ree at Korea University, Seoul. The original version of this code was used for the publication:

An Elle-based 2-D model for cation exchange reaction between garnet and biotite
Youngdo Park, Dal Park, Lynn Evans, Jin-Han Ree. Jounral of the VIrtual Explorer, Volume 14, in Numerical Modeling of Microstructures Edited By: Daniel Koehn and Anders Malthe-Sorenssen

The code was then generalised somewhat (although there are still hardwired constants for garnet & biotite) by Mark Jessell at Toulouse, so that arbitrary starting microstructures can be used. 

Usage:

elle_exchange -i file_name.elle -s 500 -f 1 -u kappa_dflt node_id attrib_val diffusion_only -n

where 	kappa_dflt is the default lattice diffusion coefficient
		node_id is not currently used
		attrib_val is not currently used
		diffusion_only if set to 1 no lattice-grain boundary exchange takes place, and the code can be used for lattice diffusion only

make_exchnge: I also wrote a simple utility called make_exchnage that creates the input file from an elle file that already has  grains defined as QUARTZ, MICA or GARNET, and already has a unode layer added. At the moment each grain must have at least one unode. 

Usage:

make_exchnge -i filename.elle -u qtz_conc bi_conc gt_conc -s 1 -f 1 -n

where each user defined conc refers to the lattice (unode) concentrations of Fe in each mineral type


A smaple shelle script and input file is included, that reproduces one of the experiemnts from the paper cited above, and includes grain boundary diffusion as part of the experiment
