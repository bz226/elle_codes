clear;
close all;
clc;

% INPUT
dim =128; % number of unodes in x- or y-direction
minstate = 0;
maxstate = 255;
outfile_name = 'states_RGB_dim'; % after the name, the dimension in 3 digits will be added

state_distribution = zeros(size((0:1:(dim^2)-1)'));
ibefore=1;

for i=dim:dim-1:((dim^2))
    state_distribution(ibefore:i) = unifrnd(minstate,maxstate,dim,1);
	ibefore=i;
end

outfile = fopen([outfile_name sprintf('%03d',dim) '.txt'],'w');
for j=0:1:(dim^2)-1
    fprintf(outfile,'%u %u\n',j,round(state_distribution(j+1)));
end
fclose(outfile);
