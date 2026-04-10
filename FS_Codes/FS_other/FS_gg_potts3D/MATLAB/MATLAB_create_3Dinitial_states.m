clear;
close all;
clc;

points = 2;
dimension = 3;
minstate = 0;
maxstate = 255;

state_distribution = zeros(size((0:1:(points^dimension)-1)'));
ibefore=1;

for i=points:points-1:((points^dimension))
    state_distribution(ibefore:i) = unifrnd(minstate,maxstate,points,1);
	ibefore=i;
end

outfile = fopen(['states_dim' sprintf('%03d',points) '.txt'],'w');
for j=0:1:(points^dimension)-1
    fprintf(outfile,'%u %u\n',j,round(state_distribution(j+1)));
end
fclose(outfile);
