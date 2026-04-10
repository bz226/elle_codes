clear;
% close all;
clc;

%% INPUT
filename = '3DPotts_demo.txt';
dim = 64;
plot_3D_slice_number = 1; % Plot the n-th the slice through the 3D dataset

%% END OF INPUT

[id,state] = import_3Dpotts_result(filename);

STATE=reshape(state(1:dim^3),dim,dim,dim);

imagesc(flip(STATE(:,:,plot_3D_slice_number)',1));
axis square;