clear;
close all;
clc;

% MATLAB script by Florian Steinbach. Please keep this header and contact
% florian.steinbach@uni-tuebingen or FlorianSteinbach@gmx.de before
% redistribution.

%% INFO
%
% Make an image with unodes plotted in CPO, strain rate or whatever look
% nicer 
%
% For this, prepare images with unodes and another set of images with
% bubbles and grain boundaries (gbs). Plot the latter ones from Elle file
% upscaled by a factor of 4. 
%
% The final scale of files (pixels x pixels) will be the one of the larger
% grain boundary and phase file.
%
% Copy all the images in the images_gbs_bubbles and images_unodes folders
%
% There needs the be an equal amount of images in each folder
% All images should be pngs
%
% IN SHORT: EVERY IMAGE IN FOLDER1 WILL BE SUPERIMPOSED ON EVERY IMAGE IN
% FOLDER2
%
%% INPUT % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
%
inpath_gbs_bubbles = 'images_gbs_bubbles/'; % with "/" at the end
inpath_unodes = 'images_unodes/'; % with "/" at the end
inpath_nativesize = 'native_sized_images/'; % with "/" at the end, only to read image size
outpath = 'final_images/'; % with "/" at the end

outroot = 'F20_final'; % without 3 digit step number and '.png'
                       % FOR UNODE IMAGE NAME: Leave empty ('')

img_format = '.png'; 
final_scale_fact = 1; % Leave at 1 for no scaling
% If you import images that have been scaled to a size of 1x1 for plotting, 
% but are actually rectangular due to pure shear.
% If this is active, you need to have the native rectangular images in a 
% separate folder called "native_sized_images". This is only used to read
% native size and scale the images back to this size
%
scaleback2pureshearbox=1; % Set to 1 to use it, to 0 to switch it off (for e.g. simple shear)
%
% Expand black pixels in GB and bubble image by N pixels by setting the 
% variable "expand" to N. Set it to 0 not to use this option.
% Note: N can be a real number (decimal place allowed) as it is actually
% the distance from nearest black pixel in an euclidean distance map.
expand = 1; % if not zero, min value is 1.0
%
% Scale the bubble and boundary image by the following factor to improve 
% final image quality (usually increase the size (scale factor >1) improves
% the quality. Set to 1 to switch this option off
scale_gb_image = 2;
%
% Sometimes, e.g. in case of ugrid images, it can be good, to first crop 
% the image you want to use as unodes image: Here, please type the ranges 
% for cropping as row_start, row_end, col_start, col_end
crop_unode_image = 0; % type 1 for yes, 0 for no, usually leave at 0
rowstart = 2074; rowend = 3269;
colstart = 31; colend = 2459;
scalebackto1x1 = 0; % usually leave at zero...only for special cases
height=0;
if scalebackto1x1==0
    height = 1200;
end
%
%
% Sometimes, e.g. in case of ugrid images, we need to transfer the input
% image in uint8, type 1 to do that, 0 not to do that (usually can be 0)
%
transfer_unode_image_uint8 = 0;
%
%% END OF INPUT % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
%
%
%
%
%% PREPARATIONS
%
cd (outpath);
if (ispc) 
    system('del *.png'); 
end
if (isunix) 
    system('rm *.png *~'); 
end
cd ..;
%
data_gbs_bubbles    = dir(fullfile([inpath_gbs_bubbles '*' img_format]));
data_unodes    = dir(fullfile([inpath_unodes '*' img_format]));
if scaleback2pureshearbox==1
    data_nativesize    = dir(fullfile([inpath_nativesize '*' img_format]));
end
n_images  = size(data_gbs_bubbles,1);
if size(data_unodes,1)~=n_images
    disp('ERROR: Two input folders should contain an equal amount of images');
    break;
end
%
%% CALCULATIONS
%
for i=1:n_images
    disp(['Preparing image ' num2str(i) ' of ' num2str(n_images)]);
    
    %% Prepare GB and bubble image
    IMG_gbs = imread([inpath_gbs_bubbles data_gbs_bubbles(i).name]);
    if (size(IMG_gbs,3)>1) 
        % make image grayscale before transferring to logical array
        IMG_gbs = rgb2gray(IMG_gbs);
    end
    IMG_gbs = logical(IMG_gbs);
    
    % Resize bubble image to increase final image quality
    if (scale_gb_image~=1)
        IMG_gbs=imresize(IMG_gbs,scale_gb_image);
    end
    % expand black pixels by N pixels
    if (expand>0)
        EDM=bwdist(~IMG_gbs,'euclidean');
        IMG_gbs(EDM<=expand)=0;
    end
    
    
    %% Prepare unode image and join images
    IMG_unodes = imread([inpath_unodes data_unodes(i).name]);
    
    if transfer_unode_image_uint8==1
        IMG_unodes = uint8( (IMG_unodes./65535).*255 );
    end
    
    if crop_unode_image==1
        IMG_unodes = IMG_unodes(rowstart:rowend,colstart:colend,:);
    end
    
    if scalebackto1x1==1
        IMG_unodes = imresize(IMG_unodes,[height,height]);
    end

    if scaleback2pureshearbox==1
        native_size = zeros(1,2);
        native_size(1) = size(imread([inpath_nativesize data_nativesize(i).name]),1);  
        native_size(2) = size(imread([inpath_nativesize data_nativesize(i).name]),2);       
    end
    unode_image_size = zeros(1,2);
    unode_image_size(1) = size(IMG_unodes,1);
    unode_image_size(2) = size(IMG_unodes,2);
    IMG_unodes = imresize(IMG_unodes,size(IMG_gbs));
    
    IMG_out = IMG_unodes;
    IMG_unodes=0;
    IMG_out(:,:,1) = IMG_out(:,:,1).*uint8(IMG_gbs);
    IMG_out(:,:,2) = IMG_out(:,:,2).*uint8(IMG_gbs);
    IMG_out(:,:,3) = IMG_out(:,:,3).*uint8(IMG_gbs);
    IMG_out = imresize(IMG_out,unode_image_size);
    
    % Scale to native size, if image was square, but actually should be
    % rectangular:
    if scaleback2pureshearbox==1
        IMG_out = imresize(IMG_out,native_size);
    end
    
    % Scale up/down image:
    IMG_out = imresize(IMG_out,final_scale_fact);
    if (strcmp(outroot,''))
        real_outroot=data_unodes(i).name;
    else
        real_outroot=outroot;
    end
    imwrite(IMG_out,[outpath real_outroot '_DD_' sprintf('%03d',i) img_format]);  

    IMG_gbs = 0;
    IMG_out = 0;
end
%
clear;
disp('Finished ...');