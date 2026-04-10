clear;
close all
clc;
%% Create a AVI movie from a set of images:

%% USER INPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Complete path to where the images are stored (WITH "/" at the end):
inpath = 'input_images/';   
outpath = 'movies/';
% Format of image, type 'png','jpg' etc.:
img_format = 'png';   
% Type the name of the desired name for your movie, without ".avi". If you
% do not additionally type a path before, the movie will be stored in the
% directory where this script is stored as well.
movie_name = [outpath 'firn_enjoys_DRX'];  
% Type how many frames per second you wish for your movie, remember: Every
% image will be one frame, so each image will be displayed for 1/fps
% seconds in your movie and the total length of the movie in seconds will 
% be number of images/fps
fps = 10; 

% If not all frames have the same size the frames are placed in the middle
% of a square box, i.e. the video will have a background: Choose the color
% for this background here as RGB value:
bgcolor = [255,255,255];

% Change frame size?? For yes set to 1, for no set to 0 
% Necessary to use if frames are not of an equal size, if not you can
% leave this at 0
change_frame_size = 1; % usually leave at 1, if not knowing what you do

% Use uncompressed AVI (0,1)? If yes, the quality is predefined and highest:
uncompr_avi = 0; 

% Write INFO-txt file at the end (0,1)?
write_info = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 1. Prepare data:
disp('Please wait, preparing the movie ...');
data    = dir(fullfile([inpath '*.' img_format]));
frames  = size(data,1);

% Pre-allocate structure array for frames:

% mov(1:frames) = struct('cdata', [],'colormap',[]);

%% 2. Find largest frame:
% 
% Explanation: Frames must not all have the same size (cf. pure shear
% models), therefore the code is searching for the largest frame size (eith
% in height or width) and uses this for the square box in the video. All
% frames are then put in the middle of this box
if (change_frame_size==1)
    boxsize = 0; % pre-allocate frame array
    for j=1:frames
        boxsize_temp = max(size(uint8(imread([inpath data(j).name]))));
        if boxsize_temp > boxsize
            boxsize = boxsize_temp;
        end
    end
end

%
%% 2. Read every image and store in movie file:
if (uncompr_avi==1)
    writerObj = VideoWriter([movie_name '.avi'],'Uncompressed AVI');
else
    writerObj = VideoWriter(movie_name);
    writerObj.Quality = 100;
end
writerObj.FrameRate = fps;
open(writerObj);
if (change_frame_size==1)
    IMG = uint8(ones(boxsize,boxsize,3)); % Set box initially for pure shear videos
end

for i=1:frames
    
    if (change_frame_size==1)
        % Fill the box again with background color pixels:
        IMG(:,:,1) = bgcolor(1);
        IMG(:,:,2) = bgcolor(2);
        IMG(:,:,3) = bgcolor(3);

        % Put the current frame in the middle of the box:
        [rows,cols,slices] = size(uint8(imread([inpath data(i).name])));


        % Find where to input the frame's matrix in IMG:
        rowstart = floor((boxsize-rows)/2)+1;
        rowend   = rowstart+rows-1;
        colstart = floor((boxsize-cols)/2)+1;
        colend   = colstart+cols-1;

        % Detect if the image is actually greyscale (which would mean 
        % slices==1) choose correct way to read either rgb or greyscale image
        if slices==1
%             disp('now');
            % Input greyscale frame's matrix in IMG at the correct position
            IMG_help =  uint8(imread([inpath data(i).name]));
            IMG(rowstart:rowend,colstart:colend,1) = IMG_help;
            IMG(rowstart:rowend,colstart:colend,2) = IMG_help;
            IMG(rowstart:rowend,colstart:colend,3) = IMG_help;               
        else 
            % Input RGB frame's matrix in IMG at the correct position
            IMG(rowstart:rowend,colstart:colend,:) = ...
                uint8(imread([inpath data(i).name]));
        end
    else
        % Just load the image
        IMG = uint8(imread([inpath data(i).name]));
    end
    % Possibility to resize if user wants that:
%     IMG=imresize(IMG,[size(IMG,1)*2,size(IMG,2)*2]);
    
    frame = im2frame(IMG);
    writeVideo(writerObj,frame);
%     mov(i)=im2frame(IMG);
end
close(writerObj);

%% 3. Store .avi file:
% movie2avi(mov,movie_name,'fps',fps)

%% 4. Write info-file:
if (write_info==1)
    if (uncompr_avi==1)
        dlmwrite([movie_name 'INFO.txt'],['Created with ' num2str(fps) ...
            ' frames per second and uncompressed AVI using VideoWriter'],'delimiter','');
    else
        dlmwrite([movie_name 'INFO.txt'],['Created with ' num2str(fps) ...
            ' frames per second and quality=100 using VideoWriter'],'delimiter','');
    end
end

disp('... Finished');
