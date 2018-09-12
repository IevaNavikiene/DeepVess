function [ ] = postProcess(inputFileName, outputFileName, maskFilePath, fullMatFileName)
%post-process the DeepVess result and extract the skeleton of 3D vessels. 
%   The mat file, the output of DeepVess with suffix of 'V_fwd.mat', with 
%   similar name to the input h5 file must be in the same folder. similar to 
%   input h5 file with '-masked.tif' sufix can mask the image. 
%   Output 'Analysis*.mat' will include the postprocessed segementation V and 
%   skeleton information. Each column of output Skel is one vessel segment.
%
%   POSTPROCESS() default value for all parameters will be used, user interface will
%       ask for the input h5 file location.
%   POSTPROCESS(pad_size, skelDilateR, vDilateR, boxFiltW, ...
%       deadEndLimit, diameterLimit, elongationLimit) user interface will
%       ask for the input file location.
%   POSTPROCESS(___, FileName, PathName) fucntion will run with no input
%       request
%
% Parameters
%     pad_size - padding to make room for dilation
%     skelDilateR - to dilate skeleton for smoothness 
%     vDilateR - to dilate segmentation to improve connectivity
%     boxFiltW - to smooth the segmentation and skeleton results
%     deadEndLimit - to remove dead end hairs
%     diameterLimit - to remove large vessels
%     elongationLimit - elongationLimit = Diameter / Length
%     FileName, PathName - input file address, if omitted, user interface 
%       will ask for the file location  
%
% Output analysis.*mat file contents
%   im - raw motion correctged image.
%   V - post-processed segmentation of im using DeepVess
%   Skel - centerlines of each vessel the Skel{1,i} contain [x, y, z] of
%    centerline of vessel i
%   C - image that centerline voxels have the value equalt to the vessel ID    
%
% Example
% ---------
% Using the default parameters and ask for the input file location. 
% postProcess();

% Copyright 2017-2018, Mohammad Haft-Javaherian. (mh973@cornell.edu)

%   References:
%   -----------
%   [1] Haft-Javaherian, M; Fang, L.; Muse, V.; Schaffer, C.B.; Nishimura, 
%       N.; & Sabuncu, M. R. (2018) Deep convolutional neural networks for 
%       segmenting 3D in vivo multiphoton images of vasculature in 
%       Alzheimer disease mouse models. *arXiv preprint, arXiv*:1801.00880.

% default parameters
pad_size = 10; % padding to make room for dilation
skelDilateR = 5; % to dilate skeleton for smoothness
vDilateR = 1; % to dilate segmentation to improve connectivity
boxFiltW = 3; % to smooth the segmentation and skeleton results
deadEndLimit = 11; % to remove dead end hairs
diameterLimit = 25; % to remove large vessels
elongationLimit = 1; % elongationLimit = Diameter / Length

im = uint8(255*(h5read(inputFileName,char("/im"))+0.5));

% read the mask image file if exits
if exist(maskFilePath, 'file')
    mask = readtif(maskFilePath);
else
    mask = ones(size(im));
end

load(fullMatFileName,'V')
% Apply the mask
V= V .* single(mask>0);

% post processing of V
V=imboxfilt3(V,3)>0.5;
V=single(imfill(imboxfilt3(single(V),3)>0.5,'hole'));

% pad array to make space for dilation
V1 = padarray(V, [pad_size, pad_size, pad_size]);
skel = Skeleton3D(V1);

% smoothing
skel = imdilate(skel, strel('sphere', skelDilateR));
skel = imboxfilt3(single(skel), boxFiltW)>0.5;
for i = 1:size(skel, 3)
    skel(:, :, i) = imfill(skel(:, :, i), 'holes');
end
skel = single(imfill(imboxfilt3(single(skel), boxFiltW)>0.5, 'hole'));
skel = Skeleton3D(skel) .* imdilate(V1, strel('sphere', vDilateR));

change = 1;
while change
% remove short dead end vessels
change = skel;
C = convn(skel, ones(3, 3, 3), 'same');
C2 = convn(skel .* (C==2), ones(3, 3, 3),'same') .* skel;
CC1 = skel .* (C==3);
CC0 = bwconncomp(CC1);
for i = 1:CC0.NumObjects
    if numel(CC0.PixelIdxList{i}) < deadEndLimit && ...
            any(C2(CC0.PixelIdxList{i}([1, end])))
        skel(CC0.PixelIdxList{i}) = 0;
    end
end    
% remove single pixel connected to node
C = convn(skel, ones(3, 3, 3),'same');
C2 = convn(C>3, ones(3, 3, 3), 'same');
CC0 = (skel>0) .* (C==2) .* C2;
skel(CC0>0) = 0;
% remove isolated pixel
C = convn(skel, ones(3, 3, 3), 'same');
skel(and(skel, C==1)) = 0;
% remove single pixle loop
C = convn(skel, ones(3, 3, 3), 'same');
C2 = convn(skel .* (C>3), ones(3, 3, 3), 'same') .* skel;
CC1 = skel .* (C==3) .* (C2==2);
for k = find(CC1)'
    CC0 = bwconncomp(skel);
    skel(k) = 0;
    CC1 = bwconncomp(skel);
    if CC0.NumObjects ~= CC1.NumObjects
        skel(k) = 1;
    end
end
% remove double pixle loop
C = convn(skel, ones(3, 3, 3), 'same');
C2 = convn(skel .* (C==2), ones(3, 3, 3), 'same') .* skel;
CC1 = skel .* (C==3);
CC0 = bwconncomp(CC1);
for i=1:CC0.NumObjects
    if numel(CC0.PixelIdxList{i})<3 
        skel(CC0.PixelIdxList{i}) = 0;
    end
end    
change = ~isempty(find(change ~= skel, 1));
end

%remove pads
skel(pad_size+1, :, :) =  any(skel(1:pad_size+1, :, :), 1);
skel(end-pad_size, :, :) =  any(skel((end-pad_size):end, :, :), 1);
skel(:, pad_size+1, :) =  any(skel(:, 1:pad_size+1, :), 2);
skel(:, end-pad_size, :) =  any(skel(:, (end-pad_size):end, :), 2);
skel(:, :, pad_size+1) =  any(skel(:, :, 1:pad_size+1), 3);
skel(:, :, end-pad_size) =  any(skel(:, :, (end-pad_size):end), 3);

skel = skel((pad_size+1):(end-pad_size), (pad_size+1):(end-pad_size), ...
    (pad_size+1):(end-pad_size));
V1 = V1((pad_size+1):(end-pad_size), (pad_size+1):(end-pad_size), ...
    (pad_size+1):(end-pad_size));

% remove rounded object and big vessels
C = convn(skel, ones(3,3,3), 'same');
CC1 = (skel>0) .* (C==3);
CC0 = bwconncomp(CC1);
D = zeros(CC0.NumObjects,1);
L = zeros(CC0.NumObjects,1);
bwDist=bwdist(~V1);
for i=1:CC0.NumObjects
    a = bwDist(CC0.PixelIdxList{i});
    a = 2 * max(a(a>0)) * (sum(a>0)>(0.5*numel(a)));
    if isempty(a)
        a=nan;
    end
    D(i)= a;
    L(i) = numel(CC0.PixelIdxList{i});
    if L(i)<(elongationLimit*D(i)) || D(i)> diameterLimit ...
            || isnan(D(i)) || D(i)==0
        CC1(CC0.PixelIdxList{i}) = 0;
    end
end
CC0 = bwconncomp(CC1);
D = zeros(CC0.NumObjects,1);
L = zeros(CC0.NumObjects,1);
for i=1:CC0.NumObjects
    a = bwDist(CC0.PixelIdxList{i});
    D(i)=2 * median(a(a>0));
    L(i) = numel(CC0.PixelIdxList{i});
    a=CC0.PixelIdxList{i};
    CC0.PixelIdxList{i}=a(V1(a)>0);
end

% Generate the skeleton outputs
Skel=cell(3, CC0.NumObjects);
C = skel;
for i=1:CC0.NumObjects
    C(CC0.PixelIdxList{i}) = i+1;
    [x,y,z] = ind2sub(size(skel), CC0.PixelIdxList{i});
    Skel{1,i} = [x,y,z];
    Skel{2,i} = 5;
    Skel{3,i} = i;
end

% Fix the path of centerlines in Skel{1,:} to have straight centerlines,
%   other wise they will be zig-zagged
for i = 1:size(Skel, 2)
    temp = Skel{1, i};
    if size(temp, 1) < 4
        continue
    end
    [Di, I] = pdist2(temp, temp, 'euclidean', 'Smallest', 3);
    A = zeros(size(temp, 1));
    for j = 1:size(temp, 1)
        for k = 2:3
            if Di(k, j) <= sqrt(3)
                A(j,I(k, j)) = 1;
            end
        end
    end
    A = single((A + A') > 0);
    BGobj = biograph(A);
    dist = allshortestpaths(BGobj, 'Directed', false);
    dist(isinf(dist)) = 0;
    [~, j] = max(dist(:));
    [S, T] = ind2sub(size(A), j(1));
    [~, path, ~] = shortestpath(BGobj, S, T);
    if ~isempty(path)
        Skel{1, i} = temp(path, :);
    end
end

% save results
save(outputFileName, 'im', 'Skel', 'C', 'V')
clear FileName