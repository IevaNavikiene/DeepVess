
function prepareImage(inPath, inFile)
% extract the vessel channel of a stack, normalize it and save it as 8 bit 
% image, then remove the motion artifact and save the result as h5 file. 
%
% Parameters
%     inPath - input path to folder of file
%
% Example
% ---------
% user interface will ask for a single tif filethat has four channel and 
%   the first channel is the vessel channel and start the image from 
%   slice 10. h5 file with similar name will be writen to dame folder. 
%
% prepareImage(10, 0, 1, 4); 

% Copyright 2017-2018, Mohammad Haft-Javaherian. (mh973@cornell.edu)

%   References:
%   -----------
%   [1] Haft-Javaherian, M; Fang, L.; Muse, V.; Schaffer, C.B.; Nishimura, 
%       N.; & Sabuncu, M. R. (2018) Deep convolutional neural networks for 
%       segmenting 3D in vivo multiphoton images of vasculature in 
%       Alzheimer disease mouse models. *arXiv preprint, arXiv*:1801.00880.

% extract the file addresses
f(1).name = inFile;
f(1).folder = inPath;
for i=1:numel(f)
    inFile = [f(i).folder, '/', f(i).name];
    outFile = [f(i).folder, '/', 'Ch4-8bit-', f(i).name];
    h5FileName = [f(i).folder, '/', 'noMotion-', 'Ch4-8bit-', ...
        f(i).name(1:end-3), 'h5'];
    % read multipage tif file
    im = readtif(inFile);
    
    % resize image
    [x,y,z] = size(im);
    im = imresize3(im, [x/2, y/2, z]);

    im = imNormalize(im);
    [nr, nc, np] = size(im);
    % write the normalized vessel channel
    writetif(uint8(255 * im), outFile)
   
    % remove the motion artifact and save the result
    inFile = outFile;
    outFile = [h5FileName(1:end-2), 'tif'];
    im = tifMotionRemoval(inFile, outFile);
    % shift im to [-0.5,0.5]
    im = single(im);
    im=im / max(im(:)) - 0.5;

    % write h5 file
    if exist(h5FileName,'file')
        delete(h5FileName)
    end

    h5create(h5FileName, '/im', size(im), 'Datatype', 'single')
    h5write(h5FileName, '/im', im)
end

end
