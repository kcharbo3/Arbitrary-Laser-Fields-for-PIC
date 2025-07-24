% Add export_fig folder and subfolders to path if not already on path
addpath(genpath('./figures'))


% Select batch of .mat files to plot, including subfolders
% files = dir('PATH/TO/FILES/*.mat')
% files = dir('./data/**/*.mat');
files = dir('./notebooks/data/*.mat');


% --- SET FLAGS
% No interpolation (low-g res plots)
use_interp3 = 0;
% Default save_path (new folder in directory)
save_path = 0;
% Do not display plots in MATLAB
hide_fig = 1;   

% Iterate plotting over each file
for i=1:length(files)
    input = strjoin({files(i).folder, files(i).name}, filesep);
    %if contains(files(i).name, "w") && contains(files(i).name, "linear")
    if contains(files(i).name, "t") && contains(files(i).name, "output_t")
        %if tru
        fprintf("\n");
        fprintf(files(i).name);
        fourier_plot(input, use_interp3, save_path, hide_fig);
    end
end
