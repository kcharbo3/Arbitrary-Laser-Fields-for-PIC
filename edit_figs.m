%export_fig folder and subfolders to path if not already on path
addpath(genpath('./figures'))


% Select batch of .mat files to plot, including subfolders
% files = dir('PATH/TO/FILES/*.mat')
% files = dir('./data/**/*.mat');
files = dir('./notebooks/data/figures/*.fig');


% --- SET FLAGS
% No interpolation (low-g res plots)
use_interp3 = 0;
% Default save_path (new folder in directory)
save_path = 0;
% Do not display plots in MATLAB
hide_fig = 1;
prefix = "/Users/kylecharbonnet/Code/UCI/barty-personal/fourier_prop_with_sim/notebooks/data/figures/"
% Iterate plotting over each file
for i=1:length(files)
    input = strjoin({files(i).folder, files(i).name}, filesep);
    if contains(files(i).name, "petal6_output_t")
        fprintf("\n");
        [filepath,name,ext] = fileparts(files(i).name)
        fprintf(files(i).name);
        fig = openfig(prefix + files(i).name);

        ax = gca;
        ax.FontSize = 18; 

        %figname = strjoin([save_path, filename], filesep);
        %fprintf("FIGNAME: %d", figname);
        %set(gcf, 'color', 'none');    
        %set(gca, 'color', 'none');
        %export_fig(sprintf(files(i).name), '-transparent', '-pdf')
        %exportgraphics(fig, strcat(prefix+"linear_input_w", '.pdf'), 'ContentType', 'image', 'BackgroundColor', 'w', 'Resolution', '300');

        exportgraphics(fig, 'test1.png', 'ContentType', 'image', 'BackgroundColor', 'k', 'Resolution', '300'); % black background
        exportgraphics(fig, 'test2.png', 'ContentType', 'image', 'BackgroundColor', 'w', 'Resolution', '300'); % white background
        
        % load exported images back in and scale to [0,1]
        % Stolen from internet
        u = imread('test1.png');
        u = double(u) / 255;
        v = imread('test2.png');
        v = double(v) / 255;
        
        delete('test1.png')
        delete('test2.png')
        
        % recover transparency as al
        al = 1 - v + u;
        al = mean(al, 3);
        al = max(0, min(1, al));
        m = al > eps;
        
        % recover rgb
        c = zeros(size(u));
        for i = 1 : 3
            ui = u(:, :, i);
            ci = c(:, :, i);
            ci(m) = ui(m) ./ al(m);
            c(:, :, i) = ci;
        end
        c = max(0, min(1, c));
        
        % figname = strcat(strjoin([save_path, filename], filesep), '.png');
        % store again
        imwrite(uint8(c*255), strcat(prefix+name, '.png'), 'Alpha', al);
        close(fig);
    end
end


% if save_path == '0'
%     pathparts = strsplit(file,filesep);
%     filename = strsplit(pathparts(end), '.');
%     filename = filename(1);
% 
%     if size(pathparts, 2) > 1
%         pathparts = strjoin(pathparts(1:end-1), filesep);
%         save_path = strcat(pathparts, '/figures');
% 
%         if ~isfolder(save_path)
%             mkdir(save_path);
%         end
% 
%     else
%         save_path = './figures';
%         mkdir(save_path);
%     end
% 
% end

% % Load data from file and add name of file
% dat = load(file);
% dat.name = file;
% 
% figname = strjoin([save_path, filename], filesep);
% 
% 
% if use_interp3 == 0
% % Save figure as FIG-file
%     savefig(figname);
% end

