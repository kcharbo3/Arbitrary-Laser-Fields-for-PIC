function fourier_plot(file, use_interp3, save_path, hide_fig)
% fourier_plot Generate 3D-projection plot of spatiotemporal E-field

arguments
    % mandatory argument
    file (1,:) string
    
    % optional arguments with default values
    use_interp3 (1,1) double = 0
    save_path (1,:) string = 0
    hide_fig (1,1) double = 0
end

% Create a default save path if none is specified
if save_path == '0'
    pathparts = strsplit(file,filesep);
    filename = strsplit(pathparts(end), '.');
    filename = filename(1);
    
    if size(pathparts, 2) > 1
        pathparts = strjoin(pathparts(1:end-1), filesep);
        save_path = strcat(pathparts, '/figures');
        
        if ~isfolder(save_path)
            mkdir(save_path);
        end
    
    else
        save_path = './figures';
        mkdir(save_path);
    end

end

% Load data from file and add name of file
dat = load(file);
dat.name = file;

% Create white -> parula colormap
CT = white_parula(256);
    
% Initialize figure
fig = figure();
% ax1 = axes('Position',[0.0435 0.0532 0.0170 0.0202]);
% ax1.PositionConstraint = 'outerposition';
pbaspect([1 0.7903 0.7903])
set(groot, 'defaultAxesLooseInset', [0.0435 0.0532 0.0170 0.0202])

% Find max value of E-field
maxval=max(dat.eFieldxt, [], 'all');

% Increase number of elements by 2N + 1 to increase resolution
if use_interp3 ~= 0
    dat.eFieldxt = interp3(dat.eFieldxt);
end

% Copy E-field to use in 3D plotting 
eField3D = dat.eFieldxt;

%fprintf("SHAPE: %d", size(eField3D));
%fprintf("MAX VAL: %d", maxval);
% Ygraph = Z, Xgraph = t, Zgraph = Y
eField3D = permute(eField3D,[3 1 2]);
eField3D = flip(eField3D, 2);

fprintf("Max %d", max(eField3D,[],'all'));
% Set all values less than 2% of max value to nan to avoid plotting 
eField3D(eField3D<maxval/200)=nan;
fprintf("Max %d", max(eField3D,[],'all'));
eField3D = imgaussfilt(eField3D, 0.8);

% Create time slices (Y-dim in plot) for 3D plotting
fprintf("Max %d", max(eField3D,[],'all'));
%fprintf("SIZE %d", size(eField3D,2));
h = slice(eField3D/max(eField3D,[],'all'), [], 1:size(eField3D,1), []);
%fprintf("HERE");
%h = slice(eField3D * 0.00001, [], [], []);
%eField3D_final = eField3D/max(eField3D,[],'all')
%eField3D_final = permute(eField3D_final,[3 2 1])
%h = slice(eField3D_final, [], 1:size(eField3D,3), []);

% Draw 3D Plot
% set(h, 'EdgeColor','interp', 'EdgeAlpha', 'interp', ...
%     'FaceColor', 'interp', 'FaceAlpha', 'interp');

set(h, 'EdgeColor','none', ...
    'FaceColor', 'interp', 'FaceAlpha', 'interp');

% Create empirical alpha map to highlight important features
alpha('color')
%alpha_map = 1./(2 + exp(0.5 - 32 .* alphamap)) + .04; % W paper
alpha_map = 1./(1 + exp(3 - 8 .* alphamap)) + .01;  % T paper
alphamap(alpha_map);
colormap parula
%colormap(slanCM('bwr'));

% Use freezeColors to use new colormap for projection plots on same axes
freezeColors;

% Figure settings
box on;
grid on;

% Extract dimensions of E-field
t_shape = size(dat.eFieldxt, 1);
x_shape = size(dat.eFieldxt, 3);
y_shape = size(dat.eFieldxt, 2);

hold on

eFieldNew = permute(dat.eFieldxt,[3 1 2]);
eFieldNew = flip(eFieldNew, 2);

% Draw XY projection (plot dimensions XZ)
%[Xxy, Yxy] = meshgrid(linspace(0, x_shape, x_shape), linspace(0, y_shape, y_shape));
[Xxy, Yxy] = meshgrid(linspace(0, t_shape, t_shape), linspace(0, y_shape, y_shape));
Onesxy=ones(y_shape, t_shape);
temp1=squeeze(sum(eFieldNew, 1))'/max(squeeze(sum(eFieldNew, 1)),[],'all');
temp1(temp1<1/100)=nan;
temp1 = imgaussfilt(temp1,0.8);
surf(Xxy,0*Onesxy,Yxy,temp1,'edgecolor','none', 'facecolor', 'flat')
colormap(CT);
alphamap(alpha_map);

% Draw YT projection  (plot dimensions YZ)
%[Xt, Yt] = meshgrid(linspace(0, t_shape, t_shape), linspace(0, x_shape, x_shape));
[Xt, Yt] = meshgrid(linspace(0, y_shape, y_shape), linspace(0, x_shape, x_shape));
Ones=ones(x_shape,y_shape);
temp2=squeeze(sum(eFieldNew,2))'/max(squeeze(sum(eFieldNew,2)),[],'all');
temp2(temp2<1/100)=nan;
temp2 = imgaussfilt(temp2,0.8);
surf(t_shape*Ones,Xt, Yt,temp2 , 'edgecolor', 'none', 'facecolor', 'flat');
colormap(CT);
alphamap(alpha_map);

% Draw XT projection (plot dimensions XY)
% [Xt2, Yt2] = meshgrid(linspace(0, x_shape, x_shape),linspace(0, t_shape, t_shape));
[Xt2, Yt2] = meshgrid(linspace(0, t_shape, t_shape),linspace(0, x_shape, x_shape));
Ones2=ones(x_shape,t_shape);
temp3=squeeze(sum(eFieldNew,3))/max(squeeze(sum(eFieldNew,3)),[],'all');
temp3(temp3<1/100)=nan;
temp3 = imgaussfilt(temp3,0.8);
surf(Xt2, Yt2,0*Ones2,temp3 , 'edgecolor', 'none', 'facecolor', 'flat');
colormap(CT);
alphamap(alpha_map);

grid on;

% Set number of ticks, 5 usually looks best
NUM_TICKS = 5;

% Decide how much relative space to add to dimension labels
spacer = 0.1;
spacer_t = 0.11;
xticks(ticker(t_shape, spacer_t, NUM_TICKS))
yticks(ticker(x_shape, spacer, NUM_TICKS))
zticks(ticker(y_shape, spacer, NUM_TICKS))

% Set plot limits to dimension sizes
xlim([0,t_shape])
ylim([0,x_shape])
zlim([0,y_shape])

% Write tick labels in proper units
% Assume we get our data in terms of cm and fs
spatial = round(dat.spatial_lim, 2, 'significant') ;
time_lim = round(dat.t_lim, 2, 'significant');
fprintf("NAME: " + dat.name);

if ~contains(dat.name, "output_t")
    omega_max = round(dat.t_lim, 4, 'significant');
    omega_min = round(dat.t_lim_start, 4, 'significant');

    omega_from_0 = omega_max - omega_min;
    tick_start_percentage = (t_shape * spacer_t) / t_shape;
    tick_end_percentage = 1 - tick_start_percentage;
    
    tick_start = round(tick_start_percentage * omega_from_0, 4, 'significant');
    tick_end = round(tick_end_percentage * omega_from_0, 4, 'significant');
    ticks = round(linspace(tick_end + omega_min, tick_start + omega_min, NUM_TICKS), 3, 'significant');
    %fprintf("TICKS: %d", ticks);
    fprintf("Omega max: %d", omega_max);
    fprintf("Omega min: %d", omega_min);
    fprintf("Omega from 0: %d", omega_from_0);
    fprintf("Start percentage: %d", tick_start_percentage);
    fprintf("tick start: %d", tick_start);

    xticklabels(cellstr(string(ticks)))
else
    tick_start_percentage = (t_shape * spacer_t) / t_shape;
    tick_end_percentage = 1 - tick_start_percentage;
    
    tick_start = round(-time_lim + (tick_start_percentage * 2 * time_lim), 2, 'significant');
    tick_end = round(-time_lim + (tick_end_percentage * 2 * time_lim), 2, 'significant');
    ticks = linspace(tick_start, tick_end, NUM_TICKS);
    xticklabels(cellstr(string(ticks)))
end

tick_start_percentage = (y_shape * spacer) / y_shape;
tick_end_percentage = 1 - tick_start_percentage;
tick_start = round(-spatial + (tick_start_percentage * 2 * spatial), 2, 'significant');
tick_end = round(-spatial + (tick_end_percentage * 2 * spatial), 2, 'significant');
ticks = linspace(tick_start, tick_end, NUM_TICKS);
zticklabels(cellstr(string(ticks)))

tick_start_percentage = (x_shape * spacer) / x_shape;
tick_end_percentage = 1 - tick_start_percentage;
tick_start = round(-spatial + (tick_start_percentage * 2 * spatial), 2, 'significant');
tick_end = round(-spatial + (tick_end_percentage * 2 * spatial), 2, 'significant');
ticks = linspace(tick_start, tick_end, NUM_TICKS);
yticklabels(cellstr(string(ticks)))
xtickangle(0)
ytickangle(0)
ztickangle(0)

% Draw labels with appropriate font size
set(gca, 'FontSize', 10)
%xlabel('Z-Displacement [microns]', 'FontSize', 12) 
%ylabel('Local Time [fs]', 'FontSize', 12) 
%zlabel('Y-Displacement [microns]', 'FontSize', 12) 

view(-1.353000000000000e+02,12.574598540145987)

figname = strjoin([save_path, filename], filesep);

% SAVED SETTINGS FOR TILTING AXES LABELS
% % Create axes
% axes1 = axes('Position',[0.13 0.11 0.775 0.815]);
% hold(axes1,'on');
%  
% % Create zlabel
% zlabel('Y-Displacement [microns]','FontSize',16);
%  
% % Create ylabel
% ylabel('Local Time [fs]','FontSize',16,'Rotation',10);
%  
% % Create xlabel
% xlabel('X-Displacement [microns]','FontSize',16,'Rotation',-11);
%  
% % Uncomment the following line to preserve the X-limits of the axes
% % xlim(axes1,[0 201]);
% % Uncomment the following line to preserve the Y-limits of the axes
% % ylim(axes1,[0 149]);
% % Uncomment the following line to preserve the Z-limits of the axes
% % zlim(axes1,[0 201]);
% view(axes1,[-135.3 12.574598540146]);
% box(axes1,'on');
% grid(axes1,'on');
% hold(axes1,'off');
% % Set the remaining axes properties
% set(axes1,'CLim',[0.00856198695831141 1],'FontSize',14,'XTick',...
% [20 60.25 100.5 140.75 181],'XTickLabel',...
% {'-7.96','-3.98','0','3.98','7.96'},'YTick',[15 44.75 74.5 104.25 134],...
% 'YTickLabel',{'-116','-58','0','58','116'},'ZTick',...
% [20 60.25 100.5 140.75 181],'ZTickLabel',...
% {'-7.96','-3.98','0','3.98','7.96'});

% Save figure as PDF
exportgraphics(fig, strcat(figname, '.pdf'), 'ContentType', 'image', 'BackgroundColor', 'w', 'Resolution', '300'); % white background

if use_interp3 == 0
% Save figure as FIG-file
    savefig(figname);
%     save('myFile.mat', 'Variablename', '-v7.3')
end

% Prevent figures from stacking
if hide_fig ~= 0
    close(fig)
end

%{
% TODO: Fix bug of pixels being off by one sometimes and breaking the save

% temporary save with black & white backgrounds
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
figname = strjoin([save_path, filename], filesep);
% store again
imwrite(uint8(c*255), strcat(strjoin([save_path, filename], filesep), '.png'), 'Alpha', al);
% imwrite(uint8(c*255), 'test.png', 'Alpha', al);

% I think this will be equivalent as above code


% export_fig out fig -m2.5 -transparent
% set(gcf, 'Color', 'none');
% export_fig(sprintf(figname), '-transparent', '-pdf')
%}

function CT = white_parula(N)
    % Start with parula
    CT = parula(N);
    
    % Set first 20 rows to RGB linearly scale form [1 1 1] to 20th parula value
    % This acts as a forced transition from white to blue
    CT(1:20,:)=interp1([0 1],[1 1 1;CT(20,:)],linspace(0,1,20),'linear','extrap');
end

end

function ticks = ticker(shape, space, NUM_TICKS)
    % shape is an x_shape, y_shape, or t_shape, and larger space increases
    % the buffer of the image
    tick_start = round(shape * space);
    tick_end = shape - tick_start;
    ticks = linspace(tick_start, tick_end, NUM_TICKS);
end

%{
  ____                 _                             
 / ___| ___   ___   __| |  _ __   _____      _____   
| |  _ / _ \ / _ \ / _` | | '_ \ / _ \ \ /\ / / __|  
| |_| | (_) | (_) | (_| | | | | |  __/\ V  V /\__ \_ 
 \____|\___/ \___/ \__,_| |_| |_|\___| \_/\_/ |___( )
                                                  |/ 
                                            _ 
  _____   _____ _ __ _   _  ___  _ __   ___| |
 / _ \ \ / / _ \ '__| | | |/ _ \| '_ \ / _ \ |
|  __/\ V /  __/ |  | |_| | (_) | | | |  __/_|
 \___| \_/ \___|_|   \__, |\___/|_| |_|\___(_)
                     |___/                    
                                                                                
                                 *%%%###%%/.                                    
                             .%,,,,,,,,,,,,,,,*##,                              
                            /,,,,,,,,**,,,,,,,,,,,,((                           
                           ,(,,,,,,*#,,&,,,,,,,,,,,,,,&,                        
                           #,,,,,,,,,,,,,,,,,,,,,,,,,,,*&,                      
                           %,,,,,%*%*,,,,,,,,,,,,,,,,,,,,,,,&                   
                           #,,,,,,,,,,,,,,,,,%*,,,,,,,,,,,,,,&                  
                           ,(,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,#/              
                            %,,,,,,,,,,,,,,,,/%/**/,,,,,,,,,,,,,,,#.            
                             &,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,*#        
                              #,,,,,,,,,,,,,,,,,%(,,,,(%...../%,%%///#@%(&&.    
                             ,/&#%%#(/*,,,,,,,#*,,,,/(.....((../(,*&........(*  
                          ** .#,*,,,/,,,,,,,,,&,,,,,%..#/.......#*%..,%(,....#  
                             *,,,#*#,,,,,,,,,,(/,,,,&....*%(...*#,,%*...,(*..%  
                              ,/,%,,,,,,,,,,,,,,%&(,,%*.......&*,,,,,,@,...,&   
                               %*(/,,,,,,,,,,,,,,,,,,#,,*//*,,,#/,,,,,,,/&,     
                               %,,%%(,,,,,,,,,,,,,,,,,,%,,,,,,,%,,,,,,,,,,(     
                                %,,,/#,,,,,,,,,,,,,,,,#*,,,,,,,,,,/%,,,,,,(     
                           .&&#@,,,,,,,,,,,,,,,,,,,*&,,,,,,,,,,,,,,,,##,,/(     
                        #/   @,,,*,,,,,,,,,,,,,,,%*,(/,(%%###%&/,,,,,,/#        
                      #     @(,/#,,,,#,,,,,,,,,*%,,,*,,,,,,,,(/  *##,,,,,*%/    
                    /*     ,#*@,,,,#/,,,,,,,,,,*/,,,,,,,%%           .&#***%    
                (#  &      ,#**&&*,,,,,,,,,,,,,,,,,,*%/                         
             %,     @      ,#***/%,,,/%*,,,,,,*%.                               
          (*        @      .%******&#,,,*@./,                                   
         @          @       @&***********#,,/                                   
        %           #       &**&********@. ,/                                   
       *,           .,      /****#%#(#@    **                                   
       %             &      .#********/(   /,                                   
      /,             /,      @*********@   /,                                   
      #               %      */********#   (/                                   
     **               ./      @**********  (%                                   
     #                 (,     //********#  #(.                                  
     #               %. #      &********@  #.(                                  
    **               (.  #     //*******@  % (                                  
    #                /.  .(     %*******@  % *.                                 
    #                (.   **    #/******@  # .(                                 
    /                &     (    ,(******&  #  (                                 
   /,                &      &    %******@  #  *                                 
   #.                (      ./   %*******%/#  .*                                
   %                ,.       #   (/********/%  #                                
   %                @         &  *(**********/%..                               
   #                %         /.  %************#&                               
   (                (          %  %*************//                              
   /                *          (. #****************                             
  .*                ,           # (/**************%.                            
  ,*                *           #./(****************                            
  ,*                (            (,#***************/                            
  ,*                %            #,%**************(.                            
  .*                @             (%**********/%@.,.                            
   /                %             #%((#%&&&%####(  *                            
   (                *.    ./      ,@##########%&   (                            
   #                 &    *,       &#########%%%. ,%                            
   **#%%*       *#&,      /        /%#######@###%,/.                            
   ,*(#,,,,&,,(/,,#       #        *%#####@#####@,&                             
   ,*(#,,,,(,,,,,,%      .(        ,%###@#######&,&                             
   ,*(/*,,#,,,/*,,%      ,*        ,@@##########%,@                             
   ,*(*/,,@,,,%,,/,      /         ,%############//                             
   **(*/,*&,,*@,,%       (         ,%############%                              
   **(#,,@%,,&#,(,       (         *%#############                              
   **(/&/ &,#, ,        ./         /#############&                              
   /,(                  *,         (#############@                              
   #.(                  (          ##############@                              
   %.(                  (          %#############@                              
   % **                ,*          &#############@                              
   %   /(,          ,((           .&#############@                              
   #                              *%#############@                              
   #                              /%#############@                              
   (                              ###############%                              
   (                              %###############                              
   /                              &##############*                              
  .*                             .%#############%.                              
   *(                            /%#############&                               
     ##&/.                     ,&###############&                               
     %########%&@@@@@&&%/*.    ,%###############&                               
     &#################/       /%################                               
    .&################%*       (###############%,                               
    /%################&,       ################%                                
    ##################@.       &###############&                                
    %#################@        #############%@#       ,.                        
   .&#################%          %,**#**,#,/(,,,/%&*,,,,/#%,                    
   (%##################        .%,,,,*,,,,,,,,,,,(%/,,,,,,,/@                   
    (&##############@/         &,,,,,,,,,,,,,,,,*#,,,,,,,,,,/&&                 
     #,,,(/,%*,,%,,,,&.  .*(#(#&/,*//(##%%&%#(/%,,,,,,,,,,,,,,*/%               
    #*,,,,,,,,,,,,,,,,,*&,,,,,,,,(&,(%%(/,,,,,,,,,,,,,,,,,,,,,,,#(              
  /#%,,,,,,,,,,,,,,,,%*,,,,,,,,,,,,*%        ,(%&&%#(//////(%&#.                
    (@/**#%@@#,,,,,/,,,,,,,,,,,,,,,,(#                                          
         ./%#*,,,*/%,,,,,,,,,,,,,,,,,#%                                         
                .(%%(/*,,,,,,,,,,,,,/%,     
    
  _____ _     _                     _        _        __               _ 
 |_   _| |__ (_)___    ___ ___   __| | ___  (_)___   / _|_ __ ___  ___| |
   | | | '_ \| / __|  / __/ _ \ / _` |/ _ \ | / __| | |_| '__/ _ \/ _ \ |
   | | | | | | \__ \ | (_| (_) | (_| |  __/ | \__ \ |  _| | |  __/  __/_| _ _ _ 
   |_| |_| |_|_|___/  \___\___/ \__,_|\___| |_|___/ |_| |_|  \___|\___(_)(_|_|_)
  _           _     __  __    _  _____ _        _    ____    _           _ _         
 | |__  _   _| |_  |  \/  |  / \|_   _| |      / \  | __ )  (_)___ _ __ ( ) |_       
 | '_ \| | | | __| | |\/| | / _ \ | | | |     / _ \ |  _ \  | / __| '_ \|/| __|      
 | |_) | |_| | |_  | |  | |/ ___ \| | | |___ / ___ \| |_) | | \__ \ | | | | |_ _ _ _ 
 |_.__/ \__,_|\__| |_|  |_/_/   \_\_| |_____/_/   \_\____/  |_|___/_| |_|  \__(_|_|_)
%}
