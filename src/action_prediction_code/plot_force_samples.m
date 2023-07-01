clear; close all;
addpath('./cmplot');

run_id = 0;

load(sprintf('result/estimated_force/regression_haf.mat'));

cls_name = {'drink', 'move', 'pound', 'pour', 'shake', ...
    'eat', 'hole', 'pick', 'fork-scratch', 'whisk', ...
    'chop', 'cut', 'poke', 'knife-scratch', 'spread', ...
    'flip', 'sponge-scratch', 'squeeze', 'wash', 'wipe'};

selected_id = randi(length(force_all), 1, 6)

nrow = 4;
ncol = 3;
figure; clf;
for i = 1 : length(selected_id)
    id = selected_id(i);
    irow = mod(i-1, ncol)+1;
    icol = floor((i-1)/ncol)*2;
    subplot(nrow, ncol, icol*ncol+irow); 
    plot(force_all{id}, 'LineWidth', 1.5); 
    ylim([0, 1]); xlim([0,size(force_all{id}, 1)]);
    title(sprintf('%s : %s', strtrim(object_id(id,:)), cls_name{action_id(id)}));
    
    subplot(nrow, ncol, (icol+1)*ncol+irow); 
    plot(gt_all{id}, 'LineWidth', 1.5); 
    ylim([0, 1]); xlim([0,size(gt_all{id}, 1)]);
    
end
hl = legend({'Ring', 'Middle', 'Pointer', 'Thumb'}, ...
       'Orientation', 'horizontal', ...
       'FontSize', 12, ...
       'FontWeight', 'bold');
   
set(hl, 'Position', [0.5, 0.05, 0.0, 0.0]);
set(gcf, 'Position', [0, 0, 1365, 750]);
writePNG(gcf, sprintf('force_comparison_r%d.png', run_id), 40, 20);

