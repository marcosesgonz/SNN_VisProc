
M = length(actions);
cm_count = confusionmat(gt_all,pred_all,'order',[1:M]);
cm_mat = cm_count./repmat(sum(cm_count,2),[1, size(cm_count,2)]);
cm_mat(isnan(cm_mat))=0;

figure;
h = imagesc(cm_mat);

axis image;
axis equal;
xlim([0.5, M+0.5]);
ylim([0.5, M+0.5]);

% set(gca, 'XAxisLocation', 'top');
set(gca, 'TickDir', 'out');
% set(gca, 'XTickLabel', actions);
set(gca, 'YTickLabel', actions);
set(gca, 'XTick', 1:M);
set(gca, 'YTick', 1:M);

colorbar;

% format_ticks(gca,action,action,1:Na,1:Na,90,0,0);
xticklabel_rotate(1:M,90,actions,'interpreter','none');

ylabel('Ground truth action classes', 'FontSize', 12);
xlabel('Predicted action classes', 'FontSize', 12);
% colormap jet

set(gca,'Position',get(gca,'Position') + [0.07 0 0 0]);

% Bottom
xlabh = get(gca,'XLabel');
set(xlabh,'Position',get(xlabh,'Position') + [0 0.20 0]);
ylabh = get(gca,'YLabel');
set(ylabh,'Position',get(ylabh,'Position') + [0.05 0 0]);

writePNG(gcf, cm_filename, 20, 20);



