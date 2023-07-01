clear; 
close all;
addpath('./cmplot');

run_id = 2;
show_figure = 1;

actions = {
    'cup - drink'
    'cup - pound'
    'cup - shake'
    'cup - move around'
    'cup - pour'
    'stone - pound'
    'stone - move around'
    'stone - play'
    'stone - grind'
    'stone - carve'
    'sponge - squeeze'
    'sponge - flip'
    'sponge - wash'
    'sponge - wipe'
    'sponge - scratch'
    'spoon - scoop'
    'spoon - stir'
    'spoon - hit'
    'spoon - eat'
    'spoon - sprinkle'
    'knife - cut'
    'knife - chop'
    'knife - poke a hole'
    'knife - peel'
    'knife - spread'
};

subject_list = {'and', 'mic', 'fer', 'kos', 'gui'};
object_list = {'cup', 'stone', 'sponge', 'spoon', 'knife'};

gt_all = [];
pred_all = [];

precision = [];
recall = [];

precision_lstm_vgg = [];
recore_lstm_vgg = [];
precision_lstm_hog = [];
recore_lstm_hog = [];
precision_hmm = [];
recall_hmm = [];
precision_svm = [];
recall_svm = [];

for si = 1 : length(subject_list)
  subject = subject_list{si};

  % baseline comparison
  gt = [];
  lstm_vgg_preds = [];
  lstm_hog_preds = [];
  lstm_withforce_pred = [];
  hmm_preds = [];
  svm_preds = [];
  offset = 0;
  for oi = 1 : length(object_list)
    lstm_vgg_test_file = sprintf('result/test_results_action_run%d/action_cls_%s_%s.mat', run_id, object_list{oi}, subject);    

    lstm_vgg_rst = load(lstm_vgg_test_file);
    
    gt = [gt; double(lstm_vgg_rst.gts'+1+offset)];
    lstm_vgg_preds = [lstm_vgg_preds; double(lstm_vgg_rst.prediction'+1+offset)];
    offset = offset + 5;
  end
  M = offset;
 
  gt_all = [gt_all; gt];
  pred_all = [pred_all; lstm_vgg_preds];
  
  [prec, rec] = precision_recall(gt, lstm_vgg_preds, M);
  precision_lstm_vgg(:,si) = prec;
  recall_lstm_vgg(:,si) = rec;
  
end


rst_per_class = [nanmean(precision, 2), nanmean(recall, 2)] * 100.0;
rst_all = mean(rst_per_class, 1);

rst_per_class_lstm_vgg = [nanmean(precision_lstm_vgg, 2), nanmean(recall_lstm_vgg, 2)] * 100.0;
rst_all_lstm_vgg = mean(rst_per_class_lstm_vgg, 1);


% print in latex format
fprintf(' Action & LSTM(Vision) \\\\ \n');
fprintf(' \\hline \n');
for i = 1 : length(actions)
  fprintf(' %s ', strrep(actions{i}, ' - ', '/'));
  fprintf('&  %.01f\\%% ', rst_per_class_lstm_vgg(i, 1));
  fprintf('\\\\ \n');
end

fprintf(' \\hline \n');
fprintf(' \\hline \n');
fprintf(' Avg. &  %.01f\\%% \\\\ \n', ...
rst_all_lstm_vgg(1));


cm_count = confusionmat(gt_all,pred_all,'order',[1:M]);
cm_mat = cm_count./repmat(sum(cm_count,2),[1, size(cm_count,2)]);
cm_mat(isnan(cm_mat))=0;

cm_filename = sprintf('confMat_r%d.png', run_id);
if show_figure,
    draw_cm;
end

rst_per_object_action = mean(reshape(rst_per_class_lstm_vgg(:, 1), 5, []), 1);

% print in latex format
fprintf('\n\n');
fprintf('Object');
for i = 1 : length(object_list)
    fprintf(' & %s ', object_list{i});
end
fprintf(' & Avg. \\\\ \n');
fprintf(' \\hline \n');
fprintf(' Vision');
for i = 1 : length(object_list)
    fprintf(' & %.01f\\%% ', rst_per_object_action(i));
end
fprintf(' & %.01f\\%% \\\\ \n', mean(rst_per_object_action));
fprintf(' \\hline \n');

