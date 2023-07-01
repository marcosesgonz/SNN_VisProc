function [prec, rec] = precision_recall(gt, pred, ncls)

  cm_count_sub = confusionmat(gt, pred, 'order', [1:ncls]);
  tp = diag(cm_count_sub);
  cls_count = sum(cm_count_sub, 2);
  fp = sum(cm_count_sub, 1)'-tp;
  fn = cls_count-tp;

  prec = tp ./ (tp + fp);
  prec(tp==0) = 0;
  prec(cls_count==0) = NaN;
  rec  = tp ./ (tp + fn);
  rec(tp==0) = 0;
  rec(cls_count==0) = NaN;

