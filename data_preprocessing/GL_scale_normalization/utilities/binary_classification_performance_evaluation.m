function perf = binary_classification_performance_evaluation(label, score)

p = sum(double(label==1));
n = sum(double(label==0));
try
    [fpr,tpr,t,auROC,optt_ROC] = perfcurve(label,score,1);
catch
    perf.at_max_f1_f1 = nan;
    perf.at_max_f1_acc = nan;
    perf.at_max_f1_sensitivity = nan;
    perf.at_max_f1_specificity = nan;
    perf.at_max_f1_precision = nan;
    perf.at_max_f1_threshold = nan;

    perf.auROC = nan;
    perf.auPRC = nan;
    perf.optt_ROC = nan;
    perf.optt_PRC = nan;
    perf.t = nan;
    perf.tn = nan;
    perf.fn = nan;
    perf.tp = nan;
    perf.fp = nan;
    perf.tpr = nan;
    perf.specificity = nan;
    perf.ppv = nan;
    perf.npv = nan;
    perf.fpr = nan;
    perf.fnr = nan;
    perf.fdr = nan;
    perf.acc = nan;
    perf.f1 = nan;
    return
end
[~,ppv,~,auPRC,optt_PRC] = perfcurve(label,score,1,'Xcrit','tpr','Ycrit','ppv');

tp = tpr * p;
fp = fpr * n;
tn = n - fp;
fn = p - tp;
specificity = tn./(tn + fp);
npv = tn./(tn + fn);
fnr = 1 - tpr;
fdr = 1 - ppv;

acc = (tp+tn)/(p+n);
f1 = 2*tp./(2*tp+fp+fn);

[M, I] = max(f1);
perf.at_max_f1_f1 = M;
perf.at_max_f1_acc = acc(I);
perf.at_max_f1_sensitivity = tpr(I);
perf.at_max_f1_specificity = specificity(I);
perf.at_max_f1_precision = ppv(I);
perf.at_max_f1_threshold = t(I);

perf.auROC = auROC;
perf.auPRC = auPRC;
perf.optt_ROC = optt_ROC;
perf.optt_PRC = optt_PRC;
perf.t = t;
perf.tn = tn;
perf.fn = fn;
perf.tp = tp;
perf.fp = fp;
perf.tpr = tpr;
perf.specificity = specificity;
perf.ppv = ppv;
perf.npv = npv;
perf.fpr = fpr;
perf.fnr = fnr;
perf.fdr = fdr;
perf.acc = acc;
perf.f1 = f1;



