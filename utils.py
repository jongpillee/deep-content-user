
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def eval_tops(prediction, truth, N):

    num_tags = prediction.shape[1]
    num_examples = prediction.shape[0]
    
    ranked = np.argsort(-prediction)
    ranked = ranked[:,:N]
    label = np.argsort(-truth)
    label = label[:,0]

    counter = 0
    for iter in range(0,num_examples):
         tmp = ismember(ranked[iter,:],label[iter])
         if np.sum(tmp) == 1:
             counter = counter + 1

    counter = np.float64(counter)
    num_examples = np.float64(num_examples)
    acc = counter/num_examples
    return acc

def ismember(A,B):
    return [np.sum(a==B) for a in A]

def eval_retrieval_perTag(prediction, truth):

    num_tags = prediction.shape[1]
    num_examples = prediction.shape[0]
    available_index = np.nonzero(np.sum(truth, axis=0))
    available_index = available_index[0]

    aroc = np.zeros(shape = (num_tags,1))
    ap = np.zeros(shape = (num_tags,1))

    for i in available_index:
        ranking = np.argsort(prediction[:,i])
        ranking = ranking[::-1]

        tp = np.zeros(shape = (num_examples,1))
        recall = np.zeros(shape = (num_examples,1))
        precision = np.zeros(shape = (num_examples,1))
        fp_rate = np.zeros(shape = (num_examples,1))
        truth_col_sum = np.sum(truth[:,i])

        for j in range(num_examples):
            index = ranking[:j+1]
            tp[j] = np.sum(truth[index,i])
            recall[j] = tp[j]/truth_col_sum
            precision[j] = tp[j]/(j+1)
            fp_rate[j] = (j+1-tp[j])/(num_examples-truth_col_sum)

        for j in range(num_examples-1):
            width = fp_rate[j+1] - fp_rate[j]
            aroc[i] = aroc[i] + width*(recall[j+1]+recall[j])*0.5


        # find levels where retrieved one is correctly identied
        recall2 = np.insert(recall, 0, 0)
        corr_index = np.where(np.diff(recall2) > 0)
        corr_index = corr_index[0]
        ap[i] = precision[corr_index].mean()


    aroc = aroc[available_index];
    ap = ap[available_index];

    return aroc, ap


def eval_retrieval(prediction, truth):

    num_tags = prediction.shape[1]
    num_examples = prediction.shape[0]
    available_index = np.nonzero(np.sum(truth, axis=0))
    available_index = available_index[0]

    aroc = np.zeros(shape = (num_tags,1))
    ap = np.zeros(shape = (num_tags,1))

    for i in available_index:
        ranking = np.argsort(prediction[:,i])
        ranking = ranking[::-1]

        tp = np.zeros(shape = (num_examples,1))
        recall = np.zeros(shape = (num_examples,1))
        precision = np.zeros(shape = (num_examples,1))
        fp_rate = np.zeros(shape = (num_examples,1))
        truth_col_sum = np.sum(truth[:,i])

        for j in range(num_examples):
            index = ranking[:j+1]
            tp[j] = np.sum(truth[index,i])
            recall[j] = tp[j]/truth_col_sum
            precision[j] = tp[j]/(j+1)
            fp_rate[j] = (j+1-tp[j])/(num_examples-truth_col_sum)

        for j in range(num_examples-1):
            width = fp_rate[j+1] - fp_rate[j]
            aroc[i] = aroc[i] + width*(recall[j+1]+recall[j])*0.5


        # find levels where retrieved one is correctly identied
        recall2 = np.insert(recall, 0, 0)
        corr_index = np.where(np.diff(recall2) > 0)
        corr_index = corr_index[0]
        ap[i] = precision[corr_index].mean()


    aroc = aroc[available_index];
    ap = ap[available_index];

    mean_aroc = aroc.mean()
    mean_ap = ap.mean()

    return mean_aroc, mean_ap

def eval_annotation(prediction, truth, emp_prob_label, num_top_tags, diverse_factor):
    
    prediction = prediction - prediction.mean(axis=0)*diverse_factor

    decision = np.zeros(prediction.shape)

    for i in range(prediction.shape[0]):
        top_index = prediction[i].argsort()[::-1][:num_top_tags]
        decision[i,top_index] = 1

    true_decision = np.multiply(decision, truth)

    # precision
    annotation_tag_index = np.where(np.logical_and(decision.sum(axis=0),truth.sum(axis=0)))
    annotation_tag_index = annotation_tag_index[0]
    non_annotation_tag_index = np.where(np.logical_and(decision.sum(axis=0) == 0, truth.sum(axis=0) > 0))
    non_annotation_tag_index= non_annotation_tag_index[0]

    word_precison = np.zeros(shape=(prediction.shape[1]))
    word_precison[annotation_tag_index] = true_decision[:,annotation_tag_index].sum(axis=0)/decision[:,annotation_tag_index].sum(axis=0)
    word_precison[non_annotation_tag_index] = emp_prob_label[non_annotation_tag_index]

    available_index = np.where(truth.sum(axis=0)>0);
    available_index = available_index[0]
    precision = word_precison[available_index].mean()

    # recall
    word_recall = true_decision.sum(axis=0)/truth.sum(axis=0)
    recall = word_recall[available_index].mean()

    # fscore
    fscore = 2*(precision*recall)/(precision+recall);

    return fscore, precision, recall
    
def eval_avg_precision_at_K(prediction, truth, K):
    
    precision = 0

    for i in range(prediction.shape[1]):
        top_index = prediction[:,i].argsort()[::-1][:K]
        precision = precision + truth[top_index,i]/K
    
    precision = precision/prediction.shape[1]

    return prediction


# compute evaluation metrics
def construct_pred_mask(tags_predicted, predictat):
    n_samples, n_tags = tags_predicted.shape
    rankings = np.argsort(-tags_predicted, axis=1)[:, :predictat]
    tags_predicted_binary = np.zeros_like(tags_predicted, dtype=bool)
    for i in xrange(n_samples):
        tags_predicted_binary[i, rankings[i]] = 1
    return tags_predicted_binary

def per_tag_prec_recall(tags_predicted_binary, tags_true_binary):
    mask = np.logical_and(tags_predicted_binary, tags_true_binary)
    prec = mask.sum(axis=0) / (tags_predicted_binary.sum(axis=0) + np.spacing(1))
    tags_true_count = tags_true_binary.sum(axis=0).astype(float)
    idx = (tags_true_count > 0)
    recall = mask.sum(axis=0)[idx] / tags_true_count[idx]
    return prec, recall


def aroc_ap(tags_true_binary, tags_predicted):
    n_tags = tags_true_binary.shape[1]
    
    auc = list()
    aprec = list()
    for i in xrange(n_tags):
        if np.sum(tags_true_binary[:, i]) != 0:
            auc.append(roc_auc_score(tags_true_binary[:, i], tags_predicted[:, i]))
            aprec.append(average_precision_score(tags_true_binary[:, i], tags_predicted[:, i]))
    return auc, aprec

def print_out_metrics(tags_true_binary, tags_predicted, predictat, diverse_factor):

    tags_predicted2 = tags_predicted - tags_predicted.mean(axis=0)*diverse_factor

    tags_predicted_binary = construct_pred_mask(tags_predicted2, predictat)
    prec, recall = per_tag_prec_recall(tags_predicted_binary, tags_true_binary)
    mprec, mrecall = np.mean(prec), np.mean(recall)
    
    print ('Precision = %.3f (%.3f)' % (mprec, np.std(prec) / np.sqrt(prec.size)))
    print ('Recall = %.3f (%.3f)' % (mrecall, np.std(recall) / np.sqrt(recall.size)))
    print ('F-score = %.3f' % (2 * mprec * mrecall / (mprec + mrecall)))

    auc, aprec = aroc_ap(tags_true_binary, tags_predicted)
    print ('AROC = %.3f (%.3f)' % (np.mean(auc), np.std(auc) / np.sqrt(len(auc))))
    print ('AP = %.3f (%.3f)' % (np.mean(aprec), np.std(aprec) / np.sqrt(len(aprec))))



