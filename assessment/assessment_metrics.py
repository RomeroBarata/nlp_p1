def precision(true_positive, false_positive):
    return true_positive / (true_positive + false_positive)

def recall(true_positive, false_negative):
    return true_positive / (true_positive + false_negative)

def f1(true_positive, false_positive, false_negative):
    prec = precision(true_positive, false_positive)
    rec = recall(true_positive, false_negative)
    return (2 * prec * rec) / (prec + rec)

def accuracy(true_positive, true_negative, false_positive, false_negative):
    return (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)

def compute_metrics(true_classes, predicted_class, interest_category):
    tp = tn = fp = fn = 0
    for i in range(len(predicted_class)):
        if predicted_class[i] in true_classes[i]:
            tp += 1
        elif predicted_class[i] == interest_category and predicted_class[i] not in true_classes[i]:
            fp += 1
        elif predicted_class[i] != interest_category and interest_category not in true_classes[i]:
            tn += 1
        else:
            fn += 1

    return {'Precision': precision(tp, fp), 'Recall': recall(tp, fn), 'F1': f1(tp, fp, fn), 'Accuracy': accuracy(tp, tn, fp, fn)}
