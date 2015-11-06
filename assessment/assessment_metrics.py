def precision(true_positive, false_positive):
    if (true_positive + false_positive) != 0:
        return true_positive / (true_positive + false_positive)
    else:
        return 0

def recall(true_positive, false_negative):
    if (true_positive + false_negative) != 0:
        return true_positive / (true_positive + false_negative)
    else:
        return 0

def f1(true_positive, false_positive, false_negative):
    prec = precision(true_positive, false_positive)
    rec = recall(true_positive, false_negative)
    if (prec + rec) != 0:
        return (2 * prec * rec) / (prec + rec)
    else:
        return 0

def accuracy(true_positive, true_negative, false_positive, false_negative):
    return (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)

def compute_metrics(true_classes, predicted_class, interest_category):
    ''' Function to compute assessment metrics. These include: precision, recall, 
        f1, accuracy, true positive, true negative, false positive, false negative, and
        also it returns the name of the class of interest as well.
        Args:
              true_classes: A list of lists where each list contains the classes of
                            individual documents.
              predicted_class: A list of predictions made by a single classifier.
              interest_category: The concept of interest of the classifier which made
                                 the predictions.
    '''
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

    return {'Precision': precision(tp, fp), 'Recall': recall(tp, fn), 'F1': f1(tp, fp, fn), 'Accuracy': accuracy(tp, tn, fp, fn),
            'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn, 'Category': interest_category}

def macro_average(performances):
    ''' Function to compute the macro metrics.
        Args:
             performances: A list of dictionaries where each dictionary contains
                           the performance metrics computed by the function compute_metrics for each trained classifier.
    '''
    num_classifiers = len(performances)
    macro_precision = macro_recall = macro_accuracy = macro_f1 = macro_tp = macro_tn = macro_fp = macro_fn = 0
    for i in range(num_classifiers):
        macro_precision += performances[i]['Precision']
        macro_recall += performances[i]['Recall']
        macro_accuracy += performances[i]['Accuracy']
        macro_f1 += performances[i]['F1']
        macro_tp += performances[i]['TP']
        macro_tn += performances[i]['TN']
        macro_fp += performances[i]['FP']
        macro_fn += performances[i]['FN']

    return {'Macro Precision': macro_precision / num_classifiers, 'Macro Recall': macro_recall / num_classifiers,
            'Macro Accuracy': macro_accuracy / num_classifiers, 'Macro F1': macro_f1 / num_classifiers,
            'Macro TP': macro_tp / num_classifiers, 'Macro TN': macro_tn / num_classifiers,
            'Macro FP': macro_fp / num_classifiers, 'Macro FN': macro_fn / num_classifiers}

def micro_average(performances):
    ''' Function to compute the micro metrics.
        Args:
             performances: A list of dictionaries where each dictionary contains
                           the performance metrics computed by the function compute_metrics for each trained classifier.
    '''
    tp = tn = fp = fn = 0
    for i in range(len(performances)):
        tp += performances[i]['TP']
        tn += performances[i]['TN']
        fp += performances[i]['FP']
        fn += performances[i]['FN']

    return {'Micro Precision': precision(tp, fp), 'Micro Recall': recall(tp, fn), 'Micro F1': f1(tp, fp, fn),
            'Micro Accuracy': accuracy(tp, tn, fp, fn), 'Micro TP': tp, 'Micro TN': tn, 'Micro FP': fp, 'Micro FN': fn}
    
