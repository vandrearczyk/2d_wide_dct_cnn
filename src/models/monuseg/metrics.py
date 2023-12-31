import numpy as np

def aggregated_jaccard_index(y_true, y_pred):
    y_true_indices = np.unique(y_true)
    y_pred_indices = np.unique(y_pred)

    y_true_indices = np.delete(y_true_indices, y_true_indices == 0)
    y_pred_indices = np.delete(y_pred_indices, y_pred_indices == 0)
    overall_correct_count = 0
    union_pixel_count = 0
    matched_pred_count = {i: 0 for i in y_pred_indices}
    for i in y_true_indices:
        nuclei_mask = y_true == i
        y_pred_match = nuclei_mask * y_pred

        if np.sum(y_pred_match) == 0:
            union_pixel_count += np.sum(nuclei_mask)
            continue
        else:
            pred_nuclei_indices = np.unique(y_pred_match)
            pred_nuclei_indices = np.delete(pred_nuclei_indices,
                                            pred_nuclei_indices == 0)
            jaccard_index = 0
            for j in pred_nuclei_indices:
                matched = y_pred == j
                intersection_tmp = np.sum(matched & nuclei_mask)
                union_tmp = np.sum(matched | nuclei_mask)
                ji_tmp = intersection_tmp / union_tmp
                if ji_tmp > jaccard_index:
                    jaccard_index = ji_tmp
                    best_match = j
                    correct_count = intersection_tmp
                    union = union_tmp

            overall_correct_count += correct_count
            union_pixel_count += union
            matched_pred_count[best_match] += 1

    unmatched_pred = [
        k for k in matched_pred_count.keys() if matched_pred_count[k] == 0
    ]
    for k in unmatched_pred:
        union_pixel_count += np.sum(y_pred == k)
    return overall_correct_count / union_pixel_count


def f_score(y_true, y_pred):
    y_true_indices = np.unique(y_true)
    y_pred_indices = np.unique(y_pred)

    y_true_indices = np.delete(y_true_indices, y_true_indices == 0)
    y_pred_indices = np.delete(y_pred_indices, y_pred_indices == 0)

    n_pred_nuclei = len(y_pred_indices)
    n_true_nuclei = len(y_true_indices)

    true_positive = 0
    matched_pred_indices = list()
    for i in y_true_indices:
        nuclei_mask = y_true == i
        y_pred_match = nuclei_mask * y_pred
        nuclei_area = np.sum(nuclei_mask)

        if np.sum(y_pred_match) == 0:
            continue
        else:
            pred_nuclei_indices = np.unique(y_pred_match)
            pred_nuclei_indices = np.delete(pred_nuclei_indices,
                                            pred_nuclei_indices == 0)
            for j in pred_nuclei_indices:
                if (np.sum(y_pred_match == j) > 0.5 * nuclei_area
                        and j not in matched_pred_indices):
                    true_positive += 1
                    matched_pred_indices.append(j)
                    continue
    false_positive = n_pred_nuclei - true_positive
    false_negative = n_true_nuclei - true_positive
    fscore = true_positive / (true_positive + 0.5 *
                              (false_positive + false_negative))

    return fscore

def confusion_terms(y_true, y_pred):
    y_true_indices = np.unique(y_true)
    y_pred_indices = np.unique(y_pred)

    y_true_indices = np.delete(y_true_indices, y_true_indices == 0)
    y_pred_indices = np.delete(y_pred_indices, y_pred_indices == 0)

    n_pred_nuclei = len(y_pred_indices)
    n_true_nuclei = len(y_true_indices)

    true_positive = 0
    matched_pred_indices = list()
    for i in y_true_indices:
        nuclei_mask = y_true == i
        y_pred_match = nuclei_mask * y_pred
        nuclei_area = np.sum(nuclei_mask)

        if np.sum(y_pred_match) == 0:
            continue
        else:
            pred_nuclei_indices = np.unique(y_pred_match)
            pred_nuclei_indices = np.delete(pred_nuclei_indices,
                                            pred_nuclei_indices == 0)
            for j in pred_nuclei_indices:
                if (np.sum(y_pred_match == j) > 0.5 * nuclei_area
                        and j not in matched_pred_indices):
                    true_positive += 1
                    matched_pred_indices.append(j)
                    break
    false_positive = n_pred_nuclei - true_positive
    false_negative = n_true_nuclei - true_positive

    return false_positive, false_negative, true_positive


def dice_coef(y_true, y_pred):
    intersection = np.sum((y_true>0) * (y_pred>0))
    union = np.sum(y_pred>0) + np.sum(y_true>0)
    return (2. * intersection + 0.000001) / (union + 0.000001)

def monuseg_metrics(y_true, y_pred):
    y_true_indices = np.unique(y_true)
    y_pred_indices = np.unique(y_pred)

    y_true_indices = np.delete(y_true_indices, y_true_indices == 0)
    y_pred_indices = np.delete(y_pred_indices, y_pred_indices == 0)

    overall_correct_count = 0
    union_pixel_count = 0
    matched_pred_count = {i: 0 for i in y_pred_indices}
    for i in y_true_indices:
        nuclei_mask = y_true == i
        y_pred_match = nuclei_mask * y_pred

        if np.sum(y_pred_match) == 0:
            union_pixel_count += np.sum(nuclei_mask)
            continue
        else:
            pred_nuclei_indices = np.unique(y_pred_match)
            pred_nuclei_indices = np.delete(pred_nuclei_indices,
                                            pred_nuclei_indices == 0)
            jaccard_index = 0
            for j in pred_nuclei_indices:
                matched = y_pred == j
                intersection_tmp = np.sum(matched & nuclei_mask)
                union_tmp = np.sum(matched | nuclei_mask)
                ji_tmp = intersection_tmp / union_tmp
                if ji_tmp > jaccard_index:
                    jaccard_index = ji_tmp
                    best_match = j
                    correct_count = intersection_tmp
                    union = union_tmp

            overall_correct_count += correct_count
            union_pixel_count += union
            matched_pred_count[best_match] += 1

    unmatched_pred = [
        k for k in matched_pred_count.keys() if matched_pred_count[k] == 0
    ]
    for k in unmatched_pred:
        union_pixel_count += np.sum(y_pred == k)
    return overall_correct_count / union_pixel_count

