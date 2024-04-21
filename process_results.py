import random
import numpy as np
import tqdm as tqdm

def get_all_numbers(test_set, results):
    num_unamb_correct_pred = 0
    num_unamb_subset = 0
    num_unamb_superset = 0
    num_unamb_noncompliance = 0

    num_amb_correct_pred = 0
    num_amb_subset = 0
    num_amb_superset = 0
    num_amb_noncompliance = 0

    for data in test_set:
        poss_options = data['poss_options']
        true_options = data['true_options']
        flex_options = data['flex_options']
        llm_preds = data['llm_preds']

        if len(poss_options) == 1:
            if set(llm_preds) == set(poss_options):
                num_unamb_correct_pred += 1
            elif set(llm_preds).issubset(set(poss_options)):
                num_unamb_subset += 1
            elif set(poss_options).issubset(set(llm_preds)):
                num_unamb_superset += 1
            else:
                num_unamb_noncompliance += 1
        else:
            if set(llm_preds) == set(poss_options):
                num_amb_correct_pred += 1
            elif set(llm_preds).issubset(set(poss_options)):
                num_amb_subset += 1
            elif set(poss_options).issubset(set(llm_preds)):
                num_amb_superset += 1
            else:
                num_amb_noncompliance += 1

    results["num_unamb_correct_pred"] = num_unamb_correct_pred
    results["num_unamb_subset"] = num_unamb_subset
    results["num_unamb_superset"] = num_unamb_superset
    results["num_unamb_noncompliance"] = num_unamb_noncompliance
    results["num_amb_correct_pred"] = num_amb_correct_pred
    results["num_amb_subset"] = num_amb_subset
    results["num_amb_superset"] = num_amb_superset
    results["num_amb_noncompliance"] = num_amb_noncompliance

    return results

def get_llm_preds_multi_label(test_set, qhat):
    for data in test_set:
        top_probs = data['top_probs']
        top_lists = data['top_lists']
        # include all options with score >= 1-qhat
        prediction_set = [
                options for options_ind, options in enumerate(top_lists)
                if top_probs[options_ind] >= 1 - qhat
            ]
        data['llm_preds'] = prediction_set
    return test_set

def get_results_multi_label(test_set):
    num_correct_pred = 0
    num_help = 0
    num_success = 0
    set_size_all = []
    num_test_data = len(test_set)

    for data in test_set:
        poss_options = data['poss_options']
        true_options = data['true_options']
        flex_options = data['flex_options']
        llm_preds = data['llm_preds']
        set_size_all.append(len(llm_preds))

        num_success += any([set(preds) == set(poss_options) for preds in llm_preds])

    # get average rate
    success_rate = num_success / num_test_data
    avg_prediction_set_size = np.mean(set_size_all)

    results = {"success_rate": success_rate, "avg_prediction_set_size": avg_prediction_set_size}

    return results


def get_results(test_set):
    num_correct_pred = 0
    num_help = 0
    num_success = 0
    set_size_all = []
    num_test_data = len(test_set)

    for data in test_set:
        poss_options = data['poss_options']
        true_options = data['true_options']
        flex_options = data['flex_options']
        llm_preds = data['llm_preds']
        set_size_all.append(len(llm_preds))

        # check coverage
        flag_coverage = not set(llm_preds).isdisjoint(true_options)

        # check help - if prediction set is not singleton, or set include option E
        flag_help = len(llm_preds) != 1 or data['add_mc_prefix'] in llm_preds
        num_help += flag_help

        # check success - same as coverage
        num_success += flag_coverage
        llm_pred_flex = [item for item in llm_preds if item not in flex_options]
        num_correct_pred += (set(llm_pred_flex) == set(poss_options))

    # get average rate
    correct_pred_rate = num_correct_pred / num_test_data
    help_rate = num_help / num_test_data
    success_rate = num_success / num_test_data
    avg_prediction_set_size = np.mean(set_size_all)

    results = {"correct_pred_rate": correct_pred_rate, "help_rate": help_rate, "success_rate": success_rate,
               "avg_prediction_set_size": avg_prediction_set_size}

    results = get_all_numbers(test_set, results)

    return results