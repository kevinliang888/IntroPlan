import random
import numpy as np

def get_top_logprobs(response):
    all_token_probs = response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    top_logprobs_full = {}
    all_tokens = ["A", "B", "C", "D", "E"]
    for i in range(len(all_token_probs)):
        cur_token = all_token_probs[i]["token"].strip()
        if cur_token in all_tokens and cur_token not in top_logprobs_full:
            top_logprobs_full[cur_token] = all_token_probs[i]["logprob"]
    return top_logprobs_full

def get_top_logprobs_orig(response):
    all_token_probs = response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    top_logprobs_full = {}
    for i in range(len(all_token_probs)):
        top_logprobs_full[all_token_probs[i]["token"]] = all_token_probs[i]["logprob"]
    return top_logprobs_full

def temperature_scaling(logits, temperature):
    logits = np.array(logits)
    logits /= temperature

    # apply softmax
    logits -= logits.max()
    logits = logits - np.log(np.sum(np.exp(logits)))
    smx = np.exp(logits)
    return smx

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

def get_non_conformity_score_multi_label(calibration_set, use_direct_preds=False):
    non_conformity_score = []
    token_all = ['A', 'B', 'C', 'D', 'E']
    for data in calibration_set:
        top_probs = data['top_probs']
        top_lists = data['top_lists']
        poss_options = data['poss_options']

        # get the softmax value of true option
        prob = [top_probs[list_ind] for list_ind, list in enumerate(top_lists) if set(list) == set(poss_options)][0]

        # get non-comformity score
        non_conformity_score.append(1 - prob)
    return non_conformity_score

def get_non_conformity_score(calibration_set, use_direct_preds=False):
    non_conformity_score = []
    token_all = ['A', 'B', 'C', 'D', 'E']
    for data in calibration_set:
        top_logprobs = data['top_logprobs']
        top_tokens = data['top_tokens']
        true_options = data['true_options']

        # normalize the five scores to sum of 1
        mc_smx_all = temperature_scaling(top_logprobs, temperature=5)

        if 'initial_preds' in data:
            initial_preds = data['initial_preds']
            mc_sum = sum([mc_smx_all[i] for i in range(len(mc_smx_all)) if top_tokens[i] in initial_preds])
            for i in range(len(mc_smx_all)):
                if top_tokens[i] in initial_preds:
                    mc_smx_all[i] = mc_smx_all[i] / mc_sum

        # get the softmax value of true option
        true_label_smx = [mc_smx_all[token_ind]
                          for token_ind, token in enumerate(top_tokens)
                          if token in true_options]
        true_label_smx = np.max(true_label_smx)

        # get non-comformity score
        non_conformity_score.append(1 - true_label_smx)
    return non_conformity_score


def get_llm_preds(test_set, qhat, use_direct_preds=False):
    for data in test_set:
        top_logprobs = data['top_logprobs']
        top_tokens = data['top_tokens']
        # normalize the five scores to sum of 1
        mc_smx_all = temperature_scaling(top_logprobs, temperature=5)

        if 'initial_preds' in data:
            initial_preds = data['initial_preds']
            mc_sum = sum([mc_smx_all[i] for i in range(len(mc_smx_all)) if top_tokens[i] in initial_preds])
            for i in range(len(mc_smx_all)):
                if top_tokens[i] in initial_preds:
                    mc_smx_all[i] = mc_smx_all[i] / mc_sum

        # include all options with score >= 1-qhat
        prediction_set = [
            token for token_ind, token in enumerate(top_tokens)
            if mc_smx_all[token_ind] >= 1 - qhat
        ]
        data['llm_preds'] = prediction_set
    return test_set