# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 13:35:03 2020
cascade15 Non dterministic model
@author: srias292
"""


from enum import Enum
import random
import math
from collections import defaultdict
import matplotlib.pyplot as plt
import pytest
from numpy.random import choice
import numpy as np
from scipy.stats import cumfreq


actions = list()
signals = list()


class Signal(Enum):
    H = 1
    L = 0


class Action(Enum):
    A = 1
    R = 0


def observed_data(n, p, v):
    # Returns the list of signals for n number of people
    signals = []
    for x in range(0, n):
        num1 = random.uniform(0, 1)
        if num1 <= p:
            signal = Signal.H if v == 1 else Signal.L
        elif num1 > p:
            signal = Signal.H if v == 0 else Signal.L
        signals.append(signal)
    return signals


def normalising_touple_dict(dict_tobe_normalised: dict) -> dict:
    total = sum(dict_tobe_normalised.values())
    if total != 0:
        for c in range(int(len(dict_tobe_normalised) / 2)):
            for v in range(2):
                dict_tobe_normalised[v, c] = dict_tobe_normalised[v, c] / total
    return dict_tobe_normalised


def normalising_single_dict(dict_tobe_normalised: dict) -> dict:
    total = sum(dict_tobe_normalised.values())
    if total != 0:
        for v in range(2):
            dict_tobe_normalised[v] = dict_tobe_normalised[v] / total
    return dict_tobe_normalised


def prior_data(num_prior_agents, p):
    prob_VC_prior_agents = defaultdict(int)
    for c in range(num_prior_agents + 1):
        for v in range(2):
            if v == 0:
                prob_VC_prior_agents[v, c] = pow((1 - p), c) * pow(
                    p, num_prior_agents - c
                )
            else:
                prob_VC_prior_agents[v, c] = pow((p), c) * pow(
                    (1 - p), num_prior_agents - c
                )
    return normalising_touple_dict(prob_VC_prior_agents)


def count_cascades(a_list, true_cascade_count, false_cascade_count, no_cascade_count):
    second_half = list()
    half = len(a_list) // 2
    second_half = a_list[half:]
    if all(a.name == "A" for a in second_half):
        true_cascade_count = true_cascade_count + 1
    elif all(a.name == "R" for a in second_half):
        false_cascade_count = false_cascade_count + 1
    else:
        no_cascade_count = no_cascade_count + 1
    return true_cascade_count, false_cascade_count, no_cascade_count


def plot_graph(
    list_p,
    true_cascade_list,
    false_cascade_list,
    no_cascade_list,
    num_prior_agents,
    real_v,
):
    plt.plot(list_p, true_cascade_list, label="up_cascade")
    plt.plot(list_p, false_cascade_list, label="down_cascade")
    plt.plot(list_p, no_cascade_list, label="no_cascade")
    plt.xlabel("Signal Accuracy (p)")
    plt.ylabel("Probabilty Ratio of Cascade")
    plt.title(
        "Graph for cascade for "
        + str(n)
        + " number of agents with "
        + str(runs)
        + " simulations with num_prior_agents "
        + str(num_prior_agents)
        + "and real_v"
        + str(real_v)
    )
    plt.text(
        1.0,
        0.4,
        "up_cascade_list"
        + str(true_cascade_list)
        + "\n down_cascade_list"
        + str(false_cascade_list)
        + "\n no_cascade_list"
        + str(no_cascade_list),
        fontsize=12,
    )
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14))
    plt.xticks(list_p, [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    plt.show()


def func_marginalise_pVC_previous_over_v(dict_to_be_marginalised: dict) -> dict:
    marginalised_dict = defaultdict(int)
    for c in range(int(len(dict_to_be_marginalised) / 2)):
        marginalised_dict[c] = (
            dict_to_be_marginalised[0, c] + dict_to_be_marginalised[1, c]
        )
    return marginalised_dict


def func_marginalise_pVC_previous_over_c(dict_to_be_marginalised: dict) -> dict:
    marginalised_dict = defaultdict(int)
    for v in range(2):
        marginalised_dict[v] = sum(
            dict_to_be_marginalised[v, ci]
            for ci in range(int(len(dict_to_be_marginalised) / 2))
        )
    return marginalised_dict


def func_adding_own_signal(
    pVC_previous: list, agent_index: int, x: Signal, p: int
) -> dict:
    # print("Adding own signal")
    pVC_given_prevC_X = defaultdict(int)
    pV_given_prevC_X = defaultdict(int)
    # marginalising over c
    pV_previous = func_marginalise_pVC_previous_over_c(pVC_previous)
    # print("pV_previous", pV_previous)

    def func_pXgivenV_prevC(x: Signal, p: int, v: int) -> dict:
        pXgivenVC = 0
        if (x == Signal.H and v == 1) or (x == Signal.L and v == 0):
            pXgivenVC = p
        else:
            pXgivenVC = 1 - p
        return pXgivenVC

    for v in range(2):
        pV_given_prevC_X[v] = pV_previous[v] * func_pXgivenV_prevC(x, p, v)

    # print("before normalised pV_given_CX", pV_given_prevC_X)

    # Normalising
    pV_given_prevC_X = normalising_single_dict(pV_given_prevC_X)
    # print("after normalised pV_given_CX", pV_given_prevC_X)

    # marginalising over v
    pC_previous = func_marginalise_pVC_previous_over_v(pVC_previous)
    # print("pVC_previous_over_v", pC_previous)

    def func_pC_given_prevC_X(pC_previous, c, x):
        if x == Signal.H:
            return pC_previous[c - 1]
        elif c in pC_previous:
            return pC_previous[c]
        else:
            return 0

    for count in range(
        agent_index + 2
    ):  # agent_index should be changed interms of length of pVC_given_prevC_X
        for v in range(2):
            pVC_given_prevC_X[v, count] = pV_given_prevC_X[v] * func_pC_given_prevC_X(
                pC_previous, count, x
            )
    return normalising_touple_dict(pVC_given_prevC_X)


def choose_action(prob_dist_c):
    # print("Choosing action")
    prob_dist_current_c = func_marginalise_pVC_previous_over_v(prob_dist_c)
    # print("prob_dist_current_c after marginalisation over v", prob_dist_current_c)
    prob_count_sum_of_lower_half = sum(
        [
            prob_dist_current_c[j]
            for j in range(math.ceil((len(prob_dist_current_c) - 1) / 2))
        ]
    )
    # print("sum_of_lower_half", prob_count_sum_of_lower_half)

    prob_count_sum_of_upper_half = sum(
        [
            prob_dist_current_c[j]
            for j in range(
                math.floor((len(prob_dist_current_c) + 1) / 2),
                len(prob_dist_current_c),
            )
        ]
    )
    # print("sum_of_upper_half", prob_count_sum_of_upper_half)
    def middle_value(dict):
        if len(dict) % 2 != 0:
            middle = dict[math.trunc(len(dict) / 2)]
        else:
            middle = 0
            # print("middle", middle)
        return middle

    middle = middle_value(prob_dist_current_c)
    choices = choice(
        ["R", "T", "A"],
        p=(prob_count_sum_of_lower_half, middle, prob_count_sum_of_upper_half),
    )
    # print("choice", choices)
    if choices == "T":
        random_number = random.randrange(2)
        if random_number == 1:
            return Action.A
        else:
            return Action.R
    elif choices == "A":
        return Action.A
    else:
        return Action.R


## -------------------------------- Simulation Starts ------------------------------------------- ##
def simulation(n, p, num_prior_agents, real_v):
    actions = []
    signals = []
    list_of_pVCj_given_Aj = [0] * (n + 1)
    prob_VC_prior_agents = prior_data(num_prior_agents, p)
    print(
        "prior data of ",
        num_prior_agents,
        " number of prior agents = ",
        prob_VC_prior_agents,
    )
    signals = observed_data(n, p, real_v)
    signals = [Signal.H, Signal.H, Signal.H, Signal.L, Signal.L]
    print("Private signals of ", n, " number of agents", signals)
    print(
        "....................................................................................."
    )
    list_of_pVCj_given_Aj[0] = prob_VC_prior_agents

    for j in range(
        n
    ):  # Agent1 starts at j = 0, and it includes prob distribution 0 & 1
        print("agent_index", j + num_prior_agents)
        if j > 0:
            print(
                "Agent ",
                num_prior_agents + j + 1,
                "sees agent ",
                num_prior_agents + j,
                "'s action",
                actions[j - 1],
            )
            # marginalise over c avoiding c's
            marginalised_over_c = defaultdict(int)
            marginalised_over_c = func_marginalise_pVC_previous_over_c(
                list_of_pVCj_given_Aj[j - 1]
            )
            print("Agent look up ", list_of_pVCj_given_Aj[j - 1])
            print("marginalised_over_c of j-2...", marginalised_over_c)
            prob_prev_x_givenV = defaultdict(int)
            prob_prev_x_givenV[Signal.H] = (
                p * marginalised_over_c[1] + (1 - p) * marginalised_over_c[0]
            )
            prob_prev_x_givenV[Signal.L] = (
                p * marginalised_over_c[0] + (1 - p) * marginalised_over_c[1]
            )

            print(
                "prob_prev_x_givenVC of",
                j + num_prior_agents - 1,
                "th agent",
                prob_prev_x_givenV,
            )

            # pad right
            def func_pad_righted_dict(Prob_dict):
                pad_righted_dict = {}
                for i in range(len(Prob_dict)):
                    pad_righted_dict[i] = Prob_dict[i]
                pad_righted_dict[len(Prob_dict)] = 0
                return pad_righted_dict

            # shift right
            def func_shift_righted_dict(Prob_dict):
                shift_righted_dict = {}
                shift_righted_dict[0] = 0
                for i in range(len(Prob_dict)):
                    shift_righted_dict[i + 1] = Prob_dict[i]
                return shift_righted_dict

            def func_sum_lower_half(prob_dist):
                sum_of_lower_half = sum(
                    [prob_dist[j] for j in range(math.ceil((len(prob_dist) - 1) / 2))]
                )
                return sum_of_lower_half

            def func_sum_upper_half(prob_dist):
                sum_of_upper_half = sum(
                    [
                        prob_dist[j]
                        for j in range(
                            math.floor((len(prob_dist) + 1) / 2), len(prob_dist),
                        )
                    ]
                )
                return sum_of_upper_half

            def middle_value(dict):
                if len(dict) % 2 != 0:
                    middle = dict[math.trunc(len(dict) / 2)]
                else:
                    middle = 0
                return middle

            def func_pA_given_X_prev_C(a, prev_c_dist, x):
                marginalised_over_v = defaultdict(int)
                marginalised_over_v = func_marginalise_pVC_previous_over_v(prev_c_dist)
                # print("marginalised_over_V", marginalised_over_v)
                pad_righted_array = func_pad_righted_dict(marginalised_over_v)
                shift_righted_array = func_shift_righted_dict(marginalised_over_v)
                # print("pad_righted_array", pad_righted_array)
                # print("shift_righted_array", shift_righted_array)

                if a == Action.A:
                    return (
                        (
                            func_sum_upper_half(shift_righted_array)
                            + middle_value(shift_righted_array) / 2
                        )
                        if x is Signal.H
                        else (
                            func_sum_upper_half(pad_righted_array)
                            + middle_value(pad_righted_array) / 2
                        )
                    )
                else:

                    return (
                        (
                            func_sum_lower_half(shift_righted_array)
                            + middle_value(shift_righted_array) / 2
                        )
                        if x is Signal.H
                        else (
                            func_sum_lower_half(pad_righted_array)
                            + middle_value(pad_righted_array) / 2
                        )
                    )

            # Bayes Rule
            prob_x_given_AVC = defaultdict(int)
            prob_x_given_AVC[Signal.H] = prob_prev_x_givenV[
                Signal.H
            ] * func_pA_given_X_prev_C(
                actions[j - 1], list_of_pVCj_given_Aj[j - 1], Signal.H
            )
            prob_x_given_AVC[Signal.L] = prob_prev_x_givenV[
                Signal.L
            ] * func_pA_given_X_prev_C(
                actions[j - 1], list_of_pVCj_given_Aj[j - 1], Signal.L
            )
            print("After Bayes rule prob_prev_x_givenVC", prob_x_given_AVC)
            prob_prevCV_givenA = defaultdict(int)
            for prevc in range(j + num_prior_agents + 1):
                for v in range(2):
                    prob_prevCV_givenA[v, prevc] = (
                        list_of_pVCj_given_Aj[j - 1][v, prevc]
                        * prob_x_given_AVC[Signal.L]
                    ) + (
                        list_of_pVCj_given_Aj[j - 1][v, prevc - 1]
                        * prob_x_given_AVC[Signal.H]
                    )

            print("prob_prevC_givenA", prob_prevCV_givenA)
            pVC_given_X = func_adding_own_signal(
                prob_prevCV_givenA, j + num_prior_agents, signals[j], p
            )
            print(
                "Agent ",
                j + 1 + num_prior_agents,
                "uses his own signal",
                signals[j],
                "to calculate prob_V_cj_given_C_Xj---------  ",
                pVC_given_X,
            )
        else:  # j == 0 case
            print(
                "Agent ",
                num_prior_agents + j + 1,
                "sees  ",
                num_prior_agents,
                "prior agent/agents",
            )
            pVC_given_X = func_adding_own_signal(
                list_of_pVCj_given_Aj[0], j + num_prior_agents, signals[j], p
            )
            print(
                "Agent ",
                j + 1 + num_prior_agents,
                "uses his own signal",
                signals[j],
                "to calculate prob_V_cj_given_C_Xj---------  ",
                pVC_given_X,
            )

        # choose action
        actions.append(choose_action(pVC_given_X))
        list_of_pVCj_given_Aj[j + 1] = pVC_given_X
        print("Agent ", j + 1 + num_prior_agents, "chooses", actions[j])
        print(
            "................................................................................"
        )
    return actions, signals


n = 100
list_p = [0.5, 0.6, 0.7, 0.8, 0.9]
runs = 1000
real_v = 1
num_prior_agents = 1
true_cascade_list = list()
false_cascade_list = list()
no_cascade_list = list()
list_a_seq_of_p = []
list_r_seq_of_p = []
dict_count_actions_for_all_ps = {}
for k in range(len(list_p)):  # changing the value for p
    print("number of agents = ", n)
    print("number of prior agents", num_prior_agents)
    print("value of p, the signal acuracy = ", list_p[k])
    print("real v = ", real_v)
    p = list_p[k]
    true_cascade_count = 0
    false_cascade_count = 0
    no_cascade_count = 0
    list_a_cum_freq = []
    list_r_cum_freq = []
    num1 = 0
    num2 = 0
    num3 = 0
    for r in range(0, runs):
        actions, signals = simulation(n, p, num_prior_agents, real_v)
        print("private signals", signals)
        print("Chosen actions", actions)
        if abs(actions.count(Action.A) - actions.count(Action.R)) < 20:
            num1 += 1
        elif actions.count(Action.A) - actions.count(Action.R) >= 20:
            num2 += 1
        elif actions.count(Action.R) - actions.count(Action.A) >= 20:
            num3 += 1
        true_cascade_count, false_cascade_count, no_cascade_count = count_cascades(
            actions, true_cascade_count, false_cascade_count, no_cascade_count
        )
        print(true_cascade_count, false_cascade_count, no_cascade_count)
        a_cum_freq = cumfreq(
            [i for i, a in enumerate(actions) if a == Action.A],
            numbins=n,
            defaultreallimits=(0, n),
        )
        # print("a_cum_freq  ", a_cum_freq.cumcount)
        list_a_cum_freq.append(a_cum_freq.cumcount)
        r_cum_freq = cumfreq(
            [i for i, a in enumerate(actions) if a == Action.R],
            numbins=n,
            defaultreallimits=(0, n),
        )
        list_r_cum_freq.append(r_cum_freq.cumcount)
        # print("r_cum_freq   ", r_cum_freq.cumcount)
        plt.plot(r_cum_freq.cumcount, color="r", label="R" if r == 0 else "")
        plt.plot(a_cum_freq.cumcount, color="g", label="A" if r == 0 else "")
        plt.xlabel("Number of agents")
        plt.ylabel("Number of adoptions/rejections")
        plt.legend(loc="upper left")
        plt.title("Signal accuracy p =  " + str(list_p[k]))
    plt.show()
    dict_count_actions_for_all_ps[p] = num1, num2, num3
    true_cascade_ratio = true_cascade_count / runs
    false_cascade_count_ratio = false_cascade_count / runs
    no_cascade_count_ratio = no_cascade_count / runs
    print(true_cascade_ratio, false_cascade_count_ratio, no_cascade_count_ratio)
    true_cascade_list.append(true_cascade_ratio)
    false_cascade_list.append(false_cascade_count_ratio)
    no_cascade_list.append(no_cascade_count_ratio)
    print("up_cascade_list", true_cascade_list)
    print("down_cascade_list", false_cascade_list)
    print("no_cascade_list", no_cascade_list)
    a_seq = np.mean(list_a_cum_freq, axis=0)
    list_a_seq_of_p.append(a_seq)
    # print("list_a_seq_of_p", list_a_seq_of_p)
    r_seq = np.mean(list_r_cum_freq, axis=0)
    list_r_seq_of_p.append(r_seq)
    # print("list_r_seq_of_p", list_r_seq_of_p)
    print("..................................................................")
