```python
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 13:35:03 2020
Deterministic and Non deterministic choice model
@author: srias292
"""

from enum import Enum
import random
import math
from collections import defaultdict
import matplotlib.pyplot as plt
import pytest
import numpy as np
from scipy.stats import cumfreq
from numpy.random import choice


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
    # Returns the normalised touple dict
    total = sum(dict_tobe_normalised.values())
    if total != 0:
        for c in range(int(len(dict_tobe_normalised) / 2)):
            for v in range(2):
                dict_tobe_normalised[v, c] = dict_tobe_normalised[v, c] / total
    return dict_tobe_normalised


def normalising_single_dict(dict_tobe_normalised: dict) -> dict:
    # Returns the normalised single dict
    total = sum(dict_tobe_normalised.values())
    if total != 0:
        for v in range(2):
            dict_tobe_normalised[v] = dict_tobe_normalised[v] / total
    return dict_tobe_normalised
```

```python

def prior_data(num_prior_agents, p):
    prob_VC_prior_agents = defaultdict(int)

```

creating normal distribution for num_prior agents

```python
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

```

Function adding own signal
```python
def func_adding_own_signal(
    pVC_previous: list, agent_index: int, x: Signal, p: int
) -> dict:
    pVC_given_prevC_X = defaultdict(int)
    pV_given_prevC_X = defaultdict(int)
    # marginalising over c
    pV_previous = func_marginalise_pVC_previous_over_c(pVC_previous)
    print("pV_previous", pV_previous)

    def func_pXgivenV_prevC(x: Signal, p: int, v: int) -> dict:
        pXgivenVC = 0
        if (x == Signal.H and v == 1) or (x == Signal.L and v == 0):
            pXgivenVC = p
        else:
            pXgivenVC = 1 - p
        return pXgivenVC

    for v in range(2):
        pV_given_prevC_X[v] = pV_previous[v] * func_pXgivenV_prevC(x, p, v)

    # Normalising
    pV_given_prevC_X = normalising_single_dict(pV_given_prevC_X)

    # marginalising over v
    pC_previous = func_marginalise_pVC_previous_over_v(pVC_previous)
    print("pVC_previous_over_v", pC_previous)

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
```

Choosing action for deterministic method

```python
def choose_action_deterministic(prob_dist_c):
    print("Choosing action for deterministic choice")
    prob_dist_current_c = func_marginalise_pVC_previous_over_v(prob_dist_c)
    # print("prob_dist_current_c after marginalisation over v", prob_dist_current_c)
    sum_of_lower_half_and_middle = sum(
        [prob_dist_current_c[j] for j in range(math.ceil(len(prob_dist_current_c) / 2))]
    )

    sum_of_upper_half_and_middle = sum(
        [
            prob_dist_current_c[j]
            for j in range(
                math.floor(len(prob_dist_current_c) / 2), len(prob_dist_current_c)
            )
        ]
    )

    print("prob_count_sum_of_lower_half with middle", sum_of_lower_half_and_middle)
    print("prob_count_sum_of_upper_half with middle", sum_of_upper_half_and_middle)
    if sum_of_lower_half_and_middle < sum_of_upper_half_and_middle:
        return Action.A
    elif sum_of_lower_half_and_middle == sum_of_upper_half_and_middle:
        random_number = random.randrange(2)
        if random_number == 1:
            return Action.A
        else:
            return Action.R
    else:
        return Action.R
```

Choosing action for non deterministic method

```python
def choose_action_non_deterministic(prob_dist_c):
    prob_dist_current_c = func_marginalise_pVC_previous_over_v(prob_dist_c)
    prob_count_sum_of_lower_half = sum(
        [
            prob_dist_current_c[j]
            for j in range(math.ceil((len(prob_dist_current_c) - 1) / 2))
        ]
    )

    prob_count_sum_of_upper_half = sum(
        [
            prob_dist_current_c[j]
            for j in range(
                math.floor((len(prob_dist_current_c) + 1) / 2),
                len(prob_dist_current_c),
            )
        ]
    )

    def middle_value(dict):
        if len(dict) % 2 != 0:
            middle = dict[math.trunc(len(dict) / 2)]
        else:
            middle = 0
        return middle

    choices = choice(
        ["R", "T", "A"],
        p=(
            prob_count_sum_of_lower_half,
            middle_value(prob_dist_current_c),
            prob_count_sum_of_upper_half,
        ),
    )
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
```

The information cascade model is used as follows.
$V$ and $C_i$ are conditionally independent given $C_{i-1}$ and $X_i$, therefore:

$P(V, C_i \mid C_{i-1}, X_i) = p(V \mid C_{i-1}, X_i) \, p(C_i \mid C_{i-1}, X_i)$

```python

## -------------------------------- Simulation Starts ------------------------------------------- ##
def simulation(n, p, num_prior_agents, real_v, choice_method):
    actions = []
    signals = []

    list_of_pVCj_given_Aj = [0] * (n + 1)
```

Each agent $i$ maintains a joint probability distribution over $V$ and $C$ that is conditional on the actions observed so far. This is $p(V,C_0) = p(V)\, p(C_0 \mid V)$ for first agent. We use a uniform prior over $V$ and a binomial distribution for $C_0$ given

$P(C_0=c \mid V=0) = (1-p)^c p^{k-c} \text{\ \ and \ \ } P(C_0=c \mid V=1) = p^c (1-p)^{k-c}$

```python
    prob_VC_prior_agents = prior_data(num_prior_agents, p)
    print(
        "prior data of ",
        num_prior_agents,
        " number of prior agents = ",
        prob_VC_prior_agents,
    )
```

Random signals either $H$ or $L$ will be created

```python
    signals = observed_data(n, p, real_v)
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
```

For $i>1$, agent $i$ will observe the actions of all prior agents $j<i$ and compute the joint conditional distribution $p(V,C_j \mid C_0, A_1, \,\cdots\!, A_j)$. Agents act and are observed sequentially, so it will already have computed $p(V,C_{j-1} \mid C_0,A_1, \cdots\!, A_{j-1})$.

Applying Bayes Rule for $j>0:$

$P(X_j \mid A_1,\cdots,A_j) \propto P(X_j \mid A_1,\,\cdots,A_{j-1}) \, P(A_j \mid X_j, A_1,\cdots,A_{j-1})$

where:
$P(X_j \mid A_1,\cdots,A_{j-1}) =
             \sum_{v\in{0,1}} P(X_j \mid V=v, A_1,\cdots,A_{j-1})\, P(V=v \mid A_1,\cdots,A_{j-1}) \\
             = \sum_{v\in{0,1}} P(X_j \mid V=v)\, P(V=v \mid A_1,\cdots,A_{j-1})$

```python
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
```

$P(A_j \mid X_j, A_1,\cdots,A_{j-1}) = P(A_j \mid X_j, C_{j-1})$
$P(A_{j} \mid X_{j}, C_{j-1})$ is calculated as shown in below for the deterministic choice model

| $X_{j}$ | Condition on $C_{j-1}$                                                                                                                                                                                   | $P(A_j\!=\!A)$   | $P(A_j\!=\!R)$   |
| ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- | ---------------- | --- |
| $H$     | $P(C_{j-1} + 1 > \tfrac{j}{2}) > P(C_{j-1} + 1 < \tfrac{j}{2})$ <br>$P(C_{j-1} + 1 > \tfrac{j}{2}) < P(C_{j-1} + 1 < \tfrac{j}{2})$ <br> $P(C_{j-1} + 1 > \tfrac{j}{2}) = P(C_{j-1} + 1 < \tfrac{j}{2})$ | 1 <br> 0 <br>0.5 | 0 <br> 1 <br>0.5 |     |
| $L$     | $P(C_{j-1} > \tfrac{j}{2}) > P(C_{j-1} < \tfrac{j}{2})$ <br> $P(C_{j-1} > \tfrac{j}{2}) < P(C_{j-1}  < \tfrac{j}{2})$ <br> $P(C_{j-1} > \tfrac{j}{2}) = P(C_{j-1} < \tfrac{j}{2})$                       | 1 <br> 0 <br>0.5 | 0 <br> 1 <br>0.5 |

```python

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

            def func_sum_lower_half_and_middle(prob_dist):
                sum_of_lower_half = sum(
                    [prob_dist[j] for j in range(math.ceil(len(prob_dist) / 2))]
                )
                return sum_of_lower_half

            def func_sum_upper_half_and_middle(prob_dist):
                sum_of_upper_half = sum(
                    [
                        prob_dist[j]
                        for j in range(
                            math.floor(len(prob_dist) / 2),
                            len(prob_dist),
                        )
                    ]
                )
                return sum_of_upper_half

            def func_pA_given_X_prev_C_deterministic(a, prev_c_dist, x):
                marginalised_over_v = defaultdict(int)
                marginalised_over_v = func_marginalise_pVC_previous_over_v(prev_c_dist)
                pad_righted_array = func_pad_righted_dict(marginalised_over_v)
                shift_righted_array = func_shift_righted_dict(marginalised_over_v)
                if a == Action.A:
                    sum_upper_half_and_middle = func_sum_upper_half_and_middle(
                        shift_righted_array
                    )
                    sum_lower_half_and_middle = func_sum_lower_half_and_middle(
                        shift_righted_array
                    )
                    if sum_upper_half_and_middle > sum_lower_half_and_middle:
                        return 1 if x is Signal.H else 0
                    elif sum_upper_half_and_middle < sum_lower_half_and_middle:
                        return 0 if x is Signal.H else 1
                    else:
                        return 0.5
                else:
                    sum_upper_half_and_middle = func_sum_upper_half_and_middle(
                        pad_righted_array
                    )
                    sum_lower_half_and_middle = func_sum_lower_half_and_middle(
                        pad_righted_array
                    )
                    if sum_upper_half_and_middle > sum_lower_half_and_middle:
                        return 1 if x is Signal.H else 0
                    elif sum_upper_half_and_middle < sum_lower_half_and_middle:
                        return 0 if x is Signal.H else 1
                    else:
                        return 0.5
```

$P(A_{j} \mid X_{j}, C_{j-1})$ is calculated as shown in below for the non deterministic choice model

| $X_{j}$ | $P(A_j\!=\!A)$                                                               | $P(A_j\!=\!R)$                                                                   |
| ------- | ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| $H$     | $P(C_{j-1} + 1 > \tfrac{j}{2}) + \tfrac{1}{2} P(C_{j-1} + 1 = \tfrac{j}{2})$ | $P(C_{j-1} + 1 < \tfrac{}{}{j}{2}) + \tfrac{1}{2} P(C_{j-1} + 1 = \tfrac{j}{2})$ |
| $L$     | $P(C_{j-1} > \tfrac{j}{2}) + \tfrac{1}{2} P(C_{j-1} = \tfrac{j}{2})$         | $P(C_{j-1} < \tfrac{j}{2}) + \tfrac{1}{2} P(C_{j-1} = \tfrac{j}{2})$             |

```python
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
                            math.floor((len(prob_dist) + 1) / 2),
                            len(prob_dist),
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

            def func_pA_given_X_prev_C_for_non_deterministic(a, prev_c_dist, x):
                marginalised_over_v = defaultdict(int)
                marginalised_over_v = func_marginalise_pVC_previous_over_v(prev_c_dist)
                print(
                    "marginalised_over_V inside the func_pA_given_X_prev_C",
                    marginalised_over_v,
                )
                print("x", x)
                pad_righted_array = func_pad_righted_dict(marginalised_over_v)
                shift_righted_array = func_shift_righted_dict(marginalised_over_v)
                print("pad_righted_array", pad_righted_array)
                print("shift_righted_array", shift_righted_array)

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

```

Applying now Bayes Rule to find
$P(X_j \mid A_1,\cdots,A_j)$

```python
             # Bayes Rule
            prob_x_given_AVC = defaultdict(int)
            if choice_method == "deterministic":
                prob_x_given_AVC[Signal.H] = prob_prev_x_givenV[
                    Signal.H
                ] * func_pA_given_X_prev_C_deterministic(
                    actions[j - 1], list_of_pVCj_given_Aj[j - 1], Signal.H
                )
                prob_x_given_AVC[Signal.L] = prob_prev_x_givenV[
                    Signal.L
                ] * func_pA_given_X_prev_C_deterministic(
                    actions[j - 1], list_of_pVCj_given_Aj[j - 1], Signal.L
                )
            else:
                prob_x_given_AVC[Signal.H] = prob_prev_x_givenV[
                    Signal.H
                ] * func_pA_given_X_prev_C_for_non_deterministic(
                    actions[j - 1], list_of_pVCj_given_Aj[j - 1], Signal.H
                )
                prob_x_given_AVC[Signal.L] = prob_prev_x_givenV[
                    Signal.L
                ] * func_pA_given_X_prev_C_for_non_deterministic(
                    actions[j - 1], list_of_pVCj_given_Aj[j - 1], Signal.L
                )
            print("After Bayes rule prob_prev_x_givenVC", prob_x_given_AVC)

```

Agent $i$ can compute $p(V,C_j \mid A_1, \cdots\!, A_j)$ as follows
$P(V=v,\, C_{j}=c \mid A_1, \cdots\!, A_{j}) ={}\\ P(V=v, C_{j-1}=c \mid A_1, \cdots\!, A_{j-1})\, P(X_{j}= L \mid A_1, \cdots\!, A_{j}) + \\ P(V= v,C_{j-1}=c-1 \mid A_1,\cdots\!, A_{j-1})\, P(X_{j}= H \mid A_1, \cdots\!, A_j)$

```python
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

```

When it is agent $i$'s turn to act, it will know $P(V=v, C_{i-1}=c \mid A_1, \cdots\!, A_{i-1})$ and will have its own signal $X_i$ to calculate $P(V=v,C_i \mid C_{i-1},X_i)$. Since $V$ and $C_i$ are conditionally independent given $C_{i-1}$ and $X_i$, we have

$P(V=v,C_i \mid C_{i-1},X_i) = P(V=v \mid C_{i-1} ,X_i) P(C_i \mid C_{i-1} ,X_i)$

where:
$P(V=v \mid C_{i-1}, X_i) \propto P(V=v \mid C_{i-1})P(X_i \mid V = v,C_{i-1})$
$P(X_i\!=\!x \mid V\!=\!v, C_{i-1}) = P(X_i\!=\!x \mid V\!=\!v)$   



$$
P(X_i\!=\!x \mid V\!=\!v) =
\begin{cases}
 p\, ,& \text{if } (v,x) \in \{(1,H), (0, L)\} \\
 1\!-\!p  ,& \text{otherwise}
 \end{cases}       
$$

and

$$
P(C_i=c \mid C_{i-1}, X_i=x) =
\begin{cases}
 P(C_i = c\!-\!1)\, ,& \text{if } x = H \\
 P(C_i = c)\, ,&\text{if } x = L
 \end{cases}       
$$



```python

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
```
Finally, As agent $i$ now has an up-to-date probability distribution over $V$, it can look up $P(V=1 \mid \cdots)$ and $P(V=0 \mid \cdots)$, and use these as the weights for the options $A$ and $R$. For deterministic choice, if one weight is larger than the other, it chooses the corresponding option. If they are equal, it tosses an evenly weighted coin to choose between the options. For non-deterministic choice, a random weighted choice is made.

```python
        # choose action
        if choice_method == "deterministic":
            actions.append(choose_action_deterministic(pVC_given_X))
        else:
            actions.append(choose_action_non_deterministic(pVC_given_X))
    
        #updating the list of previous evidence by new evidence
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
choice_method = "non_deterministic"
dict_count_actions_for_all_ps = {}
for k in range(len(list_p)):  # changing the value for p
    print("number of agents = ", n)
    print("number of prior agents", num_prior_agents)
    print("value of p, the signal acuracy = ", list_p[k])
    print("real v = ", real_v)
    p = list_p[k]
    diff_lessthan_20_percentage = 0
    significantly_more_As = 0
    significantly_more_Rs = 0
    for r in range(0, runs):
        actions, signals = simulation(n, p, num_prior_agents, real_v, choice_method)
        print("private signals", signals)
        print("Chosen actions", actions)
        if abs(actions.count(Action.A) - actions.count(Action.R)) < 20:
            diff_lessthan_20_percentage += 1
        elif actions.count(Action.A) - actions.count(Action.R) >= 20:
            significantly_more_As += 1
        elif actions.count(Action.R) - actions.count(Action.A) >= 20:
            significantly_more_Rs += 1
        a_cum_freq = cumfreq(
            [i for i, a in enumerate(actions) if a == Action.A],
            numbins=n,
            defaultreallimits=(0, n),
        )
        r_cum_freq = cumfreq(
            [i for i, a in enumerate(actions) if a == Action.R],
            numbins=n,
            defaultreallimits=(0, n),
        )
        plt.plot(r_cum_freq.cumcount, color="r", label="R" if r == 0 else "")
        plt.plot(a_cum_freq.cumcount, color="g", label="A" if r == 0 else "")
        plt.xlabel("Number of agents")
        plt.ylabel("Number of adoptions/rejections")
        plt.legend(loc="upper left")
        plt.title("Signal accuracy p =  " + str(list_p[k]))
    plt.show()
    dict_count_actions_for_all_ps[p] = (
        diff_lessthan_20_percentage,
        significantly_more_As,
        significantly_more_Rs,
    )
    print("..................................................................")

print(dict_count_actions_for_all_ps)

```
