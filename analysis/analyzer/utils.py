from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import re

def plot_histogram(filename: str, title: str, scores: List[float], min_ylim: float=1.0, max_ylim: float=6.0):
    scores = np.array(scores)
    plt.figure()
    if min_ylim > 0 and max_ylim > 0:
        plt.hist(scores, bins="auto", range=(min_ylim, max_ylim))  # arguments are passed to np.histogram
    else:
        plt.hist(scores, bins="auto")
    plt.axvline(scores.mean(), color='k', linestyle='dashed', linewidth=1)
    xmin, xmax, ymin, ymax = plt.axis()
    plt.text(scores.mean() * 1.1, ymax * 0.9, 'Mean: {:.2f}'.format(scores.mean()))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")

def plot_histogram_per_category(filename: str, title: str, scores: List[float], categories: List[str]):
    scores_per_category = {
        'score': scores,
        'category': categories,
    }
    df = pd.DataFrame(scores_per_category)
    plt.figure()
    sns.histplot(data=df, x='score', bins=200, hue='category', multiple="stack")
    scores = np.array(scores)
    plt.axvline(scores.mean(), color='k', linestyle='dashed', linewidth=1)
    xmin, xmax, ymin, ymax = plt.axis()
    plt.text(scores.mean() * 1.05, ymax * 0.9, 'Mean: {:.2f}'.format(scores.mean()))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")

def plot_categories(filename: str, title: str, x: List, labels: List, colors: Dict):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(x, labels=labels, autopct='%.1f%%',
           wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'},
           textprops={'size': 'small'},
           colors=[colors[key] for key in labels])
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")

count_relation_mapping = {
    "+": ">",
    "or more": ">=",
    "at least": ">=",
    "more than": ">",
    "longer than": ">",
    "less than": "<",
    "fewer than": "<",
    "at most": "<=",
    "or less": "<=",
    "around": "=",
    "-word": "=",
    "up to": "<=",
    " to": "--",
    "-": "--",
}

number_mapping = {"one ": 1, "two ": 2, "three ": 3, "four ": 4, "five ": 5,
                  "six ": 6, "seven ": 7, "eight ": 8, "nine ": 9, "ten ": 10}

frequency_mapping = {"once": 1, "twice": 2}

def negation_exists(text):
    return "n't " in text or "not " in text or "no " in text or "without" in text

def check_word_length(constraint, response):
    relation = ""
    for rel in count_relation_mapping:
        if rel in constraint:
            relation = count_relation_mapping[rel]
            break
    if relation == "": relation = "="
    counts = re.findall("\d+", constraint)
    counts = [int(c) for c in counts]
    word_count = response.count(" ")

    # Flip the relation in case of negation
    if relation == "<" and negation_exists(constraint):
        relation = ">="
    if relation == ">" and negation_exists(constraint):
        relation = "<="

    if len(counts) > 0:
        if relation == "<" or relation == "<=":
            count = counts[0]
            return word_count < (count + 5)
        elif relation == ">" or relation == ">=":
            count = counts[0]
            return word_count > (count - 5)
        elif relation == "--" and len(counts) > 1:
            return word_count > (counts[0] - 5) and word_count < (counts[1] + 5)
        else:  # relation == "=":
            count = counts[0]
            return word_count > (count - 5) and word_count < (count + 5)

    return None

def check_sentence_length(constraint, response):
    relation = ""
    for rel in count_relation_mapping:
        if rel in constraint:
            relation = count_relation_mapping[rel]
            break
    counts = []
    for num in number_mapping:
        if num in constraint:
            counts.append(number_mapping[num])
    if relation == "": relation = "="
    if counts == []: counts = re.findall("\d+", constraint)
    counts = [int(c) for c in counts]
    sent_count = len([sent for sent in re.split(r'[\.?!]', response) if sent.strip() != ""])

    # Flip the relation in case of negation
    if relation == "<" and negation_exists(constraint):
        relation = ">="
    if relation == ">" and negation_exists(constraint):
        relation = "<="

    if len(counts) > 0:
        if relation == "<":
            count = counts[0]
            return sent_count < count
        elif relation == ">":
            count = counts[0]
            return sent_count > count
        if relation == "<=":
            count = counts[0]
            return sent_count <= count
        elif relation == ">=":
            count = counts[0]
            return sent_count >= count
        elif relation == "--" and len(counts) > 1:
            return sent_count >= counts[0] and sent_count <= counts[1]
        else:  # relation == "=":
            count = counts[0]
            return sent_count == count

    return None

def check_paragraph_length(constraint, response):
    relation = ""
    for rel in count_relation_mapping:
        if rel in constraint:
            relation = count_relation_mapping[rel]
            break
    counts = []
    for num in number_mapping:
        if num in constraint:
            counts.append(number_mapping[num])
    if relation == "": relation = "="
    if counts == []: counts = re.findall("\d+", constraint)
    counts = [int(c) for c in counts]
    para_count = len([p for p in response.split("\n\n") if p.strip() != ""])

    # Flip the relation in case of negation
    if relation == "<" and negation_exists(constraint):
        relation = ">="
    if relation == ">" and negation_exists(constraint):
        relation = "<="

    if len(counts) > 0:
        if relation == "<":
            count = counts[0]
            return para_count < count
        elif relation == ">":
            count = counts[0]
            return para_count > count
        if relation == "<=":
            count = counts[0]
            return para_count <= count
        elif relation == ">=":
            count = counts[0]
            return para_count >= count
        elif relation == "--" and len(counts) > 1:
            return para_count >= counts[0] and para_count <= counts[1]
        else:  # relation == "=":
            count = counts[0]
            return para_count == count

    return None