from typing import List, Dict, Any, Union, Tuple, Optional
import os
import re
import sys
import json
import pandas as pd
import numpy as np
import torch # for loading training_args.bin
from collections import defaultdict
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from .classes import ModelArguments
from transformers import HfArgumentParser, TrainingArguments

def find_all_logdirs(root):
    """
        Only walk once to find all job ids a job ID is a 7 digit number.
        6410695
    """
    pattern = re.compile(r"(?<!\d)(\d{7})(?!\d)")
    for dirname, dirlist, _ in os.walk(root):
        for d in dirlist:
            m = pattern.search(d)
            if m is not None:
                yield m.group(0), os.path.join(dirname, d)


def find_all_logfiles(root):
    pattern = re.compile(r"(?<!\d)(\d{7})(?!\d)")
    for dirname, _, files in os.walk(root):
        for f in files:
            m = pattern.search(f)
            if m is not None:
                yield m.groups()[0], os.path.join(dirname, f)


def pair_log_dirs_files(ckpt_dir, log_dir):
    """
        Pair the log directories with the log files
    """
    exp_r = re.compile(r"experiment_(\d+)")
    pairings = defaultdict(lambda: defaultdict(dict))
    for job_id, dirname in find_all_logdirs(ckpt_dir):
        # the directory contains 1 or more experiment_<array_id> directories
        for exp_dir in os.listdir(dirname):
            m = exp_r.search(exp_dir)
            if m is not None:
                array_id = m.groups()[0]
                pairings[job_id][array_id]['directory'] = os.path.join(
                    dirname, exp_dir)

    log_r = re.compile(r"\d{7}_(\d+)")
    for job_id, logfile in find_all_logfiles(log_dir):
        m = log_r.search(logfile)
        if m is not None:
            array_id = m.groups()[0]
            pairings[job_id][array_id]['logfile'] = logfile
    return pairings

def parse_arguments(lines) -> Tuple[List[str], Optional[Tuple[str, Dict[str, Any]]]]:
    """
        Parse the arguments from the slurm logs and return a dictionary
    """
    # the first line contains the type of the dataclass and opens a (
    # then, we go word by word until we find the closing )
    # values are represented as key=value, which we collect in a dictionary
    # finally we return the name and the dictionary
    # if we don't find the closing ), we return None
    # if we don't find the name, we return None
    brackets = 0
    # pattern to match the name and (
    first_line_r = re.compile(r"(\w+)\(")
    # get name
    m = first_line_r.search(lines[0])
    if m is None:
        return lines, None
    name = m.group(1); brackets += 1
    if name != "ModelArguments":
        return lines, None # only for our class. TrainingArguments are saved to checkpoints
    lines[0] = lines[0][m.end():]
    body = ""
    parsed_lines = []
    while brackets > 0 and len(lines) > 0:
        # find the closing bracket and append everything to the body
        # without any whitespaces
        line = lines.pop(0); parsed_lines.append(line) # if we need to roll back
        line = ''.join(line.strip().split(' '))
        while len(line) > 0: # go character by character to find the closing bracket
            c = line[0]; line = line[1:]
            brackets += 1 if c == '(' else -1 if c == ')' else 0
            if brackets == 0:
                break
            body += c
        if len(line) > 0:
            lines.insert(0, line); parsed_lines.pop() # do not duplicate
    if brackets > 0:
        # restore the lines
        lines = parsed_lines + lines
        return lines, None
    logger.debug("dataclass body: %s" % (body,))
    # now we have the body, we can parse the key=value pairs
    # do this using eval
    try:
        arguments = eval(f'dict({body})')
    except:
        logger.warning("Could not parse arguments. body: %s" % (body,))
        return lines, None

    return lines, (name, arguments)

def parse_num_trainable_line(lines: List[str]) -> Tuple[List[str], Optional[Tuple[int,int,int]]]:
    """
    Looks something like:
    trainable params: 887,042 || all params: 125,534,212 || trainable%: 0.7066
    """
    num_trainable_r = re.compile(
        r"trainable params: ([\d,]+) \|\| all params: ([\d,]+) \|\| trainable%: (\d+\.\d+)")
    m = num_trainable_r.search(lines[0])
    if m is None:
        return lines, None
    lines.pop(0)
    num_trainable = int(m.group(1).replace(',', ''))
    num_all = int(m.group(2).replace(',', ''))
    trainable_percent = float(m.group(3))
    return lines, (num_trainable, num_all, trainable_percent)

def parse_realloc_log(lines: List[str]) -> Tuple[List[str], Optional[int]]:
    """
    Later experiments may have lines like
    Step <step>: reallocated modules
    """
    realloc_r = re.compile(r"[sS]tep (\d+): reallocated modules")
    m = realloc_r.search(lines[0])
    if m is None:
        return lines, None
    lines.pop(0)
    return lines, int(m.group(1))

def parse_metric_dict(lines: List[str]) -> Tuple[List[str], Optional[Dict[str, Any]]]:
    """
        printed as a single line dict.__str__()
    """
    line = lines[0]
    try:
        metrics = eval(line)
        lines.pop(0)
        return lines, metrics
    except:
        return lines, None

def parse_logs(logfile, errors) -> Tuple[Dict[str, Any], List[str]]:
    """
        Parse the slurm logs to extract parameters and metrics
        
        Parameters are reported as `str` representations of the TrainingArguments
        and the ModelArguments objects. We take what we can find and not care about the rest.

        Metrics are output as `str` representations of the dictionary of metrics,
        which may be training or validation reports. We identify them by their keys and collect
        them in a list of dictionaries.
    """

    with open(logfile, 'r') as f:
        lines = f.readlines()

    arguments_d = {}
    metrics_d = defaultdict(list)
    realloc_list = []
    num_param_list = []
    while len(lines) > 0:
        lines, res = parse_arguments(lines)
        if res is not None:
            name, arguments = res
            if name in arguments_d:
                logger.warning(f"Duplicate arguments for {name}")
            arguments_d[name] = arguments
            continue
        lines, res = parse_num_trainable_line(lines)
        if res is not None:
            num_trainable, num_all, trainable_percent = res
            num_param_list.append((num_trainable, num_all, trainable_percent))
            continue
        lines, res = parse_realloc_log(lines)
        if res is not None:
            realloc_list.append(res)
            continue
        lines, res = parse_metric_dict(lines)
        if res is not None:
            # create key from its contents
            key = ','.join(sorted(res.keys()))
            metrics_d[key].append(res)
            continue
        lines.pop(0)

    report = {}
    if len(arguments_d) == 0:
        errors.append('No arguments')
    report['arguments'] = arguments
    if len(metrics_d) == 0:
        errors.append('No metrics')
    report['metrics'] = metrics_d
    if len(num_param_list) == 0:
        errors.append('No num_params')
    report['num_params'] = num_param_list
    if len(realloc_list) == 0:
        errors.append('No reallocs')
    report['reallocs'] = realloc_list
    return report, errors

def parse_directory(directory, errors) -> Tuple[dict, list]:
    # parse the dynalora_logs.json
    report = {'arguments': {}}
    if not os.path.exists(os.path.join(directory, 'dynalora_logs.json')):
        errors.append('No dynalora_logs.json')
    else:
        with open(os.path.join(directory, 'dynalora_logs.json'), 'r') as f:
            dynalora_logs = json.load(f)
        report.update(dynalora_logs)
    # look for an adapter_config.json
    for dirname, _, files in os.walk(directory):
        if 'adapter_config.json' in files:
            with open(os.path.join(dirname, 'adapter_config.json'), 'r') as f:
                adapter_config = json.load(f)
            report['arguments'].update(adapter_config) # intentionally overwrite
            break
    else:
        errors.append('No adapter_config.json')
    # look for a training_args.bin
    for dirname, _, files in os.walk(directory):
        if 'training_args.bin' in files:
            training_args = torch.load(os.path.join(dirname, 'training_args.bin'))
            report['arguments'].update(training_args.to_dict())
            break
    else:
        errors.append('No training_args.bin')
    # TODO: is there anything else we need to extract?
    return report, errors


def parse_single_experiment(job_id, array_id, directory=None, logfile=None) ->Tuple[dict, list]:
    """
        Parse a single experiment
    """
    errors = []
    report = {}
    if logfile is None:
        errors.append('No log file')
    else:
        log_report, errors = parse_logs(logfile, errors)
        report.update(log_report)

    # check if directory was found
    if directory is None:
        errors.append('No directory')
    else:
        # parse the dynalora_logs.json
        dir_report, errors = parse_directory(directory, errors)
        report.update(dir_report)

    return report, errors

def main(*args):
    """
        Parse all experiments
    """
    # get all the pairings
    ckpt_dir, log_dir, save_root = args
    pairings = pair_log_dirs_files(ckpt_dir, log_dir)
    # parse everything
    results_d = defaultdict(dict)
    errors_d = defaultdict(dict)
    for job_id, exps in pairings.items():
        for array_id, files in exps.items():
            report, errors = parse_single_experiment(job_id, array_id, **files)
            results_d[job_id][array_id] = report
            errors_d[job_id][array_id] = errors
    os.makedirs(save_root, exist_ok=True)
    with open(os.path.join(save_root, 'results.json'), 'w') as f:
        json.dump(results_d, f, indent=4)
    with open(os.path.join(save_root, 'errors.json'), 'w') as f:
        json.dump(errors_d, f, indent=4)
    with open(os.path.join(save_root, 'pairings.json'), 'w') as f:
        json.dump(pairings, f, indent=4)
    with open(os.path.join(save_root, 'parser_config.json'), 'w') as f:
        json.dump({'ckpt_dir': ckpt_dir, 'log_dir': log_dir}, f, indent=4)

if __name__ == "__main__":
    main(*sys.argv[1:])