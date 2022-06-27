import itertools
import os
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
def save_fig_page(fig, path):
    pp = PdfPages(path)
    fig.savefig(pp, format="pdf")
    pp.close()
    
    
    
def find_json_files(root_path="."):
    files = os.listdir(root_path)
    json_files = []
    for f in files:
        joined_path = os.path.join(root_path, f)
        if os.path.isdir(joined_path):
            json_files.extend(find_json_files(joined_path))
        else:
            _, file_extension = os.path.splitext(joined_path)
            if file_extension == ".json":
                json_files.append(joined_path)
    return json_files

def get_data_files_from_paths(data_paths):
    data_files = []
    for data_path in data_paths:
        data_files.extend(find_json_files(data_path))
        
    return data_files
    


def get_vals_from_lamb_and_kg(lamb, k_g, prob_i, optimal_vals):
    
    for val in optimal_vals[prob_i]:
        if len(val) < 3:
            raise ValueError("wrong length of value")
        if val[0] == lamb and val[1] == k_g:
            return val[2]
    
    # no value was found
    raise ValueError("no value was found")

def get_vals_by_lamb(k_g, prob_i, optimal_vals):
    lambs = set({val[0] for val in optimal_vals[prob_i]})
    optimal = {}
    for lamb in lambs:
        optimal[lamb] = get_vals_from_lamb_and_kg(lamb, k_g, prob_i, optimal_vals)
        
    return optimal

def get_vals_by_kg(lamb, prob_i, optimal_vals):
    kgs = set({val[1] for val in optimal_vals[prob_i]})
    optimal = {}
    for k_g in kgs:
        optimal[k_g] = get_vals_from_lamb_and_kg(lamb, k_g, prob_i, optimal_vals)
        
    return optimal

def get_res_by_kg(res_i, lamb, prob_i, optimal_vals):
    return {k_g: res[res_i] for k_g, res in get_vals_by_kg(lamb, prob_i, optimal_vals).items()}

def get_res_by_lamb(res_i, k_g, prob_i, optimal_vals):
    return {lamb: res[res_i] for lamb, res in get_vals_by_lamb(k_g, prob_i, optimal_vals).items()}


def prop_via_path(path, dic):
    val = dic
    for key in path.split('.'):
        val = val[key]
    return val

def filter_runs_by_prop(prop, val, runs):
    return [run for run in runs if prop_via_path(prop, run) == val]

def filter_runs_by_props(runs, *pairs):
    res = runs
    for prop, val in pairs:
        res = filter_runs_by_prop(prop, val, res)
    return res

def check_for_intersections(run_collection):
    sets = []
    for runs in run_collection:
        sets.append(set({x['output_file_name'] for x in runs}))

    for set1, set2 in itertools.combinations(sets, 2):
        if len(set1.intersection(set2)) > 0:
            print("detected intersection!")
            
            
            
# gets x and y axis data to plot
def get_plot_data(x_var_path, y_var_paths, source):
    i_source_sorted = np.argsort([prop_via_path(x_var_path, run) for run in source])
    xs = np.array([prop_via_path(x_var_path, run) for run in source])[i_source_sorted]    
    
    yss = []
    for y_path, plot_type in y_var_paths:
        if plot_type == "line":
            yss.append(
                (
                    np.array([np.mean(prop_via_path(y_path, run)) for run in source])[i_source_sorted],
                    np.array([np.std(prop_via_path(y_path, run)) for run in source])[i_source_sorted]
                )
            )
        if plot_type == "bar":
            y = np.array([prop_via_path(y_path, run) for run in source])[i_source_sorted]
            counters = list(map(Counter, y))
            keys = sorted(set(key for counter in counters for key in counter.keys()))
            yss.append((keys, [[c[key] for c in counters] for key in keys]))


    return xs, yss

    
# gets data from fixed and free variable for single plot line
def get_plot_data_by_fixed_var(fixed_var_path, fixed_var_val, x_var_path, y_var_paths, source):
    fixed_var_filter = (fixed_var_path, fixed_var_val)
    source_by_x_var = filter_runs_by_props(source, fixed_var_filter)
    source_by_x_var_xs, source_by_x_var_yss = get_plot_data(x_var_path, y_var_paths, source_by_x_var)
    
    return source_by_x_var_xs, source_by_x_var_yss

# gets data from fixed and free variable for multiple plot lines
def get_plot_data_by_fixed_var_from_sources(fixed_var_path, fixed_var_val, x_var_path, y_var_paths, sources):
    res = []
    for source in sources:
        source_by_xvar_xs, source_by_xvar_yss = get_plot_data_by_fixed_var(
            fixed_var_path, fixed_var_val, x_var_path, y_var_paths, source
        )
        res.append((source_by_xvar_xs, source_by_xvar_yss))
        
    return res

# plots a single plot from sources
def plot_from_sources(sources, labels, title, ax, plot_type="line", bar_width=0.25):
    ax.set_title(title)
    for i, source in enumerate(sources):
        if plot_type == "line":
            ax.plot(source[0], source[1], label=labels[i])
            if len(source) > 2:
                ax.fill_between(source[0], source[1] - source[2], source[1] + source[2], alpha=0.1)
        if plot_type == "bar":
#             print("plot_from_sources bar!", source[1])
            plot_bars(source[0], source[1], labels[i], ax=ax, bar_width=bar_width)
    ax.legend()
    
# creates a figure and makes mutiple plots from sets of sources
def plot_multiple_from_plot_data_source_sets(source_sets, titles, labels, shape, figsize, bar_width=0.25, plot_type="line"):
    fig, axs = plt.subplots(shape[0], shape[1], figsize=figsize)
    axs = axs.reshape(*shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            plot_from_sources(source_sets[i][j], labels, titles[i][j], axs[i][j], bar_width=bar_width, plot_type=plot_type)

            
            
def plot_bars(xs, yss, labels, ax=None, bar_width=0.25):
#     print("plot_bars ->", xs, yss, labels, ax)
    
    if ax is None:
        fig, ax = plt.subplots(figsize =(12, 8))
    
    x_pos = range(len(yss[0]))
    offset = x_pos
    for ys, label in zip(yss, labels):
        x_ = [x + bar_width for x in offset]
#         print("ax.bar ", x_, ys, label)
        ax.bar(x_, ys, width = bar_width,
                edgecolor ='grey', label=label)
        offset = x_
    ax.set_xticks([r + (len(yss) / 2 + 0.5) * bar_width for r in x_pos], xs)
    ax.legend()

def flatten(l):
    return [x for _l in l for x in _l]

def get_plot_source_from_plot_data(plot_data):
    res = []
    for i in range(len(plot_data[0][1])):
        res.append([])
        for data_list in plot_data:
            xs, single_plot_data = data_list[0], data_list[1]
            res[i].append((xs, *single_plot_data[i]))
    return res

def parse_best_actions_plot_source(sources_best_actions):
    actions = sorted(set((a for sub in sources_best_actions for a in sub[1])))
    print("actions:", actions)

    data = []
    for sub in sources_best_actions:
        print("  sub[2]:", sub[2], len(sub[2]), [0 for _ in sub[2][0]])
        if len(sub[2]) < len(actions):
            print("    opa!")
            data.append([0 for _ in sub[2][0]])
        for x in sub[2]:
            data.append(x)
    return [(
        (sources_best_actions[0][0]),
        data
    )]

    return [(
        (sources_best_actions[0][0]),
        [x if len(x) == len(actions) else x + x for sub in sources_best_actions for x in sub[2]]
    )]
