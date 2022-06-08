import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def get_policy_size(info_run):
    explicit_graph_size = info_run['explicit_graph_size']
    explicit_graph_dc_size = info_run['explicit_graph_dc_size']
    return explicit_graph_size + explicit_graph_dc_size

def get_lamb_data_from_instance(instance_data, k_g):
    instance_data_kg_vi = np.array([
        x for x in instance_data if x['k_g'] == k_g
        and x['algorithm_dc'] == 'vi' and x['algorithm_gubs'] == 'vi'
    ])
    instance_data_kg_lamb_vi = np.array(
        [x['lamb'] for x in instance_data_kg_vi])

    instance_data_kg_vi_ao = np.array([
        x for x in instance_data if x['k_g'] == k_g
        and x['algorithm_dc'] == 'vi' and x['algorithm_gubs'] == 'ao'
    ])
    instance_data_kg_lamb_vi_ao = np.array(
        [x['lamb'] for x in instance_data_kg_vi])

    instance_data_kg_vi_ao_expansion_1 = np.array([
        x for x in instance_data
        if x['k_g'] == k_g and x['algorithm_dc'] == 'vi'
        and x['algorithm_gubs'] == 'ao' and x['expansion_levels'] == 1
    ])
    instance_data_kg_lamb_vi_ao_expansion_1 = np.array(
        [x['lamb'] for x in instance_data_kg_vi_ao_expansion_1])
    instance_data_kg_vi_ao_expansion_5 = np.array([
        x for x in instance_data
        if x['k_g'] == k_g and x['algorithm_dc'] == 'vi'
        and x['algorithm_gubs'] == 'ao' and x['expansion_levels'] == 5
    ])
    instance_data_kg_lamb_vi_ao_expansion_5 = np.array(
        [x['lamb'] for x in instance_data_kg_vi_ao_expansion_5])

    instance_data_sorted_by_lamb_vi = np.argsort(instance_data_kg_lamb_vi)
    instance_data_sorted_by_lamb_vi_ao = np.argsort(
        instance_data_kg_lamb_vi_ao)
    instance_data_sorted_by_lamb_vi_ao_expansion_1 = np.argsort(
        instance_data_kg_lamb_vi_ao_expansion_1)
    instance_data_sorted_by_lamb_vi_ao_expansion_5 = np.argsort(
        instance_data_kg_lamb_vi_ao_expansion_5)

    instance_data_kg_lamb_size_vi = np.array([
        x['explicit_graph_dc_size'] * (x['C_max'] + 1)
        for x in instance_data_kg_vi_ao_expansion_5[
            instance_data_sorted_by_lamb_vi_ao_expansion_5]
    ])
    instance_data_kg_lamb_size_vi_ao_expansion_1 = [
        get_policy_size(x) for x in instance_data_kg_vi_ao_expansion_1[
            instance_data_sorted_by_lamb_vi_ao_expansion_1]
    ]
    instance_data_kg_lamb_size_vi_ao_expansion_5 = [
        get_policy_size(x) for x in instance_data_kg_vi_ao_expansion_5[
            instance_data_sorted_by_lamb_vi_ao_expansion_5]
    ]
    instance_data_kg_lamb_size_vi_norm = instance_data_kg_lamb_size_vi / instance_data_kg_lamb_size_vi[-1]
    instance_data_kg_lamb_size_vi_ao_norm = instance_data_kg_lamb_size_vi_ao_expansion_5 / instance_data_kg_lamb_size_vi[-1]

    instance_data_kg_lamb_cmax_vi = np.array([
        x['C_max']
        for x in instance_data_kg_vi[instance_data_sorted_by_lamb_vi]
    ])
    instance_data_kg_lamb_cmax_vi_ao_expansion_1 = np.array([
        x['C_max'] for x in instance_data_kg_vi_ao_expansion_1[
            instance_data_sorted_by_lamb_vi_ao_expansion_1]
    ])
    instance_data_kg_lamb_cmax_vi_ao_expansion_5 = np.array([
        x['C_max'] for x in instance_data_kg_vi_ao_expansion_5[
            instance_data_sorted_by_lamb_vi_ao_expansion_5]
    ])
    
    # instance_data_kg_lamb_cmax_vi_norm, instance_data_kg_lamb_cmax_vi_ao_norm = [], []
    # non_zero_indexes = np.argwhere(instance_data_kg_lamb_cmax_vi != 0).reshape(-1)
    # if len(non_zero_indexes) != 0:
    #     last_non_zero_index_cmax_vi = non_zero_indexes.reshape(-1)[-1]
    #     instance_data_kg_lamb_cmax_vi_norm = instance_data_kg_lamb_cmax_vi / instance_data_kg_lamb_cmax_vi[last_non_zero_index_cmax_vi]
    #     instance_data_kg_lamb_cmax_vi_ao_norm = instance_data_kg_lamb_cmax_vi_ao_expansion_5 / instance_data_kg_lamb_cmax_vi[last_non_zero_index_cmax_vi]
    
    instance_data_kg_lamb_cmax_vi_norm, instance_data_kg_lamb_cmax_vi_ao_norm = [], []
    non_zero_indexes = np.argwhere(instance_data_kg_lamb_cmax_vi != 0).reshape(-1)
    non_zero_vals = instance_data_kg_lamb_cmax_vi[instance_data_kg_lamb_cmax_vi != 0]
    # if len(non_zero_indexes) != 0:
    if len(non_zero_vals) != 0:
        min_non_zero_val = non_zero_vals.min()
        instance_data_kg_lamb_cmax_vi_norm = instance_data_kg_lamb_cmax_vi / min_non_zero_val
        instance_data_kg_lamb_cmax_vi_ao_norm = instance_data_kg_lamb_cmax_vi_ao_expansion_5 / min_non_zero_val

    instance_data_kg_lamb_time_vi = np.array([
        x['cpu_time']
        for x in instance_data_kg_vi[instance_data_sorted_by_lamb_vi]
    ])
    instance_data_kg_lamb_time_vi_ao_expansion_1 = np.array([
        x['cpu_time'] for x in instance_data_kg_vi_ao_expansion_1[
            instance_data_sorted_by_lamb_vi_ao_expansion_1]
    ])
    instance_data_kg_lamb_time_vi_ao_expansion_5 = np.array([
        x['cpu_time'] for x in instance_data_kg_vi_ao_expansion_5[
            instance_data_sorted_by_lamb_vi_ao_expansion_5]
    ])
    instance_data_kg_lamb_time_vi_norm = instance_data_kg_lamb_time_vi / instance_data_kg_lamb_time_vi[
        -1]
    instance_data_kg_lamb_time_vi_ao_norm = instance_data_kg_lamb_time_vi_ao_expansion_5 / instance_data_kg_lamb_time_vi[
        -1]

    instance_data_kg_lamb_n_updates_vi = np.array([
        x['n_updates'] + x['n_updates_dc']
        for x in instance_data_kg_vi[instance_data_sorted_by_lamb_vi]
    ])
    instance_data_kg_lamb_n_updates_vi_ao_expansion_1 = np.array([
        x['n_updates'] + x['n_updates_dc']
        for x in instance_data_kg_vi_ao_expansion_1[
            instance_data_sorted_by_lamb_vi_ao_expansion_1]
    ])
    instance_data_kg_lamb_n_updates_vi_ao_expansion_5 = np.array([
        x['n_updates'] + x['n_updates_dc']
        for x in instance_data_kg_vi_ao_expansion_5[
            instance_data_sorted_by_lamb_vi_ao_expansion_5]
    ])

    instance_data_kg_lamb_n_updates_min = min(
        instance_data_kg_lamb_n_updates_vi.min(),
        instance_data_kg_lamb_n_updates_vi_ao_expansion_5.min())
    instance_data_kg_lamb_n_updates_vi_norm = instance_data_kg_lamb_n_updates_vi / instance_data_kg_lamb_n_updates_vi[
        -1]
    instance_data_kg_lamb_n_updates_vi_ao_norm = instance_data_kg_lamb_n_updates_vi_ao_expansion_5 / instance_data_kg_lamb_n_updates_vi[
        -1]

    k_g = f'{k_g:.0e}'
    
    return {
        f"kg{k_g}_vi":
        instance_data_kg_vi,
        f"lamb_kg{k_g}_vi":
        instance_data_kg_lamb_vi,
        f"kg{k_g}_vi_ao":
        instance_data_kg_vi_ao,
        f"lamb_kg{k_g}_vi_ao":
        instance_data_kg_lamb_vi_ao,
        f"lamb_kg{k_g}_vi_ao_expansion_1":
        instance_data_kg_lamb_vi_ao_expansion_1,
        f"lamb_kg{k_g}_vi_ao_expansion_5":
        instance_data_kg_lamb_vi_ao_expansion_5,
        f"kg{k_g}_sorted_by_lamb_vi":
        instance_data_sorted_by_lamb_vi,
        f"kg{k_g}_sorted_by_lamb_vi_ao":
        instance_data_sorted_by_lamb_vi_ao,
        f"kg{k_g}_sorted_by_lamb_vi_ao_expansion_1":
        instance_data_sorted_by_lamb_vi_ao_expansion_1,
        f"kg{k_g}_sorted_by_lamb_vi_ao_expansion_5":
        instance_data_sorted_by_lamb_vi_ao_expansion_5,
        f"lamb_kg{k_g}_size_vi":
        instance_data_kg_lamb_size_vi,
        f"lamb_kg{k_g}_size_vi_ao_expansion_1":
        instance_data_kg_lamb_size_vi_ao_expansion_1,
        f"lamb_kg{k_g}_size_vi_ao_expansion_5":
        instance_data_kg_lamb_size_vi_ao_expansion_5,
        f"lamb_kg{k_g}_size_vi_norm":
        instance_data_kg_lamb_size_vi_norm,
        f"lamb_kg{k_g}_size_vi_ao_norm":
        instance_data_kg_lamb_size_vi_ao_norm,
        f"lamb_kg{k_g}_cmax_vi":
        instance_data_kg_lamb_cmax_vi,
        f"lamb_kg{k_g}_cmax_vi_ao_expansion_1":
        instance_data_kg_lamb_cmax_vi_ao_expansion_1,
        f"lamb_kg{k_g}_cmax_vi_ao_expansion_5":
        instance_data_kg_lamb_cmax_vi_ao_expansion_5,
        f"lamb_kg{k_g}_cmax_vi_norm":
        instance_data_kg_lamb_cmax_vi_norm,
        f"lamb_kg{k_g}_cmax_vi_ao_norm":
        instance_data_kg_lamb_cmax_vi_ao_norm,
        f"lamb_kg{k_g}_n_updates_vi":
        instance_data_kg_lamb_n_updates_vi,
        f"lamb_kg{k_g}_n_updates_vi_ao_expansion_1":
        instance_data_kg_lamb_n_updates_vi_ao_expansion_1,
        f"lamb_kg{k_g}_n_updates_vi_ao_expansion_5":
        instance_data_kg_lamb_n_updates_vi_ao_expansion_5,
        f"lamb_kg{k_g}_time_vi":
        instance_data_kg_lamb_time_vi,
        f"lamb_kg{k_g}_time_vi_ao_expansion_1":
        instance_data_kg_lamb_time_vi_ao_expansion_1,
        f"lamb_kg{k_g}_time_vi_ao_expansion_5":
        instance_data_kg_lamb_time_vi_ao_expansion_5,
        f"lamb_kg{k_g}_time_vi_norm":
        instance_data_kg_lamb_time_vi_norm,
        f"lamb_kg{k_g}_time_vi_ao_norm":
        instance_data_kg_lamb_time_vi_ao_norm,
        f"lamb_kg{k_g}_n_updates_vi_norm":
        instance_data_kg_lamb_n_updates_vi_norm,
        f"lamb_kg{k_g}_n_updates_vi_ao_norm":
        instance_data_kg_lamb_n_updates_vi_ao_norm,
    }

def plot_instance_data_by_lambda(instance_data, domain_name, n_instance, k_g):
    exponent = int(np.log10(k_g))
    k_g = f'{k_g:.0e}'
    plt.figure()
    plt.title(f"{domain_name} {n_instance}" + r" - $C^+_{max}$ and $\bar{C}^+_{max}(s_0)$ x $\lambda$")
    plt.xlabel("$\lambda$")
    plt.plot(instance_data[f"lamb_kg{k_g}_vi"][instance_data[f"kg{k_g}_sorted_by_lamb_vi"]], instance_data[f"lamb_kg{k_g}_cmax_vi"], label=r"$K_g = 10^{" + f"{exponent}" + r"} - C^+_{max}$", marker="|", markersize="12")
    plt.plot(instance_data[f"lamb_kg{k_g}_vi_ao_expansion_5"][instance_data[f"kg{k_g}_sorted_by_lamb_vi_ao_expansion_5"]], instance_data[f"lamb_kg1e-{'0' if -exponent < 10 else ''}{-exponent}_cmax_vi_ao_expansion_5"], label=r"$K_g = 10^{" + f"{exponent}" + r"}- \bar{C}^+_{max}(s_0)$", marker="|", markersize="12")
    plt.legend()

    plt.figure()
    plt.title(f"{domain_name} {n_instance} - Number of updates x $\lambda$")
    plt.xlabel("$\lambda$")
    plt.ylabel("Updates")

    plt.plot(instance_data[f"lamb_kg{k_g}_vi"][instance_data[f"kg{k_g}_sorted_by_lamb_vi"]], instance_data[f"lamb_kg{k_g}_n_updates_vi"], label=r"$K_g = 10^{" + f"{exponent}" + r"}$ - eGUBS-VI", marker="|", markersize="12")
    plt.plot(instance_data[f"lamb_kg{k_g}_vi_ao_expansion_5"][instance_data[f"kg1e-{'0' if -exponent < 10 else ''}{-exponent}_sorted_by_lamb_vi_ao_expansion_5"]], instance_data[f"lamb_kg1e-{'0' if -exponent < 10 else ''}{-exponent}_n_updates_vi_ao_expansion_5"], label=r"$K_g = 10^{" + f"{exponent}" + r"}$ - eGUBS-AO*", marker="|", markersize="12")

    plt.legend()

    plt.figure()
    plt.title(f"{domain_name} {n_instance} - Number of states x $\lambda$")
    plt.xlabel("$\lambda$")
    plt.ylabel("Number of states")

    plt.plot(instance_data[f"lamb_kg{k_g}_vi"][instance_data[f"kg{k_g}_sorted_by_lamb_vi"]][:len(instance_data[f"lamb_kg{k_g}_size_vi"])], instance_data[f"lamb_kg{k_g}_size_vi"], label=r"$K_g = 10^{" + f"{exponent}" + r"}$ - eGUBS-VI", marker="|", markersize="12")
    plt.plot(instance_data[f"lamb_kg{k_g}_vi_ao_expansion_5"][instance_data[f"kg{k_g}_sorted_by_lamb_vi_ao_expansion_5"]], instance_data[f"lamb_kg1e-{'0' if -exponent < 10 else ''}{-exponent}_size_vi_ao_expansion_5"][:len(instance_data[f"lamb_kg{k_g}_vi_ao_expansion_5"])], label=r"$K_g = 10^{" + f"{exponent}" + r"}$ - eGUBS-AO*", marker="|", markersize="12")

    plt.legend()

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
    