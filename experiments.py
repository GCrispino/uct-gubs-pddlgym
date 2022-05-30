import functools
import itertools
import os

import uct_gubs.argparsing.experiments as argparsing


def parse_list_str(list_str):
    return list_str.strip("[]").split(", ")


def gen_exec_str(vars_args, combination):
    # TODO
    """
    args to set:
      2. logging level?
      3. logging output file?
    """
    params_str = functools.reduce(
        lambda x, y: f"{x}" + (f" --{y[0]} {'' if y[1] == 'True' else y[1]}"
                               if y[1] != "False" else ""),
        zip(vars_args, combination), "")
    fixed_params = " --render_and_save"
    return f"python plan.py {params_str} {fixed_params}"


args = argparsing.parse_args()

vars_args = vars(args)

combinations = list(
    itertools.product(*map(parse_list_str, vars_args.values())))

res = None
for comb in combinations:
    exec_str = gen_exec_str(vars_args, comb)
    print("exec str:", exec_str)
    res = os.system(exec_str)
    if res != 0:
        print("got error code", res)
        print("exiting...")
        break
