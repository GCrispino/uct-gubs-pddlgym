import functools
import itertools
import logging
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

# TODO -> set this up properly
logging.basicConfig(level=logging.INFO)

res = None
for i, comb in enumerate(combinations):
    exec_str = gen_exec_str(vars_args, comb)
    logging.info(f"exec str {i + 1}/{len(combinations)}: {exec_str}")
    res = os.system(exec_str)
    if res != 0:
        logging.info(f"got error code {res}")
        logging.info("exiting...")
        break
