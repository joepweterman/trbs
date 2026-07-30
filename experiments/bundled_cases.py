"""
This file contains helpers shared by the experiment scripts that read the cases
bundled with the installed vlinder package (Beerwiser, Refugee, IZZ, ...).
"""

import os
from pathlib import Path

import pandas as pd
import vlinder as vl

DATA = Path(os.path.dirname(vl.__file__)) / "data"


def is_number(value) -> bool:
    """
    This function checks whether a dependency argument is a numeric literal
    rather than a variable name. Thousands separators are allowed because the
    bundled case CSVs use them.
    :param value: the argument to check
    :return: a boolean indicating whether the argument is numeric
    """
    try:
        float(str(value).replace(",", ""))
        return True
    except (ValueError, TypeError):
        return False


def read_case_tables(case: str):
    """
    This function reads the three tables the dependency-graph experiments need.
    :param case: the name of a bundled case
    :return: a tuple of the dependencies frame, the key output names and the
        internal variable names
    """
    csv = DATA / case / "csv"
    deps = pd.read_csv(csv / "dependencies.csv", sep=";")
    kos = pd.read_csv(csv / "key_outputs.csv", sep=";")["key_output"].tolist()
    internals = pd.read_csv(csv / "decision_makers_options.csv", sep=";")["internal_variable_input"].unique().tolist()
    return deps, kos, internals
