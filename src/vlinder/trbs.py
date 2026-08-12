# pylint: disable=no-member
# pylint: disable=W0511
# pylint: disable=R0902

"""
This module contains the tRBS class. This is the parent class that deals with anything related to a Responsible
Business Simulator Case.
"""

from pathlib import Path
import os
import copy

import numpy as np
import matplotlib
import vlinder as vl
from vlinder.case_exporter import CaseExporter
from vlinder.case_importer import CaseImporter
from vlinder.evaluate import Evaluate
from vlinder.appreciate import Appreciate
from vlinder.visualize import Visualize, DependencyGraph
from vlinder.make_report import MakeReport
from vlinder.optimize import Optimize


def list_demo_cases(file_path=None):
    """This function returns all demo cases that exist in the package"""
    file_path = file_path or Path(os.path.dirname(vl.__file__)) / "data"

    try:
        case_names = [name for name in os.listdir(file_path) if (file_path / name).is_dir()]
        return case_names
    except FileNotFoundError as error:
        raise FileNotFoundError(f"The directory {file_path} does not exist.") from error


class CaseError(Exception):
    """
    This class deals with the error handling of our TheResponsibleBusinessSimulator() class.
    """

    def __init__(self, message):  # ignore warning about super-init | pylint: disable=W0231
        self.message = message

    def __str__(self):
        return f"Case Error: {self.message}"


class TheResponsibleBusinessSimulator:
    """
    This class is the base class of an tRBS-case and contains all necessary information to import data, evaluate
    dependencies and calculate appreciations.
    """

    def __init__(self, name, file_path=None, file_extension=None):
        self.file_path = file_path if file_path is not None else Path(os.path.dirname(vl.__file__)) / "data"
        self.file_extension = file_extension if file_extension is not None else "xlsx"
        self.name = name
        self.input_dict = {}
        self.dataframe_dict = {}
        self.output_dict = {}
        self.visualizer = None
        self.exporter = None
        self.report = None
        # Set by optimize(): the full result of the most recent optimization run.
        self.optimization_result = None

        self.possible_status = {0: "build", 1: "evaluate", 2: "appreciate", 3: "optimize"}
        self.status = {}

    def __str__(self):
        input_data_formatted = (
            "\n\n".join(f"{key}\n\t{value}" for key, value in self.input_dict.items())
            if self.input_dict
            else "First .build() a case to import data"
        )
        return (
            f"Case: {self.name} ({self.file_extension}) \n"
            f"Status: {self.status}\n"
            f"Data location: {self.file_path} \n"
            f"Input data: \n {input_data_formatted}"
        )

    def _get_options(self):
        """
        This function calculates the amount of different options (or calculations) of the model:
        Amount of scenarios x Amount of decision makers options x Amount of key outputs
        """
        return (
            len(self.input_dict["scenarios"])
            * len(self.input_dict["decision_makers_options"])
            * len(self.input_dict["key_outputs"])
        )

    def _status_check(self, status_codes):
        """
        This function checks whether all necessary status are present and raises an error when not
        """
        for status_code in status_codes:
            if status_code not in self.status:
                step_name = self.possible_status[status_code]
                raise CaseError(f"first {step_name} a case with .{step_name}()")

    def _set_and_reset_status(self, status_to_set):
        """
        This function sets (and if necessary re-sets) the status using the status code.
        If a step is executed with a lower status code then currently present in the dictionary,
        all higher levels are removed
        """
        self.status[status_to_set] = self.possible_status[status_to_set]
        self.status = {key: value for key, value in self.status.items() if key <= status_to_set}

    def copy(self):
        """
        Creates a deep copy of the instance.
        """
        return copy.deepcopy(self)

    def build(self):
        """This function builds all necessary elements for a generic RBS case"""
        print(f"Creating '{self.name}'")
        case_import = CaseImporter(self.file_path, self.name, self.file_extension)
        self.input_dict, self.dataframe_dict = case_import.import_case()

        # set and re-set status
        self._set_and_reset_status(0)

    def evaluate(self):
        """This function deals with the evaluation of all dependencies"""
        self._status_check([0])
        case_evaluation = Evaluate(self.input_dict)
        self.output_dict = case_evaluation.evaluate_all_scenarios()
        self._set_and_reset_status(1)

    def appreciate(self):
        """This function deals with the appreciation of the outcomes"""
        self._status_check([0, 1])
        case_appreciation = Appreciate(self.input_dict, self.output_dict)
        case_appreciation.appreciate_all_scenarios()
        self._set_and_reset_status(2)

    def visualize(self, visual_request, key, **kwargs):
        """This function deals with the visualizations of the outcomes"""
        # currently only checks for build, some visuals will also need evaluate and/or appreciate
        self._status_check([0])
        if visual_request == "dependency_graph":
            dependency_tree = DependencyGraph(self.input_dict)
            return dependency_tree.draw_graph(key, **kwargs)

        self.visualizer = Visualize(self.input_dict, self.output_dict, self._get_options())
        return self.visualizer.create_visual(visual_request, key, **kwargs)

    def transform(self, requested_format, output_path=None):
        """This function deals with transforming a case to a new format."""
        self._status_check([0])
        output_path = output_path if output_path is not None else Path.cwd() / "data"
        self.exporter = CaseExporter(output_path, self.name, self.input_dict)
        self.exporter.create_template_for_requested_format(requested_format)

    def modify(self, input_dict_key, element_key, new_value):
        """
        This function changes the value of one of the inputs in the input_dict.
        The following keys in input_dict are currently supported: key_output_weight, scenario_weight, theme_weight
        :param input_dict_key: the key in the input_dict for which the value should be changed.
        :param element_key: is the name of the element within the input_dict_key to be changed
        :param new_value: is the new value to be changed to
        """
        self._status_check([0])
        supported_input_keys = ["key_output_weight", "scenario_weight", "theme_weight"]
        if input_dict_key not in supported_input_keys:
            raise ValueError("Please specify one of", supported_input_keys)
        master_key = input_dict_key.split("_weight")[0] + "s"
        index = np.where(self.input_dict[master_key] == element_key)
        old_value = self.input_dict[input_dict_key][index]
        self.input_dict[input_dict_key][index] = new_value
        print(f"The weight for {element_key} in {input_dict_key} is changed from {old_value[0]} to {new_value}.")

    def make_report(self, scenario, page_dict=None, output_path=Path.cwd() / "reports/"):
        """This function deals with transforming a case to a Report.
        :param scenario: the selected scenario of the case
        :param output_path: desired location of the report
        """
        self._status_check([0, 1, 2])
        page_dict = {} if not page_dict else page_dict
        # Do not show the graphs in notebook when making a report
        matplotlib.pyplot.ioff()
        self.report = MakeReport(output_path, self.name, self.input_dict, self.output_dict, self.visualize, page_dict)
        location_report = self.report.create_report(scenario, output_path)
        print(location_report)

    def optimize(self, scenario, method="auto", **kwargs):
        """Find the optimal distribution of internal inputs for a scenario.

        One entry point for every optimization method. ``method`` is ``"auto"``
        (the default), a single method name, or a list of names:

          * ``"auto"``      → probe the case, let the decision tree in
            :mod:`vlinder.method_selection` pick the method that fits the shape
            of its appreciation surface, run it, and report which one it picked
            and why. The pick is also recorded on the result, as ``.selection``.
          * single name   → run that method
          * list of names → run each, print every method's appreciation +
            allocation, and keep the best (only the winner is written back)

        Supported methods: ``"grid"`` (combinatorial grid search), ``"slsqp"``
        (continuous multi-start SLSQP), ``"basin_hopping"`` (SLSQP with an
        escape loop for surfaces with more than one optimum),
        ``"genetic_algorithm"`` (derivative-free evolutionary search) and
        ``"mdbh"`` (research method: mirror descent inside an escape loop).
        Naming one overrides the automatic choice; keyword arguments override
        the solver budget that ``"auto"`` attaches to its choice.

        The optimized allocation is written to a new decision-maker option whose
        name records both the method and the scenario it was optimized for, so a
        case optimized for several scenarios keeps them apart. When the case
        configures an ``Optimize_DMO_name``, that name is the base: a configured
        "Show me what you got" becomes "Show me what you got (grid) (Base case)".
        A ``dmo_name`` passed here overrides it.

        :param scenario: scenario name (must be in input_dict["scenarios"]).
        :param method: ``"auto"``, a method name, or a list of names.
        :param kwargs: ``dmo_name`` for the optimizer's decision-maker option,
            ``new_case_name`` for the optimized case name, plus the parameters of
            the chosen solver. Grid takes ``max_combinations`` (default 60000);
            slsqp takes ``n_starts`` (default 100), ``seed``; basin_hopping takes
            ``n_hops``, ``n_starts``, ``temperature``, ``step_frac``, ``seed``;
            genetic_algorithm takes ``population_size``, ``n_generations``,
            ``crossover_prob``, ``eta_crossover``, ``eta_mutation``, ``seed``.
            A parameter passed directly goes to every method that runs. To give
            two methods different parameters, pass ``method_kwargs`` as
            ``{method name: {setting: value}}``; those win over the direct ones.
        :return: the updated ``input_dict``. The full
            :class:`~vlinder.optimize.OptimizationResult` of this run is
            available as ``self.optimization_result``.
        """
        self._status_check([0, 1, 2])
        case_optimizer = Optimize(self.input_dict, self.output_dict)

        dmo_name = kwargs.pop("dmo_name", None)
        new_case_name = kwargs.pop("new_case_name", None)
        result = case_optimizer.run(scenario, method=method, dmo_name=dmo_name, **kwargs)
        # case_optimizer.input_dict holds the winning DMO (appended/updated); sync back.
        self.input_dict = case_optimizer.input_dict
        self.optimization_result = result
        self.name = new_case_name or f"{self.name} - Optimized"
        self._set_and_reset_status(3)
        return self.input_dict
