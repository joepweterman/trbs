# pylint: disable=no-member
# pylint: disable=W0511

"""
This module contains the tRBS class. This is the parent class that deals with anything related to a Responsible
Business Simulator Case.
"""

from pathlib import Path
import os
import copy

import matplotlib
import vlinder as vl
from vlinder.modify import Modify
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


class TheResponsibleBusinessSimulator:  # pylint: disable=too-many-instance-attributes
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
        case_modifier = Modify(self.input_dict)
        self.input_dict = case_modifier.modify(input_dict_key, element_key, new_value)

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

    def optimize(self, scenario, method="basin_hopping", **kwargs):
        """
        This function deals with finding the optimal distribution of decision maker options.

        The run is routed through :meth:`vlinder.optimize.Optimize.run`, which dispatches to a
        solver class and returns the full :class:`~vlinder.optimize.OptimizationResult`.

        :param scenario: the selected scenario of the case
        :param method: a method name or a list of names; the default is ``"basin_hopping"``.
            Supported: ``"grid"``, ``"slsqp"`` and ``"basin_hopping"``. Basin-hopping is SLSQP
            wrapped in an escape loop: on a surface with one optimum it returns what SLSQP
            finds, and on a surface with several it keeps searching for the best one. ``"grid"``
            is the enumerative search, which returns nothing more precise than its step size.
        :param kwargs: ``new_dmo_name`` for the optimizer's decision-maker option and
            ``new_case_name`` for the optimized case name. The remaining keyword arguments
            reach the solver: grid takes ``max_combinations``; slsqp takes ``n_starts``
            (default 100) and ``seed``; basin_hopping takes ``n_hops``, ``n_starts``,
            ``temperature``, ``step_frac`` and ``seed``. Every solver takes ``spend_all`` and
            ``max_calculation_time``.
        :return: the :class:`~vlinder.optimize.OptimizationResult` of the run.
        """
        self._status_check([0, 1, 2])
        case_optimizer = Optimize(self.input_dict, self.output_dict)

        new_case_name = kwargs.pop("new_case_name", None)
        result = case_optimizer.run(scenario, method=method, dmo_name=kwargs.pop("new_dmo_name", None), **kwargs)
        self.input_dict = case_optimizer.input_dict
        self.name = new_case_name or f"{self.name} - Optimized"
        return result
