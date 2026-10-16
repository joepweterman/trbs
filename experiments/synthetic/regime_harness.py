"""
Regime-aware extension of the study harness, for the H3 and H4 phases.

The locked H1/H2/H5 apparatus (``study_harness.py``) can only ever generate
convex-regime cases: ``StudyHarness.ensure_case`` builds ``SyntheticCaseParams``
without a ``regime`` argument, so the regime falls back to ``"convex"``, and the
factory's own validation then forbids every non-convexity knob. Running H3
(smooth non-convex) or H4 (non-smooth) therefore needs more than compute.

This module supplies exactly what is missing and nothing else. It subclasses the
locked spec and harness rather than editing them, so the frozen apparatus stays
byte-identical and the confirmatory store of the locked phases remains
reproducible from the same file the lock commit timestamps.

Two things are added:

1. ``RegimeSpec.variant_params`` maps a variant *label* onto the factory knobs
   that build it, so one study can span several regimes and mechanisms at once.
2. ``RegimeSpec.variant_from_case_name`` recovers that label from a results row,
   because several H3/H4 variants share one appreciation shape and the row's
   ``variant`` field can no longer identify which arm of the design it came
   from. The label travels inside the case name, which every row already
   carries, so the locked execution path is not touched at all: ``run_task``,
   ``build_tasks`` and the results schema are exactly the ones the lock commit
   timestamps. The label-to-knobs mapping is recorded in ``meta.json`` next to
   the results, because it is a field of the spec.

No hypothesis is decided here, and no confirmatory run may start from this file
until the amendment that freezes the H3/H4 design is committed (pre-registration
section 7, lock-before-run).
"""

from dataclasses import dataclass, field

from case_factory import SyntheticCaseFactory, SyntheticCaseParams, manifest_path, read_manifest
from oracle import certify_case
from study_harness import StudyHarness, StudySpec


@dataclass
class RegimeSpec(StudySpec):
    """Factorial design that can span regimes, not only appreciation shapes.

    With ``variant_params`` left empty this behaves exactly like the locked
    ``StudySpec``: the variant label is the appreciation shape and the regime
    stays convex. That equivalence is asserted by the tests.
    """

    case_prefix: str = "Study"
    variant_params: dict = field(default_factory=dict)
    method_overrides: dict = field(default_factory=dict)

    def __post_init__(self):
        """Refuse a design whose labels cannot survive a round trip."""
        if "_" in self.case_prefix:
            raise ValueError("case_prefix may not contain '_': the label is parsed back out of the case name")
        for variant in self.variants:
            if "_" in variant:
                raise ValueError(f"variant label {variant!r} may not contain '_'")
            if self.variant_params and variant not in self.variant_params:
                raise KeyError(f"no factory knobs registered for variant {variant!r}")

    def variant_from_case_name(self, name):
        """
        This function recovers the variant label of one results row.
        :param name: the case name, as written in every results row
        :return: the variant label the case was built from
        """
        prefix = f"{self.case_prefix}_"
        if not name.startswith(prefix):
            raise ValueError(f"case name {name!r} does not belong to this study")
        return name.removeprefix(prefix).rsplit("_k", 1)[0]

    def factory_kwargs(self, variant):
        """
        This function returns the factory knobs that build one variant.
        :param variant: the variant label
        :return: a dictionary of SyntheticCaseParams keyword arguments
        """
        if not self.variant_params:
            return {"appreciation": variant}
        if variant not in self.variant_params:
            raise KeyError(f"no factory knobs registered for variant {variant!r}")
        return dict(self.variant_params[variant])

    def case_name(self, variant, k, seed):
        """
        This function names one case deterministically.
        :param variant: the variant label
        :param k: the number of internal variables
        :param seed: the case seed
        :return: the case name
        """
        return f"{self.case_prefix}_{variant}_k{k:02d}_s{seed}"


class RegimeHarness(StudyHarness):
    """Study harness that builds cases from a regime-aware spec."""

    def ensure_case(self, variant, k, seed):
        """
        This function generates and oracle-certifies one case if that has not
        happened yet, so a study run stays idempotent and resumable.
        :param variant: the variant label
        :param k: the number of internal variables
        :param seed: the case seed
        :return: a tuple of the case name and its manifest
        """
        name = self.spec.case_name(variant, k, seed)
        if not manifest_path(name, self.root).exists():
            params = SyntheticCaseParams(
                name=name,
                k=k,
                n_key_outputs=self.spec.n_key_outputs(k),
                seed=seed,
                **self.spec.factory_kwargs(variant),
            )
            SyntheticCaseFactory(params).write(self.root)
        if read_manifest(name, self.root)["oracle"] is None:
            certify_case(name, self.root)
        return name, read_manifest(name, self.root)

    def _case_tasks(self, name, manifest, done):
        """
        This function enumerates one case's outstanding tasks and applies the
        spec's method-budget overrides on top of the frozen defaults. H3 is
        stated on single-start SLSQP, which the locked METHOD_DEFAULTS cannot
        express; the locked task construction is reused unchanged, so only the
        budget differs and only where the spec says so.
        :param name: the case name
        :param manifest: the case manifest
        :param done: the set of task keys already in the results store
        :return: a list of task dictionaries
        """
        tasks = super()._case_tasks(name, manifest, done)
        for task in tasks:
            override = self.spec.method_overrides.get(task["method"])
            if override:
                task["method_kwargs"].update(override)
        return tasks
