"""
EQL-native Ripple Down Rules.

The rule tree is a live EQL expression DAG (``Refinement`` / ``Alternative`` / ``Add``)
and classification is plain EQL evaluation. The RDR attaches to evaluation through the
aspect-oriented :class:`~krrood.entity_query_language.evaluation.EvaluationObserver`
hooks rather than driving a bespoke traversal.
"""

from krrood.entity_query_language.rdr.backend import (
    GroundTruth,
    ModelKey,
    RDRBackend,
    key_from_attribute,
)
from krrood.entity_query_language.rdr.expert import (
    Expert,
    NoConclusionProvided,
    NoConditionsProvided,
)
from krrood.entity_query_language.rdr.interactive import IPythonInterface
from krrood.entity_query_language.rdr.interface import (
    AnswerRequest,
    CaseContext,
    ExpertAbort,
    ExpertInterface,
    FunctionInterface,
)
from krrood.entity_query_language.rdr.observer import (
    ClassificationTrace,
    ConclusionObserver,
    FiredConclusion,
    classify_case,
    trace_case,
)
from krrood.entity_query_language.rdr.rule_tree import (
    insert_alternative,
    insert_refinement,
)
from krrood.entity_query_language.rdr.rule_tree_view import (
    RuleStatus,
    RuleView,
    render_rule_tree,
    walk_rules,
)
from krrood.entity_query_language.rdr.serialization import (
    load_rdr,
    rdr_to_python,
    save_rdr,
)
from krrood.entity_query_language.rdr.decorator import RDRWrapper, rdr
from krrood.entity_query_language.rdr.file_store import RDRFileStore
from krrood.entity_query_language.rdr.function_case import FunctionCase
from krrood.entity_query_language.rdr.progress import (
    IPythonProgressBar,
    ProgressReporter,
)
from krrood.entity_query_language.rdr.serialization import save_rdr_with_case
from krrood.entity_query_language.rdr.backward_inference import (
    BackwardInferenceIndex,
    ConclusionKnowledge,
    GuardCondition,
    SufficientConditionSet,
    what_do_we_know_about,
)
from krrood.entity_query_language.rdr.condition_resolver import (
    ChainConditionResolver,
    ConditionResolver,
    CornerCaseKnowledgeResolver,
    ResolvedCondition,
    ResolutionSource,
    TargetKnowledgeResolver,
)
from krrood.entity_query_language.rdr.single_class import EQLSingleClassRDR
from krrood.entity_query_language.rdr.underspecified import (
    MultipleInferenceTargets,
    NoInferenceTarget,
    UnderspecifiedMatch,
    UnsupportedInferenceTarget,
    is_ellipsis_target,
)

__all__ = [
    "ConclusionObserver",
    "FiredConclusion",
    "classify_case",
    "trace_case",
    "ClassificationTrace",
    "RuleStatus",
    "RuleView",
    "walk_rules",
    "render_rule_tree",
    "Expert",
    "ExpertInterface",
    "IPythonInterface",
    "FunctionInterface",
    "CaseContext",
    "AnswerRequest",
    "ExpertAbort",
    "NoConditionsProvided",
    "NoConclusionProvided",
    "insert_alternative",
    "insert_refinement",
    "EQLSingleClassRDR",
    "rdr_to_python",
    "save_rdr",
    "load_rdr",
    "RDRBackend",
    "GroundTruth",
    "ModelKey",
    "key_from_attribute",
    "UnderspecifiedMatch",
    "is_ellipsis_target",
    "NoInferenceTarget",
    "MultipleInferenceTargets",
    "UnsupportedInferenceTarget",
    # progress bar
    "ProgressReporter",
    "IPythonProgressBar",
    # @rdr decorator
    "rdr",
    "RDRWrapper",
    "RDRFileStore",
    "FunctionCase",
    "save_rdr_with_case",
    # backward inference
    "BackwardInferenceIndex",
    "ConclusionKnowledge",
    "GuardCondition",
    "SufficientConditionSet",
    "what_do_we_know_about",
    # auto-condition resolution
    "ConditionResolver",
    "ChainConditionResolver",
    "TargetKnowledgeResolver",
    "CornerCaseKnowledgeResolver",
    "ResolvedCondition",
    "ResolutionSource",
]
