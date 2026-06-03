from __future__ import annotations

from dataclasses import dataclass

from railcap.data.schema import CaptionSample
from railcap.knowledge.ontology import RailwayOntology


@dataclass(frozen=True)
class CaptionKnowledgeReport:
    expected_terms: set[str]
    mentioned_terms: set[str]
    missing_terms: set[str]
    hallucinated_terms: set[str]

    @property
    def has_hallucination(self) -> bool:
        return bool(self.hallucinated_terms)


def expected_terms_from_sample(sample: CaptionSample) -> set[str]:
    return set(sample.equipment) | set(sample.faults) | set(sample.risks)


def analyze_caption(
    sample: CaptionSample,
    generated_caption: str,
    ontology: RailwayOntology,
) -> CaptionKnowledgeReport:
    expected = {ontology.canonicalize(term) for term in expected_terms_from_sample(sample)}
    mentioned = ontology.extract_terms(generated_caption)
    ontology_terms = ontology.canonical_terms

    return CaptionKnowledgeReport(
        expected_terms=expected,
        mentioned_terms=mentioned,
        missing_terms=expected - mentioned,
        hallucinated_terms=(mentioned - expected) & ontology_terms,
    )


def build_generation_prefix(sample: CaptionSample) -> str:
    parts: list[str] = []
    if sample.equipment:
        parts.append("设备：" + "、".join(sample.equipment))
    if sample.faults:
        parts.append("异常：" + "、".join(sample.faults))
    if sample.risks:
        parts.append("风险：" + "、".join(sample.risks))
    if not parts:
        return "请描述铁路巡检图像："
    return "；".join(parts) + "。请生成专业、准确、无臆测的铁路图像描述："
