from __future__ import annotations

from dataclasses import dataclass

from railcap.data.schema import CaptionSample
from railcap.knowledge.constraints import analyze_caption
from railcap.knowledge.ontology import RailwayOntology


@dataclass(frozen=True)
class DomainMetrics:
    equipment_recall: float
    fault_precision: float
    fault_recall: float
    fault_f1: float
    risk_accuracy: float
    hallucination_rate: float

    def as_dict(self) -> dict[str, float]:
        return {
            "equipment_recall": self.equipment_recall,
            "fault_precision": self.fault_precision,
            "fault_recall": self.fault_recall,
            "fault_f1": self.fault_f1,
            "risk_accuracy": self.risk_accuracy,
            "hallucination_rate": self.hallucination_rate,
        }


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_domain_metrics(
    samples: list[CaptionSample],
    predictions: list[str],
    ontology: RailwayOntology,
) -> DomainMetrics:
    if len(samples) != len(predictions):
        raise ValueError("samples and predictions must have the same length")

    equipment_hits = 0
    equipment_total = 0
    fault_tp = 0
    fault_fp = 0
    fault_fn = 0
    risk_hits = 0
    risk_total = 0
    hallucinated = 0

    for sample, prediction in zip(samples, predictions, strict=True):
        mentioned = ontology.extract_terms(prediction)

        expected_equipment = {ontology.canonicalize(x) for x in sample.equipment}
        equipment_hits += len(expected_equipment & mentioned)
        equipment_total += len(expected_equipment)

        expected_faults = {ontology.canonicalize(x) for x in sample.faults}
        predicted_faults = mentioned & set(ontology.faults)
        fault_tp += len(expected_faults & predicted_faults)
        fault_fp += len(predicted_faults - expected_faults)
        fault_fn += len(expected_faults - predicted_faults)

        expected_risks = {ontology.canonicalize(x) for x in sample.risks}
        if expected_risks:
            risk_total += 1
            if expected_risks & mentioned:
                risk_hits += 1

        report = analyze_caption(sample, prediction, ontology)
        hallucinated += int(report.has_hallucination)

    fault_precision = _safe_div(fault_tp, fault_tp + fault_fp)
    fault_recall = _safe_div(fault_tp, fault_tp + fault_fn)
    fault_f1 = _safe_div(2 * fault_precision * fault_recall, fault_precision + fault_recall)

    return DomainMetrics(
        equipment_recall=_safe_div(equipment_hits, equipment_total),
        fault_precision=fault_precision,
        fault_recall=fault_recall,
        fault_f1=fault_f1,
        risk_accuracy=_safe_div(risk_hits, risk_total),
        hallucination_rate=_safe_div(hallucinated, len(samples)),
    )
