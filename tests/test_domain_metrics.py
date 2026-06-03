from railcap.data.schema import CaptionSample
from railcap.knowledge.ontology import RailwayOntology
from railcap.metrics.domain_metrics import compute_domain_metrics


def test_compute_domain_metrics():
    ontology = RailwayOntology(
        equipment=("钢轨", "扣件"),
        faults=("扣件松动",),
        risks=("设备失效风险",),
        aliases={},
    )
    samples = [
        CaptionSample(
            image="a.jpg",
            caption="",
            equipment=["钢轨", "扣件"],
            faults=["扣件松动"],
            risks=["设备失效风险"],
        )
    ]
    metrics = compute_domain_metrics(samples, ["钢轨和扣件可见，存在扣件松动和设备失效风险"], ontology)

    assert metrics.equipment_recall == 1.0
    assert metrics.fault_f1 == 1.0
    assert metrics.risk_accuracy == 1.0
    assert metrics.hallucination_rate == 0.0
