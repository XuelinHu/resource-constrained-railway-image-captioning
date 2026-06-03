from railcap.knowledge.ontology import RailwayOntology


def test_extract_terms_with_aliases(tmp_path):
    ontology_path = tmp_path / "terms.yaml"
    ontology_path.write_text(
        """
equipment:
  - 钢轨
faults:
  - 异物侵限
risks:
  - 行车安全风险
aliases:
  钢轨:
    - 铁轨
  异物侵限:
    - 异物侵入限界
""".strip(),
        encoding="utf-8",
    )
    ontology = RailwayOntology.from_yaml(ontology_path)

    assert ontology.extract_terms("铁轨旁存在异物侵入限界，可能造成行车安全风险") == {
        "钢轨",
        "异物侵限",
        "行车安全风险",
    }
