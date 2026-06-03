from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RailwayOntology:
    equipment: tuple[str, ...]
    faults: tuple[str, ...]
    risks: tuple[str, ...]
    aliases: dict[str, tuple[str, ...]]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RailwayOntology":
        with Path(path).open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        aliases = {
            str(term): tuple(str(alias) for alias in values)
            for term, values in (raw.get("aliases") or {}).items()
        }
        return cls(
            equipment=tuple(str(x) for x in raw.get("equipment", [])),
            faults=tuple(str(x) for x in raw.get("faults", [])),
            risks=tuple(str(x) for x in raw.get("risks", [])),
            aliases=aliases,
        )

    @property
    def canonical_terms(self) -> set[str]:
        return set(self.equipment) | set(self.faults) | set(self.risks)

    @property
    def all_surface_terms(self) -> set[str]:
        terms = set(self.canonical_terms)
        for values in self.aliases.values():
            terms.update(values)
        return terms

    def canonicalize(self, term: str) -> str:
        if term in self.canonical_terms:
            return term
        for canonical, aliases in self.aliases.items():
            if term in aliases:
                return canonical
        return term

    def extract_terms(self, text: str) -> set[str]:
        found: set[str] = set()
        for term in self.all_surface_terms:
            if term and term in text:
                found.add(self.canonicalize(term))
        return found
