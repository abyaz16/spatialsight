"""Scoring ID reconciliation against MOT17, across every sequence.

The question this answers is narrow and the only one worth asking of this
layer: given the fragmented identities an upstream tracker produces, how much
of the truth does our reconciliation put back, and how often does it put back
something false?

Two numbers, and they are not symmetric.

* **Fragments** are one person reported as several entities. The count is too
  high; every reported history is real.
* **Merges** are several people reported as one entity. The count is too low
  *and* the entity has a history that never happened -- a dwell assembled
  from two different people. Nothing downstream can detect it, so a merge is
  strictly worse than a fragment and the parameters are chosen accordingly.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .datasets.mot import fragment, identity_errors, load_sequence, true_tracks
from .eval import score_tracking
from .tracker import TrackerParams, reconcile

DEFAULT_SEQUENCES = (
    "MOT17-02-FRCNN",
    "MOT17-04-FRCNN",
    "MOT17-05-FRCNN",
    "MOT17-09-FRCNN",
    "MOT17-10-FRCNN",
    "MOT17-11-FRCNN",
    "MOT17-13-FRCNN",
)


@dataclass
class SequenceResult:
    sequence: str
    fps: float
    frames: int
    boxes: int
    true_identities: int
    observed_fragments: int
    entities: int
    merges: int
    fragments: int
    ambiguous: int
    id_switches: int

    @property
    def merge_rate(self) -> float:
        return self.merges / self.true_identities if self.true_identities else 0.0

    @property
    def recovered(self) -> float:
        """Share of the upstream over-count that reconciliation removed."""
        excess = self.observed_fragments - self.true_identities
        if excess <= 0:
            return 1.0
        return max(0.0, (self.observed_fragments - self.entities) / excess)


@dataclass
class MotReport:
    patience: int
    gap_seconds: float
    results: list[SequenceResult] = field(default_factory=list)

    def totals(self) -> dict:
        return {
            "true_identities": sum(r.true_identities for r in self.results),
            "observed_fragments": sum(r.observed_fragments for r in self.results),
            "entities": sum(r.entities for r in self.results),
            "merges": sum(r.merges for r in self.results),
            "fragments": sum(r.fragments for r in self.results),
        }

    def lines(self) -> list[str]:
        out = [
            f"MOT17 identity reconciliation "
            f"(upstream patience {self.patience} frames, gap window {self.gap_seconds}s)",
            "",
            f"  {'sequence':<18}{'fps':>4}{'true':>6}{'observed':>10}{'ours':>6}"
            f"{'MERGES':>8}{'frags':>7}{'recovered':>11}",
            "  " + "-" * 70,
        ]
        for r in self.results:
            out.append(
                f"  {r.sequence:<18}{r.fps:>4.0f}{r.true_identities:>6}"
                f"{r.observed_fragments:>10}{r.entities:>6}{r.merges:>8}"
                f"{r.fragments:>7}{r.recovered:>10.0%}"
            )
        t = self.totals()
        excess = t["observed_fragments"] - t["true_identities"]
        recovered = (
            (t["observed_fragments"] - t["entities"]) / excess if excess > 0 else 1.0
        )
        out += [
            "  " + "-" * 70,
            f"  {'TOTAL':<18}{'':>4}{t['true_identities']:>6}"
            f"{t['observed_fragments']:>10}{t['entities']:>6}{t['merges']:>8}"
            f"{t['fragments']:>7}{recovered:>10.0%}",
            "",
            f"  merge rate   {t['merges'] / t['true_identities']:.1%} of true identities",
            f"  without us   {t['observed_fragments']} entities for "
            f"{t['true_identities']} people "
            f"({t['observed_fragments'] / t['true_identities']:.1f}x over-count)",
            f"  with us      {t['entities']} entities "
            f"({t['entities'] / t['true_identities']:.1f}x over-count)",
        ]
        return out

    def as_dict(self) -> dict:
        return {
            "patience": self.patience,
            "gap_seconds": self.gap_seconds,
            "totals": self.totals(),
            "sequences": [asdict(r) for r in self.results],
        }


def run_mot(
    root: str | Path,
    *,
    sequences: tuple[str, ...] = DEFAULT_SEQUENCES,
    patience: int = 5,
    gap_seconds: float = 1.0,
    min_visibility: float = 0.25,
) -> MotReport:
    root = Path(root)
    report = MotReport(patience=patience, gap_seconds=gap_seconds)
    for name in sequences:
        directory = root / name
        if not (directory / "gt" / "gt.txt").exists():
            continue
        sequence = load_sequence(directory, min_visibility=min_visibility)
        detections, observed = fragment(sequence, patience=patience)
        params = TrackerParams.for_fps(
            sequence.meta.fps, gap_seconds, min_detections=3
        )
        recon = reconcile(detections, params)
        errors = identity_errors(recon.tracks, sequence)
        metrics = score_tracking(recon.tracks, true_tracks(sequence))
        report.results.append(
            SequenceResult(
                sequence=sequence.name,
                fps=sequence.meta.fps,
                frames=sequence.meta.frame_count,
                boxes=len(sequence.boxes),
                true_identities=sequence.identities,
                observed_fragments=observed,
                entities=len(recon.tracks),
                merges=errors["merges"],
                fragments=errors["fragments"],
                ambiguous=recon.ambiguous,
                id_switches=metrics.id_switches,
            )
        )
    return report


def write_report(report: MotReport, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    return path
