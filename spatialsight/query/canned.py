"""Ten questions with reference SQL written by hand.

They serve three purposes at once, which is why they are worth maintaining:

1. **The demo runs without a key.** Every canned question answers from the
   database alone, so the UI is fully explorable offline and on the morning
   the credits run out.
2. **They are the NL->SQL benchmark.** When the key is live, Muse Spark is
   asked each question in plain English and its SQL is compared against the
   reference result set. That turns "the text-to-SQL works" into a number.
3. **They are a schema regression test.** If a column is renamed, these fail
   immediately rather than at the demo.

The hard ones are deliberate. `concurrent_dock_occupancy` needs a self-join
on overlapping intervals, and `median_dwell` needs a median in a dialect with
no median function. Both are places a language model reliably stumbles, so
both are measured rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CannedQuestion:
    key: str
    question: str
    sql: str
    note: str = ""
    must_have: tuple[str, ...] = ()
    """Reference columns any correct answer has to carry.

    The rule is what the question's own words demand, applied uniformly:
    a "which zone" superlative demands the zone, "how many" demands the
    count, "with the frame to seek to" demands the frame. Supporting
    detail the reference happens to select is not required.

    Stated per question rather than inferred from `sql`, because otherwise
    the benchmark's definition of "correct" is whatever columns the reference
    query happens to select. A model that answers "the five longest dwells,
    with the frame to seek to" with exactly those two columns is right, and a
    benchmark that calls it wrong is measuring the reference, not the model.
    """


QUESTIONS: tuple[CannedQuestion, ...] = (
    CannedQuestion(
        key="entity_count",
        question="How many entities were tracked, broken down by class?",
        sql="""
SELECT class, COUNT(*) AS entities
FROM entities
GROUP BY class
ORDER BY entities DESC
""",
            must_have=('class', 'entities'),
    ),
    CannedQuestion(
        key="longest_average_dwell",
        question="Which zone had the longest average dwell?",
        sql="""
SELECT zone,
       ROUND(AVG(dwell_s), 1) AS avg_dwell_s,
       COUNT(*) AS visits
FROM events
GROUP BY zone
ORDER BY avg_dwell_s DESC
""",
            must_have=('zone',),
    ),
    CannedQuestion(
        key="busiest_zone",
        question="Which zone was visited most often?",
        sql="""
SELECT zone, COUNT(*) AS visits, COUNT(DISTINCT entity_id) AS distinct_entities
FROM events
GROUP BY zone
ORDER BY visits DESC
""",
            must_have=('zone',),
    ),
    CannedQuestion(
        key="longest_dwells",
        question="Show the five longest dwells, with the frame to seek to.",
        sql="""
SELECT entity_id, class, zone, dwell_s, frame_start, entered_at
FROM events
ORDER BY dwell_s DESC
LIMIT 5
""",
            must_have=('dwell_s', 'frame_start'),
    ),
    CannedQuestion(
        key="time_per_entity_in_dock",
        question="Which entity spent the most total time in the loading dock?",
        sql="""
SELECT entity_id, class,
       ROUND(SUM(dwell_s), 1) AS total_dwell_s,
       COUNT(*) AS visits
FROM events
WHERE zone = 'loading_dock'
GROUP BY entity_id, class
ORDER BY total_dwell_s DESC
""",
            must_have=('entity_id',),
    ),
    CannedQuestion(
        key="multi_zone_entities",
        question="Which entities visited more than one zone?",
        sql="""
SELECT entity_id, class,
       COUNT(DISTINCT zone) AS zones_visited,
       GROUP_CONCAT(DISTINCT zone) AS zones
FROM events
GROUP BY entity_id, class
HAVING COUNT(DISTINCT zone) > 1
ORDER BY zones_visited DESC
""",
            must_have=('entity_id',),
    ),
    CannedQuestion(
        key="concurrent_dock_occupancy",
        question=(
            "Was a person ever in the loading dock at the same time as a vehicle?"
        ),
        sql="""
SELECT p.entity_id AS person_id,
       v.entity_id AS vehicle_id,
       MAX(p.entered_at, v.entered_at) AS overlap_start_s,
       MIN(p.exited_at, v.exited_at) AS overlap_end_s,
       ROUND(MIN(p.exited_at, v.exited_at) - MAX(p.entered_at, v.entered_at), 1)
           AS overlap_s,
       p.frame_start AS person_frame
FROM events p
JOIN events v
  ON p.clip = v.clip
 AND p.zone = v.zone
 AND p.class = 'person'
 AND v.class = 'vehicle'
 AND p.entered_at < v.exited_at
 AND v.entered_at < p.exited_at
WHERE p.zone = 'loading_dock'
ORDER BY overlap_s DESC
""",
        note="Interval-overlap self-join. Models usually get the inequality backwards.",
            must_have=('person_id', 'vehicle_id'),
    ),
    CannedQuestion(
        key="median_dwell",
        question="What is the median dwell time per zone?",
        sql="""
SELECT zone,
       ROUND(AVG(dwell_s), 1) AS median_dwell_s
FROM (
  SELECT zone, dwell_s,
         ROW_NUMBER() OVER (PARTITION BY zone ORDER BY dwell_s) AS rn,
         COUNT(*) OVER (PARTITION BY zone) AS n
  FROM events
)
WHERE rn IN ((n + 1) / 2, (n + 2) / 2)
GROUP BY zone
ORDER BY median_dwell_s DESC
""",
        note="SQLite has no median; this is the window-function construction.",
            must_have=('zone', 'median_dwell_s'),
    ),
    CannedQuestion(
        key="open_at_clip_end",
        question="Which visits were still in progress when the clip ended?",
        sql="""
SELECT entity_id, class, zone, entered_at, dwell_s, frame_start
FROM events
WHERE closed = 0
ORDER BY dwell_s DESC
""",
            must_have=('entity_id', 'zone'),
    ),
    CannedQuestion(
        key="entries_per_minute",
        question="How many zone entries happened in each minute of the clip?",
        sql="""
SELECT CAST(entered_at / 60 AS INTEGER) AS minute,
       COUNT(*) AS entries,
       GROUP_CONCAT(DISTINCT zone) AS zones
FROM events
GROUP BY minute
ORDER BY minute
""",
            must_have=('minute', 'entries'),
    ),
)

BY_KEY = {q.key: q for q in QUESTIONS}
