"""End-to-end: the synthetic scene, whose answers are known by construction.

This is the test that would catch a regression anywhere between segmentation
and SQL. It asserts the counts a person watching the simulated clip would
write down.
"""

from __future__ import annotations

from spatialsight.backends.synthetic import SyntheticBackend, default_scene
from spatialsight.db import connect
from spatialsight.runner import run

ZONES = "zones/synthetic-01.json"


def _run(tmp_path):
    return run(
        backend=SyntheticBackend("synthetic-01"),
        clip="synthetic-01",
        zones_path=ZONES,
        db_path=tmp_path / "test.sqlite",
    )


def test_entity_count_matches_the_scene(tmp_path):
    report, _, _ = _run(tmp_path)
    truth = len(default_scene().actors)
    assert report.entities == truth, (
        f"scene has {truth} actors; pipeline reported {report.entities}"
    )


def test_raw_ids_exceed_entities_so_the_repair_is_actually_exercised(tmp_path):
    report, recon, _ = _run(tmp_path)
    assert report.raw_object_ids > report.entities
    assert recon.stitches >= 1, "the injected identity split must be repaired"


def test_every_event_has_a_positive_dwell_and_a_known_zone(tmp_path):
    report, _, events = _run(tmp_path)
    assert events
    for e in events:
        assert e.dwell_s >= 0.0
        assert e.zone in report.zones
        assert e.frame_end >= e.frame_start
        assert 0.0 <= e.peak_overlap <= 1.0


def test_events_reference_real_entities(tmp_path):
    _, recon, events = _run(tmp_path)
    known = {t.entity_id for t in recon.tracks}
    assert {e.entity_id for e in events} <= known


def test_rows_land_in_sqlite_and_join(tmp_path):
    db = tmp_path / "test.sqlite"
    _run(tmp_path)
    con = connect(db, read_only=True)
    try:
        orphans = con.execute(
            "SELECT COUNT(*) FROM events e LEFT JOIN entities n"
            "  ON e.clip = n.clip AND e.entity_id = n.entity_id"
            " WHERE n.entity_id IS NULL"
        ).fetchone()[0]
        assert orphans == 0, "every event must join to an entity"
        assert con.execute("SELECT COUNT(*) FROM events").fetchone()[0] > 0
    finally:
        con.close()


def test_rerunning_replaces_rather_than_duplicates(tmp_path):
    db = tmp_path / "test.sqlite"
    _run(tmp_path)
    con = connect(db, read_only=True)
    first = con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    con.close()

    _run(tmp_path)
    con = connect(db, read_only=True)
    try:
        assert con.execute("SELECT COUNT(*) FROM events").fetchone()[0] == first
    finally:
        con.close()


def test_the_vehicle_is_found_in_the_loading_dock(tmp_path):
    # Actor 1 is scripted to sit in the dock for most of the first minute.
    _, _, events = _run(tmp_path)
    dock = [e for e in events if e.zone == "loading_dock" and e.label == "vehicle"]
    assert dock, "the scripted forklift dwell must appear as an event"
    assert max(e.dwell_s for e in dock) > 20.0


# ----- the schema migration -------------------------------------------------


def test_a_database_from_before_the_colour_columns_is_upgraded(tmp_path):
    """`CREATE TABLE IF NOT EXISTS` does nothing to an existing table.

    Without the migration, a new column never appears in a database someone
    has already ingested into, and the first query for it fails with 'no such
    column' on data that looks perfectly fine.
    """
    import sqlite3

    from spatialsight.db import connect

    path = tmp_path / "old.sqlite"
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE entities (entity_id INTEGER NOT NULL, clip TEXT NOT NULL,"
        " class TEXT NOT NULL, first_seen REAL NOT NULL, last_seen REAL NOT NULL,"
        " frame_first INTEGER NOT NULL, frame_last INTEGER NOT NULL,"
        " PRIMARY KEY (clip, entity_id));"
        "INSERT INTO entities VALUES (1,'old-clip','person',0.0,1.0,0,2);"
    )
    con.commit()
    con.close()

    upgraded = connect(path)
    try:
        columns = {r[1] for r in upgraded.execute("PRAGMA table_info(entities)")}
        assert {"colour", "colour_hex", "colour_stability"} <= columns
        # And the rows that were already there survived it.
        assert upgraded.execute("SELECT COUNT(*) FROM entities").fetchone()[0] == 1
    finally:
        upgraded.close()


def test_migrating_twice_is_a_no_op(tmp_path):
    from spatialsight.db import _migrate, connect

    path = tmp_path / "m.sqlite"
    con = connect(path)
    try:
        assert _migrate(con) == []
    finally:
        con.close()


def test_colour_is_written_for_a_clip_with_pixels(tmp_path):
    """End to end: painted actors in, colour column out."""
    import sqlite3

    from spatialsight.backends.synthetic import SyntheticBackend
    from spatialsight.runner import run

    path = tmp_path / "c.sqlite"
    report, _, _ = run(
        backend=SyntheticBackend("synthetic-01"),
        clip="synthetic-01",
        zones_path="zones/synthetic-01.json",
        db_path=path,
    )
    assert report.coloured == report.entities
    assert report.colour_conflicts == 2  # the scripted ID switch

    con = sqlite3.connect(path)
    try:
        rows = dict(
            con.execute(
                "SELECT entity_id, colour FROM entities WHERE clip='synthetic-01'"
            )
        )
        assert rows[1] == "red"
        assert rows[2] == "blue"
    finally:
        con.close()


def test_colour_is_null_when_the_clip_has_no_pixels(tmp_path):
    """A cached run over a `synthetic://` clip has masks but no frames.

    NULL is the honest answer, and a query for a red box then returns
    nothing rather than something invented.
    """
    import sqlite3

    from spatialsight.backends.synthetic import SyntheticBackend
    from spatialsight.runner import run

    path = tmp_path / "n.sqlite"
    report, _, _ = run(
        backend=SyntheticBackend("synthetic-01"),
        clip="synthetic-01",
        zones_path="zones/synthetic-01.json",
        db_path=path,
        with_colour=False,
    )
    assert report.coloured == 0

    con = sqlite3.connect(path)
    try:
        colours = [r[0] for r in con.execute("SELECT colour FROM entities")]
        assert colours and all(c is None for c in colours)
    finally:
        con.close()
