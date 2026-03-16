"""Graph store adapter: SQLite."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

from archrag.domain.models import Entity, Relation
from archrag.ports.graph_store import GraphStorePort


class SQLiteGraphStore(GraphStorePort):
    """Persist KG in a local SQLite database."""

    def __init__(self, db_path: str = "data/archrag.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    # ── schema ──

    def _create_tables(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS entities (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                entity_type TEXT NOT NULL DEFAULT '',
                source_chunk_ids TEXT NOT NULL DEFAULT '[]',
                embedding   TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(name);

            CREATE TABLE IF NOT EXISTS relations (
                id          TEXT PRIMARY KEY,
                source_id   TEXT NOT NULL,
                target_id   TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                weight      REAL NOT NULL DEFAULT 1.0,
                source_chunk_ids TEXT NOT NULL DEFAULT '[]',
                FOREIGN KEY (source_id) REFERENCES entities(id),
                FOREIGN KEY (target_id) REFERENCES entities(id)
            );
            CREATE INDEX IF NOT EXISTS idx_rel_src ON relations(source_id);
            CREATE INDEX IF NOT EXISTS idx_rel_tgt ON relations(target_id);
            """
        )
        self._conn.commit()

    # ── write ──

    def save_entity(self, entity: Entity) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO entities
               (id, name, description, entity_type, source_chunk_ids, embedding)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                entity.id,
                entity.name,
                entity.description,
                entity.entity_type,
                json.dumps(entity.source_chunk_ids),
                json.dumps(entity.embedding) if entity.embedding else None,
            ),
        )
        self._conn.commit()

    def save_entities(self, entities: list[Entity]) -> None:
        self._conn.executemany(
            """INSERT OR REPLACE INTO entities
               (id, name, description, entity_type, source_chunk_ids, embedding)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [
                (
                    e.id,
                    e.name,
                    e.description,
                    e.entity_type,
                    json.dumps(e.source_chunk_ids),
                    json.dumps(e.embedding) if e.embedding else None,
                )
                for e in entities
            ],
        )
        self._conn.commit()

    def save_relation(self, relation: Relation) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO relations
               (id, source_id, target_id, description, weight, source_chunk_ids)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                relation.id,
                relation.source_id,
                relation.target_id,
                relation.description,
                relation.weight,
                json.dumps(relation.source_chunk_ids),
            ),
        )
        self._conn.commit()

    def save_relations(self, relations: list[Relation]) -> None:
        self._conn.executemany(
            """INSERT OR REPLACE INTO relations
               (id, source_id, target_id, description, weight, source_chunk_ids)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [
                (
                    r.id,
                    r.source_id,
                    r.target_id,
                    r.description,
                    r.weight,
                    json.dumps(r.source_chunk_ids),
                )
                for r in relations
            ],
        )
        self._conn.commit()

    # ── read ──

    def _row_to_entity(self, row: tuple) -> Entity:
        return Entity(
            id=row[0],
            name=row[1],
            description=row[2],
            entity_type=row[3],
            source_chunk_ids=json.loads(row[4]),
            embedding=json.loads(row[5]) if row[5] else None,
        )

    def _row_to_relation(self, row: tuple) -> Relation:
        return Relation(
            id=row[0],
            source_id=row[1],
            target_id=row[2],
            description=row[3],
            weight=row[4],
            source_chunk_ids=json.loads(row[5]),
        )

    def get_entity(self, entity_id: str) -> Entity | None:
        cur = self._conn.execute("SELECT * FROM entities WHERE id=?", (entity_id,))
        row = cur.fetchone()
        return self._row_to_entity(row) if row else None

    def get_all_entities(self) -> list[Entity]:
        cur = self._conn.execute("SELECT * FROM entities")
        return [self._row_to_entity(r) for r in cur.fetchall()]

    def get_entity_by_name(self, name: str) -> Entity | None:
        cur = self._conn.execute(
            "SELECT * FROM entities WHERE LOWER(name)=LOWER(?)", (name,)
        )
        row = cur.fetchone()
        return self._row_to_entity(row) if row else None

    def get_relations_for(self, entity_id: str) -> list[Relation]:
        cur = self._conn.execute(
            "SELECT * FROM relations WHERE source_id=? OR target_id=?",
            (entity_id, entity_id),
        )
        return [self._row_to_relation(r) for r in cur.fetchall()]

    def get_all_relations(self) -> list[Relation]:
        cur = self._conn.execute("SELECT * FROM relations")
        return [self._row_to_relation(r) for r in cur.fetchall()]

    def get_neighbours(self, entity_id: str) -> list[Entity]:
        rels = self.get_relations_for(entity_id)
        ids: set[str] = set()
        for r in rels:
            if r.source_id == entity_id:
                ids.add(r.target_id)
            else:
                ids.add(r.source_id)
        return [e for eid in ids if (e := self.get_entity(eid)) is not None]

    def clear(self) -> None:
        self._conn.executescript("DELETE FROM relations; DELETE FROM entities;")
        self._conn.commit()

    def delete_entity(self, entity_id: str) -> None:
        self._conn.execute("DELETE FROM relations WHERE source_id=? OR target_id=?", (entity_id, entity_id))
        self._conn.execute("DELETE FROM entities WHERE id=?", (entity_id,))
        self._conn.commit()

    def search_entities_by_name(self, query: str) -> list[Entity]:
        cur = self._conn.execute(
            "SELECT * FROM entities WHERE LOWER(name) LIKE LOWER(?)",
            (f"%{query}%",),
        )
        return [self._row_to_entity(r) for r in cur.fetchall()]

    def clone(self) -> "SQLiteGraphStore":
        """Create an independent copy via sqlite3 backup API."""
        fd, tmp_path = tempfile.mkstemp(suffix=".db", prefix="archrag_graph_")
        import os
        os.close(fd)
        dst_conn = sqlite3.connect(tmp_path)
        self._conn.backup(dst_conn)
        dst_conn.close()
        return SQLiteGraphStore(db_path=tmp_path)
