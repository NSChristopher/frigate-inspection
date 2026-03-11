"""Peewee migrations -- 036_create_person_tables.py.

Creates personentity and personobservation tables for the person entity
tracking layer (Layers 2-4). The vec_face_observations virtual table is
created separately via SqliteVecQueueDatabase.create_person_embeddings_tables()
when the vec extension is loaded (embeddings process).
"""

import peewee as pw

SQL = pw.SQL


def migrate(migrator, database, fake=False, **kwargs):
    migrator.sql(
        """
        CREATE TABLE IF NOT EXISTS personentity (
            id VARCHAR(36) NOT NULL PRIMARY KEY,
            known_name VARCHAR(100) NULL,
            face_centroid BLOB NULL,
            first_seen DATETIME NOT NULL,
            last_seen DATETIME NOT NULL,
            observation_count INTEGER NOT NULL DEFAULT 0,
            cameras_seen TEXT NOT NULL,
            confidence REAL NULL,
            is_known INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    migrator.sql(
        """
        CREATE TABLE IF NOT EXISTS personobservation (
            id VARCHAR(36) NOT NULL PRIMARY KEY,
            event_id VARCHAR(30) NOT NULL,
            camera VARCHAR(20) NOT NULL,
            timestamp DATETIME NOT NULL,
            face_embedding BLOB NULL,
            face_quality_score REAL NULL,
            face_bbox TEXT NULL,
            body_embedding BLOB NULL,
            bbox TEXT NULL,
            snapshot_path TEXT NULL,
            zones TEXT NULL,
            path_data TEXT NULL,
            avg_speed REAL NULL,
            velocity_angle REAL NULL,
            dwell_time REAL NULL,
            person_entity_id VARCHAR(36) NULL
        )
        """
    )
    migrator.sql(
        'CREATE INDEX IF NOT EXISTS personobservation_event_id ON personobservation (event_id)'
    )
    migrator.sql(
        'CREATE INDEX IF NOT EXISTS personobservation_camera ON personobservation (camera)'
    )
    migrator.sql(
        'CREATE INDEX IF NOT EXISTS personobservation_person_entity_id ON personobservation (person_entity_id)'
    )


def rollback(migrator, database, fake=False, **kwargs):
    migrator.sql("DROP TABLE IF EXISTS personobservation")
    migrator.sql("DROP TABLE IF EXISTS personentity")
