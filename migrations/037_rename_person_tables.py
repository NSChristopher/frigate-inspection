"""Peewee migrations -- 037_rename_person_tables.py.

Renames person_entity -> personentity and person_observation -> personobservation
to match Peewee's default table naming convention used by all other Frigate models.
"""

import peewee as pw

SQL = pw.SQL


def migrate(migrator, database, fake=False, **kwargs):
    migrator.sql("ALTER TABLE person_observation RENAME TO personobservation")
    migrator.sql("ALTER TABLE person_entity RENAME TO personentity")


def rollback(migrator, database, fake=False, **kwargs):
    migrator.sql("ALTER TABLE personentity RENAME TO person_entity")
    migrator.sql("ALTER TABLE personobservation RENAME TO person_observation")
