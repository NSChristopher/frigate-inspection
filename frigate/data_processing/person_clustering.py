"""Identity clustering: link person observations to person entities by face similarity."""

import logging
import uuid

import numpy as np

from frigate.config import FrigateConfig
from frigate.models import PersonEntity, PersonObservation
from frigate.util.builtin import deserialize, serialize

logger = logging.getLogger(__name__)


def run_clustering(db, config: FrigateConfig) -> None:
    """Link unclustered person observations to entities by face embedding similarity."""
    if not config.person_entity.enabled:
        return
    pe_config = config.person_entity
    threshold = pe_config.similarity_threshold
    # Cosine distance in vec0: 0 = identical, 2 = opposite. We want distance <= 1 - threshold.
    max_distance = 1.0 - threshold

    try:
        unclustered = (
            PersonObservation.select(
                PersonObservation.id,
                PersonObservation.face_embedding,
                PersonObservation.event_id,
                PersonObservation.camera,
                PersonObservation.timestamp,
)
            .where(
                PersonObservation.person_entity_id.is_null(True),
                PersonObservation.face_embedding.is_null(False),
            )
            .limit(50)
        )
    except Exception as e:
        logger.debug("Person clustering: no observations or table missing: %s", e)
        return

    if not unclustered:
        return

    for obs in unclustered:
        if obs.face_embedding is None:
            continue
        try:
            emb_bytes = obs.face_embedding
            if isinstance(emb_bytes, memoryview):
                emb_bytes = bytes(emb_bytes)
            embedding = np.array(deserialize(emb_bytes), dtype=np.float32)
        except Exception as e:
            logger.warning("Person clustering: failed to deserialize embedding %s: %s", obs.id, e)
            continue

        try:
            cursor = db.execute_sql(
                """
                SELECT id, distance
                FROM vec_face_observations
                WHERE face_embedding MATCH ?
                AND k = 20
                ORDER BY distance
                """,
                (serialize(embedding),),
            )
            rows = cursor.fetchall() if cursor else []
        except Exception as e:
            logger.debug("Person clustering: vec search failed: %s", e)
            continue

        if not rows:
            continue

        # Find best match that already has an entity
        entity_id = None
        similar_unclustered = []
        for row in rows:
            obs_id, distance = row[0], row[1]
            if distance > max_distance:
                break
            if obs_id == obs.id:
                continue
            other = PersonObservation.get_or_none(PersonObservation.id == obs_id)
            if other is None:
                continue
            if other.person_entity_id is not None:
                if entity_id is None:
                    entity_id = other.person_entity_id
            else:
                similar_unclustered.append(obs_id)

        if entity_id is None and similar_unclustered:
            entity_id = str(uuid.uuid4())
            PersonEntity.insert(
                {
                    PersonEntity.id: entity_id,
                    PersonEntity.known_name: None,
                    PersonEntity.face_centroid: emb_bytes,
                    PersonEntity.first_seen: obs.timestamp,
                    PersonEntity.last_seen: obs.timestamp,
                    PersonEntity.observation_count: 1,
                    PersonEntity.cameras_seen: [obs.camera],
                    PersonEntity.confidence: float(threshold),
                    PersonEntity.is_known: False,
                }
            ).execute()
            PersonObservation.update(
                person_entity_id=entity_id
            ).where(PersonObservation.id == obs.id).execute()
            for other_id in similar_unclustered[:10]:
                PersonObservation.update(
                    person_entity_id=entity_id
                ).where(PersonObservation.id == other_id).execute()
            _update_entity_stats(entity_id)
            continue

        if entity_id is not None:
            PersonObservation.update(
                person_entity_id=entity_id
            ).where(PersonObservation.id == obs.id).execute()
            _update_entity_stats(entity_id)


def _update_entity_stats(entity_id: str) -> None:
    """Refresh first_seen, last_seen, observation_count, cameras_seen for an entity."""
    try:
        obs_list = list(
            PersonObservation.select()
            .where(PersonObservation.person_entity_id == entity_id)
            .order_by(PersonObservation.timestamp)
        )
    except Exception:
        return
    if not obs_list:
        return
    first_ts = obs_list[0].timestamp
    last_ts = obs_list[-1].timestamp
    cameras = list({o.camera for o in obs_list})
    PersonEntity.update(
        first_seen=first_ts,
        last_seen=last_ts,
        observation_count=len(obs_list),
        cameras_seen=cameras,
    ).where(PersonEntity.id == entity_id).execute()
