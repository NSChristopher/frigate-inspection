"""Person entity APIs."""

import logging
from typing import Any, Optional

from fastapi import APIRouter, Body, Request, UploadFile
from fastapi.params import Depends
from fastapi.responses import JSONResponse
from peewee import DoesNotExist
from pydantic import BaseModel

from frigate.api.auth import allow_any_authenticated, require_role
from frigate.api.defs.tags import Tags
from frigate.embeddings import EmbeddingsContext
from frigate.models import PersonEntity, PersonObservation


class PersonNameBody(BaseModel):
    name: Optional[str] = None


class PersonMergeBody(BaseModel):
    source_entity_id: str
    target_entity_id: str


class PersonSplitBody(BaseModel):
    entity_id: str
    observation_ids: list[str]

logger = logging.getLogger(__name__)

router = APIRouter(tags=[Tags.persons])


def _format_ts(val) -> Optional[float]:
    """Return a timestamp as a float (epoch seconds), handling both float and datetime."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if hasattr(val, "timestamp"):
        return val.timestamp()
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _entity_to_dict(e: PersonEntity) -> dict[str, Any]:
    return {
        "id": e.id,
        "known_name": e.known_name,
        "first_seen": _format_ts(e.first_seen),
        "last_seen": _format_ts(e.last_seen),
        "observation_count": e.observation_count,
        "cameras_seen": e.cameras_seen or [],
        "confidence": e.confidence,
        "is_known": e.is_known,
    }


def _observation_to_dict(o: PersonObservation) -> dict[str, Any]:
    return {
        "id": o.id,
        "event_id": o.event_id,
        "camera": o.camera,
        "timestamp": _format_ts(o.timestamp),
        "face_quality_score": o.face_quality_score,
        "snapshot_path": o.snapshot_path,
        "zones": o.zones,
        "path_data": o.path_data,
        "avg_speed": o.avg_speed,
        "velocity_angle": o.velocity_angle,
        "dwell_time": o.dwell_time,
        "person_entity_id": o.person_entity_id,
    }


@router.get(
    "/persons",
    dependencies=[Depends(allow_any_authenticated())],
    summary="List person entities",
    description="Returns a list of person entities with observation counts and last seen.",
)
def list_persons(
    request: Request,
    limit: int = 100,
    offset: int = 0,
):
    query = PersonEntity.select().order_by(PersonEntity.last_seen.desc())
    total = query.count()
    entities = list(query.limit(limit).offset(offset))
    return JSONResponse(
        content={
            "total": total,
            "limit": limit,
            "offset": offset,
            "entities": [_entity_to_dict(e) for e in entities],
        }
    )


@router.get(
    "/persons/{entity_id}",
    dependencies=[Depends(allow_any_authenticated())],
    summary="Get one person entity",
    description="Returns a single person entity and its observation count.",
)
def get_person(request: Request, entity_id: str):
    try:
        entity = PersonEntity.get_by_id(entity_id)
    except DoesNotExist:
        return JSONResponse(
            status_code=404,
            content={"message": "Person entity not found.", "success": False},
        )
    return JSONResponse(content=_entity_to_dict(entity))


@router.get(
    "/persons/{entity_id}/observations",
    dependencies=[Depends(allow_any_authenticated())],
    summary="List observations for an entity",
    description="Returns paginated observations for a person entity.",
)
def list_entity_observations(
    request: Request,
    entity_id: str,
    limit: int = 50,
    offset: int = 0,
):
    try:
        PersonEntity.get_by_id(entity_id)
    except DoesNotExist:
        return JSONResponse(
            status_code=404,
            content={"message": "Person entity not found.", "success": False},
        )
    query = (
        PersonObservation.select()
        .where(PersonObservation.person_entity_id == entity_id)
        .order_by(PersonObservation.timestamp.desc())
    )
    total = query.count()
    observations = list(query.limit(limit).offset(offset))
    return JSONResponse(
        content={
            "total": total,
            "limit": limit,
            "offset": offset,
            "observations": [_observation_to_dict(o) for o in observations],
        }
    )


@router.post(
    "/persons/search",
    dependencies=[Depends(allow_any_authenticated())],
    summary="Search person entities by face image",
    description="Upload a face image; returns matching person entities by embedding similarity.",
)
async def search_persons(request: Request, file: UploadFile):
    if request.app.embeddings is None:
        return JSONResponse(
            status_code=503,
            content={
                "message": "Person entity / face search is not available.",
                "success": False,
            },
        )
    if not file.content_type or not file.content_type.startswith("image/"):
        return JSONResponse(
            status_code=400,
            content={"message": "Upload must be an image.", "success": False},
        )
    image_bytes = await file.read()
    if not image_bytes:
        return JSONResponse(
            status_code=400,
            content={"message": "Empty image.", "success": False},
        )
    context: EmbeddingsContext = request.app.embeddings
    results = context.search_person_face(image_bytes)
    # Map observation_id -> (distance); then group by person_entity_id, keep best distance
    entity_best: dict[str, float] = {}
    for obs_id, distance in results:
        obs = PersonObservation.get_or_none(PersonObservation.id == obs_id)
        if obs is None or obs.person_entity_id is None:
            continue
        eid = obs.person_entity_id
        if eid not in entity_best or distance < entity_best[eid]:
            entity_best[eid] = distance
    # Build list of entities with best distance (lower = more similar for cosine distance)
    out = []
    for eid, dist in sorted(entity_best.items(), key=lambda x: x[1])[:20]:
        try:
            entity = PersonEntity.get_by_id(eid)
            out.append({**_entity_to_dict(entity), "distance": dist})
        except DoesNotExist:
            pass
    return JSONResponse(content={"matches": out})


@router.put(
    "/persons/{entity_id}/name",
    dependencies=[Depends(require_role(["admin"]))],
    summary="Set or update known name",
    description="Assign or update the known name for a person entity.",
)
def set_person_name(
    request: Request,
    entity_id: str,
    body: PersonNameBody = Body(...),
):
    try:
        entity = PersonEntity.get_by_id(entity_id)
    except DoesNotExist:
        return JSONResponse(
            status_code=404,
            content={"message": "Person entity not found.", "success": False},
        )
    name = body.name
    entity.known_name = name.strip() if name and name.strip() else None
    entity.is_known = entity.known_name is not None
    entity.save()
    return JSONResponse(content={**_entity_to_dict(entity), "success": True})


@router.post(
    "/persons/merge",
    dependencies=[Depends(require_role(["admin"]))],
    summary="Merge two person entities",
    description="Merge the second entity into the first; all observations are reassigned.",
)
def merge_persons(request: Request, body: PersonMergeBody = Body(...)):
    source_entity_id = body.source_entity_id
    target_entity_id = body.target_entity_id
    if source_entity_id == target_entity_id:
        return JSONResponse(
            status_code=400,
            content={"message": "Source and target must differ.", "success": False},
        )
    try:
        source = PersonEntity.get_by_id(source_entity_id)
        target = PersonEntity.get_by_id(target_entity_id)
    except DoesNotExist:
        return JSONResponse(
            status_code=404,
            content={"message": "Person entity not found.", "success": False},
        )
    updated = (
        PersonObservation.update(person_entity_id=source_entity_id).where(
            PersonObservation.person_entity_id == target_entity_id
        )
    ).execute()
    # Recompute source stats (first_seen, last_seen, observation_count, cameras_seen)
    obs = (
        PersonObservation.select()
        .where(PersonObservation.person_entity_id == source_entity_id)
        .order_by(PersonObservation.timestamp)
    )
    timestamps = [o.timestamp for o in obs]
    cameras = list({o.camera for o in obs})
    source.first_seen = min(timestamps) if timestamps else source.first_seen
    source.last_seen = max(timestamps) if timestamps else source.last_seen
    source.observation_count = len(timestamps)
    source.cameras_seen = cameras
    if target.known_name and not source.known_name:
        source.known_name = target.known_name
        source.is_known = True
    source.save()
    target.delete_instance()
    return JSONResponse(
        content={
            "success": True,
            "merged_into": source_entity_id,
            "observations_reassigned": updated,
        }
    )


@router.post(
    "/persons/split",
    dependencies=[Depends(require_role(["admin"]))],
    summary="Split a person entity",
    description="Move the given observations out of the entity (they become unclustered).",
)
def split_person(request: Request, body: PersonSplitBody = Body(...)):
    entity_id = body.entity_id
    observation_ids = body.observation_ids
    if not observation_ids:
        return JSONResponse(
            status_code=400,
            content={"message": "observation_ids required.", "success": False},
        )
    try:
        entity = PersonEntity.get_by_id(entity_id)
    except DoesNotExist:
        return JSONResponse(
            status_code=404,
            content={"message": "Person entity not found.", "success": False},
        )
    updated = (
        PersonObservation.update(person_entity_id=None).where(
            PersonObservation.person_entity_id == entity_id,
            PersonObservation.id.in_(observation_ids),
        )
    ).execute()
    # Recompute entity stats
    remaining = list(
        PersonObservation.select().where(
            PersonObservation.person_entity_id == entity_id
        )
    )
    if not remaining:
        entity.delete_instance()
        return JSONResponse(
            content={
                "success": True,
                "observations_unlinked": updated,
                "entity_removed": True,
            }
        )
    timestamps = [o.timestamp for o in remaining]
    entity.first_seen = min(timestamps)
    entity.last_seen = max(timestamps)
    entity.observation_count = len(remaining)
    entity.cameras_seen = list({o.camera for o in remaining})
    entity.save()
    return JSONResponse(
        content={
            "success": True,
            "observations_unlinked": updated,
        }
    )
