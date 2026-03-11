"""Store person observations with face embeddings for identity clustering."""

import datetime
import logging
import os
from typing import Any, Optional

import cv2
import numpy as np

from frigate.config import FrigateConfig
from frigate.const import MODEL_CACHE_DIR
from frigate.data_processing.common.face.model import (
    ArcFaceRecognizer,
    FaceNetRecognizer,
    FaceRecognizer,
)
from frigate.models import PersonObservation
from frigate.util.builtin import EventsPerSecond, InferenceSpeed, serialize, to_relative_box
from frigate.util.image import area

from ..types import DataProcessorMetrics
from .api import RealTimeProcessorApi

logger = logging.getLogger(__name__)

MAX_DETECTION_HEIGHT = 1080


class PersonEntityProcessor(RealTimeProcessorApi):
    """Stores face embeddings and observation metadata per person detection."""

    def __init__(
        self,
        config: FrigateConfig,
        metrics: DataProcessorMetrics,
        db,
    ):
        super().__init__(config, metrics)
        self.db = db
        self.pe_config = config.person_entity
        self.requires_face_detection = "face" not in self.config.objects.all_objects
        self.face_detector: Optional[cv2.FaceDetectorYN] = None
        self.recognizer: Optional[FaceRecognizer] = None
        self.event_observation_count: dict[str, int] = {}
        self.person_entity_speed = InferenceSpeed(self.metrics.person_entity_speed)
        self.person_entity_fps = EventsPerSecond()
        self.person_entity_fps.start()

        if self.requires_face_detection:
            self._init_face_detector()

        if self.pe_config.embedding_model.value == "facenet":
            self.recognizer = FaceNetRecognizer(self.config)
        else:
            self.recognizer = ArcFaceRecognizer(self.config)

    def _init_face_detector(self) -> None:
        download_path = os.path.join(MODEL_CACHE_DIR, "facedet")
        model_files = {
            "facedet.onnx": os.environ.get("GITHUB_ENDPOINT", "https://github.com")
            + "/NickM-27/facenet-onnx/releases/download/v1.0/facedet.onnx",
        }
        if not all(
            os.path.exists(os.path.join(download_path, n)) for n in model_files
        ):
            from frigate.util.downloader import ModelDownloader

            downloader = ModelDownloader(
                model_name="facedet",
                download_path=download_path,
                file_names=list(model_files.keys()),
                download_func=lambda p: ModelDownloader.download_from_url(
                    model_files[os.path.basename(p)], p
                ),
                complete_func=self._build_face_detector,
            )
            downloader.ensure_model_files()
        else:
            self._build_face_detector()

    def _build_face_detector(self) -> None:
        path = os.path.join(MODEL_CACHE_DIR, "facedet/facedet.onnx")
        if os.path.exists(path):
            self.face_detector = cv2.FaceDetectorYN.create(
                path, "", (320, 320), 0.5, 0.3
            )

    def _detect_face(
        self, input_img: np.ndarray, threshold: float
    ) -> Optional[tuple[int, int, int, int]]:
        if self.face_detector is None:
            return None
        if input_img.shape[0] > MAX_DETECTION_HEIGHT:
            scale = MAX_DETECTION_HEIGHT / input_img.shape[0]
            new_w = int(scale * input_img.shape[1])
            input_img = cv2.resize(input_img, (new_w, MAX_DETECTION_HEIGHT))
            scale_factor = scale
        else:
            scale_factor = 1.0
        self.face_detector.setInputSize((input_img.shape[1], input_img.shape[0]))
        _, faces = self.face_detector.detect(input_img)
        if faces is None:
            return None
        best = None
        for f in faces:
            if f[-1] < threshold:
                continue
            x, y, w, h = f[0:4].astype(int)
            x = int(max(0, x) / scale_factor)
            y = int(max(0, y) / scale_factor)
            w = int(w / scale_factor)
            h = int(h / scale_factor)
            bbox = (x, y, x + w, y + h)
            if best is None or area(bbox) > area(best):
                best = bbox
        return best

    def _get_face_frame(
        self, obj_data: dict[str, Any], frame: np.ndarray, camera: str
    ) -> Optional[tuple[np.ndarray, tuple[int, int, int, int], tuple]]:
        """Return (face_frame_bgr, face_bbox_pixels, face_bbox_relative) or None."""
        person_box = obj_data.get("box")
        if not person_box:
            return None
        detect_config = self.config.cameras[camera].detect
        width, height = detect_config.width, detect_config.height

        if self.requires_face_detection:
            rgb = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB_I420)
            left, top, right, bottom = person_box
            person = rgb[top:bottom, left:right]
            face_box = self._detect_face(
                person, self.config.face_recognition.detection_threshold
            )
            if not face_box:
                return None
            face_frame = person[
                max(0, face_box[1]) : min(person.shape[0], face_box[3]),
                max(0, face_box[0]) : min(person.shape[1], face_box[2]),
            ]
            if area(face_box) < self.config.cameras[camera].face_recognition.min_area:
                return None
            face_frame = cv2.cvtColor(face_frame, cv2.COLOR_RGB2BGR)
            face_bbox_rel = to_relative_box(
                person.shape[1], person.shape[0],
                (face_box[0], face_box[1], face_box[2], face_box[3]),
            )
            return face_frame, face_box, face_bbox_rel
        else:
            attrs = obj_data.get("current_attributes") or []
            face_attr = None
            for a in attrs:
                if a.get("label") != "face":
                    continue
                if face_attr is None or a.get("score", 0) > face_attr.get("score", 0):
                    face_attr = a
            if not face_attr or not face_attr.get("box"):
                return None
            face_box = tuple(face_attr["box"])
            if area(face_box) < self.config.cameras[camera].face_recognition.min_area:
                return None
            bgr = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
            face_frame = bgr[
                max(0, face_box[1]) : min(frame.shape[0], face_box[3]),
                max(0, face_box[0]) : min(frame.shape[1], face_box[2]),
            ]
            face_bbox_rel = to_relative_box(width, height, face_box)
            return face_frame, face_box, face_bbox_rel

    def process_frame(self, obj_data: dict[str, Any], frame: np.ndarray) -> None:
        if obj_data.get("label") != "person":
            return
        camera = obj_data.get("camera")
        if not camera or camera not in self.config.cameras:
            return
        if not self.config.cameras[camera].enabled:
            return
        if not self.pe_config.enabled:
            return

        event_id = obj_data.get("id")
        if not event_id:
            return

        count = self.event_observation_count.get(event_id, 0)
        if count >= self.pe_config.max_observations_per_event:
            return

        face_result = self._get_face_frame(obj_data, frame, camera)
        if face_result is None:
            return
        face_frame, face_bbox_px, face_bbox_rel = face_result

        if self.recognizer is None:
            return
        start = datetime.datetime.now().timestamp()
        result = self.recognizer.get_embedding(face_frame)
        self.person_entity_fps.update()
        self.person_entity_speed.update(datetime.datetime.now().timestamp() - start)
        if result is None:
            return
        embedding, blur_reduction = result
        face_quality = max(0.0, 1.0 - blur_reduction)
        if face_quality < self.pe_config.min_face_quality:
            return

        obs_id = event_id if count == 0 else f"{event_id}_{count}"
        detect_config = self.config.cameras[camera].detect
        w, h = detect_config.width, detect_config.height
        person_box = obj_data.get("box")
        bbox_rel = (
            to_relative_box(w, h, tuple(person_box)) if person_box else None
        )
        zones = obj_data.get("entered_zones")
        path_data = obj_data.get("path_data")
        avg_speed = obj_data.get("average_estimated_speed")
        velocity_angle = obj_data.get("velocity_angle")
        timestamp = obj_data.get("frame_time") or start

        try:
            PersonObservation.insert(
                {
                    PersonObservation.id: obs_id,
                    PersonObservation.event_id: event_id,
                    PersonObservation.camera: camera,
                    PersonObservation.timestamp: timestamp,
                    PersonObservation.face_embedding: embedding.tobytes(),
                    PersonObservation.face_quality_score: face_quality,
                    PersonObservation.face_bbox: list(face_bbox_rel),
                    PersonObservation.body_embedding: None,
                    PersonObservation.bbox: list(bbox_rel) if bbox_rel else None,
                    PersonObservation.snapshot_path: None,
                    PersonObservation.zones: zones,
                    PersonObservation.path_data: path_data,
                    PersonObservation.avg_speed: avg_speed,
                    PersonObservation.velocity_angle: velocity_angle,
                    PersonObservation.dwell_time: None,
                    PersonObservation.person_entity_id: None,
                }
            ).execute()
        except Exception as e:
            logger.warning("Failed to insert person observation: %s", e)
            return

        # vec_face_observations is 512-dim (ArcFace); skip for FaceNet (128-dim)
        if (
            getattr(self.db, "load_vec_extension", False)
            and self.pe_config.embedding_model.value == "arcface"
            and len(embedding) == 512
        ):
            try:
                self.db.execute_sql(
                    """
                    INSERT OR REPLACE INTO vec_face_observations(id, face_embedding)
                    VALUES(?, ?)
                    """,
                    (obs_id, serialize(embedding)),
                )
            except Exception as e:
                logger.warning("Failed to insert face embedding into vec table: %s", e)

        self.event_observation_count[event_id] = count + 1

    def get_embedding_for_image(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Get face embedding for an image (e.g. uploaded for search). Returns None if no face."""
        img = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if img is None:
            return None
        if self.requires_face_detection:
            face_box = self._detect_face(img, 0.5)
            if not face_box:
                return None
            face_frame = img[
                face_box[1] : face_box[3],
                face_box[0] : face_box[2],
            ]
        else:
            face_frame = img
        if self.recognizer is None:
            return None
        result = self.recognizer.get_embedding(face_frame)
        if result is None:
            return None
        embedding, _ = result
        return embedding

    def handle_request(
        self, topic: str, request_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        return None

    def expire_object(self, object_id: str, camera: str) -> None:
        if object_id in self.event_observation_count:
            del self.event_observation_count[object_id]
