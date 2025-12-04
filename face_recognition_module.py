"""Reusable helpers for loading and recognizing faces using `face_recognition`.
"""

import os
from typing import List, Tuple

import cv2
import face_recognition
import numpy as np


def load_known_faces(directory: str) -> Tuple[List[np.ndarray], List[str]]:
    """Load all faces from the given directory.

    Each image file (*.jpg, *.jpeg, *.png) is encoded and returned as a single
    face encoding plus its corresponding name (derived from the filename).
    """

    encodings: List[np.ndarray] = []
    names: List[str] = []

    if not os.path.isdir(directory):
        return encodings, names

    for filename in os.listdir(directory):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(directory, filename)
            image = face_recognition.load_image_file(path)
            face_encs = face_recognition.face_encodings(image)
            if not face_encs:
                continue
            encodings.append(face_encs[0])
            names.append(os.path.splitext(filename)[0])

    return encodings, names


def recognize_faces_in_frame(
    frame_bgr: np.ndarray,
    known_face_encodings: List[np.ndarray],
    known_face_names: List[str],
    process_this_frame: bool = True,
) -> Tuple[List[Tuple[int, int, int, int]], List[str], bool]:
    """Run face recognition on a single BGR frame.

    Returns (face_locations, face_names, next_process_flag).

    - `face_locations` are in the original frame coordinate space.
    - `face_names` are labels matched against the known encodings.
    - `next_process_flag` toggles whether the next frame should be processed
      (to mimic the "every other frame" optimization from the demo).
    """

    face_locations: List[Tuple[int, int, int, int]] = []
    face_names: List[str] = []

    if frame_bgr is None:
        return face_locations, face_names, not process_this_frame

    if process_this_frame:
        small_frame = cv2.resize(frame_bgr, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        rgb_small = np.ascontiguousarray(rgb_small)

        raw_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, raw_locations)

        for encoding, (top, right, bottom, left) in zip(face_encodings, raw_locations):
            matches = face_recognition.compare_faces(known_face_encodings, encoding)
            name = "Unknown"

            if known_face_encodings:
                face_distances = face_recognition.face_distance(
                    known_face_encodings, encoding
                )
                best_match_index = int(np.argmin(face_distances))
                if matches and matches[best_match_index]:
                    name = known_face_names[best_match_index]

            # Scale back up face locations to original frame size
            face_locations.append((top * 4, right * 4, bottom * 4, left * 4))
            face_names.append(name)

    next_flag = not process_this_frame
    return face_locations, face_names, next_flag


__all__ = ["load_known_faces", "recognize_faces_in_frame"]

