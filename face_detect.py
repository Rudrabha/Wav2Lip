import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_detection = mp.solutions.face_detection


def face_rect(images):
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    ) as face_detection:
        for image_cv2 in images:
            # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
            results = face_detection.process(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))

            # Draw face detections of each face.
            if not results.detections:
                yield None
            for detection in results.detections:
                yield _get_bounding_rect(image_cv2, detection)


def _get_bounding_rect(
    image: mp_drawing.np.ndarray,
    detection: mp_drawing.detection_pb2.Detection,
):
    """
    Stolen from mediapipe.solutions.drawing_utils.draw_detection()
    """
    if not detection.location_data:
        return
    if image.shape[2] != mp_drawing._BGR_CHANNELS:
        raise ValueError("Input image must contain three channel bgr data.")
    image_rows, image_cols, _ = image.shape

    location = detection.location_data

    # get bounding box if exists.
    if not location.HasField("relative_bounding_box"):
        return
    relative_bounding_box = location.relative_bounding_box
    rect_start_point = mp_drawing._normalized_to_pixel_coordinates(
        relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols, image_rows
    )
    rect_end_point = mp_drawing._normalized_to_pixel_coordinates(
        relative_bounding_box.xmin + relative_bounding_box.width,
        relative_bounding_box.ymin + relative_bounding_box.height,
        image_cols,
        image_rows,
    )

    return *rect_start_point, *rect_end_point

