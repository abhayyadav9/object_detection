def build_response(detections, persons, activities, raw_text, meaning):
    return {
        "success": True,
        "detections": detections,
        "human_activities": activities,
        "human_count": len(persons),
        "ocr_text": raw_text,
        "ocr_meaning": meaning
    }
