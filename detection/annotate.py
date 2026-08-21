"""Draws person boxes, status labels, optional helmet boxes, and a summary
overlay directly onto the (BGR) frame with OpenCV.
"""

import cv2

from detection.association import HELMET, NO_HELMET

# BGR colors, one per status - visually distinct but not hardcoded elsewhere.
STATUS_COLORS = {
    HELMET: (0, 170, 0),
    NO_HELMET: (0, 0, 230),
}
STATUS_LABELS = {
    HELMET: "HELMET",
    NO_HELMET: "NO HELMET",
}


def annotate_frame(frame, person_statuses, counts, compliance, draw_helmet_boxes=True):
    annotated = frame.copy()

    for ps in person_statuses:
        color = STATUS_COLORS[ps.status]
        x1, y1, x2, y2 = (int(v) for v in ps.bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label_lines = [f"#{ps.track_id}", STATUS_LABELS[ps.status]]
        if ps.confidence is not None:
            label_lines.append(f"{ps.confidence:.2f}")
        label = " ".join(label_lines)

        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(annotated, (x1, max(0, y1 - text_h - 8)), (x1 + text_w + 6, y1), color, -1)
        cv2.putText(
            annotated, label, (x1 + 3, max(12, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA,
        )

        if draw_helmet_boxes and ps.helmet_bbox is not None:
            hx1, hy1, hx2, hy2 = (int(v) for v in ps.helmet_bbox)
            cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), color, 1)

    compliance_text = f"{compliance * 100:.0f}%" if compliance is not None else "N/A"
    summary_lines = [
        f"People: {counts['people']}",
        f"Helmet: {counts['helmet']}",
        f"No Helmet: {counts['no_helmet']}",
        f"Compliance: {compliance_text}",
    ]
    _draw_summary_box(annotated, summary_lines)
    return annotated


def _draw_summary_box(frame, lines, origin=(10, 10)):
    x, y = origin
    line_h = 22
    box_w = 210
    box_h = line_h * len(lines) + 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)
    for i, line in enumerate(lines):
        cv2.putText(
            frame, line, (x + 10, y + 22 + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
