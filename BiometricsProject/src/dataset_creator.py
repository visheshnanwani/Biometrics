import cv2
import numpy as np
import os
from pathlib import Path


def create_sample_signatures():
    """Create synthetic sample signatures for testing"""
    output_dir = Path('data/samples')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sample signature templates (simple drawings)
    signatures = {
        'john_doe': [
            [(100, 100), (150, 120), (200, 80), (250, 130), (300, 100)],
            [(100, 150), (120, 180), (140, 160), (160, 190), (180, 170)]
        ],
        'jane_smith': [
            [(80, 120), (130, 100), (180, 140), (230, 110), (280, 130)],
            [(80, 170), (100, 190), (120, 165), (140, 195), (160, 175)]
        ]
    }

    for name, strokes in signatures.items():
        # Create a white canvas
        canvas = np.ones((400, 400, 3), dtype=np.uint8) * 255

        # Draw signature strokes
        for stroke in strokes:
            for i in range(len(stroke) - 1):
                cv2.line(canvas, stroke[i], stroke[i + 1], (0, 0, 0), 3)

        # Save the signature
        output_path = output_dir / f"{name}_signature.png"
        cv2.imwrite(str(output_path), canvas)
        print(f"Created sample signature: {output_path}")


if __name__ == "__main__":
    create_sample_signatures()