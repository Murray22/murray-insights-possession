"""Possession models: FrameTable (+ optional Smart-Tagging-like intervals) -> labels H / A / D / X.

Every model is a function `predict(table, intervals=None, **params) -> np.ndarray of '<U1'` with one
label per table frame. The registry in mtah.modelbench lists the versions that were scored.
"""
