"""
AnnouncementFilter — subject-level keep/drop gate.
No scoring logic; filtering is purely subject-based.
"""
from __future__ import annotations

from models import Announcement
from screener.filter_config import FilterConfig


class AnnouncementFilter:
    """
    Filters announcements based solely on their subject field.
    Returns True only if subject is in keep_subjects and NOT in drop_subjects.
    """

    def __init__(self, config: FilterConfig) -> None:
        self._cfg = config

    def should_keep(self, ann: Announcement) -> bool:
        if ann.subject in self._cfg.drop_subjects:
            return False
        if ann.subject not in self._cfg.keep_subjects:
            return False
        return True
