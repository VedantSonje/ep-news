"""
Enrichment package — adds structured metadata to every announcement.

SectorTagger       : defence / railway / solar / pharma / IT / PSU / PLI …
NoveltyDetector    : is this announcement genuinely new or a repeat?
MaterialityExtractor: order value, relative size, catalyst type
"""
from enrichment.sector_tagger import SectorTagger
from enrichment.novelty_detector import NoveltyDetector
from enrichment.materiality import MaterialityExtractor, MaterialityInfo

__all__ = ["SectorTagger", "NoveltyDetector", "MaterialityExtractor", "MaterialityInfo"]
