"""
SectorTagger — assigns one or more sector/theme labels to an announcement.

Labels are used for:
  - Filtering ("show me solar EP stocks")
  - Agent context ("BEL is a defence PSU")
  - Sector-rotation analysis

All matching is regex-based — zero LLM calls, runs in microseconds.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SectorPattern:
    name: str
    pattern: re.Pattern


_SECTORS: list[SectorPattern] = [
    SectorPattern("defence", re.compile(
        r"defence|defense|military|army|navy|air force|"
        r"DRDO|ISRO|missile|warship|ammunition|radar|"
        r"satellite|aerospace|ordnance|GRSE|BEL|HAL|"
        r"BEML|Mazagon|Garden Reach|Premier Explosives|"
        r"Avantel|Data Patterns",
        re.IGNORECASE,
    )),
    SectorPattern("railway", re.compile(
        r"railway|rail|metro|EMU|coach|locomotive|wagon|"
        r"RVNL|DFCC|IRFC|Titagarh|Texmaco|Jupiter Wagons|"
        r"loco|train|track laying|rolling stock|signalling",
        re.IGNORECASE,
    )),
    SectorPattern("solar_renewables", re.compile(
        r"solar|photovoltaic|PV module|wind energy|"
        r"renewable|BESS|battery storage|green hydrogen|"
        r"clean energy|EV charging|Waaree|Premier Energies|"
        r"Acme Solar|Inox Green|Solex|Ujaas Energy",
        re.IGNORECASE,
    )),
    SectorPattern("power", re.compile(
        r"\bpower\b|electricity|transmission|distribution|"
        r"NTPC|NHPC|SJVN|CESC|Torrent Power|Adani Power|"
        r"substation|transformer|Voltamp|Apar Industries|"
        r"KEC|Sterlite Power|POWERGRID",
        re.IGNORECASE,
    )),
    SectorPattern("infrastructure", re.compile(
        r"highway|road|bridge|tunnel|airport|port|"
        r"waterway|irrigation|canal|dam|flyover|"
        r"EPC|construction|L&T|Afcons|NCC|KNR|PNC|"
        r"HG Infra|Ceigall|Dilip Buildcon|Gawar|"
        r"desilting|sewerage|water supply|Jal Jeevan",
        re.IGNORECASE,
    )),
    SectorPattern("pharma_healthcare", re.compile(
        r"pharma|drug|medicine|ANDA|NDA|USFDA|US FDA|"
        r"API|generic|biosimilar|clinical trial|"
        r"WHO prequalification|WHO PQ|hospital|healthcare|"
        r"diagnostic|Lupin|Cipla|Sun Pharma|Aurobindo|"
        r"Divi|Laurus|Caplin Point|Anuh Pharma",
        re.IGNORECASE,
    )),
    SectorPattern("IT_technology", re.compile(
        r"\bIT\b|software|digital|cloud|SaaS|AI\b|"
        r"artificial intelligence|machine learning|"
        r"cybersecurity|data center|Infosys|TCS|Wipro|"
        r"HCL|Coforge|Mphasis|Hexaware|R Systems|"
        r"Newgen|Onward Tech|KPIT|Zensar",
        re.IGNORECASE,
    )),
    SectorPattern("PSU_government", re.compile(
        r"public sector|PSU|CPSE|ministry|government|"
        r"central government|state government|CPWD|"
        r"GAIL|Coal India|BHEL|SAIL|NMDC|NLCINDIA|"
        r"Railtel|MTNL|BEL|HAL|Mazagon|"
        r"tender|bid award|L1|government contract",
        re.IGNORECASE,
    )),
    SectorPattern("PLI_manufacturing", re.compile(
        r"PLI|Production Linked Incentive|"
        r"Make in India|Atmanirbhar|"
        r"semiconductor|electronics manufacturing|"
        r"mobile phone|EV battery|ACC battery|"
        r"white goods|medical device",
        re.IGNORECASE,
    )),
    SectorPattern("capital_goods", re.compile(
        r"capital goods|engineering|industrial|"
        r"machinery|equipment|pump|valve|compressor|"
        r"Bharat Forge|Thermax|Cummins|Kirloskar|"
        r"Shanthi Gears|Elecon|BEML|Texmaco|"
        r"new plant|greenfield|brownfield|capacity expansion|"
        r"commissioning|CAPEX|capex",
        re.IGNORECASE,
    )),
    SectorPattern("real_estate", re.compile(
        r"real estate|realty|housing|apartment|villa|"
        r"residential|commercial property|office space|"
        r"Godrej Properties|Sobha|Prestige|"
        r"DLF|Oberoi|Brigade|Puravankara|Mahindra Life",
        re.IGNORECASE,
    )),
    SectorPattern("banking_finance", re.compile(
        r"bank|NBFC|microfinance|housing finance|"
        r"NPA|credit|loan|AUM|deposit|CASA|"
        r"Manappuram|Muthoot|SBFC|Spandana|Aavas|"
        r"Home First|Poonawalla|AU Bank|RBL",
        re.IGNORECASE,
    )),
    SectorPattern("FMCG_consumer", re.compile(
        r"FMCG|consumer goods|food|beverage|"
        r"personal care|retail|brand|Dabur|Marico|"
        r"Jyothy Labs|Emami|Bajaj Consumer|"
        r"restaurant|QSR|Sapphire Foods|Devyani",
        re.IGNORECASE,
    )),
]


class SectorTagger:
    """
    Assigns sector/theme tags to an announcement by running all
    sector patterns against the combined subject + details + company text.
    Returns a sorted list of matched sector names.
    """

    def tag(self, subject: str, details: str, company: str = "") -> list[str]:
        """Return list of matching sector tags (e.g. ['defence', 'PSU_government'])."""
        text = f"{company} {subject} {details}"
        matched = [
            sp.name
            for sp in _SECTORS
            if sp.pattern.search(text)
        ]
        return sorted(set(matched))

    def tag_str(self, subject: str, details: str, company: str = "") -> str:
        """Comma-separated string of tags for storage."""
        return ",".join(self.tag(subject, details, company))
