"""Knowledge Base Parser and Indexer for ZentroMart.

Extracts all markdown documents with frontmatter and [SECTION-ID] headings,
preserves status, effective_from dates, document types, and precedence hierarchy.
Also indexes structured catalogs (products, pincodes) for entity-augmented RAG.
"""
import re
import csv
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path
from src.config import KB_DIR, DATA_DIR, DOC_PRECEDENCE


@dataclass
class KBChunk:
    chunk_id: str
    section_id: str
    doc_id: str
    doc_title: str
    doc_type: str
    version: str
    effective_from: str
    status: str
    heading: str
    content: str
    source_file: str
    precedence_rank: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def searchable_text(self) -> str:
        return f"[{self.section_id}] {self.doc_title} - {self.heading}\n{self.content}"


@dataclass
class ProductRecord:
    sku: str
    name: str
    category: str
    return_category: str
    price_inr: int
    fulfilled_by: str
    assured: bool
    warranty_months: int

    @property
    def searchable_text(self) -> str:
        assured_str = "ZentroMart Assured" if self.assured else "Standard"
        return (f"Product SKU {self.sku}: {self.name}, Category: {self.category} ({self.return_category}), "
                f"Price: Rs. {self.price_inr}, Fulfilled by: {self.fulfilled_by}, "
                f"{assured_str}, Warranty: {self.warranty_months} months.")


class KnowledgeBase:
    def __init__(self, kb_dir: Optional[Path] = None, data_dir: Optional[Path] = None):
        self.kb_dir = Path(kb_dir) if kb_dir else KB_DIR
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.chunks: List[KBChunk] = []
        self.chunks_by_section_id: Dict[str, List[KBChunk]] = {}
        self.products: Dict[str, ProductRecord] = {}
        self.pincodes: Dict[str, Dict[str, Any]] = {}
        self._load_kb()
        self._load_catalogs()

    def _parse_frontmatter(self, text: str) -> tuple[Dict[str, str], str]:
        """Parse YAML-style frontmatter between --- markers."""
        metadata = {}
        body = text
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                fm_text = parts[1]
                body = parts[2]
                for line in fm_text.strip().split("\n"):
                    if ":" in line:
                        k, v = line.split(":", 1)
                        metadata[k.strip()] = v.strip()
        return metadata, body

    def _load_kb(self):
        """Load and chunk all markdown files in kb/."""
        if not self.kb_dir.exists():
            return

        section_regex = re.compile(r"^##\s+\[([A-Za-z0-9_\-\.]+)\]\s*(.*?)$", re.MULTILINE)

        for filepath in sorted(self.kb_dir.glob("*.md")):
            content = filepath.read_text(encoding="utf-8")
            fm, body = self._parse_frontmatter(content)

            doc_id = fm.get("doc_id", filepath.stem)
            doc_title = fm.get("title", filepath.stem.replace("_", " ").title())
            doc_type = fm.get("doc_type", "general")
            version = fm.get("version", "1.0")
            effective_from = fm.get("effective_from", "2026-01-01")
            status = fm.get("status", "current")
            precedence = DOC_PRECEDENCE.get(doc_type, 99)
            if status == "superseded":
                precedence += 50  # Superseded documents have strictly lower precedence

            # Find all section headers
            matches = list(section_regex.finditer(body))
            if not matches:
                # Document without section IDs, store as single chunk
                chunk = KBChunk(
                    chunk_id=f"{doc_id}-0",
                    section_id=doc_id,
                    doc_id=doc_id,
                    doc_title=doc_title,
                    doc_type=doc_type,
                    version=version,
                    effective_from=effective_from,
                    status=status,
                    heading=doc_title,
                    content=body.strip(),
                    source_file=filepath.name,
                    precedence_rank=precedence,
                    metadata=fm
                )
                self.chunks.append(chunk)
                self.chunks_by_section_id.setdefault(doc_id, []).append(chunk)
                continue

            for i, match in enumerate(matches):
                sec_id = match.group(1).strip()
                heading = match.group(2).strip()
                start_pos = match.end()
                end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(body)
                sec_body = body[start_pos:end_pos].strip()

                chunk = KBChunk(
                    chunk_id=f"{doc_id}-{sec_id}",
                    section_id=sec_id,
                    doc_id=doc_id,
                    doc_title=doc_title,
                    doc_type=doc_type,
                    version=version,
                    effective_from=effective_from,
                    status=status,
                    heading=heading,
                    content=sec_body,
                    source_file=filepath.name,
                    precedence_rank=precedence,
                    metadata=fm
                )
                self.chunks.append(chunk)
                self.chunks_by_section_id.setdefault(sec_id, []).append(chunk)

    def _load_catalogs(self):
        """Load products and pincodes from data/."""
        products_file = self.data_dir / "products.csv"
        if products_file.exists():
            with open(products_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sku = row.get("sku", "").strip()
                    if sku:
                        try:
                            price = int(row.get("price_inr", 0))
                        except ValueError:
                            price = 0
                        try:
                            warranty = int(row.get("warranty_months", 0))
                        except ValueError:
                            warranty = 0
                        self.products[sku] = ProductRecord(
                            sku=sku,
                            name=row.get("name", ""),
                            category=row.get("category", ""),
                            return_category=row.get("return_category", ""),
                            price_inr=price,
                            fulfilled_by=row.get("fulfilled_by", ""),
                            assured=row.get("assured", "False").lower() == "true",
                            warranty_months=warranty
                        )

        pincodes_file = self.data_dir / "pincodes.csv"
        if pincodes_file.exists():
            with open(pincodes_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pin = row.get("pincode", "").strip()
                    if pin:
                        self.pincodes[pin] = {
                            "pincode": pin,
                            "city": row.get("city", ""),
                            "state": row.get("state", ""),
                            "serviceable": row.get("serviceable", "True").lower() == "true",
                            "cod_available": row.get("cod_available", "True").lower() == "true",
                            "express_available": row.get("express_available", "False").lower() == "true",
                        }

    def get_chunk_by_section_id(self, sec_id: str) -> Optional[KBChunk]:
        """Get best chunk for a section ID (prioritizing current status)."""
        chunks = self.chunks_by_section_id.get(sec_id)
        if not chunks:
            return None
        # Return chunk with lowest precedence rank (highest priority)
        return sorted(chunks, key=lambda c: c.precedence_rank)[0]
