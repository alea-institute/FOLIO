#!/usr/bin/env python3
"""
WebProtégé Merge Pipeline for FOLIO

Extracts genuine content changes from GitHub's FOLIO.owl and applies them
to a WebProtégé export, producing a merge-ready file that WebProtégé can ingest.

Uses rdflib for semantic diffing (what changed?), then text operations to apply
changes to the WebProtégé file (preserving its formatting).
"""

import argparse
import logging
import os
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field

import rdflib
from rdflib import RDF, RDFS, OWL, URIRef, Literal
from rdflib.namespace import SKOS

log = logging.getLogger(__name__)

FOLIO_NS = "https://folio.openlegalstandard.org/"

# SKOS label predicates whose *removals* are safe to propagate to WebProtégé.
# These are multi-valued, non-structural labels: dropping one never leaves a
# concept without a name. Deliberately excludes skos:prefLabel (every concept
# needs one; a "removed" prefLabel is almost always a rename handled elsewhere)
# and skos:definition (handled as a 1:1 update). folio:-prefixed values are
# filtered out too — those are WebProtégé's relationship-label representation,
# not genuine content (the inverse of the label-normalization handling above).
REMOVABLE_LABEL_PREDICATES = {
    "altLabel": SKOS.altLabel,
    "hiddenLabel": SKOS.hiddenLabel,
}

# Regex patterns for OWL/XML block extraction
IRI_COMMENT_RE = re.compile(r"^\s*<!-- (https?://\S+) -->\s*$")
CLASS_BLOCK_RE = re.compile(
    r"([ \t]*<!-- (?P<iri>https?://\S+) -->\s*\n"
    r"\s*<owl:Class rdf:about=\"(?P=iri)\">"
    r".*?"
    r"</owl:Class>)",
    re.DOTALL,
)


@dataclass
class SemanticDiff:
    """Categorized differences between GitHub and WebProtégé OWL files."""

    new_classes: list = field(default_factory=list)  # list of IRI strings
    new_alt_labels: dict = field(default_factory=dict)  # IRI -> list of Literal
    label_normalizations: dict = field(default_factory=dict)  # IRI -> (plain_label, folio_label)
    definition_updates: dict = field(default_factory=dict)  # IRI -> new definition Literal
    new_restrictions: dict = field(default_factory=dict)  # IRI -> list of (prop, value) tuples
    new_other_triples: list = field(default_factory=list)  # (s, p, o) triples
    removed_labels: dict = field(default_factory=dict)  # IRI -> list of (pred_local, Literal) to delete
    removals: list = field(default_factory=list)  # (s, p, o) triples removed in GH (logged, not applied)


def load_webprotege_base(path: str) -> str:
    """Load WebProtégé base file from .zip or .owl path. Returns file content as string."""
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            owl_files = [
                n for n in zf.namelist()
                if n.endswith(".owl") and "REVISION-HEAD" in n
            ]
            if not owl_files:
                # Fall back to any .owl file in the zip
                owl_files = [n for n in zf.namelist() if n.endswith(".owl")]
            if not owl_files:
                raise FileNotFoundError(f"No .owl file found in {path}")
            owl_file = owl_files[0]
            log.info("Extracting %s from zip", owl_file)
            return zf.read(owl_file).decode("utf-8")
    else:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()


def _parse_graph(content_or_path: str, is_path: bool = True) -> rdflib.Graph:
    """Parse OWL/XML into an rdflib Graph."""
    g = rdflib.Graph()
    if is_path:
        g.parse(content_or_path, format="xml")
    else:
        g.parse(data=content_or_path, format="xml")
    return g


def _is_blank_node_triple(s, p, o):
    """Check if a triple involves blank nodes (restriction reordering noise)."""
    return isinstance(s, rdflib.BNode) or isinstance(o, rdflib.BNode)


def _get_classes(g: rdflib.Graph) -> set:
    """Get all named class IRIs from a graph."""
    classes = set()
    for s in g.subjects(RDF.type, OWL.Class):
        if isinstance(s, URIRef):
            classes.add(s)
    return classes


def _get_restrictions_for_class(g: rdflib.Graph, cls: URIRef) -> set:
    """Get restriction (property, value) pairs for a class via rdfs:subClassOf."""
    restrictions = set()
    for obj in g.objects(cls, RDFS.subClassOf):
        if isinstance(obj, rdflib.BNode):
            prop = None
            val = None
            for p2, o2 in g.predicate_objects(obj):
                if p2 == OWL.onProperty:
                    prop = o2
                elif p2 in (OWL.someValuesFrom, OWL.allValuesFrom,
                            OWL.hasValue, OWL.minCardinality,
                            OWL.maxCardinality, OWL.cardinality):
                    val = (p2, o2)
            if prop and val:
                restrictions.add((prop, val[0], val[1]))
    return restrictions


def compute_semantic_diff(gh_path: str, wp_content: str) -> SemanticDiff:
    """Parse both files into rdflib Graphs and compute semantic differences."""
    log.info("Parsing GitHub OWL file...")
    gh_graph = _parse_graph(gh_path, is_path=True)
    log.info("GitHub graph: %d triples", len(gh_graph))

    log.info("Parsing WebProtégé OWL content...")
    wp_graph = _parse_graph(wp_content, is_path=False)
    log.info("WebProtégé graph: %d triples", len(wp_graph))

    diff = SemanticDiff()

    # Find new classes
    gh_classes = _get_classes(gh_graph)
    wp_classes = _get_classes(wp_graph)
    new_class_iris = gh_classes - wp_classes
    diff.new_classes = sorted(str(c) for c in new_class_iris)
    if diff.new_classes:
        log.info("Found %d new classes", len(diff.new_classes))
        for c in diff.new_classes:
            log.info("  New class: %s", c)

    # For existing classes, find new altLabels, definition updates, label normalizations
    common_classes = gh_classes & wp_classes
    for cls in common_classes:
        cls_str = str(cls)

        # Check altLabels
        gh_alt = set(gh_graph.objects(cls, SKOS.altLabel))
        wp_alt = set(wp_graph.objects(cls, SKOS.altLabel))
        new_alt = gh_alt - wp_alt
        if new_alt:
            diff.new_alt_labels[cls_str] = sorted(new_alt, key=str)

        # Check label removals (altLabel/hiddenLabel present in WP but gone in GH).
        # Only treat a label as removed when its *surface string* is entirely
        # absent from GH for this concept — never when it merely moved to another
        # slot (e.g. altLabel→prefLabel) or was re-tagged with a language. Since
        # this script does not re-add prefLabels, deleting a reclassified label
        # would otherwise drop the surface form. folio:-prefixed values are
        # WebProtégé's relationship-label representation, not editorial content.
        gh_label_strings = {
            str(v) for lp in (RDFS.label, SKOS.prefLabel, SKOS.altLabel, SKOS.hiddenLabel)
            for v in gh_graph.objects(cls, lp)
        }
        for pred_local, pred in REMOVABLE_LABEL_PREDICATES.items():
            gh_vals = set(gh_graph.objects(cls, pred))
            wp_vals = set(wp_graph.objects(cls, pred))
            removed = [
                v for v in (wp_vals - gh_vals)
                if isinstance(v, Literal)
                and not str(v).startswith("folio:")
                and str(v) not in gh_label_strings
            ]
            for v in sorted(removed, key=str):
                diff.removed_labels.setdefault(cls_str, []).append((pred_local, v))

        # Check label normalization (folio: prefix moved from rdfs:label to skos:notation)
        gh_labels = set(gh_graph.objects(cls, RDFS.label))
        wp_labels = set(wp_graph.objects(cls, RDFS.label))
        gh_notations = set(gh_graph.objects(cls, SKOS.notation))
        wp_notations = set(wp_graph.objects(cls, SKOS.notation))

        for wp_label in wp_labels:
            label_str = str(wp_label)
            if label_str.startswith("folio:"):
                plain = label_str[len("folio:"):]
                # Check if GH has the plain label and folio: as notation
                plain_lit = Literal(plain)
                folio_lit = Literal(label_str)
                if (plain_lit in gh_labels or Literal(plain) in gh_labels) and \
                   folio_lit in gh_notations:
                    diff.label_normalizations[cls_str] = (plain, label_str)

        # Check definition updates
        gh_defs = set(gh_graph.objects(cls, SKOS.definition))
        wp_defs = set(wp_graph.objects(cls, SKOS.definition))
        new_defs = gh_defs - wp_defs
        removed_defs = wp_defs - gh_defs
        if new_defs and not removed_defs:
            # New definitions added (not replacing existing ones)
            for d in new_defs:
                diff.definition_updates.setdefault(cls_str, []).append(d)
        elif new_defs and removed_defs:
            # Definition changed — only if it's a 1:1 replacement
            if len(gh_defs) == 1 and len(wp_defs) == 1:
                diff.definition_updates[cls_str] = list(gh_defs)

        # Check new restrictions
        gh_restrictions = _get_restrictions_for_class(gh_graph, cls)
        wp_restrictions = _get_restrictions_for_class(wp_graph, cls)
        new_restrictions = gh_restrictions - wp_restrictions
        if new_restrictions:
            diff.new_restrictions[cls_str] = sorted(new_restrictions, key=str)

    # Check for non-class triples that differ (annotations section, etc.)
    # We handle these via label normalization detection above
    # Log removals for awareness
    for s, p, o in wp_graph:
        if _is_blank_node_triple(s, p, o):
            continue
        if (s, p, o) not in gh_graph:
            if isinstance(s, URIRef) and str(s).startswith(FOLIO_NS):
                # Only track removals of folio entities, skip noise
                if p == RDFS.label and str(o).startswith("folio:"):
                    # This is a label normalization, not a removal
                    continue
                # Skip label removals we apply explicitly (tracked separately)
                applied = diff.removed_labels.get(str(s), [])
                if any(pred == str(p).split("#")[-1] and lit == o for pred, lit in applied):
                    continue
                diff.removals.append((str(s), str(p), str(o)))

    return diff


def extract_xml_block(content: str, iri: str) -> str | None:
    """Extract the <!-- IRI --> comment + <owl:Class ...>...</owl:Class> block for an IRI."""
    # Escape IRI for regex
    iri_escaped = re.escape(iri)
    pattern = re.compile(
        r"([ \t]*<!-- " + iri_escaped + r" -->\s*\n"
        r"\s*<owl:Class rdf:about=\"" + iri_escaped + r"\">"
        r".*?"
        r"</owl:Class>)",
        re.DOTALL,
    )
    m = pattern.search(content)
    if m:
        return m.group(1)
    return None


def _find_insertion_point(wp_content: str, new_iri: str) -> int | None:
    """Find the alphabetical insertion point for a new class IRI in the WP file.

    Classes are ordered by IRI in the file. Find the comment block whose IRI
    would immediately follow the new IRI alphabetically, and insert before it.
    """
    # Collect all IRI comments with their positions
    iri_positions = []
    for m in IRI_COMMENT_RE.finditer(wp_content):
        iri_positions.append((m.group(1), m.start()))

    if not iri_positions:
        return None

    # Find the first IRI that would come after the new IRI
    for iri, pos in iri_positions:
        if iri > new_iri:
            # Walk back to find the start of the blank lines before this comment
            # The pattern is typically: \n\n\n    <!-- IRI -->
            search_start = max(0, pos - 10)
            while search_start > 0 and wp_content[search_start - 1] in ("\n", " ", "\t"):
                search_start -= 1
            # Move past the last content character
            if search_start > 0:
                search_start += 1
            return search_start

    # New IRI goes after all existing ones — find the end of the last class block
    # before </rdf:RDF>
    return None


def apply_changes(wp_content: str, diff: SemanticDiff, gh_content: str) -> str:
    """Apply semantic diff changes to WebProtégé content using text surgery."""
    result = wp_content
    changes_applied = 0

    # 1. Insert new classes
    for iri in diff.new_classes:
        block = extract_xml_block(gh_content, iri)
        if not block:
            log.warning("Could not extract block for new class %s from GitHub file", iri)
            continue

        insert_pos = _find_insertion_point(result, iri)
        if insert_pos is None:
            # Insert before </rdf:RDF>
            rdf_end = result.rfind("</rdf:RDF>")
            if rdf_end == -1:
                log.error("Could not find </rdf:RDF> in WebProtégé content")
                continue
            insert_pos = rdf_end

        # Format the block with proper spacing (3 blank lines between blocks)
        formatted_block = f"\n\n\n    {block.strip()}\n    "
        result = result[:insert_pos] + formatted_block + result[insert_pos:]
        changes_applied += 1
        log.info("Inserted new class: %s", iri)

    # 2. Add new altLabels to existing classes
    for iri, labels in diff.new_alt_labels.items():
        # Find the closing </owl:Class> for this IRI's block
        iri_escaped = re.escape(iri)
        block_pattern = re.compile(
            r"(<owl:Class rdf:about=\"" + iri_escaped + r"\">)"
            r"(.*?)"
            r"(</owl:Class>)",
            re.DOTALL,
        )
        m = block_pattern.search(result)
        if not m:
            log.warning("Could not find class block for %s to add altLabels", iri)
            continue

        existing_block = m.group(2)
        close_tag = m.group(3)
        insert_point = m.start(3)

        # Build altLabel lines
        alt_lines = []
        for label in labels:
            lang = label.language
            label_str = _xml_escape(str(label))
            if lang:
                alt_lines.append(
                    f'        <skos:altLabel xml:lang="{lang}">{label_str}</skos:altLabel>'
                )
            else:
                alt_lines.append(f"        <skos:altLabel>{label_str}</skos:altLabel>")

        if alt_lines:
            insert_text = "\n".join(alt_lines) + "\n"
            # Insert before </owl:Class>, preserving indentation
            result = result[:insert_point] + insert_text + "    " + result[insert_point:]
            changes_applied += len(alt_lines)
            log.info("Added %d altLabel(s) to %s", len(alt_lines), iri)

    # 3. Apply label normalizations (folio: prefix → skos:notation)
    for iri, (plain_label, folio_label) in diff.label_normalizations.items():
        iri_escaped = re.escape(iri)
        # Find the block for this IRI
        block_pattern = re.compile(
            r"(<(?:owl:Class|owl:AnnotationProperty|owl:ObjectProperty|owl:DatatypeProperty|owl:NamedIndividual)"
            r" rdf:about=\"" + iri_escaped + r"\">)"
            r"(.*?)"
            r"(</(?:owl:Class|owl:AnnotationProperty|owl:ObjectProperty|owl:DatatypeProperty|owl:NamedIndividual)>)",
            re.DOTALL,
        )
        m = block_pattern.search(result)
        if not m:
            log.warning("Could not find block for %s to normalize label", iri)
            continue

        block_start = m.start()
        block_end = m.end()
        block_text = m.group(0)

        folio_escaped = _xml_escape(folio_label)
        plain_escaped = _xml_escape(plain_label)

        # Replace <rdfs:label>folio:X</rdfs:label> with <rdfs:label>X</rdfs:label>
        old_label = f"<rdfs:label>{folio_escaped}</rdfs:label>"
        new_label = f"<rdfs:label>{plain_escaped}</rdfs:label>"

        if old_label in block_text:
            new_block = block_text.replace(old_label, new_label, 1)

            # Add skos:notation if not already present
            notation_tag = f"<skos:notation>{folio_escaped}</skos:notation>"
            if notation_tag not in new_block:
                # Insert after the rdfs:label line
                label_line_end = new_block.find(new_label) + len(new_label)
                # Find the end of this line
                newline_pos = new_block.find("\n", label_line_end)
                if newline_pos != -1:
                    indent = "        "
                    notation_line = f"\n{indent}{notation_tag}"
                    new_block = new_block[:newline_pos] + notation_line + new_block[newline_pos:]

            result = result[:block_start] + new_block + result[block_end:]
            changes_applied += 1
            log.info("Normalized label for %s: %s → %s + notation", iri, folio_label, plain_label)

    # 4. Update definitions
    for iri, new_defs in diff.definition_updates.items():
        if len(new_defs) != 1:
            log.warning("Skipping definition update for %s: expected 1 def, got %d", iri, len(new_defs))
            continue
        new_def = str(new_defs[0])
        iri_escaped = re.escape(iri)
        block_pattern = re.compile(
            r"(<(?:owl:Class|owl:AnnotationProperty|owl:ObjectProperty|owl:DatatypeProperty|owl:NamedIndividual|rdf:Description)"
            r"(?: rdf:about=\"" + iri_escaped + r"\">))"
            r"(.*?)"
            r"(</(?:owl:Class|owl:AnnotationProperty|owl:ObjectProperty|owl:DatatypeProperty|owl:NamedIndividual|rdf:Description)>)",
            re.DOTALL,
        )
        m = block_pattern.search(result)
        if not m:
            log.warning("Could not find block for %s to update definition", iri)
            continue

        block_text = m.group(0)
        block_start = m.start()
        block_end = m.end()

        # Replace existing definition
        def_pattern = re.compile(
            r"<skos:definition>.*?</skos:definition>",
            re.DOTALL,
        )
        new_def_escaped = _xml_escape(new_def)
        new_def_tag = f"<skos:definition>{new_def_escaped}</skos:definition>"

        if def_pattern.search(block_text):
            new_block = def_pattern.sub(new_def_tag, block_text, count=1)
            result = result[:block_start] + new_block + result[block_end:]
            changes_applied += 1
            log.info("Updated definition for %s", iri)

    # 5. Insert new restrictions
    for iri, restrictions in diff.new_restrictions.items():
        for prop, restriction_type, value in restrictions:
            iri_escaped = re.escape(iri)
            block_pattern = re.compile(
                r"(<owl:Class rdf:about=\"" + iri_escaped + r"\">)"
                r"(.*?)"
                r"(</owl:Class>)",
                re.DOTALL,
            )
            m = block_pattern.search(result)
            if not m:
                log.warning("Could not find class block for %s to add restriction", iri)
                continue

            close_pos = m.start(3)
            restriction_type_local = restriction_type.split("#")[-1] if "#" in str(restriction_type) else str(restriction_type).split("/")[-1]

            restriction_block = (
                "        <rdfs:subClassOf>\n"
                "            <owl:Restriction>\n"
                f'                <owl:onProperty rdf:resource="{prop}"/>\n'
                f'                <owl:{restriction_type_local} rdf:resource="{value}"/>\n'
                "            </owl:Restriction>\n"
                "        </rdfs:subClassOf>\n"
            )

            result = result[:close_pos] + restriction_block + "    " + result[close_pos:]
            changes_applied += 1
            log.info("Added restriction to %s: %s %s %s", iri, prop, restriction_type_local, value)

    # 6. Remove labels deleted in GitHub (e.g. a stray altLabel). Surgical,
    #    line-level removal scoped to the target class block only.
    for iri, items in diff.removed_labels.items():
        iri_escaped = re.escape(iri)
        block_pattern = re.compile(
            r"(<owl:Class rdf:about=\"" + iri_escaped + r"\">)"
            r"(.*?)"
            r"(</owl:Class>)",
            re.DOTALL,
        )
        m = block_pattern.search(result)
        if not m:
            log.warning("Could not find class block for %s to remove labels", iri)
            continue

        block_start, block_end = m.start(), m.end()
        block_text = m.group(0)
        new_block = block_text
        for pred_local, label in items:
            new_block, removed = _remove_label_line(new_block, pred_local, label)
            if removed:
                changes_applied += 1
                log.info("Removed %s '%s' from %s", pred_local, str(label), iri)
            else:
                log.warning(
                    "Label %s '%s' not found in %s block (already absent?)",
                    pred_local, str(label), iri,
                )

        if new_block != block_text:
            result = result[:block_start] + new_block + result[block_end:]

    log.info("Total changes applied: %d", changes_applied)
    return result


def _remove_label_line(block_text: str, pred_local: str, label) -> tuple[str, bool]:
    """Remove a single SKOS label line (e.g. <skos:altLabel>X</skos:altLabel>)
    from a class block, matching its optional xml:lang and the whole physical
    line (leading indentation + trailing newline). Returns (new_block, removed?).

    A plain (no-lang) literal only matches the no-attribute tag, so it will never
    clobber a language-tagged variant of the same string, and vice versa.
    """
    tag = re.escape(f"skos:{pred_local}")
    label_escaped = re.escape(_xml_escape(str(label)))
    lang = label.language if isinstance(label, Literal) else None
    lang_attr = r' xml:lang="' + re.escape(lang) + r'"' if lang else r""
    pattern = re.compile(
        r"[ \t]*<" + tag + lang_attr + r">" + label_escaped + r"</" + tag + r">[ \t]*\r?\n"
    )
    new_block, n = pattern.subn("", block_text, count=1)
    return new_block, n > 0


def _xml_escape(text: str) -> str:
    """Escape special XML characters in text content."""
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace('"', "&quot;")
    text = text.replace("'", "&apos;")
    return text


def _xml_unescape(text: str) -> str:
    """Unescape XML entities back to plain text (for comparison)."""
    text = text.replace("&apos;", "'")
    text = text.replace("&quot;", '"')
    text = text.replace("&gt;", ">")
    text = text.replace("&lt;", "<")
    text = text.replace("&amp;", "&")
    return text


def validate_output(output_content: str, output_path: str) -> bool:
    """Validate the output file for both XML and RDF validity."""
    valid = True

    # XML validation
    log.info("Validating XML structure...")
    try:
        ET.fromstring(output_content.encode("utf-8"))
        log.info("  XML validation: PASSED")
    except ET.ParseError as e:
        log.error("  XML validation: FAILED - %s", e)
        valid = False

    # RDF validation
    log.info("Validating RDF/OWL content...")
    try:
        g = rdflib.Graph()
        g.parse(data=output_content, format="xml")
        triple_count = len(g)
        class_count = len(list(g.subjects(RDF.type, OWL.Class)))
        log.info("  RDF validation: PASSED (%d triples, %d named classes)", triple_count, class_count)
    except Exception as e:
        log.error("  RDF validation: FAILED - %s", e)
        valid = False

    return valid


def print_summary(diff: SemanticDiff, output_path: str):
    """Print a human-readable summary of changes."""
    print("\n" + "=" * 60)
    print("WEBPROTÉGÉ MERGE SUMMARY")
    print("=" * 60)
    print(f"Output file: {output_path}")
    print()

    if diff.new_classes:
        print(f"New classes ({len(diff.new_classes)}):")
        for c in diff.new_classes:
            print(f"  + {c}")
        print()

    if diff.new_alt_labels:
        total_labels = sum(len(v) for v in diff.new_alt_labels.values())
        print(f"New altLabels ({total_labels} across {len(diff.new_alt_labels)} classes):")
        for iri, labels in sorted(diff.new_alt_labels.items()):
            for label in labels:
                print(f"  + {iri.split('/')[-1]}: {label}")
        print()

    if diff.label_normalizations:
        print(f"Label normalizations ({len(diff.label_normalizations)}):")
        for iri, (plain, folio) in sorted(diff.label_normalizations.items()):
            print(f"  ~ {iri.split('/')[-1]}: {folio} → label:{plain} + notation:{folio}")
        print()

    if diff.definition_updates:
        print(f"Definition updates ({len(diff.definition_updates)}):")
        for iri in sorted(diff.definition_updates):
            print(f"  ~ {iri.split('/')[-1]}")
        print()

    if diff.new_restrictions:
        total_r = sum(len(v) for v in diff.new_restrictions.values())
        print(f"New restrictions ({total_r} across {len(diff.new_restrictions)} classes):")
        for iri in sorted(diff.new_restrictions):
            print(f"  + {iri.split('/')[-1]}: {len(diff.new_restrictions[iri])} restriction(s)")
        print()

    if diff.removed_labels:
        total_removed = sum(len(v) for v in diff.removed_labels.values())
        print(f"Label removals applied ({total_removed} across {len(diff.removed_labels)} classes):")
        for iri, items in sorted(diff.removed_labels.items()):
            for pred_local, label in items:
                print(f"  - {iri.split('/')[-1]}: {pred_local} '{label}'")
        print()

    if diff.removals:
        print(f"Other removals detected ({len(diff.removals)}) — NOT applied (review manually):")
        for s, p, o in diff.removals[:10]:
            print(f"  - {s.split('/')[-1]} {p.split('#')[-1] if '#' in p else p.split('/')[-1]} {str(o)[:60]}")
        if len(diff.removals) > 10:
            print(f"  ... and {len(diff.removals) - 10} more")
        print()

    total_changes = (
        len(diff.new_classes)
        + sum(len(v) for v in diff.new_alt_labels.values())
        + len(diff.label_normalizations)
        + len(diff.definition_updates)
        + sum(len(v) for v in diff.new_restrictions.values())
        + sum(len(v) for v in diff.removed_labels.values())
    )
    print(f"Total changes applied: {total_changes}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a WebProtégé-compatible merge file from GitHub's FOLIO.owl"
    )
    parser.add_argument(
        "-g", "--github-owl",
        required=True,
        help="Path to the GitHub FOLIO.owl file",
    )
    parser.add_argument(
        "-w", "--webprotege-input",
        required=True,
        help="Path to WebProtégé export (.zip or .owl)",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output path for the merge-ready .owl file",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Load inputs
    log.info("Loading GitHub OWL: %s", args.github_owl)
    with open(args.github_owl, "r", encoding="utf-8") as f:
        gh_content = f.read()

    log.info("Loading WebProtégé input: %s", args.webprotege_input)
    wp_content = load_webprotege_base(args.webprotege_input)

    # Compute semantic diff
    diff = compute_semantic_diff(args.github_owl, wp_content)

    # Apply changes
    result = apply_changes(wp_content, diff, gh_content)

    # Write output
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(result)
    log.info("Wrote output to %s", args.output)

    # Validate
    valid = validate_output(result, args.output)

    # Summary
    print_summary(diff, args.output)

    if not valid:
        print("\nWARNING: Output file failed validation. Review errors above.")
        sys.exit(1)
    else:
        print("\nOutput file is valid and ready for WebProtégé import.")


if __name__ == "__main__":
    main()
