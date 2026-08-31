"""
Speaker assignment and merging pipeline steps (ADR-008).

Extracted verbatim from the repository-root transcribe.py during the ADR-008
library extraction. Only the mechanical transformations allowed by ADR-008
were applied: print() -> emit("speakers", ...), module-local logger, and
imports resolved to the whycast package modules.
"""

import hashlib
import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, Optional

from whycast._deps import OpenAI, BadRequestError, openai_available
from whycast.config import (
    MAX_TOKENS,
    OPENAI_SPEAKER_MODEL,
    OPENAI_SPEAKER_REASONING_EFFORT,
    PROMPT_SPEAKER_ASSIGN_FILE,
)
from whycast.episodes import SPEAKER_MAP_SUFFIX as _SPEAKER_MAP_SUFFIX
from whycast.errors import SpeakerMappingError
from whycast.events import emit
from whycast.io_utils import atomic_write_text
from whycast.pipeline.llm import process_with_openai, read_prompt_file
from whycast.pipeline.outputs import write_all_format
from transcript_formatter import format_transcript_with_headers

logger = logging.getLogger(__name__)


def merge_speaker_lines(transcript_text: str) -> str:
    """
    Merge consecutive lines that have the same speaker label into paragraphs.
    Removes duplicate speaker labels, keeping only one at the beginning of each merged section.

    Args:
        transcript_text: The original transcript text with speaker labels

    Returns:
        Merged transcript text with speaker paragraphs
    """
    if not transcript_text.strip():
        return transcript_text

    lines = transcript_text.strip().split('\n')
    merged_lines = []
    current_speaker = None
    current_content = []
    current_use_brackets = True  # Track the format being used

    # Pattern to match speaker labels - both bracketed [Speaker] and non-bracketed Speaker:
    speaker_pattern_bracketed = re.compile(r'^\[([^\]]+)\]\s*(.*)$')
    speaker_pattern_colon = re.compile(r'^([^:]+):\s*(.*)$')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Try bracketed format first [Speaker]
        match = speaker_pattern_bracketed.match(line)
        if match:
            speaker_label = match.group(1)
            content = match.group(2).strip()
            use_brackets = True
        else:
            # Try colon format Speaker:
            match = speaker_pattern_colon.match(line)
            if match:
                speaker_label = match.group(1).strip()
                content = match.group(2).strip()
                use_brackets = False
            else:
                match = None

        if match:
            # If this is the same speaker as the previous line, accumulate content
            if speaker_label == current_speaker and current_content:
                if content:  # Only add non-empty content
                    current_content.append(content)
            else:
                # Different speaker or first line - output previous speaker's content
                if current_speaker and current_content:
                    merged_content = ' '.join(current_content)
                    if current_use_brackets:
                        merged_lines.append(f"[{current_speaker}] {merged_content}")
                    else:
                        merged_lines.append(f"{current_speaker}: {merged_content}")

                # Start new speaker section
                current_speaker = speaker_label
                current_content = [content] if content else []
                current_use_brackets = use_brackets
        else:
            # Line without speaker label - treat as continuation of current speaker
            if current_speaker and line:
                current_content.append(line)
            elif line:
                # Line without speaker and no current speaker - add as is
                merged_lines.append(line)

    # Don't forget the last speaker's content
    if current_speaker and current_content:
        merged_content = ' '.join(current_content)
        if current_use_brackets:
            merged_lines.append(f"[{current_speaker}] {merged_content}")
        else:
            merged_lines.append(f"{current_speaker}: {merged_content}")

    return '\n\n'.join(merged_lines)

def write_merged_transcript(transcript_text: str, base_path: str) -> str:
    """
    Create a merged transcript file where consecutive lines from the same speaker are combined.

    Args:
        transcript_text: The original transcript text
        base_path: Base path for output file (without extension)

    Returns:
        Path to the merged transcript file
    """
    try:
        merged_text = merge_speaker_lines(transcript_text)
        merged_file = f"{base_path}_merged.txt"

        # backup=False at every artifact call site (the io_utils default
        # stays True): a re-run would leave a .bak beside every artifact. The
        # atomic replace already guarantees no partial file; the backup only
        # kept the previous version.
        #
        # Worth being blunt about, because this file invites hand editing more
        # than any other: <base>_merged.txt is a GENERATED ARTIFACT. Any full
        # run rewrites it, there is no .bak (by decision, ADR-009) and
        # podcasts/ is gitignored, so an edit made here is gone the next run.
        # It is also the first file webui/runner.py:_read_transcript picks up
        # as the input for a cheap re-run, which makes a hand edit *look* like
        # it stuck - the speaker and postprocess jobs honour it right up until
        # the next full run overwrites it. Corrections belong in a pipeline
        # INPUT (the speaker mapping, vocabulary.json, a prompt), never here.
        atomic_write_text(merged_file, merged_text, backup=False)

        logging.info(f"Merged transcript written to: {merged_file}")
        emit("speakers", f"✅ Merged transcript saved: {os.path.basename(merged_file)}")

        return merged_file
    except Exception as e:
        logging.error(f"Error writing merged transcript: {str(e)}")
        emit("speakers", f"❌ Error creating merged transcript: {str(e)}")
        return ""

def analyze_speakers_with_o4(transcript: str, output_basename: str = None, output_dir: str = None) -> Optional[Dict[str, str]]:
    """
    Use o4 model to analyze transcript and create detailed speaker mapping.

    Args:
        transcript: The original transcript text with SPEAKER_XX labels
        output_basename: Base name for output files
        output_dir: Directory for output files

    Returns:
        Dictionary mapping SPEAKER_XX to final labels, or None if failed
    """
    if not openai_available:
        logging.warning("OpenAI not available, skipping speaker analysis")
        return None

    logging.info("Running detailed speaker analysis with o4 model")
    emit("speakers", "🧠 Analyzing speakers with o4 reasoning model...")

    try:
        # Read the speaker analysis prompt
        analysis_prompt_file = os.path.join(os.path.dirname(PROMPT_SPEAKER_ASSIGN_FILE), "speaker_analysis_prompt.txt")
        analysis_prompt = read_prompt_file(analysis_prompt_file)

        if not analysis_prompt:
            logging.error("Speaker analysis prompt not found")
            return None

        # Speaker work is the one step on its own model and its own reasoning
        # budget (ADR-003): it is judgement over the whole transcript, where
        # the summary steps are compression.
        analysis_result = process_with_openai(transcript, analysis_prompt, OPENAI_SPEAKER_MODEL,
                                              max_tokens=MAX_TOKENS * 2,
                                              reasoning_effort=OPENAI_SPEAKER_REASONING_EFFORT)

        if not analysis_result:
            logging.error("Speaker analysis failed")
            return None

        # Parse the mapping from the analysis result
        speaker_mapping = parse_speaker_mapping_from_analysis(analysis_result)

        # Save detailed analysis to file
        if output_basename and output_dir:
            analysis_file = os.path.join(output_dir, f"{output_basename}_speaker_analysis.txt")
            try:
                # Same pieces, same order, same newlines as the old sequence of
                # f.write() calls - joined up front so the whole report lands
                # in one atomic replace (ADR-008, whycast/io_utils.py).
                report = [
                    "DETAILED SPEAKER ANALYSIS REPORT\n",
                    "=" * 50 + "\n\n",
                    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
                    f"Model: {OPENAI_SPEAKER_MODEL}\n",
                    f"Transcript: {output_basename}\n\n",
                    analysis_result,
                    f"\n\n{'='*50}\n",
                    "EXTRACTED MAPPING:\n",
                ]
                for original, mapped in speaker_mapping.items():
                    report.append(f"{original} → {mapped}\n")
                atomic_write_text(analysis_file, "".join(report), backup=False)

                logging.info(f"Speaker analysis saved to: {analysis_file}")
                emit("speakers", f"📄 Speaker analysis report saved: {os.path.basename(analysis_file)}")

            except Exception as e:
                logging.error(f"Error saving speaker analysis: {str(e)}")

        # Log the mapping
        logging.info(f"Speaker mapping extracted: {speaker_mapping}")
        emit("speakers", f"🎭 Speaker mapping created:")
        for original, mapped in speaker_mapping.items():
            emit("speakers", f"   {original} → {mapped}")

        return speaker_mapping

    except Exception as e:
        logging.error(f"Error in speaker analysis: {str(e)}")
        emit("speakers", f"❌ Speaker analysis failed: {str(e)}")
        return None

#: Longest string still plausibly a person's name or a role ("Dave Borghuis",
#: "Host"). Anything longer is the model explaining itself rather than
#: answering, and apply_speaker_mapping_programmatically would paste it into
#: the transcript as if someone were called that.
MAX_SPEAKER_LABEL_CHARS = 40

#: The heading that introduces the model's decision. Matched case-insensitively
#: and without requiring a colon: o4-mini writes "FINAL MAPPING FOR TRANSCRIPT:"
#: and gpt-5.6-sol writes "## FINAL MAPPING FOR TRANSCRIPT".
_MAPPING_HEADING = re.compile(r"(?:FINAL|SPEAKER)\s+MAPPING", re.IGNORECASE)

#: One decision. The label may be bracketed or not, and the arrow may be ASCII
#: or unicode. SPEAKER_UNKNOWN matches too: the old pattern demanded
#: SPEAKER_\d+, which silently dropped it.
#:
#: A colon is deliberately NOT a separator here. Both models end their answer
#: with a "CONFIDENCE SUMMARY" listing "SPEAKER_00: high", and reading those as
#: decisions overwrites the names with confidence words.
_MAPPING_LINE = re.compile(
    r"^\s*\[?(SPEAKER_[A-Z0-9_]+)\]?\s*(?:->|→)\s*(\S.*?)\s*$"
)

#: The same decision, but allowed to sit mid-sentence. Only used by the last
#: resort, where there is no block to anchor to.
_MAPPING_ANYWHERE = re.compile(
    r"\[?(SPEAKER_[A-Z0-9_]+)\]?\s*(?:->|→)\s*(\S.*?)\s*$"
)

#: An all-caps line ending in a colon starts a new section ("CONFIDENCE
#: SUMMARY:", "EXTRACTED MAPPING:") and therefore ends the mapping block.
_SECTION_HEADER = re.compile(r"^[A-Z][A-Z0-9 _-]*:$")

#: Rules, code fences and other decoration between the heading and the entries.
_DECORATION = re.compile(r"^[\s=*_`~-]*$|^```")


def _plausible_speaker_label(raw: str, speaker: str) -> Optional[str]:
    """Clean one parsed label, or return None when it is not a name at all.

    The model is asked for a name and usually gives one, but it also writes
    prose around its answer. The previous last-resort pattern scanned that
    prose, and on one run handed back a 120-character sentence as the name of
    SPEAKER_01. Silently accepting that is worse than dropping it: the label
    goes straight into the transcript.
    """
    label = raw.strip().strip("`").strip()
    label = re.sub(r"^\*\*(.*)\*\*$", r"\1", label).strip()
    label = label.strip("[]").strip()
    label = re.sub(r"[\s.,;:]+$", "", label).strip()

    if not label:
        return None
    if len(label) > MAX_SPEAKER_LABEL_CHARS:
        logging.warning(
            "Ignoring the label for %s: %d characters is prose, not a name (%r)",
            speaker, len(label), label[:60],
        )
        return None
    if re.search(r"[.!?]\s", label):
        logging.warning("Ignoring the label for %s: reads as a sentence (%r)", speaker, label)
        return None
    return label


def _mapping_from_final_block(analysis_text: str) -> Dict[str, str]:
    """Read the decisions under the LAST mapping heading.

    Only that block is authoritative. Everything before it is the model
    reasoning out loud, where a line such as "SPEAKER_03 -> could be almost
    anyone here" is commentary, not a decision.
    """
    headings = list(_MAPPING_HEADING.finditer(analysis_text))
    if not headings:
        return {}

    mapping: Dict[str, str] = {}
    for line in analysis_text[headings[-1].end():].split("\n"):
        stripped = line.strip()
        if not stripped or _DECORATION.match(stripped):
            continue
        # A new markdown section closes the block, but only once something has
        # been read: the heading itself is often followed by one.
        if mapping and (stripped.startswith("#") or _SECTION_HEADER.match(stripped)):
            break
        match = _MAPPING_LINE.match(stripped)
        if not match:
            continue
        speaker, raw = match.group(1), match.group(2)
        label = _plausible_speaker_label(raw, speaker)
        if label:
            mapping[f"[{speaker}]"] = label
    return mapping


def _mapping_from_speaker_sections(analysis_text: str) -> Dict[str, str]:
    """Fallback: per-speaker sections ending in "- Final Label:".

    o4-mini's house style, and the only shape the parser read reliably before,
    so it stays as a second chance for answers with no closing block.
    """
    mapping: Dict[str, str] = {}
    current = None
    for line in analysis_text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("SPEAKER_") and stripped.endswith(":"):
            current = stripped[:-1]
        elif current and stripped.startswith("- Final Label:"):
            label = _plausible_speaker_label(stripped[len("- Final Label:"):], current)
            if label:
                mapping[f"[{current}]"] = label
            current = None
    return mapping


def _mapping_from_anywhere(analysis_text: str) -> Dict[str, str]:
    """Last resort: arrow lines anywhere in the answer.

    Some answers name the speakers without ever writing a closing block. This
    used to be an unguarded regex over the whole document, which is how a
    sentence of commentary once became a speaker's name; every candidate now
    has to look like a name before it is accepted.
    """
    mapping: Dict[str, str] = {}
    for line in analysis_text.split("\n"):
        match = _MAPPING_ANYWHERE.search(line.strip())
        if not match:
            continue
        label = _plausible_speaker_label(match.group(2), match.group(1))
        if label:
            mapping[f"[{match.group(1)}]"] = label
    return mapping


def parse_speaker_mapping_from_analysis(analysis_text: str) -> Dict[str, str]:
    """Read the speaker mapping out of the analysis model's answer.

    ADR-004 has the model decide who each ``SPEAKER_xx`` is and has code apply
    that decision. This function is the seam between the two, and it has to
    survive two models writing the same answer in different shapes: o4-mini in
    per-speaker sections, gpt-5.6-sol in a fenced block under a markdown
    heading.

    Args:
        analysis_text: The analysis model's full answer.

    Returns:
        ``{"[SPEAKER_00]": "Nancy", ...}``, empty when nothing could be read.
        ``SPEAKER_UNKNOWN`` is included when the model labelled it: it carries
        real segments - 45 of 155 in episode 1 - and is not a marker to skip.
    """
    if not analysis_text:
        return {}

    try:
        mapping = _mapping_from_final_block(analysis_text)
        if not mapping:
            mapping = _mapping_from_speaker_sections(analysis_text)
        if not mapping:
            mapping = _mapping_from_anywhere(analysis_text)
        if mapping:
            logging.info(f"Parsed speaker mapping: {mapping}")
        else:
            logging.warning("No speaker mapping could be read from the analysis")
        return mapping
    except Exception as e:
        logging.error(f"Error parsing speaker mapping: {str(e)}")
        return {}


# ---------------------------------------------------------------------------
# The persisted speaker mapping (ADR-010)
# ---------------------------------------------------------------------------
#
# Until ADR-010 the mapping only ever lived in memory: analyze_speakers_with_o4
# produced it, speaker_assignment_step spent it, and the only trace left on disk
# was <base>_speaker_analysis.txt - prose, written for a human to read, which
# nothing parses back. So every re-run paid a reasoning model to rediscover
# names that were already right, and a person who saw "Nancy" where it should
# say "Ad" had nowhere to put the correction: editing the generated transcript
# is editing a file the next run overwrites, without even a .bak since ADR-009.
#
# <base>_speakers.json is that missing place. It is a pipeline INPUT: the
# pipeline reads it, a person may edit it, and the pipeline only writes it when
# it does not exist yet (or when the editor saves one). Hence backup=True on the
# write - the one call site in this package that keeps a .bak, because what it
# overwrites may be somebody's typing rather than a regenerable artifact.

#: Filename suffix of the persisted mapping, next to the episode's other files.
#: Re-exported, not redefined: :mod:`whycast.episodes` has to recognise exactly
#: this name while scanning ``podcasts/`` (ADR-010 wants the mapping listed as a
#: kind of its own, not as an unmatched stray), so the literal lives there and
#: the reader, the writer and the scanner cannot drift apart on a rename.
SPEAKER_MAP_SUFFIX = _SPEAKER_MAP_SUFFIX

#: Values the ``source`` field may hold: who decided these names.
SPEAKER_MAP_SOURCES = ("llm", "human")


def speaker_map_path(output_basename: str, output_dir: str) -> str:
    """Path of the persisted mapping for one episode.

    Args:
        output_basename: Episode base name, e.g. ``episode_13``.
        output_dir: Directory the episode's files live in.

    Returns:
        ``<output_dir>/<output_basename>_speakers.json``.
    """
    return os.path.join(output_dir, f"{output_basename}{SPEAKER_MAP_SUFFIX}")


def fingerprint_transcript(transcript: str) -> str:
    """SHA-256 of the transcript a mapping was derived from.

    Cheap and deterministic: the same text always gives the same hex digest, and
    a re-transcription - different words, different segment boundaries, and
    quite possibly different ``SPEAKER_xx`` cluster ids for the same people -
    gives a different one. That difference is the whole staleness signal.

    The only normalisation is line endings. ``atomic_write_text`` writes with
    ``newline=None``, so a transcript that came back from disk on Windows
    carries CRLF while the same text held in memory carries LF; hashing them
    differently would flag a mapping stale for having survived a file round
    trip. Nothing else is touched - not whitespace, not case - because every
    other difference is a real difference in the transcript.
    """
    normalised = (transcript or "").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def mapping_is_stale(record: Dict[str, Any], transcript_fingerprint: Optional[str]) -> bool:
    """Was this mapping made for a different transcript than the one at hand?

    Args:
        record: A loaded mapping file, in its record form.
        transcript_fingerprint: :func:`fingerprint_transcript` of the transcript
            about to be processed, or None when the caller has no transcript to
            compare against (the editor listing the names, say).

    Returns:
        True only when both fingerprints are known and they differ.

    A missing fingerprint on either side means "no opinion", never "stale". The
    file a person writes by hand is exactly ``{"SPEAKER_00": "Nancy"}`` with no
    fingerprint in it, and refusing to apply that would break the one case the
    format exists to support.
    """
    if not transcript_fingerprint:
        return False
    saved = (record or {}).get("transcript_fingerprint")
    if not saved:
        return False
    return saved != transcript_fingerprint


def load_speaker_map_record(
    output_basename: str,
    output_dir: str,
    transcript_fingerprint: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Read ``<base>_speakers.json``, whole, as a normalised record.

    The record form is what the file holds and what the editor wants to see::

        {"speakers": {"SPEAKER_00": "Nancy"}, "source": "llm",
         "transcript_fingerprint": "<sha256>", "updated_at": "<ISO 8601>"}

    A file that is just ``{"SPEAKER_00": "Nancy"}`` loads too, and comes back
    wrapped in the same shape with ``source`` "human": a person who writes the
    obvious thing by hand should not have to know about the envelope, and if
    they wrote it by hand, a human is precisely where it came from.

    Args:
        output_basename: Episode base name.
        output_dir: Directory the episode's files live in.
        transcript_fingerprint: Fingerprint of the transcript being processed.
            Pass it to have staleness checked; leave it out to read the file as
            it stands.

    Returns:
        The record, or None when there is no mapping file (or nowhere to look
        for one, when either argument is missing).

    Raises:
        SpeakerMappingError: The file exists but does not parse, holds something
            that is not a mapping, or was made for a different transcript. Never
            a silent fall back to the paid analysis - see the class docstring.
    """
    if not output_basename or not output_dir:
        return None

    path = speaker_map_path(output_basename, output_dir)
    if not os.path.isfile(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
    except json.JSONDecodeError as e:
        raise _malformed_mapping(path, f"it is not valid JSON ({e})") from e
    except UnicodeDecodeError as e:
        raise _malformed_mapping(path, f"it is not valid UTF-8 ({e})") from e
    except OSError as e:
        raise _malformed_mapping(path, f"it could not be read ({e})") from e

    record = _record_from_document(document, path)

    if mapping_is_stale(record, transcript_fingerprint):
        # Ordered by what the reader has to DO, not by what is most interesting
        # to explain. webui.runner._short_error caps this at 500 characters for
        # the job's error column, so whatever sits last is what a person loses:
        # the two routes out therefore come before the reasoning, and the
        # reasoning - which is the part that survives fine in the ADR and in
        # events.jsonl - comes last on purpose. Naming the remedy as "remove the
        # field" rather than "paste this 64-character hash" is the same economy:
        # shorter to read, shorter to type, and it cannot be mistyped.
        raise SpeakerMappingError(
            f"The speaker mapping in {path} was made for a different "
            f"transcript. Nothing has been deleted. To keep these names, "
            f"remove the \"transcript_fingerprint\" field from that file; to "
            f"let the model work them out again, delete the file. It records "
            f"{record['transcript_fingerprint'][:12]}, this transcript hashes "
            f"to {transcript_fingerprint[:12]}. Diarization labels are per-run "
            f"cluster ids, not identities, so a re-transcription can leave "
            f"SPEAKER_00 pointing at somebody else."
        )

    return record


def load_speaker_mapping(
    output_basename: str,
    output_dir: str,
    transcript_fingerprint: Optional[str] = None,
) -> Optional[Dict[str, str]]:
    """The saved mapping for one episode, or None when there is none.

    The names-only view of :func:`load_speaker_map_record`, for callers that do
    not care where the names came from. Keys come back bare - ``SPEAKER_00``,
    not ``[SPEAKER_00]`` - which is what the file format says and what
    :func:`apply_speaker_mapping_programmatically` accepts either way.

    Returns the mapping only when the file exists, parses, and is not stale for
    ``transcript_fingerprint``; the other outcomes raise (see the record
    loader), because a mapping that cannot be trusted must be visible, not
    quietly replaced by a paid model call.
    """
    record = load_speaker_map_record(
        output_basename, output_dir, transcript_fingerprint=transcript_fingerprint
    )
    return record["speakers"] if record else None


def save_speaker_mapping(
    mapping: Dict[str, str],
    output_basename: str,
    output_dir: str,
    transcript_fingerprint: Optional[str] = None,
    source: str = "llm",
) -> str:
    """Write ``<base>_speakers.json`` atomically, keeping a backup.

    Args:
        mapping: ``{"SPEAKER_00": "Nancy"}``. Bracketed keys are accepted -
            everything :func:`parse_speaker_mapping_from_analysis` produces is
            bracketed - and stored bare, so the file always looks the way the
            format documents it.
        output_basename: Episode base name.
        output_dir: Directory the episode's files live in.
        transcript_fingerprint: :func:`fingerprint_transcript` of the transcript
            these names were read off. Recorded so a later re-transcription can
            be recognised as making them stale. None writes no fingerprint,
            which means "applies to any transcript" - fine for names typed by a
            person who knows the episode.
        source: ``"llm"`` when a model produced this, ``"human"`` when a person
            did. Only tells the reader who to trust; nothing branches on it.

    Returns:
        The path written.

    Raises:
        SpeakerMappingError: The mapping is empty or not a mapping of strings -
            refusing to write a file we would refuse to read back.
        OSError: The write failed (see :func:`whycast.io_utils.atomic_write_text`).

    ``backup=True`` here, alone in this package. ADR-009 dropped the ``.bak``
    beside every generated artifact because a re-run doubled the file count for
    copies of things the pipeline can simply make again. This file is not that:
    it is human input, and the run that overwrites it may be overwriting an
    edit somebody typed. One kept copy is the difference between an undo and an
    apology.
    """
    speakers = _clean_mapping(
        mapping, speaker_map_path(output_basename, output_dir), "it is empty"
    )
    if source not in SPEAKER_MAP_SOURCES:
        raise SpeakerMappingError(
            f"Unknown speaker mapping source {source!r}; expected one of "
            f"{', '.join(SPEAKER_MAP_SOURCES)}."
        )

    record: Dict[str, Any] = {
        "speakers": dict(sorted(speakers.items())),
        "source": source,
        "transcript_fingerprint": transcript_fingerprint,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }

    path = speaker_map_path(output_basename, output_dir)
    # indent + a trailing newline because a person opens this in an editor, and
    # ensure_ascii=False because the names here are Dutch as often as not and
    # "Björn" helps nobody read their own correction back.
    atomic_write_text(
        path,
        json.dumps(record, indent=2, ensure_ascii=False) + "\n",
        backup=True,
    )
    logging.info("Speaker mapping saved to %s (source=%s)", path, source)
    return path


def _record_from_document(document: Any, path: str) -> Dict[str, Any]:
    """Normalise whatever the file held into the record form.

    Two accepted shapes, told apart by one rule: a top-level ``speakers`` key
    holding an object means the full record; anything else means the object
    *is* the mapping. A ``speakers`` key holding something that is not an
    object is a mistake, not a speaker named "speakers" - real labels are
    ``SPEAKER_xx`` - so it is reported rather than guessed at.
    """
    if not isinstance(document, dict):
        raise _malformed_mapping(
            path,
            f"the top level is a {type(document).__name__}, not a JSON object of "
            f"speaker labels",
        )

    if "speakers" in document:
        speakers = document["speakers"]
        if not isinstance(speakers, dict):
            raise _malformed_mapping(
                path,
                f'"speakers" holds a {type(speakers).__name__} instead of an '
                f"object of speaker labels",
            )
        record = dict(document)
        source = record.get("source")
        if source not in SPEAKER_MAP_SOURCES:
            # An unreadable source is not worth failing a run over: the names
            # are still there, and "who typed this" is provenance, not data.
            source = "human" if source is None else str(source)
    else:
        # The bare form a person writes by hand. If they typed it, a human is
        # where it came from, and it carries no fingerprint - which means it is
        # never stale, on purpose (see mapping_is_stale).
        record = {}
        speakers = document
        source = "human"

    record["speakers"] = _clean_mapping(
        speakers, path, "it maps no speakers at all"
    )
    record["source"] = source
    record.setdefault("transcript_fingerprint", None)
    record.setdefault("updated_at", None)

    # The one field the staleness check reads, so the one field whose type has
    # to be right. A number here would compare unequal to the real fingerprint
    # and then blow up while the message tried to quote it, turning "your file
    # has a typo" into a stack trace.
    saved = record["transcript_fingerprint"]
    if saved is not None and not isinstance(saved, str):
        raise _malformed_mapping(
            path,
            f'"transcript_fingerprint" is a {type(saved).__name__}; it must be '
            f'the sha256 of the transcript as text, or absent to mean "these '
            f'names hold whatever the transcript says"',
        )

    return record


def _clean_mapping(mapping: Any, path: str, empty_reason: str) -> Dict[str, str]:
    """Validate a ``{label: name}`` object and return it with bare keys.

    Brackets are stripped from the keys, so ``[SPEAKER_00]`` - what the analysis
    parser produces, and what a person copying from the analysis report would
    paste - stores and loads as ``SPEAKER_00``.

    An empty mapping raises rather than returning ``{}``. A falsy return would
    be indistinguishable from "no file", which sends the caller straight to the
    paid analysis: exactly the silent fallback ADR-010 forbids. A file that maps
    nobody can only be a mistake, and saying so is cheaper than paying for it.
    """
    if not isinstance(mapping, dict):
        raise _malformed_mapping(
            path, f"it is a {type(mapping).__name__}, not an object of speaker labels"
        )

    cleaned: Dict[str, str] = {}
    for label, name in mapping.items():
        if not isinstance(name, str):
            raise _malformed_mapping(
                path,
                f"the name for {label!r} is a {type(name).__name__}; every name "
                f"must be text",
            )
        key = str(label).strip()
        if key.startswith("[") and key.endswith("]"):
            key = key[1:-1].strip()
        if not key:
            raise _malformed_mapping(path, "it contains an empty speaker label")
        # A name that is only whitespace used to be stored as "", which is a
        # non-empty dict and so passed the "it is empty" guard below - and then
        # apply_speaker_mapping_programmatically replaced [SPEAKER_00] with a
        # bare ":" and the line lost its speaker. The module's stated principle
        # is refusing to write a file we would refuse to read back; this is one
        # we were happy to do both with.
        stripped = name.strip()
        if not stripped:
            raise _malformed_mapping(
                path,
                f"the name for {key!r} is blank; every label listed must be "
                f"given a name, and a label you do not want to rename should "
                f"be left out of the file entirely",
            )
        cleaned[key] = stripped

    if not cleaned:
        raise _malformed_mapping(path, empty_reason)
    return cleaned


def _malformed_mapping(path: str, reason: str) -> SpeakerMappingError:
    """The one wording for "this file exists but cannot be used"."""
    return SpeakerMappingError(
        f"The speaker mapping in {path} cannot be used: {reason}. It is a "
        f"hand-editable JSON file; the simplest form that works is "
        f'{{"SPEAKER_00": "Nancy", "SPEAKER_01": "Ad"}}. Fix it, or delete it to '
        f"let the model work the names out again. It is deliberately not being "
        f"replaced by a model call, because that would hide the mistake and "
        f"charge you for the privilege."
    )


def analyze_transcript_changes(original: str, processed: str) -> Dict[str, Any]:
    """Analyze changes between original and processed transcript."""

    orig_stats = {
        'words': len(original.split()),
        'lines': original.count('\n'),
        'chars': len(original),
        'speakers': len(re.findall(r'\[SPEAKER_\d+\]', original))
    }

    proc_stats = {
        'words': len(processed.split()),
        'lines': processed.count('\n'),
        'chars': len(processed),
        'speakers': len(re.findall(r'\[[^\]]+\]', processed))
    }

    return {
        'word_retention': (proc_stats['words'] / orig_stats['words']) * 100 if orig_stats['words'] > 0 else 100,
        'line_retention': (proc_stats['lines'] / orig_stats['lines']) * 100 if orig_stats['lines'] > 0 else 100,
        'char_retention': (proc_stats['chars'] / orig_stats['chars']) * 100 if orig_stats['chars'] > 0 else 100,
        'speaker_conversion': proc_stats['speakers'] / orig_stats['speakers'] if orig_stats['speakers'] > 0 else 0,
        'orig_stats': orig_stats,
        'proc_stats': proc_stats
    }

def apply_speaker_mapping_programmatically(transcript: str, speaker_mapping: Dict[str, str]) -> str:
    """
    Apply speaker mapping programmatically without using AI.

    Args:
        transcript: Original transcript with [SPEAKER_XX] tags
        speaker_mapping: Dictionary mapping [SPEAKER_XX] to clean labels

    Returns:
        Transcript with speaker labels replaced

    The name is substituted *literally*. It arrives from a hand-edited JSON
    file, so it is arbitrary text, and ``re.sub`` reads its replacement
    argument as a template: a name containing ``\\1`` raises "invalid group
    reference" and kills the run after the GPU has already been paid for,
    while the far likelier ``C:\\temp`` expands ``\\t`` into a real tab and
    corrupts the transcript without saying anything at all. Passing a function
    makes the replacement a value instead of a program. The default argument is
    what binds the current name rather than the loop variable.
    """
    if not transcript or not speaker_mapping:
        return transcript

    result = transcript

    # Sort mappings by key length (longest first) to avoid partial replacements
    sorted_mappings = sorted(speaker_mapping.items(), key=lambda x: len(x[0]), reverse=True)

    for speaker_tag, clean_label in sorted_mappings:
        # Ensure speaker_tag has brackets if not already present
        if not speaker_tag.startswith('['):
            speaker_tag = f"[{speaker_tag}]"

        # Create the replacement pattern
        # Replace [SPEAKER_XX] with "CleanLabel:"
        pattern = re.escape(speaker_tag)
        replacement = f"{clean_label}:"

        # Replace all occurrences
        result = re.sub(pattern, lambda _match, text=replacement: text, result)

    return result

def attribute_unknown_speakers(
    transcript: str,
    speaker_mapping: Dict[str, str] = None,
    output_basename: str = None,
    output_dir: str = None,
) -> str:
    """Give ``[SPEAKER_UNKNOWN]`` segments back to the speaker around them.

    Diarization marks a segment ``SPEAKER_UNKNOWN`` when it cannot place the
    voice in a cluster - a crosstalk moment, a cough, a two-word interjection.
    One rule recovers most of them: if the speaker *before* the gap and the
    speaker *after* it are the same person, nobody took the floor, and the
    segment belongs to them. Anything less certain is left alone and collapses
    to a generic ``Speaker:`` in :func:`handle_unknown_speakers`.

    Args:
        transcript: Transcript with ``[SPEAKER_UNKNOWN]`` segments.
        speaker_mapping: Unused; kept so the signature still matches the
            callers that pass it positionally.
        output_basename: Unused; see ``speaker_mapping``.
        output_dir: Unused; see ``speaker_mapping``.

    Returns:
        The transcript with confidently-attributable segments reassigned and
        the rest handed to :func:`handle_unknown_speakers`.

    This used to send every batch of five segments to ``gpt-4o`` before
    applying that same rule. The model's answer was written to
    ``<base>_unknown_attribution_batch_N.txt`` and then *never read*: the
    attributions came from ``previous_speaker == next_speaker`` either way, in
    the ``try`` branch and in the ``except`` branch alike. Measured on a real
    episode the output was byte-for-byte identical with the API answering and
    with the API failing - 12 paid calls, ~25k prompt tokens, zero effect. It
    fired on 100% of this corpus (every transcript here has 33-117 unknown
    segments), including on the saved-mapping path, which the same run
    announces as "Speaker analysis skipped - no model call needed". Spending
    money to contradict your own progress message is worse than spending it
    for nothing.

    So the rule stands on its own, which is also what ADR-004 asks for:
    judgement produces a mapping, deterministic code applies it. Attributing a
    gap from the speakers on either side of it is application, not judgement.

    Worth knowing, and left exactly as it was: when
    :func:`speaker_assignment_programmatic` calls this, the mapping has
    *already* been applied, so the neighbouring lines read ``Nancy:`` rather
    than ``[SPEAKER_00]`` and the scan below finds no previous or next speaker
    at all. Every gap therefore falls through to the generic ``Speaker:``. That
    is the pre-existing behaviour and is not changed here - it is also the
    mechanism that made the paid call measurably pointless, since neither the
    ``try`` branch nor the ``except`` branch could attribute anything. The rule
    does its work when this is called on a transcript that still carries
    labels, which is how ``transcribe.py`` uses it.
    """
    lines = transcript.split('\n')
    unknown_segments = []

    for i, line in enumerate(lines):
        if '[SPEAKER_UNKNOWN]' not in line:
            continue
        segment = {
            'line_number': i + 1,
            'previous_speaker': None,
            'next_speaker': None,
            'line_index': i,
        }

        # Find previous speaker
        for j in range(i - 1, -1, -1):
            if '[SPEAKER_' in lines[j] and '[SPEAKER_UNKNOWN]' not in lines[j]:
                segment['previous_speaker'] = lines[j].split(']')[0] + ']'
                break

        # Find next speaker
        for j in range(i + 1, len(lines)):
            if '[SPEAKER_' in lines[j] and '[SPEAKER_UNKNOWN]' not in lines[j]:
                segment['next_speaker'] = lines[j].split(']')[0] + ']'
                break

        unknown_segments.append(segment)

    if not unknown_segments:
        emit("speakers", "✅ No SPEAKER_UNKNOWN segments found")
        return transcript

    emit(
        "speakers",
        f"📊 Attributing {len(unknown_segments)} SPEAKER_UNKNOWN segments from "
        f"their neighbours (no model call needed)",
    )

    attributions = {}
    for seg in unknown_segments:
        previous, following = seg['previous_speaker'], seg['next_speaker']
        if previous and previous == following:
            attributions[seg['line_index']] = previous

    if attributions:
        modified_lines = lines.copy()
        for line_idx, new_speaker in attributions.items():
            modified_lines[line_idx] = modified_lines[line_idx].replace(
                '[SPEAKER_UNKNOWN]', new_speaker
            )
        transcript = '\n'.join(modified_lines)
        emit(
            "speakers",
            f"✅ Applied {len(attributions)} SPEAKER_UNKNOWN attributions "
            f"(same speaker before and after)",
        )

    remaining_unknown = transcript.count('[SPEAKER_UNKNOWN]')
    if remaining_unknown > 0:
        emit(
            "speakers",
            f"📋 {remaining_unknown} SPEAKER_UNKNOWN segments stay unattributed "
            f"- the floor changed hands across them",
        )
        transcript = handle_unknown_speakers(transcript)

    return transcript


#: The name this function carried while it made a paid call that changed
#: nothing. Kept because ``transcribe.py`` imports it by name and that module
#: is the legacy entry point the golden tests still compare against. The
#: "_with_ai" half is now a lie, which is rather the point: there is nothing
#: left here for a model to do.
attribute_unknown_speakers_with_ai = attribute_unknown_speakers


#: A *numbered* diarization label, the only kind a mapping is expected to name.
#: ``[SPEAKER_UNKNOWN]`` is deliberately excluded: it means diarization could
#: not place the voice, not that the mapping forgot somebody.
_NUMBERED_LABEL_RE = re.compile(r'\[SPEAKER_\d+\]')


def transcript_speaker_labels(transcript: str) -> set:
    """The numbered speaker labels a transcript actually contains, bare.

    ``"[SPEAKER_00] hi"`` gives ``{"SPEAKER_00"}``. This is the set a saved
    mapping is checked against, so that a mapping which can do nothing at all
    is caught before it is applied rather than after somebody reads the result.
    """
    return {tag[1:-1] for tag in _NUMBERED_LABEL_RE.findall(transcript or "")}


def _report_unmapped_labels(assigned: str) -> None:
    """Say whether any numbered label survived the mapping. ADR-010's signal."""
    remaining = sorted(set(_NUMBERED_LABEL_RE.findall(assigned)))
    if remaining:
        logging.warning(f"Unmapped speakers found: {remaining}")
        emit(
            "speakers",
            f"⚠️ Unmapped speakers: {', '.join(remaining)} - these keep a "
            f"generic label. If the mapping was made for an earlier "
            f"transcript, the names it does apply may be on the wrong lines.",
            level="warning",
        )
    else:
        emit("speakers", "✅ All speaker tags successfully mapped")


def _check_mapping_covers_transcript(
    speaker_mapping: Dict[str, str],
    transcript: str,
    output_basename: str,
    output_dir: str,
) -> None:
    """Refuse a saved mapping that names none of this transcript's speakers.

    Args:
        speaker_mapping: The names loaded from ``<base>_speakers.json``.
        transcript: The transcript about to be assigned.
        output_basename: Episode base name, for naming the file in the message.
        output_dir: Directory the episode's files live in.

    Raises:
        SpeakerMappingError: The transcript has numbered labels and the mapping
            names not one of them.

    The realistic mistake is a dropped zero: somebody types ``SPEAKER_1`` where
    the transcript says ``SPEAKER_01``. Nothing downstream notices. The mapping
    takes precedence, the paid analysis is skipped, every replacement misses,
    and :func:`handle_unknown_speakers` turns the real labels into ``Speaker
    00:`` - a run that reports success and produces a transcript strictly worse
    than the one the model would have made, for a file the person believed was
    their correction.

    Total miss raises; a *partial* mapping only warns, in
    :func:`_report_unmapped_labels`. Naming one speaker out of three is a
    supported edit - the web UI's blank-name field is how you say "leave this
    one alone" - whereas naming none of them cannot be anything but a typo.
    Raising matches what ADR-010 already does for a stale file and a malformed
    one: *report it, so a typo in a hand-edited file is visible rather than
    expensively papered over.*

    A transcript carrying no numbered labels at all (only ``SPEAKER_UNKNOWN``)
    is not this mistake and is left to the warning path: there is nothing for
    any mapping to match, so the file tells us nothing about itself.
    """
    present = transcript_speaker_labels(transcript)
    if not present:
        return
    named = {
        label[1:-1].strip() if label.startswith("[") and label.endswith("]") else label
        for label in (str(key).strip() for key in speaker_mapping)
    }
    if named & present:
        return

    path = speaker_map_path(output_basename, output_dir) if output_dir else "the mapping"
    raise SpeakerMappingError(
        f"The speaker mapping in {path} names none of the speakers in this "
        f"transcript, so applying it would leave every one of them unnamed. "
        f"Nothing has been deleted. It names {', '.join(sorted(named)) or 'nothing'}; "
        f"this transcript has {', '.join(sorted(present))}. Check for a typo - "
        f"SPEAKER_1 and SPEAKER_01 are different labels - or delete the file to "
        f"let the model work the names out again."
    )


def handle_unknown_speakers(transcript: str) -> str:
    """
    Handle any remaining [SPEAKER_XX] or [SPEAKER_UNKNOWN] tags that weren't mapped.

    Args:
        transcript: Transcript that may still have unmapped speaker tags

    Returns:
        Transcript with unknown speakers replaced with generic labels
    """
    # Handle [SPEAKER_UNKNOWN] tags
    transcript = re.sub(r'\[SPEAKER_UNKNOWN\]', 'Speaker:', transcript)

    # Handle any remaining [SPEAKER_XX] tags with numbers
    def replace_numbered_speaker(match):
        speaker_num = match.group(1)
        return f"Speaker {speaker_num}:"

    transcript = re.sub(r'\[SPEAKER_(\d+)\]', replace_numbered_speaker, transcript)

    # Handle any other bracketed speaker tags
    transcript = re.sub(r'\[SPEAKER_[^\]]*\]', 'Speaker:', transcript)

    return transcript

def speaker_assignment_programmatic(
    transcript: str,
    speaker_mapping: Dict[str, str],
    output_basename: str = None,
    output_dir: str = None
) -> Optional[str]:
    """
    Apply speaker mapping programmatically (replaces the AI-based assignment).

    Args:
        transcript: Original transcript with [SPEAKER_XX] tags
        speaker_mapping: Dictionary mapping speaker tags to clean labels
        output_basename: Base name for output files
        output_dir: Directory for output files

    Returns:
        Transcript with speaker labels replaced
    """
    try:
        # Pre-processing statistics
        input_lines = transcript.count('\n')
        input_words = len(transcript.split())
        input_chars = len(transcript)

        logging.info(f"Programmatic speaker assignment input: {input_lines} lines, {input_words} words, {input_chars} chars")
        emit("speakers", f"🔄 Applying speaker mapping programmatically...")
        emit("speakers", f"📊 Input: {input_words} words, {input_lines} lines")

        # Apply the speaker mapping
        result = apply_speaker_mapping_programmatically(transcript, speaker_mapping)

        # Verify the replacement worked. This has to happen HERE - after the
        # mapping and before anything sweeps up - because the two steps below
        # both rewrite every remaining [SPEAKER_*] tag. The check used to sit
        # after them, where re.findall could never match anything, so the run
        # unconditionally reported "All speaker tags successfully mapped" no
        # matter how wrong the mapping was. ADR-010 leans on this exact warning
        # as its stale-mapping signal; a signal that cannot fire is worse than
        # no signal, because it reads as an all-clear.
        #
        # Numbered labels only. [SPEAKER_UNKNOWN] is diarization saying "I could
        # not place this voice", not a name the mapping failed to supply, and it
        # appears in every transcript in this corpus - warning on it would bury
        # the real signal in noise on 100% of runs.
        _report_unmapped_labels(result)

        # Attribute SPEAKER_UNKNOWN segments from their neighbours, then
        # collapse whatever is left to generic labels.
        result = attribute_unknown_speakers(result, speaker_mapping, output_basename, output_dir)

        # Handle any remaining unknown/unmapped speakers with fallback
        result = handle_unknown_speakers(result)

        # Post-processing statistics
        output_lines = result.count('\n')
        output_words = len(result.split())
        output_chars = len(result)

        # Calculate retention (should be nearly 100% for programmatic replacement)
        word_retention = (output_words / input_words) * 100 if input_words > 0 else 100
        line_retention = (output_lines / input_lines) * 100 if input_lines > 0 else 100
        char_retention = (output_chars / input_chars) * 100 if input_chars > 0 else 100

        logging.info(f"Programmatic speaker assignment output: {output_lines} lines, {output_words} words, {output_chars} chars")
        logging.info(f"Retention rates: {word_retention:.1f}% words, {line_retention:.1f}% lines, {char_retention:.1f}% chars")

        emit("speakers", f"✅ Programmatic assignment complete!")
        emit("speakers", f"📈 Retention: {word_retention:.1f}% words, {line_retention:.1f}% lines")

        # Add header and footer formatting
        if output_basename:
            try:
                formatted_result = format_transcript_with_headers(result, output_basename)

                # Save files if output directory provided
                if output_dir:
                    write_all_format(formatted_result, f"{output_basename}_speaker_assignment", output_dir)
                    emit("speakers", f"✅ Files saved: {output_basename}_speaker_assignment.*")

                return formatted_result
            except Exception as e:
                logging.error(f"Error in formatting: {str(e)}")
                emit("speakers", f"⚠️ Formatting error, returning unformatted result: {str(e)}")
                return result

        return result

    except Exception as e:
        logging.error(f"Error in programmatic speaker assignment: {str(e)}")
        emit("speakers", f"❌ Error in programmatic assignment: {str(e)}")
        return None

def speaker_assignment_step(transcript: str, output_basename: str = None, output_dir: str = None) -> Optional[str]:
    """
    Two-phase speaker assignment, with the saved mapping taking precedence.

    Phase 1 decides which name belongs to which ``SPEAKER_xx`` label; phase 2
    applies that decision to the transcript with plain string replacement
    (ADR-004 - judgement and application stay apart). What ADR-010 adds is that
    phase 1 now has a cheaper first answer: ``<base>_speakers.json``.

    Precedence, in full:

    1. The file is there, parses, and its fingerprint matches (or it has none):
       those are the names. The paid analysis does not run at all.
    2. The file is there but stale - it records a different transcript than this
       one - :class:`SpeakerMappingError`. A re-transcription renumbers the
       diarization clusters, so applying it unchecked would confidently
       misattribute; the person decides whether the names still hold.
    3. No file: the model works the names out as it always did, and the result
       is saved so the next run is free and there is something to correct.
    4. The file is there and malformed: :class:`SpeakerMappingError`. A typo in
       a hand-edited file has to be visible, never papered over by a paid call.

    The OpenAI availability check now guards only branch 3. Branch 1 needs no
    model - that is the point of it - and refusing to apply names a person
    already typed because no API key is configured would be an odd way to
    honour their correction.
    """
    try:
        if not any("SPEAKER_" in line for line in transcript.splitlines()):
            logging.info("No speaker tags found, skipping speaker assignment")
            return None

        logging.info("Starting two-phase speaker assignment")

        # Phase 1a: the saved mapping, if there is one worth having.
        fingerprint = fingerprint_transcript(transcript)
        record = load_speaker_map_record(
            output_basename, output_dir, transcript_fingerprint=fingerprint
        )

        if record:
            speaker_mapping = record["speakers"]
            # Before anything is applied and before the paid call is skipped:
            # a mapping that matches no label in this transcript is a typo, not
            # a decision, and it must not pass for one.
            _check_mapping_covers_transcript(
                speaker_mapping, transcript, output_basename, output_dir
            )
            _announce_saved_mapping(record, output_basename, output_dir, fingerprint)
        else:
            if not openai_available:
                logging.warning("OpenAI not available, skipping speaker assignment")
                return None

            # Phase 1b: no saved mapping - ask the model, then keep the answer.
            _announce_missing_mapping(output_basename, output_dir)
            speaker_mapping = analyze_speakers_with_o4(transcript, output_basename, output_dir)

            if not speaker_mapping:
                logging.warning("Speaker analysis failed, falling back to original method")
                emit("speakers", "⚠️  Advanced speaker analysis failed, using fallback method")

                # Fallback to original method
                return speaker_assignment_fallback(transcript, output_basename, output_dir)

            _remember_speaker_mapping(
                speaker_mapping, output_basename, output_dir, fingerprint
            )

        # Phase 2: Apply the mapping programmatically (NEW - replaces AI call)
        logging.info("Applying speaker mapping programmatically")
        emit("speakers", "🔄 Applying speaker assignments...")

        try:
            assigned_text = speaker_assignment_programmatic(
                transcript,
                speaker_mapping,
                output_basename,
                output_dir
            )

            if not assigned_text:
                logging.warning("Programmatic assignment failed, trying fallback")
                emit("speakers", "⚠️ Programmatic assignment failed, using fallback method")
                return speaker_assignment_fallback(transcript, output_basename, output_dir)

        except Exception as e:
            logging.error(f"Error in programmatic assignment: {str(e)}")
            emit("speakers", f"❌ Error in programmatic assignment: {str(e)}")
            return speaker_assignment_fallback(transcript, output_basename, output_dir)

        # The programmatic assignment already includes content validation and header/footer formatting
        # Return the result directly since it's already been processed completely
        logging.info("Speaker assignment completed successfully")
        emit("speakers", "✅ Speaker assignment completed with headers and footers")

        return assigned_text

    except SpeakerMappingError as e:
        # Deliberately not swallowed into the None below. None means "nothing to
        # assign", which the web UI reports as a job that finished with a shrug;
        # this is a file on disk that needs a person to look at it, and it has
        # to arrive as a failure with the message attached (ADR-008: library
        # code raises, the worker translates).
        logging.error(f"Speaker mapping unusable: {e}")
        emit("speakers", f"❌ {e}", level="error")
        raise

    except Exception as e:
        logging.error(f"Error in speaker assignment: {str(e)}")
        emit("speakers", f"❌ Speaker assignment failed: {str(e)}")
        return None


def _announce_saved_mapping(
    record: Dict[str, Any],
    output_basename: str,
    output_dir: str,
    transcript_fingerprint: Optional[str] = None,
) -> None:
    """Say which names are being used and where they came from.

    Somebody who cannot tell whether their correction was picked up will not
    trust the correction, and will go back to editing the transcript by hand -
    the thing ADR-009 and ADR-010 exist to make unnecessary. So the file is
    named, its provenance is named, and the names themselves are listed exactly
    the way :func:`analyze_speakers_with_o4` lists its own.

    A file with no ``transcript_fingerprint`` gets one extra line. ADR-010's
    Must says a re-transcription marks a mapping stale "rather than deleting
    it. The file records which transcript it was made for" - but the bare
    hand-written form the format exists to support, ``{"SPEAKER_00": "Nancy"}``,
    records nothing, and :func:`mapping_is_stale` reads a missing fingerprint
    as "no opinion", never as stale. That is the right default (refusing it
    would break the one case the bare form is for) and it does mean the file a
    person is most likely to type by hand is the one a re-transcription cannot
    invalidate. So it is said out loud instead: the check was not made, and
    here is the cost of that if the audio was re-transcribed.
    """
    name = os.path.basename(speaker_map_path(output_basename, output_dir))
    origin = (
        "edited by hand" if record.get("source") == "human"
        else "written by the model on an earlier run"
    )
    emit("speakers", f"📌 Using the saved speaker mapping from {name} ({origin})")
    emit("speakers", "   Speaker analysis skipped - no model call needed")
    if transcript_fingerprint and not record.get("transcript_fingerprint"):
        emit(
            "speakers",
            f"   {name} records no transcript fingerprint, so it applies to any "
            f"transcript and cannot be flagged stale. If this episode was "
            f"re-transcribed since the names were typed, check them: "
            f"diarization labels are per-run cluster ids, not identities.",
            level="warning",
        )
    for original, mapped in record["speakers"].items():
        emit("speakers", f"   {original} → {mapped}")


def _announce_missing_mapping(output_basename: str, output_dir: str) -> None:
    """Say that there is no saved mapping, so the model is about to be asked."""
    if output_basename and output_dir:
        name = os.path.basename(speaker_map_path(output_basename, output_dir))
        emit("speakers", f"🔎 No saved speaker mapping ({name}); asking the model")
    else:
        emit("speakers", "🔎 No output directory for a saved mapping; asking the model")


def _remember_speaker_mapping(
    speaker_mapping: Dict[str, str],
    output_basename: str,
    output_dir: str,
    fingerprint: str,
) -> None:
    """Persist a freshly derived mapping so the next run need not pay for it.

    Best effort, on purpose. By the time this runs the expensive part is done
    and the transcript can still be assigned; failing the run because the
    mapping could not be written would throw away a paid call to protect a
    convenience. The failure is reported, not hidden.

    It will not overwrite a mapping that appeared while the analysis ran. This
    is only reached on ADR-010's branch 3 - there was no file when the step
    started - so a file here now means somebody saved one in the meantime, and
    the paid analysis can take tens of seconds while the editor takes one
    click. Overwriting would flip ``source`` from "human" back to "llm" and
    replace a person's typing with the model's guess, unannounced, leaving the
    human version only in the one-deep ``.bak`` that the next such run
    consumes. ADR-010's Must Not covers this as squarely as it covers deletion:
    an unannounced automatic replacement of human work is the same loss with
    better manners. The model's answer is simply dropped; the person's names
    are the ones the *next* run will use, which is what they asked for.
    """
    if not output_basename or not output_dir:
        return
    if os.path.isfile(speaker_map_path(output_basename, output_dir)):
        name = os.path.basename(speaker_map_path(output_basename, output_dir))
        logging.warning(
            "Not overwriting %s: it appeared while the speaker analysis ran", name
        )
        emit(
            "speakers",
            f"📌 {name} was saved while the analysis was running, so the "
            f"model's answer has been discarded rather than overwriting it. "
            f"The saved names are what the next run will use.",
            level="warning",
        )
        return
    try:
        path = save_speaker_mapping(
            speaker_mapping,
            output_basename,
            output_dir,
            transcript_fingerprint=fingerprint,
            source="llm",
        )
    except (OSError, SpeakerMappingError) as e:
        logging.warning(f"Could not save the speaker mapping: {e}")
        emit(
            "speakers",
            f"⚠️ Could not save the speaker mapping for re-use: {e}",
            level="warning",
        )
        return
    emit(
        "speakers",
        f"💾 Speaker mapping saved to {os.path.basename(path)} - correct a name "
        f"there and the next run uses your version instead of asking the model",
    )

def speaker_assignment_fallback(transcript: str, output_basename: str = None, output_dir: str = None) -> Optional[str]:
    """Fallback to original speaker assignment method."""
    logging.info("Using fallback speaker assignment method")
    emit("speakers", "🔄 Using fallback speaker assignment...")

    try:
        # Pre-processing: Count input characteristics
        input_words = len(transcript.split())
        input_lines = transcript.count('\n')
        input_chars = len(transcript)

        logging.info(f"Fallback speaker assignment input: {input_lines} lines, {input_words} words, {input_chars} chars")

        # Read speaker assignment prompt
        prompt = read_prompt_file(PROMPT_SPEAKER_ASSIGN_FILE)
        if not prompt:
            logging.error("Speaker assignment prompt not found")
            return None

        # Add preservation instructions to prompt
        enhanced_prompt = f"""INPUT STATISTICS (for preservation verification):
- Lines: {input_lines}
- Words: {input_words}
- Characters: {input_chars}

Your output should have similar statistics (±10% variance acceptable).

{prompt}

CRITICAL: The output length should be nearly identical to input length. Only change speaker labels, preserve everything else exactly."""

        # Process with OpenAI
        assigned_text = process_with_openai(
            transcript,
            enhanced_prompt,
            OPENAI_SPEAKER_MODEL,
            max_tokens=MAX_TOKENS * 2,
            reasoning_effort=OPENAI_SPEAKER_REASONING_EFFORT
        )

        if not assigned_text:
            logging.error("Fallback speaker assignment failed")
            return None

        # Content preservation check
        changes = analyze_transcript_changes(transcript, assigned_text)

        logging.info(f"Fallback content analysis: {changes['word_retention']:.1f}% retention")
        emit("speakers", f"📈 Fallback retention: {changes['word_retention']:.1f}%")

        # Check for significant content loss
        if changes['word_retention'] < 90.0:
            logging.warning(f"Fallback significant content loss: {changes['word_retention']:.1f}% retention")
            if changes['word_retention'] < 70.0:
                logging.error("Fallback excessive content loss - using original transcript")
                assigned_text = transcript

        return assigned_text

    except Exception as e:
        logging.error(f"Error in fallback speaker assignment: {str(e)}")
        return None
