"""
Speaker assignment and merging pipeline steps (ADR-008).

Extracted verbatim from the repository-root transcribe.py during the ADR-008
library extraction. Only the mechanical transformations allowed by ADR-008
were applied: print() -> emit("speakers", ...), module-local logger, and
imports resolved to the whycast package modules.
"""

import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, Optional

from whycast._deps import OpenAI, BadRequestError, openai_available
from whycast.config import MAX_TOKENS, OPENAI_SPEAKER_MODEL, PROMPT_SPEAKER_ASSIGN_FILE
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

        # Use o4 model for analysis (assuming it's configured in OPENAI_SPEAKER_MODEL)
        analysis_result = process_with_openai(transcript, analysis_prompt, OPENAI_SPEAKER_MODEL, max_tokens=MAX_TOKENS * 2)

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

def parse_speaker_mapping_from_analysis(analysis_text: str) -> Dict[str, str]:
    """
    Parse speaker mapping from the o4 analysis result.

    Args:
        analysis_text: The analysis result from o4 model

    Returns:
        Dictionary mapping SPEAKER_XX to final labels
    """
    mapping = {}

    try:
        # Look for the "FINAL MAPPING FOR TRANSCRIPT:" section
        lines = analysis_text.split('\n')
        in_mapping_section = False

        for line in lines:
            line = line.strip()

            # Check if we've reached the mapping section
            if "FINAL MAPPING FOR TRANSCRIPT:" in line or "SPEAKER MAPPING:" in line:
                in_mapping_section = True
                continue

            # Stop at next major section or examples
            if in_mapping_section and (line.startswith(('##', '===', 'CONFIDENCE SUMMARY:', '**EXAMPLES'))) or line == '':
                if line.startswith(('##', '===', 'CONFIDENCE SUMMARY:', '**EXAMPLES')):
                    break
                continue

            # Parse mapping lines like "SPEAKER_00 → Sarah" or "SPEAKER_00 → Host"
            if in_mapping_section and ('→' in line or '->' in line):
                # Split on either arrow type
                if '→' in line:
                    parts = line.split('→')
                else:
                    parts = line.split('->')

                if len(parts) == 2:
                    original = parts[0].strip()
                    mapped = parts[1].strip()

                    # Clean up the original speaker label
                    if not original.startswith('SPEAKER_'):
                        continue

                    # Keep mapped label simple - it will be used as "Label:" without brackets
                    # Store mapping as [SPEAKER_XX] → Label (no brackets in the mapped value)
                    mapping[f"[{original}]"] = mapped

        # Fallback: look for individual speaker sections
        if not mapping:
            current_speaker = None
            current_label = None

            for line in lines:
                line = line.strip()

                # Look for speaker headers like "SPEAKER_00:"
                if line.startswith('SPEAKER_') and line.endswith(':'):
                    current_speaker = f"[{line[:-1]}]"  # Remove colon, add brackets
                    current_label = None

                # Look for Final Label lines (now expecting simple labels)
                elif current_speaker and line.startswith('- Final Label:'):
                    label_part = line.replace('- Final Label:', '').strip()
                    # Remove any brackets from the response if present
                    if label_part.startswith('[') and label_part.endswith(']'):
                        label_part = label_part[1:-1]
                    current_label = label_part  # No brackets in the mapped value

                    mapping[current_speaker] = current_label
                    current_speaker = None
                    current_label = None

        # If still no mapping found, try simple pattern matching
        if not mapping:
            import re
            # Look for any SPEAKER_XX → Label patterns (without requiring brackets)
            pattern = r'(SPEAKER_\d+)\s*[→\->\s]+\s*([^\n\[]+?)(?:\n|$)'
            matches = re.findall(pattern, analysis_text)
            for original, mapped in matches:
                clean_mapped = mapped.strip()
                # Remove any trailing punctuation or brackets
                clean_mapped = re.sub(r'[\[\]]+$', '', clean_mapped).strip()
                mapping[f"[{original}]"] = clean_mapped  # No brackets in the mapped value

        logging.info(f"Parsed speaker mapping: {mapping}")
        return mapping

    except Exception as e:
        logging.error(f"Error parsing speaker mapping: {str(e)}")
        return {}

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
        result = re.sub(pattern, replacement, result)

    return result

def attribute_unknown_speakers_with_ai(transcript: str, speaker_mapping: Dict[str, str] = None, output_basename: str = None, output_dir: str = None) -> str:
    """
    Use AI to intelligently attribute SPEAKER_UNKNOWN segments to most likely speakers
    based on conversational flow and context analysis.

    Args:
        transcript: Transcript with SPEAKER_UNKNOWN segments to analyze
        speaker_mapping: Current speaker mapping (for reference)
        output_basename: Base name for output files
        output_dir: Directory for output files

    Returns:
        Transcript with SPEAKER_UNKNOWN segments attributed to specific speakers where possible
    """
    import re
    from typing import List, Dict, Any
    from pathlib import Path

    emit("speakers", "\n🤖 Analyzing SPEAKER_UNKNOWN segments with AI...")

    # Extract SPEAKER_UNKNOWN segments with context
    lines = transcript.split('\n')
    unknown_segments = []

    for i, line in enumerate(lines):
        if '[SPEAKER_UNKNOWN]' in line:
            # Get context: previous 3 lines and next 3 lines
            start_idx = max(0, i - 3)
            end_idx = min(len(lines), i + 4)
            context_lines = lines[start_idx:end_idx]

            segment = {
                'line_number': i + 1,
                'content': line.strip(),
                'context': '\n'.join(context_lines),
                'previous_speaker': None,
                'next_speaker': None,
                'line_index': i
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

    emit("speakers", f"📊 Found {len(unknown_segments)} SPEAKER_UNKNOWN segments to analyze")

    # Load the attribution prompt
    prompt_file = Path("prompts/speaker_unknown_attribution_prompt.txt")
    if not prompt_file.exists():
        emit("speakers", f"❌ Attribution prompt not found: {prompt_file}")
        return handle_unknown_speakers(transcript)  # Fallback to simple replacement

    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            attribution_prompt = f.read()
    except Exception as e:
        emit("speakers", f"❌ Error loading attribution prompt: {e}")
        return handle_unknown_speakers(transcript)  # Fallback

    # Analyze segments in batches to manage API costs and context
    batch_size = 5
    attributions = {}

    for i in range(0, len(unknown_segments), batch_size):
        batch = unknown_segments[i:i + batch_size]

        # Create analysis request for this batch
        batch_text = "\n\n".join([
            f"SEGMENT {j+1} (Line {seg['line_number']}):\n"
            f"Content: {seg['content']}\n"
            f"Previous: {seg['previous_speaker']} → Next: {seg['next_speaker']}\n"
            f"Context:\n{seg['context']}"
            for j, seg in enumerate(batch)
        ])

        analysis_prompt = f"""{attribution_prompt}

ANALYZE THESE SPEAKER_UNKNOWN SEGMENTS:

{batch_text}

SPEAKER MAPPING REFERENCE:
{speaker_mapping if speaker_mapping else "No speaker mapping available"}

Provide analysis for each segment in the specified format."""

        try:
            # Use OpenAI API for attribution analysis
            client = OpenAI()

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.1
            )

            analysis_result = response.choices[0].message.content

            # Parse attribution results and apply high-confidence ones
            # For now, we'll save the analysis for review
            if output_dir and output_basename:
                analysis_file = Path(output_dir) / f"{output_basename}_unknown_attribution_batch_{i//batch_size + 1}.txt"
                atomic_write_text(analysis_file, "".join([
                    f"BATCH {i//batch_size + 1} ATTRIBUTION ANALYSIS\n",
                    "=" * 50 + "\n\n",
                    analysis_result,
                ]), backup=False)
                emit("speakers", f"📝 Saved attribution analysis: {analysis_file}")

            # For now, extract simple patterns for automatic attribution
            # Pattern: Same speaker before and after UNKNOWN
            for seg in batch:
                if seg['previous_speaker'] and seg['next_speaker']:
                    if seg['previous_speaker'] == seg['next_speaker']:
                        # High confidence: same speaker continuation
                        attributions[seg['line_index']] = seg['previous_speaker']
                        emit("speakers", f"✅ High confidence attribution: Line {seg['line_number']} → {seg['previous_speaker']}")

        except Exception as e:
            emit("speakers", f"⚠️ Error in AI attribution analysis: {e}")
            # Continue with pattern-based fallback for this batch
            for seg in batch:
                if seg['previous_speaker'] and seg['next_speaker']:
                    if seg['previous_speaker'] == seg['next_speaker']:
                        attributions[seg['line_index']] = seg['previous_speaker']

    # Apply attributions to transcript
    if attributions:
        modified_lines = lines.copy()
        for line_idx, new_speaker in attributions.items():
            original_line = modified_lines[line_idx]
            modified_line = original_line.replace('[SPEAKER_UNKNOWN]', new_speaker)
            modified_lines[line_idx] = modified_line

        transcript = '\n'.join(modified_lines)
        emit("speakers", f"✅ Applied {len(attributions)} SPEAKER_UNKNOWN attributions")

    # Handle any remaining SPEAKER_UNKNOWN with fallback
    remaining_unknown = transcript.count('[SPEAKER_UNKNOWN]')
    if remaining_unknown > 0:
        emit("speakers", f"📋 {remaining_unknown} SPEAKER_UNKNOWN segments remain - applying fallback")
        transcript = handle_unknown_speakers(transcript)

    return transcript


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

        # Use AI to intelligently attribute SPEAKER_UNKNOWN segments before fallback
        result = attribute_unknown_speakers_with_ai(result, speaker_mapping, output_basename, output_dir)

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

        # Verify the replacement worked
        remaining_speakers = re.findall(r'\[SPEAKER_[^\]]*\]', result)
        if remaining_speakers:
            logging.warning(f"Unmapped speakers found: {remaining_speakers}")
            emit("speakers", f"⚠️ Unmapped speakers: {remaining_speakers}")
        else:
            emit("speakers", f"✅ All speaker tags successfully mapped")

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
    Two-phase speaker assignment:
    1. Detailed speaker analysis with o4 model
    2. Apply the mapping using structured assignment prompt
    """
    if not openai_available:
        logging.warning("OpenAI not available, skipping speaker assignment")
        return None

    logging.info("Starting two-phase speaker assignment")

    try:
        if any("SPEAKER_" in line for line in transcript.splitlines()):
            # Store original for comparison
            original_text = transcript

            # Phase 1: Detailed speaker analysis with o4 model
            speaker_mapping = analyze_speakers_with_o4(transcript, output_basename, output_dir)

            if not speaker_mapping:
                logging.warning("Speaker analysis failed, falling back to original method")
                emit("speakers", "⚠️  Advanced speaker analysis failed, using fallback method")

                # Fallback to original method
                return speaker_assignment_fallback(transcript, output_basename, output_dir)

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

        else:
            logging.info("No speaker tags found, skipping speaker assignment")
            return None

    except Exception as e:
        logging.error(f"Error in speaker assignment: {str(e)}")
        emit("speakers", f"❌ Speaker assignment failed: {str(e)}")
        return None

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
            max_tokens=MAX_TOKENS * 2
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
