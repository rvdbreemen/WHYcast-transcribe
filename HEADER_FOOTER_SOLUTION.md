# Header and Footer Integration Solution

## Problem
The user wanted the header and footer (disclaimer and WikiMedia tags) to be added back to the speaker assignment output, but with the correct episode number instead of generic placeholders.

## Solution Design

### 1. Transcript Formatter Module (`transcript_formatter.py`)
Created a dedicated module to handle episode information extraction and formatting:

#### Episode Number Extraction
- **Function**: `extract_episode_number(output_basename)`
- **Supports multiple patterns**:
  - `ep29` → `29`
  - `episode_42` → `42`
  - `Episode_35` → `35`
  - `E28` → `28`
  - `episode-1` → `1`
  - `Whycast33` → `33`
  - `whycast_episode_4_-_fire_` → `4`

#### Header and Footer Formatting
- **Function**: `format_transcript_with_headers(transcript, output_basename)`
- **Adds proper header**:
  ```
  == Disclaimer ==
  [Standard disclaimer text about AI-generated content]
  
  == Transcript 29 ==
  ```
- **Adds proper footer**:
  ```
  [[Category:WHYcast]] [[Category:transcription]] [[Episode number::29| ]]
  ```

### 2. Integration into Main Pipeline

#### Modified `transcribe.py`
1. **Import**: Added `from transcript_formatter import format_transcript_with_headers`
2. **Processing Flow**:
   - AI processes transcript (clean speaker assignment)
   - Content validation and preservation checks
   - **NEW**: Apply header/footer formatting with correct episode info
   - Save formatted result to files
   - Return formatted transcript

#### Updated Speaker Assignment Flow
```python
# AI processing (clean output)
assigned_text = process_with_openai(transcript, full_prompt, ...)

# Content preservation checks
changes = analyze_transcript_changes(original_text, assigned_text)

# Add headers/footers with episode info
formatted_transcript = format_transcript_with_headers(assigned_text, output_basename)

# Save and return formatted result
return formatted_transcript
```

## Benefits

### 1. Separation of Concerns
- **AI Processing**: Focuses only on speaker label replacement
- **Formatting**: Handled separately with proper episode data
- **Clean Architecture**: Easy to maintain and modify

### 2. Accurate Episode Information
- **Before**: Generic `<<episode number>>` placeholders
- **After**: Actual episode numbers like `29`, `42`, `35`
- **Robust Pattern Matching**: Handles various naming conventions

### 3. Proper Content Metrics
- **AI Output**: Clean transcript with accurate retention stats
- **Final Output**: Properly formatted with headers/footers
- **No Content Inflation**: Headers added after metrics calculation

### 4. Maintained Functionality
- **Headers**: Standard disclaimer about AI generation
- **Episode Identification**: Proper WikiMedia category tags
- **Flexibility**: Easy to modify header/footer content

## Example Output

### Input
```
[SPEAKER_00] Welcome back to the show everyone.
[SPEAKER_01] Thanks for having me on.
```

### AI Processing Result
```
Host: Welcome back to the show everyone.
Guest: Thanks for having me on.
```

### Final Formatted Output
```
== Disclaimer ==
This is the full transcript generated using with AI tools and some human oversight...

== Transcript 29 ==
Host: Welcome back to the show everyone.
Guest: Thanks for having me on.

[[Category:WHYcast]] [[Category:transcription]] [[Episode number::29| ]]
```

## Testing
Created test scripts to validate:
- Episode number extraction accuracy
- Header/footer formatting
- Integration with speaker assignment pipeline
- Content preservation metrics

This solution provides the best of both worlds: clean AI processing with accurate content metrics, plus proper episode-specific formatting for final output.
