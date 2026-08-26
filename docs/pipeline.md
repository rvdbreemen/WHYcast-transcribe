# The WHYcast pipeline

How an episode becomes a transcript, a summary and a blog post, which files
each step reads and writes, and which job re-runs which part.

Drawn from the code, not from memory: `whycast/pipeline/workflow.py` for the
stages, `whycast/pipeline/postprocess.py` for the language-model steps, and
`webui/runner.py` for the job types.

## The pipeline

Rounded boxes are steps, plain boxes are files on disk. Anything with `<base>`
is per episode - `episode_13_summary.txt` and so on.

```mermaid
flowchart TD
    RSS[/"RSS feed"/] -->|download, verified against<br/>Content-Length and feed size| MP3["&lt;base&gt;.mp3"]

    MP3 --> PREP("1. Prepare audio<br/><i>normalise for diarization</i>")
    PREP --> WAV["temp 16k mono<br/><i>deleted after the run</i>"]

    WAV --> DIA("2. Diarize<br/><i>pyannote 3.1 · GPU</i>")
    DIA --> SEG(["speaker segments<br/><i>in memory</i>"])

    WAV --> TRANS("3. Transcribe<br/><i>faster-whisper large-v3 · GPU</i>")
    SEG --> TRANS
    VOCAB[/"vocabulary.json<br/><b>input</b>"/] -.->|corrections| TRANS

    TRANS --> TXT["&lt;base&gt;_transcript.txt"]
    TRANS --> TS["&lt;base&gt;_ts.txt<br/><i>with timestamps</i>"]
    TXT --> MERGE("Merge speaker lines")
    MERGE --> MRG["&lt;base&gt;_merged.txt"]

    MRG --> SPK("4a. Speaker assignment")
    MAP[/"&lt;base&gt;_speakers.json<br/><b>input · editable</b>"/] -->|when present, the paid<br/>analysis is skipped| SPK
    SPK -->|when absent, the model<br/>decides and it is saved| MAP
    SPK --> ANA["&lt;base&gt;_speaker_analysis.txt"]
    SPK --> ASG["&lt;base&gt;_speaker_assignment<br/>.txt/.html/.wiki"]

    ASG --> CLEAN("4b. Cleanup")
    CLEAN --> CLN["&lt;base&gt;_cleaned.txt"]

    CLN --> SUM("4c. Summary")
    SUM --> SUMF["&lt;base&gt;_summary<br/>.txt/.html/.wiki"]

    CLN --> BLOG("4d. Blog")
    SUMF --> BLOG
    BLOG --> BLOGF["&lt;base&gt;_blog<br/>.txt/.html/.wiki"]

    CLN --> ALT("4e. Alternative blog")
    SUMF --> ALT
    ALT --> ALTF["&lt;base&gt;_blog_alt1<br/>.txt/.html/.wiki"]

    CLN --> HIST("4f. History extraction")
    HIST --> HISTF["&lt;base&gt;_history<br/>.txt/.html/.wiki"]

    PROMPTS[/"prompts/*.txt<br/><b>input</b>"/] -.-> CLEAN
    PROMPTS -.-> SUM
    PROMPTS -.-> BLOG
    PROMPTS -.-> ALT
    PROMPTS -.-> HIST
    PROMPTS -.-> SPK

    classDef gpu fill:#e8f0fe,stroke:#3b6ea5,color:#12263f
    classDef paid fill:#fdf2e3,stroke:#8a6100,color:#3b2b00
    classDef input fill:#eaf5ec,stroke:#2f6b3a,color:#12331a
    class DIA,TRANS gpu
    class SPK,CLEAN,SUM,BLOG,ALT,HIST paid
    class VOCAB,MAP,PROMPTS input
```

Blue steps hold the GPU. Amber steps call the paid OpenAI API. Green boxes are
**inputs**: files a person edits and the pipeline reads. They are never
overwritten by a run, and never moved to the backup - that distinction is what
ADR-010 and ADR-011 are about.

## Which job re-runs what

Every job below starts by moving the artifacts it will replace into
`backups/<base>/<timestamp>/`, so a re-run never overwrites the previous
version (ADR-011).

```mermaid
flowchart LR
    subgraph feed["From the feed"]
        FL["fetch_latest<br/><i>GPU · paid</i>"]
        FA["fetch_all<br/><i>download only</i>"]
    end

    subgraph whole["Whole episode"]
        FE["full_episode<br/><i>GPU · paid</i>"]
        FO["force_episode<br/><i>GPU · paid</i>"]
        RT["retranscribe<br/><i>GPU · free</i>"]
    end

    subgraph parts["Part of an episode"]
        PP["postprocess<br/><i>paid</i>"]
        SP["speakers<br/><i>paid</i>"]
    end

    subgraph single["One step"]
        S1["step_cleanup"]
        S2["step_summary"]
        S3["step_blog"]
        S4["step_blog_alt1"]
        S5["step_history"]
    end

    A(["audio"]) --> FE & FO & RT
    RSS2(["RSS"]) --> FL & FA
    T(["transcript"]) --> PP & SP
    C(["cleaned"]) --> S2 & S3 & S4 & S5
    T --> S1
    SUM2(["summary"]) --> S3 & S4

    RT -.->|"stops after<br/>the transcript"| T
    SP -.-> T
    PP -.-> C

    classDef free fill:#eaf5ec,stroke:#2f6b3a,color:#12331a
    class RT,FA free
```

`retranscribe` is the only GPU job that spends nothing: it redoes the audio
half and stops, which is what you want after changing the Whisper model, the
vocabulary or the diarization settings. Its counterpart is the single-step
re-run: one model call to redo one output, instead of four to see one change.

## The two rules that shape all of this

**Inputs are read, outputs are written, and nothing is both by accident.**
`vocabulary.json`, `prompts/*.txt` and `<base>_speakers.json` are human input:
edited by a person, read by the pipeline, backed up with a `.bak` when saved,
and never regenerated. Everything else is an artifact: regenerable, no `.bak`,
moved aside before a run replaces it.

`<base>_merged.txt` is the one file that is genuinely both - speaker assignment
writes it, post-processing reads it - which is why a run copies it to the
backup instead of moving it away.

**A step that does not run does not lose its output.** The speaker analysis is
skipped when a saved mapping exists, and the alternative blog is skipped when
its prompt file is absent. Their artifacts are moved aside at the start of a
run like any other, and put back at the end when the run turns out not to have
replaced them.

## Related decisions

- **ADR-001**, **ADR-002** - the GPU half: faster-whisper on CUDA, pyannote for diarization.
- **ADR-003**, **ADR-005** - per-task model selection and recursive summarization.
- **ADR-004**, **ADR-010** - speaker assignment in two phases, and the mapping as an editable input.
- **ADR-006**, **ADR-007** - vocabulary corrections and file-based prompts.
- **ADR-008**, **ADR-009**, **ADR-011** - the web UI over this library, and how artifacts are written and preserved.
