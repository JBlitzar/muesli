# Muesli – ARCHITECTURE.md  
_Last updated: 2025-07-16_

## 1. High-Level Overview
Muesli is an offline-first desktop application for speech transcription and summarisation.  
The architecture follows a **modular, hexagonal** (ports-and-adapters) pattern:

```
┌────────────┐     Qt/QML        ┌────────────┐
│   UI/QML   │  ─────────────▶  │  Core App  │◀─┐
└────────────┘   Signal / Slot   └────────────┘  │
      ▲                                         │
      │                                         ▼
      │                             Job API / Callbacks
      │                                         │
┌────────────┐   Core Models   ┌────────────┐   │
│Transcriber │◀───────────────▶│   Job Q    │───┤
└────────────┘                 └────────────┘   │
      ▲                                         ▼
      │                        HTTP JSON        │
┌────────────┐                 ┌────────────┐   │
│   LLM      │◀───────────────▶│ Ollama API │◀──┘
└────────────┘                 └────────────┘
```

* **Core** (Python, `muesli.core`) owns configuration, the job queue, and orchestrates all modules.  
* **UI** (`muesli.ui`) is a thin PySide6+QML layer that talks to Core via Qt signals/slots.  
* **Transcription** (`muesli.transcription`) wraps `whisper.cpp` for file & streaming inference.  
* **LLM** (`muesli.llm`) interfaces with a local Ollama server for summarisation.  
* **Data Store**: SQLite for metadata + filesystem JSONL for transcripts (handled by Core).  

No network calls are made by default; Ollama is expected to run locally.

---

## 2. Module Responsibilities

| Package | Role | Key Classes / Entry Points |
|---------|------|----------------------------|
| **muesli.core** | Orchestrator, config validation, job scheduling, in-memory state of transcripts/summaries. | `MuesliApp`, `AppConfig`, `JobQueue` |
| **muesli.ui** | Desktop UX. Emits user intents (`startTranscription`, `generateSummary`), renders live state via QML bindings. | `MainWindow`, `TranscriptModel`, `SummaryModel` |
| **muesli.transcription** | Interaction with `whisper.cpp`; supports batch and microphone streams. | `WhisperTranscriber`, `TranscriptionStreamProcessor` |
| **muesli.llm** | Summary generation through local LLM. Handles prompt templates & streaming output. | `OllamaClient`, `TranscriptSummarizer` |
| **models** | Pydantic data classes reused across modules. | `AudioFile`, `Transcript`, `Summary`, `TranscriptSegment` |

---

## 3. Data Flow

### 3.1 File Transcription

1. **UI** calls `MuesliApp.transcribe_file(path)`.
2. **Core** adds a `JobType.TRANSCRIPTION` to **JobQueue**.
3. Worker thread invokes **WhisperTranscriber** on the audio file.  
   Progress updates → `Job.update_progress()` → Qt signal to UI.
4. Segments appended to `Transcript`; when done, `transcription_complete` signal fires.
5. If `auto_summarize` is enabled, Core enqueues a **SUMMARIZATION** job.

### 3.2 Live (Streaming) Transcription

1. **UI** triggers `start_streaming_transcription()`.
2. **TranscriptionStreamProcessor** captures microphone audio, applies VAD, writes temp WAV chunks.
3. Each chunk is synchronously passed to **WhisperTranscriber**; returning segments are time-shifted and emitted via `_on_stream_segment()` -> UI.
4. On stop, remaining buffer is flushed; transcript is marked complete.

### 3.3 Summarisation

1. Core enqueues **SUMMARIZATION** job with transcript id.
2. Worker formats prompt via `prompt_templates.get_summary_prompt()`.
3. **OllamaClient** POSTs to `http://localhost:11434/api/generate`.
4. Generated text stored in `Summary`; `summarization_complete` signal notifies UI.

---

## 4. Component Interfaces

### 4.1 Core ↔ UI (Qt Signals)

| Signal | Args | Emitted When |
|--------|------|--------------|
| `transcription_started` | `transcript_id` | job queued / stream started |
| `transcription_progress` | `id`, `progress:float`, `msg` | batch progress updates |
| `transcription_complete` | `id` | finished (file or stream) |
| `summarization_started` | `summary_id` | job queued |
| `summarization_complete` | `summary_id` | summary ready |
| `*_failed` | `id`, `error` | job or stream error |

### 4.2 Core ↔ Transcriber

* Synchronous `WhisperTranscriber.transcribe()` for files.  
* For live mode `TranscriptionStreamProcessor` holds a reference to same `WhisperTranscriber` instance.

### 4.3 Core ↔ LLM

* `OllamaClient.generate(prompt)` returns string.  
* Optional `generate_streaming()` yields chunks for progressive UI update.

---

## 5. Job Queue

* Thin wrapper around `concurrent.futures.ThreadPoolExecutor`.
* Priority first-in handling, dependency resolution, cancel / timeout stubs.
* New background tasks simply register a handler:

```python
job_queue.register_handler(JobType.EXPORT, export_handler)
```

---

## 6. Persistence

| Artifact | Location | Format |
|----------|----------|--------|
| App config | `~/.muesli/config.yaml` | YAML |
| Models | `~/.muesli/models/whisper` | GGML / GGUF |
| DB | `~/.muesli/data/muesli.db` | SQLite (metadata / FTS) |
| Transcripts | `~/.muesli/data/<id>.jsonl` | one JSON per segment |

---

## 7. Extension Points

1. **Job Handlers** – register additional `JobType` for export, translation, etc.  
2. **Prompt Templates** – `llm.prompt_templates.register_custom_template()` to add domain-specific styles.  
3. **Plugin System (future)** – planned entry-point discovery (`importlib.metadata`) to auto-load extra processing modules (e.g., diarisation, sentiment).  
4. **UI QML Components** – QML is hot-reloadable; new panes can bind to Core signals without modifying Python.

---

## 8. Design Decisions

* **Single source of truth**: Core module owns runtime state; UI is reactive.
* **No encryption module**: All artefacts remain on local disk by design; simplifies threat model.
* **Streaming first**: Live microphone pipeline shaped early to keep latency budgets visible.
* **Local LLM**: Avoids TOS / privacy issues of remote APIs; Ollama chosen for minimal ops cost.
* **Ascii diagrams** over images to stay VCS-friendly.

---

## 9. Future Work

| Area | Notes |
|------|-------|
| Diarisation | Integrate pyannote-audio offline pipeline via `JobType.DIARISATION`. |
| Translation | Reuse Whisper multilingual output or LLM post-process. |
| File export pipeline | `JobType.EXPORT` handlers for Obsidian MD, CSV, PDF. |
| Plugin SDK | Sandboxed subprocess execution, manifest validation. |

---

End of file.
