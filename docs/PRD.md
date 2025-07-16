# Muesli – Product Requirements Document (PRD)

_Last updated: 2025-07-16_

## 1. Product Overview

Muesli is an **offline-first, privacy-centric voice transcription and summarization desktop application**.  
It delivers the convenience of modern AI transcription apps (e.g., Granola) while guaranteeing that **all audio, transcripts, and model inference remain on the user’s machine**.

**Core differentiators**

1. 100 % on-device processing – no cloud upload, no telemetry.
2. Pluggable LLM pipeline – default to `ollama llama3:8b-instruct`, but configurable.
3. Native cross-platform UI (Windows, macOS, Linux) built with **PySide6 + QML** for performance and system integration.
4. Modular architecture enabling future add-ons: diarization, topic tagging, sentiment, translation.

---

## 2. Problem Statement

Knowledge workers, researchers, and privacy-sensitive users record an increasing amount of audio (meetings, interviews, lectures). Existing transcription tools either:

- **Upload audio to third-party servers**, risking data leakage and regulatory violations, or
- **Provide sub-par UX** when run locally.

There is a market gap for a **beautiful, performant, fully private desktop transcription app** that matches cloud solutions in accuracy and convenience.

---

## 3. User Personas

| Persona                    | Profile                       | Needs                                                                      | Pain Points                                                                  |
| -------------------------- | ----------------------------- | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Privacy-First Professional | Lawyer, therapist, journalist | Accurate transcripts stored locally; compliance with confidentiality rules | Cannot legally use cloud transcription; manual note-taking is time-consuming |
| Researcher / Academic      | PhD student, professor        | Bulk transcription of interviews & lectures; keyword search                | Existing open-source tools are CLI-only and fragile                          |
| Remote Knowledge Worker    | Product manager, consultant   | Quick meeting summaries; integration with task tools                       | Cloud apps violate company policy; raw transcripts are overwhelming          |
| Power User / Hacker        | Developer, OSS contributor    | Custom models, fine-tuning, scripting hooks                                | Proprietary apps lack extensibility                                          |

---

## 4. Feature Requirements

### 4.1 Must-Have (v1)

1. **Local Audio Import**

   - Drag-and-drop common formats (wav, mp3, m4a, flac).
   - Batch import with progress indication.

2. **On-Device Transcription**

   - Powered by `whisper.cpp` with GPU/CPU auto-selection.
   - Language auto-detection; manual override.

3. **Editable Transcript Viewer**

   - Rich text with timestamps.
   - Inline editing & correction.

4. **Search & Navigate**

   - Full-text search across transcripts.
   - Clickable search results jump to timestamp.

5. **Summary Generation**

   - Uses local LLM (Ollama) to produce bullet/paragraph summaries.
   - Configurable prompt templates.

6. **Export & Share**

   - Plain text, Markdown, SRT, and JSON export.
   - No hidden metadata; optional encryption on disk.

7. **Privacy Dashboard**
   - Toggle telemetry (default off).
   - View/clear cache, model files, and logs.

### 4.2 Should-Have (v1.x)

1. Speaker diarization (offline).
2. Translation to target language.
3. Clipboard & global hotkey recording.

### 4.3 Nice-to-Have

1. Real-time transcription overlay.
2. Integration plugins (Obsidian, VS Code).
3. Custom fine-tune UI for Whisper/LLM.

---

## 5. Technical Architecture

### 5.1 High-Level Components

```
┌───────────┐      ┌─────────────┐      ┌──────────────┐
│   UI/UX   │←────→│  App Core   │←────→│  Data Store  │
└───────────┘      │ (Python)    │      │ (SQLite+FS)  │
                   │             │
                   │             └─┬──────────┐
                   │               │          │
                   ↓               ↓          ↓
            ┌──────────────┐ ┌──────────┐ ┌─────────────┐
            │ whisper.cpp  │ │  Ollama  │ │  Plugins    │
            │  wrapper     │ │ (LLM API)│ │  (optional) │
            └──────────────┘ └──────────┘ └─────────────┘
```

### 5.2 Key Technologies

- **PySide6 + QML** – native UI, dark/light themes.
- **whisper.cpp** – C++ inference library, accessed via FFI bindings; models stored in `~/.muesli/models/whisper`.
- **Ollama** – lightweight local LLM server; default model `llama3:8b-instruct`; user-selectable via settings.
- **SQLite** – metadata DB; transcripts stored as `.jsonl` plus optional encrypted blobs.
- **Python ≥ 3.11** – orchestrating layer with modular packages:
  - `muesli.core`: data models (Pydantic v2), job queue.
  - `muesli.transcription`: wrapper around whisper.cpp.
  - `muesli.llm`: prompt templates, summary generation.
  - `muesli.ui`: QML components.
  - `muesli.security`: encryption, permissions, audit logging.
- **Plugin Framework**
  - Entry-point discovery (`importlib.metadata`).
  - Sandboxed execution (sub-process, restricted Python).

### 5.3 Modularity Principles

1. **Loose coupling**: Each component communicates via explicit interfaces/events.
2. **Config-as-Code**: `muesli.yaml` with Pydantic validation; env var overrides.
3. **Replaceable models**: Drop-in GGUF/ggml Whisper models and any Ollama-compatible LLM.

---

## 6. Privacy & Security Considerations

1. **Zero Network Dependence by Default**
   - App launches without internet. External calls (model download) require explicit consent.
2. **Local Encryption**
   - AES-256 encryption for audio/transcripts with user-defined passphrase.
3. **Secure Model Sources**
   - SHA256 hash verification of downloaded models.
4. **Sandboxed LLM**
   - Ollama runs in user space; restricts network via `--offline`.
5. **Logging**
   - Structured logs stored locally; redacted of PII.
   - Verbose mode toggle.
6. **Open-Source Transparency**
   - Source licensed under AGPL-v3; reproducible builds.

---

## 7. UI / UX Specifications

### 7.1 Design Language

- Dark theme by default; auto matches OS theme.
- Minimalistic toolbar with large import button.
- Side panel: Library list, filters, tags.
- Main panel: Transcript editor with waveform timeline.

### 7.2 User Flows

1. **Import → Transcribe → Review → Summarize → Export**
2. **Library Search → Open Transcript → Edit / Tag**

### 7.3 Accessibility

- Keyboard shortcuts for all actions.
- WCAG 2.1 AA color contrast.
- Adjustable font size.

### 7.4 Mock Screen Definitions

1. **Dashboard.qml** – recent files, stats.
2. **TranscriptView.qml** – split waveform & text.
3. **Settings.qml** – models, privacy, advanced.

---

## 8. Implementation Roadmap

| Phase                         | Timeline    | Milestones                                                   |
| ----------------------------- | ----------- | ------------------------------------------------------------ |
| **0. Foundations**            | Month 1     | Repo scaffolding, CI, Pydantic config, basic PySide6 window. |
| **1. Core Transcription**     | Months 2-3  | Integrate whisper.cpp; batch import UI; progress dialogs.    |
| **2. LLM Summaries**          | Month 4     | Bundle Ollama; summary prompt templates; settings panel.     |
| **3. Library & Search**       | Month 5     | SQLite schema; full-text search; global filters.             |
| **4. Export & Formats**       | Month 6     | Markdown, SRT, JSON export; encryption.                      |
| **5. Privacy Hardening**      | Month 7     | Hash checks, network toggles, audit logs.                    |
| **6. Beta Release**           | Month 8     | Cross-platform installers, feedback loop.                    |
| **7. Post-Beta Enhancements** | Months 9-12 | Diarization, translations, plugin SDK V1.                    |

---

## 9. Success Metrics

1. **Accuracy** – WER within 1 % of OpenAI Whisper base on LibriSpeech test.
2. **Performance** – Transcription speed ≥ 1.2× real-time on M1 Pro, 16-thread CPU mode ≤ 1.0×.
3. **Privacy Score** – No outbound network calls in default run (verified by egress monitor).
4. **User Adoption** – 5 000 downloads and 500 MAU within 3 months of launch.
5. **User Satisfaction** – Average rating ≥ 4.5/5 on feedback survey (ease of use & trust).
6. **Crash-Free Sessions** – ≥ 99.5 % in beta telemetry (opt-in only).

---

## 10. Risks & Mitigations

| Risk                       | Impact                    | Mitigation                                     |
| -------------------------- | ------------------------- | ---------------------------------------------- |
| Large model footprint      | High disk usage           | Offer model size choices; prune unused models. |
| GPU compatibility issues   | Poor perf on some devices | Provide CPU fallback; detect hardware.         |
| Legal/compliance ambiguity | Trust barrier             | Clear privacy policy; open-source audit.       |
| UI complexity creep        | Delivery delay            | Strict scope control; incrementally ship.      |

---

## 11. Appendix

- Reference competitors: Granola, MacWhisper, Obsidian Audio Notes.
- Whisper model sizes: tiny (75 MB) → large-v3 (2.9 GB).

---

End of document.
