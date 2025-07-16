# Muesli

Offline-first, privacy-centric voice transcription **and** summarisation desktop application.

Muesli delivers the convenience of modern cloud transcription tools while ensuring that **all audio, transcripts and AI inference stay on your machine**. No uploads, no telemetry, no compromises.

---

## Key Features

- **100 % on-device processing** – powered by `whisper.cpp` and a local LLM (default: `ollama llama3:8b-instruct`).
- **Streaming transcription** – start/stop recording from the UI; text appears in real-time.
- **Editable transcript viewer** – timestamped rich text with inline corrections.
- **One-click summaries** – generate concise bullet or paragraph summaries locally.
- **Search & navigation** – full-text search across your library, jump to any timestamp.
- **Multi-format export** – Plain text, Markdown, SRT or JSON.
- **Cross-platform UI** – PySide6 + QML, dark/light themes, keyboard shortcuts.
- **Completely private** – network disabled by default; files remain on disk unless you choose to share them.

---

## Requirements

| Component | Minimum Version |
|-----------|-----------------|
| Python    | 3.11            |
| Poetry    | 1.6             |
| PySide6   | 6.5             |
| whisper.cpp model | any GGUF / ggml model (tiny–large-v3) |
| Ollama (optional, for summaries) | 0.1.32 |

macOS (M-series & Intel), Windows 10 +, and modern Linux distributions are supported.

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/JBlitzar/muesli.git
   cd muesli
   ```

2. **Install dependencies with Poetry**
   ```bash
   poetry install --with dev
   ```

3. **Download a Whisper model**  
   (until automatic download is implemented)
   ```bash
   mkdir -p ~/.muesli/models/whisper
   # example for the small model
   curl -L -o ~/.muesli/models/whisper/ggml-small.bin \
        https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin
   ```

4. **(Optional) Pull an LLM model with Ollama**
   ```bash
   # Install Ollama from https://ollama.ai
   ollama pull llama3:8b-instruct
   ```

---

## Usage

### Graphical Interface

```bash
poetry run muesli
```

* Press **Start Recording** to begin live transcription.  
* Press **Stop Recording** to end the session.  
* Click **Generate Summary** to create a local LLM summary once transcription stops.

### Command-line Mode

```bash
# Transcribe a file
poetry run muesli --transcribe path/to/audio.wav

# Headless (no UI) with verbose logging
poetry run muesli --no-ui -v --transcribe interview.mp3
```

Configuration is stored in `~/.muesli/config.yaml`.  
Environment variables prefixed with `MUESLI_` override any key (e.g. `MUESLI_TRANSCRIPTION_MODEL=small`).

---

## Project Structure

```
muesli/
├── core/            # Application orchestrator, config, models, job queue
├── transcription/   # whisper.cpp wrapper + streaming processor
├── llm/             # Ollama client, prompt templates, summariser
├── ui/              # PySide6 widgets & QML front-end
│   └── qml/         # QML files (main.qml, Dashboard.qml, etc.)
└── main.py          # CLI & GUI entry point
```

Each sub-package exposes a clean public interface; the **core** package is the single point of coordination between modules.

---

## License

Muesli is released under the **GNU Affero General Public License v3.0** (AGPL-3.0).  
See the [`LICENSE`](LICENSE) file for the full text.

Commercial enquiries or custom licensing? Please reach out via the project’s issue tracker.

---
