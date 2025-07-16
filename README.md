# Muesli

Offline-first, privacy-centric voice **transcription** and **summarisation** desktop application powered by  
[whisper.cpp](https://github.com/ggerganov/whisper.cpp) and a local LLM served by [Ollama](https://ollama.ai/).

― Inspired by the lean “one-file” style of `agentic-assistant`, Muesli keeps the codebase small while offering a polished Qt / QML interface.

---

## Features

- 🎙️ Real-time microphone transcription with optional VAD
- 📂 Drag-and-drop or “Open File” transcription for WAV / MP3 / M4A / FLAC / OGG
 - 🧠 Local LLM summarisation – generated automatically right after transcription
 - 📝 Single combined markdown view (summary **first**, transcript below a `---` separator)
 - 💾 Save combined content as **.md**, **.txt**, or **.srt** (auto-generated block)
- 🌗 Dark / Light / System theme
- 💻 Runs completely offline – network calls can be disabled in `config.yml`
- 🪄 Single-executable build possible via `pyinstaller`

---

## Project Structure

```
muesli/
├── main.py            # Application entry-point & orchestration
├── models.py          # Pydantic data models (AudioFile, Transcript, Summary…)
├── whisper_wrapper.py # Thin subprocess wrapper around whisper.cpp
├── stream_processor.py# Microphone capture & streaming transcription
├── ollama_client.py   # Subprocess wrapper around the Ollama CLI
├── summarizer.py      # Generates summaries with the LLM client
└── ui/
    ├── main_window.py # PySide6 main window (widgets)
    └── qml/           # Optional QML front-end
        └── main.qml
```

No databases, message queues or complex dependency-injection frameworks – the **`MuesliApp`** class in `main.py` wires everything together.

---

## Requirements

| Requirement            | Purpose                          |
| ---------------------- | -------------------------------- |
| Python **3.8 +**       | Core application                 |
| `whisper.cpp` binary   | On-device speech-to-text         |
| `ffmpeg` (recommended) | Decode/convert non-WAV audio     |
| Ollama (optional)      | Local LLM for summaries          |
| **Poetry**             | Dependency management (optional) |

Python deps are listed in `pyproject.toml` – main ones are **PySide6**, **pydantic**, **PyAudio**, **numpy**.

---

## Installation

### 1. Clone & set up Python environment

```bash
git clone https://github.com/JBlitzar/muesli
cd muesli
# (recommended) create a virtual-env
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# install Python dependencies
pip install -r requirements.txt
```

### 2. Build whisper.cpp (required)

```bash
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
cmake -B build
cmake --build build -j --config Release
# Add build/ (contains `main` / `whisper`) to your PATH, or move the binary
```

`Muesli` will auto-download the GGML model you choose (default **medium**).  
You can also drop a file such as:

- `ggml-medium.bin` **or**
- `ggml-medium.en.bin`

into `~/.muesli/models/whisper/` – the wrapper looks for both names.

### 3. (Optional) Install Ollama

```bash
curl https://ollama.ai/install.sh | sh
ollama pull llama3:8b-instruct
```

### 4. Run

```bash
python main.py                 # normal
python main.py --verbose       # debug logs
python main.py --transcribe path/to/audio.wav  # CLI mode
```

---

## Usage

### Graphical UI

1. Start `muesli` (double-click or `python -m muesli`)
2. Click **Open File** _or_ **Record**
3. When you stop recording (or when file transcription ends) a summary is generated automatically and shown above the transcript.
4. Save the combined markdown via **File → Save Content…**

### Command-line shortcuts

```
--no-ui              run headless
--config path.yml    custom configuration
--transcribe file    transcribe and print to stdout
-v / --verbose       debug logs
```

Environment variable overrides follow the pattern `MUESLI_SECTION_KEY=value`, e.g.

```bash
export MUESLI_TRANSCRIPTION_MODEL=small
export MUESLI_LLM_PROVIDER=none
```

---

## Optional Configuration (`muesli.yml`)

```yaml
transcription:
  model: medium # tiny / base / small / medium / large / large-v3
  auto_language_detection: true
llm:
  provider: ollama
  model_name: llama3:8b-instruct
privacy:
  allow_network: false
ui:
  theme: system
```

Place it in:

- `./muesli.yml` (project root)
- `~/.muesli/config.yml` (user)

The application runs fine without any configuration file – sensible defaults are baked in.

---

## Component Details

| Component                                                | Notes                                                                                                    |
| -------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| **WhisperTranscriber** (`whisper_wrapper.py`)            | Spawns whisper.cpp via `subprocess`, streams progress, verifies model checksum.                          |
| **TranscriptionStreamProcessor** (`stream_processor.py`) | Captures microphone audio with PyAudio, performs VAD, chunks audio into ~5 s segments, feeds to Whisper. |
| **OllamaClient** (`ollama_client.py`)                    | Thin subprocess wrapper around the `ollama run` CLI (no HTTP), supports streaming responses.             |
| **TranscriptSummarizer** (`summarizer.py`)               | Builds prompt templates and calls `OllamaClient` to get summary.                                         |
| **MuesliApp** (`main.py`)                                | Loads config, sets up components, exposes Qt signals for UI.                                             |
| **UI** (`ui/`)                                           | PySide6 widgets + optional QML; auto-updates via Qt signals.                                             |

---

## Development

```bash
poetry run black .       # format
poetry run isort .       # import order
poetry run ruff .        # lint
poetry run mypy muesli   # type-check
poetry run pytest -q     # tests (minimal)
```

---

## Troubleshooting

| Symptom / Log message                          | Possible cause & fix                                                                                                                        |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `whisper.cpp binary not found`                 | Ensure you compiled the binary (`make`) and that `whisper`, `whisper-cpp`, or `main` is in your `$PATH`, **or** provide `--whisper-binary`. |
| `Model file not found…`                        | Place **either** `ggml-medium.bin` **or** `ggml-medium.en.bin` in `~/.muesli/models/whisper/`, or allow auto-download (default).            |
| `Failed to parse JSON output from whisper.cpp` | Older builds don’t support `--output-json`. Re-compile latest `whisper.cpp` (`git pull && make`) **or** let Muesli fall back to TXT parse.  |
| Silence / no transcription                     | Verify your input device using `pyaudio` list in Python or system preferences; disable VAD in config to test.                               |
| `ffmpeg` errors decoding files                 | Install `ffmpeg` and ensure it’s on PATH; or convert the audio to WAV manually.                                                             |

### whisper.cpp CLI cheat-sheet

```
whisper -m ggml-medium.en.bin \
        -f audio.wav \
        --language en \
        --output-dir out/ \
        --output-json transcript.json \
        --output-txt true \
        --beam-size 5
```

Muesli internally generates a similar command; use it as a reference when debugging.

---

## Roadmap

- Speaker diarisation
- Multi-file batch mode
- Electron-free mobile build via Qt-for-Android/iOS

---

## License

The source code is released under the **GNU GPLv3 License**.  
Whisper model weights are distributed under their respective licenses (MIT for GGML binaries).  
“Whisper” and “GPT” are trademarks of their respective owners.

Enjoy your breakfast 🥣.  
– **Muesli** team
