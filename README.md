# CircuitSense

**Turn electrical diagrams into structured, analyzable models.**

CircuitSense is an engineering analysis tool that takes electrical schematics, ladder-logic diagrams, and wiring diagrams and converts them into structured models that can be analyzed by deterministic software.

Instead of asking an AI model to perform electrical calculations, CircuitSense separates **diagram understanding** from **engineering analysis**:

**Image → Recognition → Structured Model → Deterministic Analysis**

AI is optional and is used where language and interpretation are useful—not as the source of the electrical calculations.

## What CircuitSense Can Do

### Circuit Analysis

* Detect common circuit components and connections
* Read component values using OCR
* Solve DC circuits
* Calculate node voltages, branch currents, and component power
* Perform what-if analysis
* Check continuity, nets, and potential shorts
* Identify missing or insufficient information

### AC Analysis

* Analyze RLC circuits using phasor methods
* Calculate voltage and current magnitude and phase
* Determine average and reactive power
* Perform frequency sweeps
* Generate Bode plots
* Identify cutoff frequency, resonance, and bandwidth

### Ladder Logic

* Recognize rungs and contacts
* Identify NO/NC contacts and coils
* Generate Boolean representations
* Explain ladder programs in plain English
* Simulate ladder logic
* Generate PLC I/O wiring diagrams

### KiCad / SPICE Netlists

CircuitSense can import KiCad/SPICE-style netlists for cases where exact connectivity is already available.

This provides a direct path from a circuit description to analysis without relying on image recognition.

## Why CircuitSense?

Electrical engineering software is extremely good at solving circuits when the circuit is already represented as structured data.

The difficult part is often getting from:

> **A picture of a circuit**

to:

> **A machine-readable electrical model**

CircuitSense focuses on that bridge.

The recognition layer extracts components, labels, values, and connections. The resulting model is then passed to deterministic engineering algorithms for analysis.

This makes the numerical results reproducible and inspectable rather than dependent on an AI model "guessing" the answer.

## Quick Demo

The repository includes sample diagrams that can be loaded directly from the application.

Some examples include:

* BJT amplifier analysis
* Inverting op-amp analysis
* RC low-pass frequency response
* Circuit connectivity checks
* Missing-value solving
* Ladder-logic analysis and simulation

## Running Locally

### Windows

Run:

```bat
run.bat
```

Or manually:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn circuitsense.api:app
```

### Linux / macOS

```bash
./run.sh
```

The application will start locally at:

```text
http://127.0.0.1:8000
```

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## Optional AI Features

CircuitSense can operate without an AI API key for its core analysis functionality.

AI features can be enabled by setting:

```text
GEMINI_API_KEY=your_key_here
```

You can either export the variable in your environment or place it in a local `.env` file based on `.env.example`.

The AI layer is used for features such as:

* Explaining ladder programs
* Answering questions about an analyzed system
* Helping identify unsupported/unknown symbols

**Never commit your API key or `.env` file to the repository.**

## Testing

Install the development dependencies:

```bash
pip install -r requirements-dev.txt
```

Then run:

```bash
pytest
```

## Project Structure

```text
CircuitSense/
├── circuitsense/      # Application and analysis engine
├── samples/           # Example diagrams and netlists
├── static/            # Frontend assets
├── tests/             # Automated tests
├── Dockerfile         # Container configuration
├── run.bat            # Windows launcher
├── run.sh             # Linux/macOS launcher
├── requirements.txt   # Runtime dependencies
└── README.md
```

## Technology

CircuitSense is built around:

* Python
* FastAPI
* OpenCV
* OCR
* Deterministic circuit-solving algorithms
* Ladder-logic parsing and simulation
* Optional Google Gemini integration
* Docker

## Status

CircuitSense is an active engineering project.

The recognition pipeline currently focuses on supported diagram styles and component types. Unsupported or ambiguous elements are identified rather than silently assumed.
