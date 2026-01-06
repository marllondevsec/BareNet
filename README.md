# BareNet

BareNet is a local-first, privacy-focused network reconnaissance and discovery tool designed for cybersecurity research and defensive analysis. The project aims to provide large-scale URL and service discovery without relying on centralized servers, external APIs, or cloud-based infrastructure.

Unlike traditional scanners or OSINT platforms, BareNet is designed to run entirely on the user's machine, giving full control over data sources, filtering logic, and execution flow.

---

## Key Features

* Local-first execution (no external servers required)
* Modular search engine architecture
* Pluggable filters for protocol and security properties
* Designed for large-scale URL and endpoint discovery
* Suitable for research, blue team analysis, and controlled lab environments
* CLI-oriented and scriptable

---

## Project Philosophy

BareNet is built around a simple principle:

> If you want to reduce noise, fingerprints, and external dependency risks, you must control the entire pipeline.

The tool avoids reliance on third-party SaaS platforms, indexed search engines, or telemetry-heavy tooling. All discovery logic runs locally, making it suitable for air-gapped environments, controlled networks, and forensic research.

---

## Architecture Overview

BareNet follows a modular and extensible design:

* **CLI Layer** – Argument parsing and user interaction
* **Search Engines** – Multiple discovery backends (file-based, search-based, etc.)
* **Filters** – Post-processing modules to refine results
* **HTTP Client** – Lightweight networking abstraction
* **Cache Layer** – Optional local caching to reduce repeated requests

Each component can be extended independently.

---

## Directory Structure

```
barenet/
├── __init__.py
├── __main__.py
├── cli.py            # Command-line interface
├── config.py         # Global configuration
├── httpclient.py     # HTTP request abstraction
├── cache.py          # Local cache handling
├── browser.py        # Network interaction logic
├── utils.py          # Shared utilities
├── search/           # Search engine modules
│   ├── base.py
│   ├── duckduckgo.py
│   └── localfile.py
├── filters/          # Result filtering modules
│   ├── base.py
│   ├── http_only.py
│   └── no_hsts.py
examples/
└── urls.txt
```

---

## Installation

BareNet currently requires Python 3.10 or newer.

```bash
git clone https://github.com/yourusername/barenet.git
cd barenet
pip install -r requirements.txt
```

---

## Usage

Run BareNet as a Python module:

```bash
python -m barenet [options]
```

Example (local file discovery):

```bash
python -m barenet --source localfile --input examples/urls.txt
```

Filters can be chained to refine results:

```bash
python -m barenet --source localfile --filter http_only --filter no_hsts
```

---

## Search Engines

Search engines define how endpoints are discovered. Each engine implements a common interface and can be enabled or disabled independently.

Current engines:

* **localfile** – Reads URLs from local files
* **duckduckgo** – Uses public search results (optional, controlled)

---

## Filters

Filters are applied after discovery to refine results based on protocol, headers, or security properties.

Examples:

* `http_only` – Keeps only HTTP endpoints
* `no_hsts` – Excludes endpoints enforcing HSTS

---

## Use Cases

* Network surface mapping
* Service exposure analysis
* Security research and experimentation
* Blue team reconnaissance
* Academic and lab-based studies

---

## Security & Ethics

BareNet is intended for **defensive security research**, controlled environments, and systems you own or are explicitly authorized to test.

Unauthorized scanning or reconnaissance of third-party systems may be illegal and unethical.

---

## Roadmap

* Additional local discovery engines
* Advanced protocol fingerprinting
* Improved caching strategies
* Export formats (JSON, CSV)
* Plugin system for third-party modules

---

## License

This project is released under the MIT License.

---

## Disclaimer

This tool is provided "as is" without warranty of any kind. The authors are not responsible for misuse or damages resulting from its use.

