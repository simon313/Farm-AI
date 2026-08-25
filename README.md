# Farm AI System

An AI-powered farm intelligence platform that watches a woods-raised pig farm through cameras and sensors, scores captured moments through an editorial filter, and assembles them into content.

## ⚠️ Important: Keep Your Configuration Private

This repository is **open source**, but your **configuration files contain sensitive information** and should **never be committed to version control**. This includes:

- **`config/cameras.yaml`** — contains camera IP addresses and network details
- **`config/animals.yaml`** — contains your farm's animal roster
- **`config/farm.yaml`** — contains operational configuration
- **`.env`** — contains API keys and passwords

### Setup Instructions

1. **Clone this repository to your own private location:**
   ```bash
   git clone https://github.com/simon313/Farm-AI.git
   cd Farm-AI
   git remote set-url origin <your-private-repo-url>
   ```

2. **Create your private configuration files from the examples:**
   ```bash
   cp config/cameras.yaml.example config/cameras.yaml
   cp config/animals.yaml.example config/animals.yaml
   cp config/farm.yaml.example config/farm.yaml
   cp config/.env.example .env
   ```

3. **Edit each file with your actual values:**
   - `config/cameras.yaml` — your camera IPs, ports, and labels
   - `config/animals.yaml` — your farm's animals
   - `config/farm.yaml` — your farm identity and operational settings
   - `.env` — your API keys and camera credentials

4. **Verify `.gitignore` protects your files:**
   The repository includes `.gitignore` rules that prevent these files from being accidentally committed:
   ```
   config/cameras.yaml
   config/animals.yaml
   config/farm.yaml
   .env
   ```

5. **Push only to your private repository:**
   ```bash
   git add -A
   git commit -m "Initial setup with private configuration"
   git push origin main
   ```

## Architecture

The system runs a four-gate editorial pipeline on video clips from farm cameras:

- **Gate 1** — Technical quality check (brightness, sharpness, shake)
- **Gate 2** — Motion detection and activity level
- **Gate 3** — Emotional register scoring via Claude vision AI
- **Gate 4** — Farm vibe scoring and context analysis

Clips that pass all gates are stored as `Moment` objects in the SQLite database.

## Installation

```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

## Running

```bash
python main.py
```

The system will:
1. Load your configuration from `config/*.yaml`
2. Start camera streams via RTSP
3. Run motion detection and editorial gates
4. Serve a dashboard on `http://localhost:8000`

## Environment Variables

Set these in your `.env` file (see `.env.example`):

- `CAMERA_USERNAME` — Username for RTSP camera access (default: `admin`)
- `CAMERA_PASSWORD` — Password for RTSP camera access
- `ANTHROPIC_API_KEY` — Your Claude API key for AI vision gates

## Testing

```bash
pytest
```

Tests are mocked — no actual API keys or cameras needed.

## Security Notes

- **Never commit `.env` or `config/` YAML files** to any public or shared repository
- Always use environment variables for secrets
- If you accidentally commit sensitive data, rotate your credentials immediately and use `git filter-branch` or `BFG` to remove it from history
- Keep your private fork synchronized with upstream security updates

## License

[Add your license here]
