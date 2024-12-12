<!-- markdownlint-disable MD029 -->

# Web App for NBA Player Data Exploration

- hosted on [streamlit](https://nba-player-stats.streamlit.app).

## Development

1. setup and activate python virtual environment. For example,

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. install dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

3. install pre-commit hook

```bash
pre-commit install
```

3. run app

```bash
streamlit run ./src/app.py
```
