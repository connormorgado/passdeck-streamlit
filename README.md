# PassDeck (Streamlit)

Satellite pass predictor with a sky-track plot, for small ops teams / cubesat operators.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

This opens the dashboard in your browser at `http://localhost:8501`.

## Notes

- Live TLEs are pulled from Celestrak by NORAD ID whenever you pick a preset or
  type an ID — no manual copy-pasting or CORS issues, since this runs server-side
  in Python instead of a browser.
- You can still paste a custom TLE directly if you're working with a satellite
  that isn't on Celestrak yet (e.g. a not-yet-cataloged cubesat).
- Orbit propagation uses SGP4 via the `skyfield` library — the same standard
  algorithm used by the JS version, just computed in Python.
