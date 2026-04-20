# Web Deployment

The GitHub Pages site is served from `docs/`.

## Layout

- `docs/index.html`
  - Landing page that links to the playable modes.
- `docs/twoplayer/`
  - Source of truth for the two-player browser game.
  - `index.html` is the pygbag loader page served by GitHub Pages.
  - `main.py`, `pygbag.json`, and `assets/` are the tracked app sources that get packed into `penfootballweb.apk`.
- `docs/singleplayer/`
  - Source of truth for the single-player browser game.
  - `main.py`, `pen_football_web.py`, and `assets/` are packed into `build/web/singleplayer.apk`.
  - `index.html` is the AI selection page shown before launching the game.
- `docs/*/*.apk`
  - Build artifacts consumed by the pygbag loader pages.

## Rebuilding

Run from the repo root:

```bash
python docs/build_web_bundle.py twoplayer
python docs/build_web_bundle.py singleplayer
python docs/build_web_bundle.py all
```

`singleplayer` also refreshes the duplicate `penfootball_singleplayer.apk` copy in `docs/singleplayer/build/web/`.

## Local Testing

Do not open the HTML files directly with `file://...` for local testing.
The pygbag pages fetch `.apk` bundles and browser assets, and many browsers block that when the page is opened from disk.

Serve `docs/` over HTTP instead:

```bash
python docs/serve_docs.py --open
```

Or use the standard library server:

```bash
python -m http.server 8000 --directory docs
```

Then open `http://127.0.0.1:8000/`.

## Updating The AI Opponent

1. Export a checkpoint to browser weights:

```bash
python main/export_json.py --checkpoint ... --output docs/singleplayer/assets/model_weights.json
```

2. Rebuild the single-player bundle:

```bash
python docs/build_web_bundle.py singleplayer
```

To refresh the curated multi-bot picker:

```bash
python main/export_web_bots.py
python docs/build_web_bundle.py singleplayer
```

## Notes

- The two-player source was extracted from the original `penfootballweb.apk` so it is now editable in plain files.
- The loader HTML stays checked in separately from the APK so we can tweak the web page without unpacking artifacts.
