# Frontend images

Two images ship with the viewer and are referenced by the shell. Both are missing from
this checkout; `test_web_assets.py` fails until they are added here.

| file              | referenced by      | what it is                                          |
| ----------------- | ------------------ | --------------------------------------------------- |
| `aicor-logo.png`  | `index.html`       | the AICOR logo in the top bar, light/inverted artwork so it sits on the dark bar |
| `ai-picture.png`  | `app.css` (`.stage-bg`) | the lab photo used as the blurred 3D-stage backdrop |

Referencing an image that is not here fails silently in the browser, which is why the
asset test checks every `<img src>` and every CSS `url(...)`.
