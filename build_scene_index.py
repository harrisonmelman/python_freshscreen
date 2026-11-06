"""Build a single HTML page of links from a folder of display JSONs.

Each display JSON holds an object_displayName and a view_neuroglancerURL. The
links are embedded in the page, so it needs no server: open it directly, attach
it to an email, or put it on a shared drive.

    python build_scene_index.py path/to/json_dir
    python build_scene_index.py path/to/json_dir -o shared/scenes.html --title "New scenes"
"""
import argparse
import html
import json
import sys
from pathlib import Path

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: system-ui, sans-serif; line-height: 1.4; max-width: 44rem; margin: 2rem auto; padding: 0 1rem; }}
  h1 {{ font-size: 1.5rem; margin-bottom: 0.25rem; }}
  p {{ margin-top: 0; opacity: 0.7; }}
  ul {{ list-style: none; padding: 0; }}
  li {{ padding: 0.6rem 0; border-top: 1px solid color-mix(in srgb, currentColor 20%, transparent); }}
  a {{ font-size: 1.05rem; }}
  small {{ display: block; opacity: 0.7; overflow-wrap: anywhere; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p>{count} scene{plural}. Each link opens in Neuroglancer.</p>
<ul>
{items}
</ul>
</body>
</html>
"""

ITEM = '<li><a href="{url}" target="_blank" rel="noopener">{name}</a><small>{file}</small></li>'


def load_scenes(json_dir):
    """Return the display JSONs in json_dir sorted by name, warning about any other file."""
    scenes = []
    for path in sorted(Path(json_dir).glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            scenes.append({
                "name": data["object_displayName"],
                "url": data["view_neuroglancerURL"],
                "file": path.name,
            })
        except (ValueError, KeyError, TypeError) as e:
            print(f"Skipping {path.name}: not a display JSON ({e!r})", file=sys.stderr)
    return sorted(scenes, key=lambda scene: scene["name"].lower())


def render_page(scenes, title):
    items = "\n".join(
        ITEM.format(**{key: html.escape(value) for key, value in scene.items()})
        for scene in scenes
    )
    return PAGE.format(
        title=html.escape(title),
        count=len(scenes),
        plural="" if len(scenes) == 1 else "s",
        items=items,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("json_dir", help="folder containing display JSONs")
    parser.add_argument("-o", "--output", default="scene_index.html", help="HTML file to write")
    parser.add_argument("--title", default="Neuroglancer scenes", help="page heading")
    args = parser.parse_args()

    scenes = load_scenes(args.json_dir)
    if not scenes:
        sys.exit(f"No display JSONs found in {args.json_dir}")

    Path(args.output).write_text(render_page(scenes, args.title), encoding="utf-8")
    print(f"Wrote {len(scenes)} scenes to {args.output}")


if __name__ == "__main__":
    main()
