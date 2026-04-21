import csv
import json
import os
from pathlib import Path

from export_json import export_checkpoint_to_json


ROOT = Path(__file__).resolve().parent.parent
RATINGS_CSV = ROOT / "results" / "eval_reports" / "stochastic" / "ratings.csv"
ASSETS_DIR = ROOT / "docs" / "singleplayer" / "assets"
BOT_DIR = ASSETS_DIR / "bots"
MANIFEST_PATH = ASSETS_DIR / "bots_manifest.json"
DEFAULT_WEIGHTS_PATH = ASSETS_DIR / "model_weights.json"


BOT_SPECS = [
    {
        "id": "rookie_1621",
        "name": "Rookie",
        "checkpoint": ROOT / "checkpoints" / "league_stable_seeded" / "checkpoint_5000000.pth",
    },
    {
        "id": "club_1661",
        "name": "Club",
        "checkpoint": ROOT / "checkpoints" / "league_stable_seeded" / "checkpoint_6000000.pth",
    },
    {
        "id": "ranked_1723",
        "name": "Ranked",
        "checkpoint": ROOT / "checkpoints" / "league_stable_seeded" / "checkpoint_1000000.pth",
    },
    {
        "id": "elite_1772",
        "name": "Elite",
        "checkpoint": ROOT / "checkpoints" / "league_stable_seeded" / "checkpoint_8000000.pth",
    },
    {
        "id": "master_1836",
        "name": "Master",
        "checkpoint": ROOT / "checkpoints" / "league_stable_seeded" / "checkpoint_11000000.pth",
    },
    {
        "id": "champion_1953",
        "name": "Champion",
        "checkpoint": ROOT / "checkpoints" / "league_stable_seeded" / "checkpoint_29000000.pth",
    },
    {
        "id": "veteran_1798",
        "name": "Veteran",
        "checkpoint": ROOT / "checkpoints" / "league_stable_curated_run1" / "checkpoint_22000000.pth",
    },
    {
        "id": "grandmaster_1902",
        "name": "Grandmaster",
        "checkpoint": ROOT / "checkpoints" / "league_stable_curated_run1" / "checkpoint_35000000.pth",
    },
    {
        "id": "titan_2006",
        "name": "Titan",
        "checkpoint": ROOT / "checkpoints" / "league_stable_curated_run1" / "checkpoint_47000000.pth",
    },
    {
        "id": "legend_2036",
        "name": "Legend",
        "checkpoint": ROOT / "checkpoints" / "league_stable_curated_run1" / "checkpoint_54000000.pth",
    },
]


def load_ratings():
    ratings = {}
    with RATINGS_CSV.open() as f:
        for row in csv.DictReader(f):
            ratings[row["model_key"]] = row
    return ratings


def main():
    ratings = load_ratings()
    BOT_DIR.mkdir(parents=True, exist_ok=True)

    bots = []
    for spec in BOT_SPECS:
        checkpoint = spec["checkpoint"].resolve()
        ratings_row = ratings.get(str(checkpoint))
        if ratings_row is None:
            raise KeyError(f"No rating found for checkpoint: {checkpoint}")

        output_rel = Path("assets") / "bots" / f"{spec['id']}.json"
        output_path = ASSETS_DIR / "bots" / f"{spec['id']}.json"
        export_checkpoint_to_json(str(checkpoint), str(output_path))

        bot_entry = {
            "id": spec["id"],
            "name": spec["name"],
            "display_label": f"BT {float(ratings_row['bt_rating']):.0f}",
            "rating_system": "bt_rating",
            "rating": float(ratings_row["bt_rating"]),
            "conservative_rating": float(ratings_row["bt_conservative_rating"]),
            "elo": float(ratings_row["elo"]),
            "weights_path": output_rel.as_posix(),
            "source_label": ratings_row["label"],
        }
        bots.append(bot_entry)

    bots.sort(key=lambda bot: bot["rating"])
    manifest = {
        "default_bot": "legend_2036",
        "bots": bots,
    }

    with MANIFEST_PATH.open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    default_bot = next(bot for bot in bots if bot["id"] == manifest["default_bot"])
    export_checkpoint_to_json(
        str(next(spec["checkpoint"] for spec in BOT_SPECS if spec["id"] == manifest["default_bot"]).resolve()),
        str(DEFAULT_WEIGHTS_PATH),
    )
    print(f"Wrote bot manifest to {MANIFEST_PATH}")
    print(f"Default bot: {default_bot['name']} ({default_bot['rating']:.2f})")


if __name__ == "__main__":
    main()
