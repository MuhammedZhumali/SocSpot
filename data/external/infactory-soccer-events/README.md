---
version: 1.0.0
license: other
license_name: proprietary
license_link: LICENSE
task_categories:
  - video-classification
  - feature-extraction
language:
  - en
tags:
  - sports
  - soccer
  - football
  - video
  - events
  - action-recognition
annotations_creators:
  - machine-generated
pretty_name: Soccer (Football) In-Match Events
size_categories:
  - n<1K
---

# Soccer (Football) In-Match Events

A curated sample dataset of in-match soccer/football events, automatically identified using multi-modal models from Infactory. This dataset sample contains video clips and metadata for three key event types: **yellow cards**, **red cards**, and **goals**. Infactory models are capable of detecting corner kicks, blocked shots, saved shots, substitutions, and many more types of events in gameplay footage. 

## Dataset Description

This dataset provides 211 annotated video clips extracted from professional soccer broadcasts, totaling approximately **3.5 hours** (12,677 seconds) of footage and **190,000 frames**. Each sample includes:

- A video clip (MP4, H.264/AVC, 1280×720 @ 15 FPS)
- Rich metadata including teams, competition, and precise timing

Average clip duration is **60 seconds**, making this suitable for video classification and temporal action recognition tasks.

### Preview

<p align="center">
  <img src="previews/combined_preview.jpg?v=4" alt="Sample frames"/>
  <br/>
  <b>Sample event frames:</b> Goal | Yellow Card | Red Card
</p>

### Event Categories

| Category | Description | Samples |
|----------|-------------|---------|
| `yellow_card` | Referee shows yellow card to a player | 100 |
| `red_card` | Referee shows red card to a player | 11 |
| `goal` | Goal being scored | 100 |

### Source Data

- **Competition**: Serie A (Italian top-flight football)
- **Season**: 2024-25
- **Provider**: [Infront Italy](https://www.infront.sport/)
- **Broadcast Quality**: HD 720p (1280x720)

## Dataset Structure

```
infactory-ai/soccer-events/
├── README.md                    # Dataset card
├── metadata.csv                 # Index with all metadata
├── dataset_info.json            # Dataset statistics
└── data/
    ├── {uuid}.json              # Event metadata (211 files)
    └── {uuid}.mp4               # Video clips (211 files)
```

### Metadata Fields

| Field | Type | Description |
|-------|------|-------------|
| `asset_id` | string | Unique identifier (UUID) |
| `event_type` | string | Original event type name |
| `event_category` | string | Normalized: `yellow_card`, `red_card`, `goal` |
| `event_subtype` | string | Sub-category (e.g., "Header", "Penalty") |
| `team` | string | Team involved: "Home" or "Away" |
| `home_team` | string | Home team name |
| `away_team` | string | Away team name |
| `competition` | string | Competition name |
| `game_date` | string | Match date (YYYY-MM-DD) |
| `season` | string | Season (e.g., "2024-25") |
| `starttimecode` | string | Clip start (HH:MM:SS:FF) |
| `endtimecode` | string | Clip end (HH:MM:SS:FF) |
| `frame_rate` | string | Video frame rate |
| `duration_seconds` | float | Clip duration in seconds |

## Usage

### Loading Metadata

```python
import pandas as pd

df = pd.read_csv("hf://datasets/infactory-ai/soccer-events/metadata.csv")

# Filter by event type
goals = df[df["event_category"] == "goal"]
red_cards = df[df["event_category"] == "red_card"]
yellow_cards = df[df["event_category"] == "yellow_card"]

print(f"Goals: {len(goals)}, Red Cards: {len(red_cards)}, Yellow Cards: {len(yellow_cards)}")
```

### Downloading and Playing Videos

```python
from huggingface_hub import hf_hub_download
import cv2

# Download a specific video
video_path = hf_hub_download(
    repo_id="infactory-ai/soccer-events",
    filename="data/0293e19b-7281-4f97-b2e9-57b3cee260b2.mp4",
    repo_type="dataset"
)

# Load with OpenCV
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {fps} FPS, {frame_count} frames, {frame_count/fps:.1f}s duration")

# Read frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Process frame...
cap.release()
```

### Extracting Frames for Training

```python
from huggingface_hub import hf_hub_download
import cv2
import pandas as pd

# Load metadata
df = pd.read_csv("hf://datasets/infactory-ai/soccer-events/metadata.csv")

# Download and extract frames from a goal video
row = df[df["event_category"] == "goal"].iloc[0]
video_path = hf_hub_download(
    repo_id="infactory-ai/soccer-events",
    filename=f"data/{row['mp4_file']}",
    repo_type="dataset"
)

cap = cv2.VideoCapture(video_path)
frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame_rgb)
cap.release()

print(f"Extracted {len(frames)} frames from {row['event_type']} event")
```

### Using with PyTorch DataLoader

```python
import torch
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import hf_hub_download
import pandas as pd
import cv2

class SoccerEventsDataset(Dataset):
    def __init__(self, event_category=None):
        self.df = pd.read_csv("hf://datasets/infactory-ai/soccer-events/metadata.csv")
        if event_category:
            self.df = self.df[self.df["event_category"] == event_category]
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = hf_hub_download(
            repo_id="infactory-ai/soccer-events",
            filename=f"data/{row['mp4_file']}",
            repo_type="dataset"
        )
        # Load video frames as tensor...
        return {"video_path": video_path, "label": row["event_category"]}

# Create dataloader for goals only
dataset = SoccerEventsDataset(event_category="goal")
loader = DataLoader(dataset, batch_size=4, shuffle=True)
```

## License

This dataset contains proprietary content from [Infront Italy](https://www.infront.sport/). Usage is restricted to non-commercial research purposes. For commercial and large-scale needs, contact Infactory.

## Citation

```bibtex
@dataset{soccer_events_2026,
  title={Soccer (Football) In-Match Events},
  author={John Kanalakis, Valentino Constantinou},
  year={2026},
  publisher={Infactory},
  url={https://huggingface.co/datasets/infactory-ai/soccer-events}
}
```
