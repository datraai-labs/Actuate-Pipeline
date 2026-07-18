# Actuate examples

Each script is self-contained and runnable. Start here:

```bash
pip install -e ".[all]"
python examples/01_process_local_video.py      # runs on the bundled session_001 capture
```

| Script | What it shows | Needs |
|---|---|---|
| `01_process_local_video.py` | local video → LeRobot v3 | the bundled `processed/session_001` |
| `02_process_huggingface.py` | `hf://` dataset → processed | network (downloads `lerobot/pusht`) |
| `03_process_stereo.py` | stereo video, auto rig-detect | your own side-by-side clip |
| `04_batch_processing.py` | a folder of videos → dataset | a `./my_videos/` folder |
| `05_custom_embodiment.py` | process for a robot embodiment | the bundled capture |
| `06_compare_depth_models.py` | reach the per-frame depth | the bundled capture |
| `07_inspect_quality.py` | read the L4 certificate | the bundled capture |

Scripts 01, 05, 06, 07 run against the checked-in `processed/session_001` with no extra
inputs. 02 downloads a small public dataset. 03/04 need your own video(s).

Perception is GPU-heavy — these cap at `max_frames` 20–45 for a fast run on a 4 GB card.
