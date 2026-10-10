# A better model for Dhaka City

Start with [progress.md](progress.md), then read the [architecture and concepts guide](outputs/ARCHITECTURE.md). This workspace is shared by Codex and a teammate/other agent.

| Path | Purpose |
| --- | --- |
| `progress.md` | Shared status and session handoff; links to the canonical file in `outputs/` |
| `AGENTS.md` | Instructions for agents working in this workspace |
| `collaboration/codex/` | Codex working notes and detailed handoff |
| `collaboration/teammate/` | Teammate notes; teammate owns these files |
| `outputs/detection-pilot/` | Existing shared pilot code, frames, predictions, annotations and metrics |
| `outputs/claude/` | Claude deliverables, including plate-detector model-selection research |
| `outputs/teammate/` | Independent teammate deliverables before integration |
| `outputs/progress.md` | Canonical shared progress file; edit this or the root symlink, not two copies |
| `work/detection-pilot/` | Existing model weights, Python environment, diagnostics and logs |
| `work/teammate/` | Teammate scratch files |
| `work/video-tools/` | Local ffmpeg dependency |

The original video remains in Downloads. This workspace is a private Git repository. Model weights in `work/detection-pilot/` are committed (user-approved). Source videos, extracted frames, all JPG/PNG views, and the Python environment are git-ignored; do not commit footage-derived images without the user's explicit authorization. Ownership notes reduce accidental conflicts but do not provide locking; coordinate before editing the same files.

## Workspace location

Canonical location: `/Volumes/Projects/A better model for Dhaka City`. Open this directory for shared work. The previous Codex task directory is a compatibility symlink to this directory; it is not a second copy. The Python environment launchers still reference that compatibility path. Source footage remains in Downloads.
