# Shared roadside plate detection workspace

Read `progress.md` before starting work. It is the shared handoff record.

## Session protocol
- At session start, read the current scope, ownership table, and pending work in `progress.md`.
- Claim a bounded task and its files in the ownership table before editing. Do not overwrite a teammate's active files. Claims are coordination notes, not automatic filesystem locks.
- Use `collaboration/codex/` and `collaboration/teammate/` for separate working notes; use `work/<owner>/` for new scratch work and `outputs/<owner>/` for new independent deliverables.
- Before ending each session, update your own ownership row, completed work, exact resume steps, checks/results, and blockers in `progress.md`. Preserve the other contributor's notes. Re-read immediately before writing to avoid overwriting recent edits.
- Keep existing dataset, model, script, and output paths stable. Do not move or regenerate shared artifacts without checking ownership and downstream references.
- Only one contributor should edit the shared progress file at a time; keep updates short. If a collision occurs, merge both updates rather than discarding either.

## Current scientific constraints
- Current task: number-plate detection only.
- Per-file exclusion, not a general rule: for the current source video `2025-12-31-111331_2025-12-31-131239_1000004$1$0$0.mp4` only, skip the first 900 seconds (a protest is recorded there). Other footage has no default skip; record any exclusion per file.
- Rickshaws (COCO `bicycle` proposals) are kept and tracked as vehicles, labelled `rickshaw`, with no BRTA registration plate expected. Do not report a registration plate for a rickshaw; municipal rickshaw tags are not registration plates.
- Report precision and recall, matching IoU and confidence threshold, evaluation region and sample size. Do not describe detection recall as OCR accuracy.
- Existing annotations are provisional agent-drawn labels, not independently adjudicated human ground truth.
- The six fresh timestamps use the same camera/session and may repeat vehicles. They are not an independent vehicle or camera holdout.
- Hybrid settings in `outputs/detection-pilot/check_protocol.json` are frozen for that check. Do not tune on the check sample and call it held-out performance.
- No 95% claim without supporting evaluation. No tracker or violation implementation claims from the current detection run.
- Keep source footage local unless the user explicitly authorizes uploading it.
