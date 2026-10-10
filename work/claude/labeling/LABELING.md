# Labelling guide: plates for the detector fine-tune

Open the labeller from the Claude app (launch config `plate-labeller`), or run it yourself:

```bash
python3 "work/claude/labeling/server.py"
```

Then open http://localhost:8765 in a browser. Frames stay on this machine.

## What to label
- **Every physical registration plate you can identify**, including blurry or unreadable ones, on any motor vehicle. Draw a tight box around the plate only, not the bumper.
- Set the vehicle type with keys `1`–`5`: car/SUV/microbus, motorcycle, **CNG**, bus/truck/pickup, other (plate mode) or rickshaw (vehicle mode). Legunas don't run on this route, so there is no leguna class. These tags let us measure recall per vehicle type; the detector itself learns a single `plate` class.
- **Ignore (`I`)** anything you are unsure about: a plate too far or small to tell, heavily occluded or cut off by the frame edge, dealer placeholders, painted duplicate numbers, and **rickshaw municipal tags**. Ignored regions count as neither right nor wrong.
- **Not plates** (delete the suggestion): signs, shop boards, bus route text, the painted numbers on CNG bumpers, stickers.
- Rickshaws are vehicles with **no registration plate**. They appear with a pink box marked "rickshaw · no registration".

## Workflow
1. Dashed boxes are model suggestions. **Everything on screen when you save is saved as truth**, so delete the wrong ones.
2. Press `Tab` to jump box by box (it zooms in). Fix each box, then draw any plates the models missed.
3. Press `Enter` to save and go to the next frame. Use `Skip` for an unusable frame.
4. Frames are ordered so that any prefix covers the whole time range. 30–40 frames already makes a useful first fine-tune; all 119 is better.
5. The 9 purple-outlined frames (≥1950 s) are the **locked test**. Label them, but they are never used for training.

## Then train
```bash
work/claude/kaggle/run_training.sh
```
