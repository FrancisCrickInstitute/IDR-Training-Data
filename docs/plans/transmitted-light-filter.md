# Plan: keep transmitted-light channels out of the training data

**Status:** planned, not started (recorded 2026-11-26).

## Problem

The generator picks source and target channels at random. IDR channel metadata is often missing or inaccurate, so transmitted-light channels (brightfield, DIC, phase contrast) have ended up in earlier training data as source or target channels. Bleed-through between a fluorescence channel and a transmitted-light channel isn't the effect we are modelling, so these pairs are noise.

### What the metadata looks like (from test runs on 2026-11-26)

- OMERO channel names are often placeholders. For example, image 3260779 (idr0025) has channels named `0`, `1`, `2`.
- IDR often has an image-level `Channels` map annotation that is more informative, e.g. `ch00:DAPI(nuclei);ch01:Alexa 488(target protein);ch02:Alexa 555(microtubules)`.
- That annotation doesn't always match the channel order. For example, idr0033 lists 6 stains for a 5-channel image, because WGA and phalloidin share a channel.

## Approach: layered checks, cheapest first

Each check labels each channel `fluorescence`, `transmitted` or `unknown`, and records which check made the call.

1. **OMERO channel metadata other than the name.** `image.getChannels()` gives a `ChannelWrapper` for each channel, and `getLogicalChannel()` on it gives the following fields. Check the exact wrapper method names against omero-py before relying on them.
   - illumination: `Transmitted` vs `Epifluorescence`
   - contrast method: `Brightfield`, `DIC`, `Phase`, …
   - excitation and emission wavelengths. Transmitted light usually has no emission wavelength.
   - fluorophore (`getFluor()`)

   These fields are often empty on IDR, but when they are filled in they settle the question.
2. **Keyword search** of the channel names and of the `Channels` annotation (image level, then study level). Words to look for: brightfield, bright field, BF, DIC, Nomarski, phase, PC, trans, transmitted, TL. Match short tokens such as BF, PC and TL as whole words only, to avoid false positives.
   - If the annotation mentions transmitted light but can't be matched to a channel number reliably, skip the whole image (conservative).
3. **Pixel statistics**, which work when there is no metadata. Fluorescence has a dark background with a sparse bright tail. Transmitted light has a bright, fairly even background, with structure that deviates both up and down. Candidate measures (none depend on the stored pixel scale):
   - median / 99.9th percentile (high for transmitted light)
   - skewness (strongly positive for fluorescence; near zero or negative for transmitted light)
   - spread of the background, e.g. (p50 − p1) / (p99.9 − p1)
   - fraction of pixels near the background level

## What happens to a flagged channel

- A `transmitted` channel is never used as a source or target.
- If at least two `fluorescence` channels remain, pick the pair from those. Otherwise skip the image and pick another (this counts as an attempt under `MAX_ATTEMPTS_PER_IMAGE`).
- How `unknown` channels are handled is an open decision (see below).

## Logging

For each channel, add to `metadata.csv`:
- the label
- which check decided it (`omero_metadata` / `keyword` / `pixel_stats` / none)
- the pixel statistics
- the raw OMERO fields (illumination, contrast method, wavelengths)

This lets the rules be audited and adjusted after a run without re-downloading anything.

## Rollout

1. **Phase 1:** implement checks 1 and 2 and act on them. Compute and log the check-3 statistics, but don't act on them yet.
2. **Calibrate:**
   - Take channels labelled confidently by checks 1–2 (e.g. DAPI/Hoechst vs "Brightfield"/"DIC") as a free labelled set.
   - Add a small hand-labelled sample.
   - Choose check-3 cut-offs from the logged statistics.
3. **Phase 2:** turn check 3 on with those cut-offs.
4. **Existing data:** run the check-3 statistics over previously generated TIFFs to flag likely transmitted-light pairs for review or removal.
   - Source TIFFs show the source channel directly.
   - The mixed TIFF is target + bleed, so it mainly reflects the target channel.
   - The statistics don't depend on scale, so the older per-image normalisation doesn't matter.

## Open decisions (Dave)

- The cut-offs for the pixel statistics (to be set from calibration data, not guessed).
- How to treat `unknown` channels: reject them (cleaner data, fewer pairs) or accept them.
- Whether to drop flagged pairs from existing data automatically or only list them for review.

## Notes

- IDR currently has a known issue where many images fail to load (`ResourceError: Error instantiating pixel buffer`). It isn't related to this plan, but it limits how much test or calibration data can be fetched until it's fixed.
