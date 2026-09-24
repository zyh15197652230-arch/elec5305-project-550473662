# From VAD to Learned Speech-Presence Probability for Adaptive Speech Enhancement

**Course:** ELEC5305
**Student ID:** 550473662

## Project Overview
This project investigates the critical challenge of updating noise Power Spectral Density (PSD) during continuous speech in non-stationary acoustic environments. Conventional enhancement systems rely on hard Voice Activity Detection (VAD) gating, which often fails to update the noise model when the background environment changes rapidly beneath active speech. 

To address this, this project compares three generations of speech-presence estimation techniques driving a common Wiener enhancement backend:
1. **System A:** Classical Hard frame-level VAD (baseline).
2. **System B:** Statistical soft Speech-Presence Probability (IMCRA).
3. **System C:** Modern Learned SPP (Pretrained TCN model).

By evaluating PSD estimation error, adaptation time, SI-SDR, and STOI across controlled dynamic noise transitions, this project aims to quantify the trade-offs between estimation accuracy, robustness in unseen noise, and computational complexity.

## Current Progress (Preliminary Submission)
- [x] Initialized MATLAB project structure.
- [x] Implemented Baseline System A (Hard-VAD Wiener Filter).
- [x] Developed custom scripts to synthesize controlled dynamic noise mixtures (e.g., precise SNR level transitions at specific timestamps).
- [x] Visualized critical failure cases of Hard VAD in tracking dynamic noise transitions.
