# From Hard-VAD to Learned Speech-Presence Probability in Dynamic Acoustic Environments

**Course:** ELEC5305 Acoustics, Speech and Signal Processing  
**Student ID:** 550473662  
**Name:** Yuhuan Zhang  

## 🎯 Quick Links
- **Project Video Demonstration:** [Watch on YouTube](https://youtu.be/SrJ-hilw1kc)
- **Final Research Report:** [Download PDF](./ELEC5305_Final_Report_550473662.pdf)

## 📖 Project Overview
This project investigates the failure mechanics of traditional Power Spectral Density (PSD) tracking algorithms under severe non-stationary conditions, culminating in the development of a resilient data-driven alternative. 

By simulating an abrupt 10 dB to 0 dB SNR drop at 3.0 seconds, this project quantitatively evaluates three evolutionary approaches driving a common Decision-Directed Wiener backend:
1. **System A:** Baseline Hard-VAD (Suffers from threshold lock-up).
2. **System B:** Statistical Soft-SPP via IMCRA (Experiences 0.8s minimum-tracking lag).
3. **System C:** Proposed Lightweight Learned SPP via Multi-Frame Contextual Neural Network (Achieves instantaneous adaptation using scale-invariant features).

## 🚀 Key Achievements (Final Submission)
- [x] Synthesized a precise step-function dynamic noise dataset (10dB -> 0dB transition).
- [x] Implemented baseline Hard-VAD and observed the permanent algorithmic lock-up.
- [x] Implemented IMCRA and identified its sliding-window latency bottleneck.
- [x] Developed and trained the proposed System C (Lightweight MLP) on scale-invariant spectral contrast features.
- [x] Achieved a breakthrough **+3.63 dB Post-Transition SNR Improvement** in the extreme 0 dB region using System C, outperforming both baseline systems.
- [x] Completed the 12-page comprehensive final research report.

## 🛠️ How to Run the Code
1. Clone this repository to your local machine.
2. Open MATLAB.
3. Run `baseline_A_hard_vad.m` to observe the Hard-VAD lock-up.
4. Run `system_B_imcra.m` to visualize the minimum-tracking lag.
5. Run `system_C_learned_spp.m` to test the proposed neural approach.
6. Run `evaluate_metrics.m` to generate the final quantitative comparison table (Global MSE, SNR, and Post-3s Improvement).
