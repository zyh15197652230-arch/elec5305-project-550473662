% baseline_A_hard_vad.m
% System A: Baseline Hard-VAD Wiener Enhancement
% This script implements a classical Wiener filter driven by a hard decision 
% Voice Activity Detector (VAD) for noise Power Spectral Density (PSD) estimation.

clear; clc; close all;

%% 1. Parameter Initialization
% File paths (adapted to your actual folder names)
inputFile = fullfile('DatasetMixed', 'dev_spk1_01_dynamic_10dB_to_0dB.wav'); 
outputFile = fullfile('DatasetMixed', 'dev_spk1_01_dynamic_10dB_to_0dB_enhanced.wav');

% Read noisy audio
[noisy_sig, fs] = audioread(inputFile);
noisy_sig = noisy_sig(:, 1); 
sig_len = length(noisy_sig);

% STFT Parameters (Standard for 16kHz audio)
win_len = 512;               % Window length (32ms)
hop_size = 256;              % Hop size (16ms, 50% overlap)
nfft = 512;                  % FFT points
win = hanning(win_len);      % Analysis window

% VAD & Tracking Parameters
alpha_dd = 0.98;             % Smoothing factor for Decision-Directed a priori SNR
noise_update_alpha = 0.90;   % Smoothing factor for noise PSD update
vad_threshold = 1.5;         % Hard VAD energy threshold multiplier (tune based on dev set)

%% 2. Framing and Initialization
num_frames = floor((sig_len - win_len) / hop_size) + 1;
half_bin = nfft / 2 + 1;

% Pre-allocate memory
enhanced_frames = zeros(half_bin, num_frames);
enhanced_sig = zeros(sig_len, 1);

% Initialize Noise PSD (Assume first 5 frames are noise-only)
init_noise_frames = min(5, num_frames);
noise_psd = zeros(half_bin, 1);
for i = 1:init_noise_frames
    idx = (i-1)*hop_size + 1;
    frame = noisy_sig(idx : idx+win_len-1) .* win;
    frame_fft = fft(frame, nfft);
    noise_psd = noise_psd + (abs(frame_fft(1:half_bin)).^2) / init_noise_frames;
end

% Trackers
prev_enhanced_mag = zeros(half_bin, 1);
vad_decision = zeros(num_frames, 1);

%% 3. Frame-by-Frame Processing (Hard VAD + Wiener Filter)
for m = 1:num_frames
    % Extract current frame and apply window
    start_idx = (m-1)*hop_size + 1;
    frame = noisy_sig(start_idx : start_idx+win_len-1) .* win;
    
    % STFT
    frame_fft = fft(frame, nfft);
    noisy_mag = abs(frame_fft(1:half_bin));
    noisy_phase = angle(frame_fft(1:half_bin));
    noisy_psd = noisy_mag.^2;
    
    % --- Hard VAD Decision ---
    % Calculate frame energy
    frame_energy = sum(noisy_psd);
    noise_energy = sum(noise_psd);
    
    % Binary decision: 1 if speech present, 0 if noise only
    if frame_energy > vad_threshold * noise_energy
        vad_decision(m) = 1; % Speech
    else
        vad_decision(m) = 0; % Pause (Noise-only)
    end
    
    % --- Noise PSD Update ---
    % Update noise estimate ONLY when VAD detects a pause
    if vad_decision(m) == 0
        noise_psd = noise_update_alpha * noise_psd + (1 - noise_update_alpha) * noisy_psd;
    end
    
    % --- SNR Estimation ---
    % A posteriori SNR
    gamma = noisy_psd ./ noise_psd;
    
    % A priori SNR (Decision-Directed approach)
    if m == 1
        xi = max(gamma - 1, 0);
    else
        xi = alpha_dd * (prev_enhanced_mag.^2 ./ noise_psd) + (1 - alpha_dd) * max(gamma - 1, 0);
    end
    % Limit a priori SNR to prevent musical noise
    xi = max(xi, 10^(-25/10)); 
    
    % --- Wiener Gain Calculation ---
    wiener_gain = xi ./ (1 + xi);
    
    % Apply Gain
    enhanced_mag = wiener_gain .* noisy_mag;
    prev_enhanced_mag = enhanced_mag;
    
    % Reconstruct positive half of spectrum
    enhanced_frames(:, m) = enhanced_mag .* exp(1j * noisy_phase);
end

%% 4. ISTFT (Overlap-Add)
for m = 1:num_frames
    start_idx = (m-1)*hop_size + 1;
    
    % Mirror spectrum for IFFT
    half_spec = enhanced_frames(:, m);
    full_spec = [half_spec; conj(half_spec(end-1:-1:2))];
    
    % IFFT
    time_frame = real(ifft(full_spec, nfft));
    
    % Overlap-add (Apply synthesis window to maintain energy)
    enhanced_sig(start_idx : start_idx+win_len-1) = ...
        enhanced_sig(start_idx : start_idx+win_len-1) + time_frame(1:win_len) .* win;
end

% Normalize to prevent clipping
enhanced_sig = enhanced_sig / max(abs(enhanced_sig));

% Save output
audiowrite(outputFile, enhanced_sig, fs);
disp(['Enhancement complete. File saved to: ', outputFile]);

%% 5. Quick Visualization
figure('Name', 'System A Baseline Results');
subplot(3,1,1);
plot((1:sig_len)/fs, noisy_sig);
title('Noisy Speech (5dB SNR)'); xlabel('Time (s)'); ylabel('Amplitude');

subplot(3,1,2);
plot((1:num_frames) * (hop_size/fs), vad_decision, 'r', 'LineWidth', 1.5);
title('Hard VAD Decision (1=Speech, 0=Noise)'); xlabel('Time (s)'); ylim([-0.2 1.2]);

subplot(3,1,3);
plot((1:sig_len)/fs, enhanced_sig);
title('Enhanced Speech (Wiener Filter)'); xlabel('Time (s)'); ylabel('Amplitude');