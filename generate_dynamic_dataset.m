% generate_dynamic_dataset.m
% Milestone 2: Generate dynamic noise mixture (Level change at t = 3s)
clear; clc; close all;

% Configuration: File paths
clean_file = fullfile('DatasetClean_SpeechDev', 'dev_spk1_01.flac');
fan_file = fullfile('DatasetNoiseStationary', 'fan_noise.wav');
output_dir = 'DatasetMixed';
if ~exist(output_dir, 'dir')
    mkdir(output_dir); 
end

% 1. Read clean speech
[clean, fs] = audioread(clean_file);
clean = clean(:,1);
N = length(clean);

% 2. Set transition time to 3 seconds
trans_time = 3.0; 
trans_idx = round(trans_time * fs);

% Ensure speech is longer than 3s; otherwise, transition at the midpoint
if trans_idx >= N
    trans_idx = round(N / 2);
end

% 3. Read and process stationary fan noise
[fan, fs_fan] = audioread(fan_file);
if fs_fan ~= fs
    fan = resample(fan, fs, fs_fan);
end
fan = fan(:,1);

% Concatenate noise if it is shorter than the speech signal
while length(fan) < N
    fan = [fan; fan];
end
fan = fan(1:N);

% 4. Calculate scaling factors for the two segments (Pre-transition: 10dB, Post-transition: 0dB)
snr1 = 10; 
snr2 = 0;

speech_pow1 = sum(clean(1:trans_idx).^2) / trans_idx;
speech_pow2 = sum(clean(trans_idx+1:end).^2) / (N - trans_idx);
fan_pow1 = sum(fan(1:trans_idx).^2) / trans_idx;
fan_pow2 = sum(fan(trans_idx+1:end).^2) / (N - trans_idx);

scale1 = sqrt((speech_pow1 / (10^(snr1/10))) / (fan_pow1 + eps));
scale2 = sqrt((speech_pow2 / (10^(snr2/10))) / (fan_pow2 + eps));

% Construct the dynamic noise profile
dynamic_noise = zeros(N, 1);
dynamic_noise(1:trans_idx) = fan(1:trans_idx) * scale1;
dynamic_noise(trans_idx+1:end) = fan(trans_idx+1:end) * scale2;

% 5. Mix clean speech with dynamic noise and save
mixed_dynamic = clean + dynamic_noise;
out_file = fullfile(output_dir, 'dev_spk1_01_dynamic_10dB_to_0dB.wav');
audiowrite(out_file, mixed_dynamic, fs);

disp(['Dynamic noise mixture generated: ', out_file]);

% 6. Plot the construction of dynamic noise for verification
time_axis = (0:N-1)/fs;
figure('Name', 'Dynamic Noise Construction');
subplot(2,1,1);
plot(time_axis, clean); 
title('Clean Speech'); xlabel('Time (s)'); ylabel('Amplitude');

subplot(2,1,2);
plot(time_axis, dynamic_noise, 'r'); 
title('Dynamic Fan Noise (10dB -> 0dB at 3s)');
xlabel('Time (s)'); ylabel('Amplitude');
xline(time_axis(trans_idx), 'k--', 'LineWidth', 2); % Draw transition boundary