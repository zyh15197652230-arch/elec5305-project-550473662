%% ELEC5305 Project: Comprehensive Quantitative Evaluation (Systems A, B, and C)
% Calculates Global MSE, Global SNR, and Post-Transition SNR (t > 3s)
% Student ID: 550473662

clc; clear; close all;

%% 1. Load Audio Files
clean_file = fullfile('DatasetClean_SpeechDev', 'dev_spk1_01.flac');
noisy_file = fullfile('DatasetMixed', 'dev_spk1_01_dynamic_10dB_to_0dB.wav');
sys_A_file = fullfile('DatasetMixed', 'dev_spk1_01_dynamic_10dB_to_0dB_enhanced.wav');
sys_B_file = 'enhanced_system_B_imcra.wav';
sys_C_file = 'enhanced_system_C_learned.wav';

[x_clean, fs] = audioread(clean_file);
[x_noisy, ~]  = audioread(noisy_file);
[x_sysA, ~]   = audioread(sys_A_file);
[x_sysB, ~]   = audioread(sys_B_file);
[x_sysC, ~]   = audioread(sys_C_file);

% Ensure mono and align lengths
min_len = min([length(x_clean), length(x_noisy), length(x_sysA), length(x_sysB), length(x_sysC)]);
x_clean = x_clean(1:min_len, 1);
x_noisy = x_noisy(1:min_len, 1);
x_sysA  = x_sysA(1:min_len, 1);
x_sysB  = x_sysB(1:min_len, 1);
x_sysC  = x_sysC(1:min_len, 1);

%% 2. Optimal Least-Squares Scale Alignment (Standard for Speech Enhancement Evaluation)
% Removes arbitrary volume scaling differences introduced by audiowrite normalization
align_scale = @(est, ref) est * ((ref' * est) / (est' * est + eps));

x_sysA_aligned = align_scale(x_sysA, x_clean);
x_sysB_aligned = align_scale(x_sysB, x_clean);
x_sysC_aligned = align_scale(x_sysC, x_clean);

%% 3. Compute Global Metrics (Full Utterance: 0s to End)
mse_noisy = mean((x_clean - x_noisy).^2);
mse_sysA  = mean((x_clean - x_sysA_aligned).^2);
mse_sysB  = mean((x_clean - x_sysB_aligned).^2);
mse_sysC  = mean((x_clean - x_sysC_aligned).^2);

snr_noisy = 10 * log10(sum(x_clean.^2) / sum((x_clean - x_noisy).^2));
snr_sysA  = 10 * log10(sum(x_clean.^2) / sum((x_clean - x_sysA_aligned).^2));
snr_sysB  = 10 * log10(sum(x_clean.^2) / sum((x_clean - x_sysB_aligned).^2));
snr_sysC  = 10 * log10(sum(x_clean.^2) / sum((x_clean - x_sysC_aligned).^2));

%% 4. Compute Post-Transition Metrics (t > 3.0s, 0dB Abrupt Noise Region)
idx_post = round(3.0 * fs) + 1 : min_len;
c_post   = x_clean(idx_post);

snr_post_noisy = 10 * log10(sum(c_post.^2) / sum((c_post - x_noisy(idx_post)).^2));
snr_post_sysA  = 10 * log10(sum(c_post.^2) / sum((c_post - x_sysA_aligned(idx_post)).^2));
snr_post_sysB  = 10 * log10(sum(c_post.^2) / sum((c_post - x_sysB_aligned(idx_post)).^2));
snr_post_sysC  = 10 * log10(sum(c_post.^2) / sum((c_post - x_sysC_aligned(idx_post)).^2));

%% 5. Display Formatted Table in Command Window
fprintf('\n================================================================================\n');
fprintf('             ELEC5305 Final Quantitative Evaluation (Systems A, B, C)           \n');
fprintf('================================================================================\n');
fprintf('Metric               | Noisy Input | System A (Hard) | System B (IMCRA) | System C (Learned)\n');
fprintf('---------------------|-------------|-----------------|------------------|-------------------\n');
fprintf('Global MSE (x10^-3)  | %11.4f | %15.4f | %16.4f | %17.4f\n', ...
    mse_noisy*1000, mse_sysA*1000, mse_sysB*1000, mse_sysC*1000);
fprintf('Global Output SNR(dB)| %11.4f | %15.4f | %16.4f | %17.4f\n', ...
    snr_noisy, snr_sysA, snr_sysB, snr_sysC);
fprintf('Global SNR Improv.   | %11s | %12.4f dB | %13.4f dB | %14.4f dB\n', ...
    'N/A', snr_sysA - snr_noisy, snr_sysB - snr_noisy, snr_sysC - snr_noisy);
fprintf('Post-3s SNR (0dB Reg)| %11.4f | %15.4f | %16.4f | %17.4f\n', ...
    snr_post_noisy, snr_post_sysA, snr_post_sysB, snr_post_sysC);
fprintf('Post-3s SNR Improv.  | %11s | %12.4f dB | %13.4f dB | %14.4f dB\n', ...
    'N/A', snr_post_sysA - snr_post_noisy, snr_post_sysB - snr_post_noisy, snr_post_sysC - snr_post_noisy);
fprintf('================================================================================\n\n');

% Save to CSV/TXT for Final Report
fileID = fopen('quantitative_results_final.txt','w');
fprintf(fileID, 'Metric, Noisy Input, System A (Hard VAD), System B (IMCRA), System C (Learned SPP)\n');
fprintf(fileID, 'Global_MSE_1e-3, %.4f, %.4f, %.4f, %.4f\n', mse_noisy*1000, mse_sysA*1000, mse_sysB*1000, mse_sysC*1000);
fprintf(fileID, 'Global_SNR_dB, %.4f, %.4f, %.4f, %.4f\n', snr_noisy, snr_sysA, snr_sysB, snr_sysC);
fprintf(fileID, 'Post3s_SNR_dB, %.4f, %.4f, %.4f, %.4f\n', snr_post_noisy, snr_post_sysA, snr_post_sysB, snr_post_sysC);
fclose(fileID);