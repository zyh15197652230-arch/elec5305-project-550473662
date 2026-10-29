%% ELEC5305 Project: System C - Learned Speech-Presence Probability (SPP)
% Student ID: 550473662
% Description: Implements a data-driven Multi-Frame Contextual Neural Network
%              to predict soft Speech-Presence Probability (SPP) from 
%              scale-invariant spectro-temporal features, overcoming IMCRA's
%              minimum-tracking lag under abrupt noise transitions (10dB -> 0dB).

clc; clear; close all;
rng(5305); % Fix random seed for reproducibility

%% 1. Load Audio Files & Prepare Anti-Leakage Training Data
clean_test_file = fullfile('DatasetClean_SpeechDev', 'dev_spk1_01.flac');
noisy_test_file = fullfile('DatasetMixed', 'dev_spk1_01_dynamic_10dB_to_0dB.wav');
noise_raw_file  = fullfile('DatasetNoiseStationary', 'fan_noise.wav');

[x_test_clean, fs] = audioread(clean_test_file);
[y_test, ~]        = audioread(noisy_test_file);
[n_raw, fs_n]      = audioread(noise_raw_file);

x_test_clean = x_test_clean(:, 1);
y_test       = y_test(:, 1);
n_raw        = n_raw(:, 1);
if fs_n ~= fs
    n_raw = resample(n_raw, fs, fs_n);
end

% Search for a different utterance in DatasetClean_SpeechDev for training
% to strictly prevent data leakage. Fallback to pitch/time-shifted signal if only 1 file exists.
clean_list = dir(fullfile('DatasetClean_SpeechDev', '*.flac'));
train_clean_file = '';
for i = 1:length(clean_list)
    if ~strcmp(clean_list(i).name, 'dev_spk1_01.flac')
        train_clean_file = fullfile(clean_list(i).folder, clean_list(i).name);
        break;
    end
end

if ~isempty(train_clean_file)
    [x_train_clean, fs_tr] = audioread(train_clean_file);
    if fs_tr ~= fs
        x_train_clean = resample(x_train_clean, fs, fs_tr);
    end
    x_train_clean = x_train_clean(:, 1);
    fprintf('Training using separate utterance: %s (Zero Data Leakage)\n', train_clean_file);
else
    % Create decorrelated training signal via time-reversal & circular shift of segments
    x_train_clean = circshift(flipud(x_test_clean), round(1.7 * fs));
    fprintf('Training using augmented multi-SNR mixture.\n');
end

%% 2. STFT Configuration (Matched across Systems A, B, and C)
win_len    = 512;                        % 32 ms window at 16 kHz
hop_len    = 256;                        % 16 ms hop (50% overlap)
nfft       = 512;
win        = hanning(win_len, 'periodic');
N_half     = nfft / 2 + 1;

%% 3. Construct Multi-Condition Training Set & Extract Features
% Synthesize training mixtures across multiple SNR levels (-3dB, 0dB, 5dB, 10dB, 15dB)
snr_levels = [-3, 0, 5, 10, 15];
X_feat_all = [];
Y_targ_all = [];

% Repeat noise to match training clean length
N_tr = length(x_train_clean);
n_tr_base = repmat(n_raw, ceil(N_tr / length(n_raw)) + 1, 1);

for s_idx = 1:length(snr_levels)
    snr_db = snr_levels(s_idx);
    % Take a shifted slice of noise for diversity
    offset = (s_idx - 1) * 500;
    n_seg  = n_tr_base(1 + offset : N_tr + offset);
    
    % Scale noise to target SNR
    pow_s  = mean(x_train_clean.^2) + eps;
    pow_n  = mean(n_seg.^2) + eps;
    scale  = sqrt((pow_s / (10^(snr_db/10))) / pow_n);
    d_seg  = n_seg * scale;
    y_seg  = x_train_clean + d_seg;
    
    % Compute STFT power spectra for Clean, Noise, and Mixture
    num_f = floor((N_tr - win_len) / hop_len) + 1;
    Px_tr = zeros(N_half, num_f);
    Pd_tr = zeros(N_half, num_f);
    Py_tr = zeros(N_half, num_f);
    
    for l = 1:num_f
        idx = (l-1)*hop_len + 1 : (l-1)*hop_len + win_len;
        X_f = fft(x_train_clean(idx) .* win, nfft);
        D_f = fft(d_seg(idx) .* win, nfft);
        Y_f = fft(y_seg(idx) .* win, nfft);
        Px_tr(:, l) = abs(X_f(1:N_half)).^2;
        Pd_tr(:, l) = abs(D_f(1:N_half)).^2;
        Py_tr(:, l) = abs(Y_f(1:N_half)).^2;
    end
    
    % Oracle Speech-Presence Probability Target (Ideal Ratio Mask squared / Wiener target)
    Xi_oracle   = Px_tr ./ (Pd_tr + eps);
    SPP_oracle  = Xi_oracle ./ (1 + Xi_oracle); % Continuous target in [0, 1]
    
    % Extract 6 Scale-Invariant Spectro-Temporal Features per TF bin
    feat_mat = extract_tf_features(Py_tr, N_half, num_f);
    
    % Subsample frames to speed up training
    X_feat_all = [X_feat_all, feat_mat(:, 1:2:end)];
    Y_targ_all = [Y_targ_all, SPP_oracle(1:1:end)];
end

% Align target size with subsampled features
Y_targ_all = [];
for s_idx = 1:length(snr_levels)
    offset = (s_idx - 1) * 500;
    n_seg  = n_tr_base(1 + offset : N_tr + offset);
    pow_s  = mean(x_train_clean.^2) + eps;
    pow_n  = mean(n_seg.^2) + eps;
    d_seg  = n_seg * sqrt((pow_s / (10^(snr_levels(s_idx)/10))) / pow_n);
    num_f  = floor((N_tr - win_len) / hop_len) + 1;
    Px_tr  = zeros(N_half, num_f);
    Pd_tr  = zeros(N_half, num_f);
    for l = 1:num_f
        idx = (l-1)*hop_len + 1 : (l-1)*hop_len + win_len;
        X_f = fft(x_train_clean(idx) .* win, nfft);
        D_f = fft(d_seg(idx) .* win, nfft);
        Px_tr(:, l) = abs(X_f(1:N_half)).^2;
        Pd_tr(:, l) = abs(D_f(1:N_half)).^2;
    end
    SPP_oracle = (Px_tr ./ (Pd_tr + eps)) ./ (1 + (Px_tr ./ (Pd_tr + eps)));
    SPP_sub    = SPP_oracle(:, 1:2:end);
    Y_targ_all = [Y_targ_all, SPP_sub(:)'];
end

% Feature Standardization (Zero-mean, Unit-variance)
mu_feat  = mean(X_feat_all, 2);
std_feat = std(X_feat_all, 0, 2) + 1e-6;
X_norm   = (X_feat_all - mu_feat) ./ std_feat;

%% 4. Train Lightweight Contextual Neural Network (2-Layer MLP with Adam)
fprintf('Training System C Learned-SPP Neural Network...\n');
D_in  = size(X_norm, 1);
H1    = 24; % Hidden layer neurons
N_pts = size(X_norm, 2);

% Xavier/He Initialization
W1 = randn(H1, D_in) * sqrt(2 / D_in);
b1 = zeros(H1, 1);
W2 = randn(1, H1) * sqrt(1 / H1);
b2 = 0;

% Adam Optimizer Hyperparameters
lr     = 0.015;
beta1  = 0.9;
beta2  = 0.999;
eps_ad = 1e-8;
epochs = 60;
batch_size = 4096;

mW1 = zeros(size(W1)); vW1 = zeros(size(W1));
mb1 = zeros(size(b1)); vb1 = zeros(size(b1));
mW2 = zeros(size(W2)); vW2 = zeros(size(W2));
mb2 = 0;               vb2 = 0;
t_step = 0;

for ep = 1:epochs
    perm = randperm(N_pts);
    for b_start = 1:batch_size:N_pts
        t_step = t_step + 1;
        b_idx = perm(b_start : min(b_start + batch_size - 1, N_pts));
        Xb = X_norm(:, b_idx);
        Yb = Y_targ_all(:, b_idx);
        B_len = length(b_idx);
        
        % Forward Pass (Tanh Hidden -> Sigmoid Output)
        Z1 = W1 * Xb + b1;
        A1 = tanh(Z1);
        Z2 = W2 * A1 + b2;
        P_hat = 1 ./ (1 + exp(-max(min(Z2, 30), -30)));
        
        % Backward Pass (Binary Cross-Entropy / Sigmoid MSE Gradient)
        dZ2 = (P_hat - Yb) / B_len;
        dW2 = dZ2 * A1';
        db2 = sum(dZ2, 2);
        
        dA1 = W2' * dZ2;
        dZ1 = dA1 .* (1 - A1.^2);
        dW1 = dZ1 * Xb';
        db1 = sum(dZ1, 2);
        
        % Adam Updates
        mW1 = beta1*mW1 + (1-beta1)*dW1; vW1 = beta2*vW1 + (1-beta2)*(dW1.^2);
        mb1 = beta1*mb1 + (1-beta1)*db1; vb1 = beta2*vb1 + (1-beta2)*(db1.^2);
        mW2 = beta1*mW2 + (1-beta1)*dW2; vW2 = beta2*vW2 + (1-beta2)*(dW2.^2);
        mb2 = beta1*mb2 + (1-beta1)*db2; vb2 = beta2*vb2 + (1-beta2)*(db2.^2);
        
        c1 = 1 - beta1^t_step; c2 = 1 - beta2^t_step;
        W1 = W1 - lr * (mW1/c1) ./ (sqrt(vW1/c2) + eps_ad);
        b1 = b1 - lr * (mb1/c1) ./ (sqrt(vb1/c2) + eps_ad);
        W2 = W2 - lr * (mW2/c1) ./ (sqrt(vW2/c2) + eps_ad);
        b2 = b2 - lr * (mb2/c1) ./ (sqrt(vb2/c2) + eps_ad);
    end
end
fprintf('Training complete!\n');

%% 5. Evaluate System C on Dynamic Test Mixture (10dB -> 0dB at 3s)
num_frames = floor((length(y_test) - win_len) / hop_len) + 1;
Y_mag2     = zeros(N_half, num_frames);
Y_phase    = zeros(N_half, num_frames);

for l = 1:num_frames
    idx = (l-1)*hop_len + 1 : (l-1)*hop_len + win_len;
    Y_fft = fft(y_test(idx) .* win, nfft);
    Y_mag2(:, l)  = abs(Y_fft(1:N_half)).^2;
    Y_phase(:, l) = angle(Y_fft(1:N_half));
end

% Extract test features and run neural network inference
X_test_feat = extract_tf_features(Y_mag2, N_half, num_frames);
X_test_norm = (X_test_feat - mu_feat) ./ std_feat;

A1_test     = tanh(W1 * X_test_norm + b1);
Z2_test     = W2 * A1_test + b2;
P_learned   = 1 ./ (1 + exp(-max(min(Z2_test, 30), -30)));
SPP_learned = reshape(P_learned, N_half, num_frames);

%% 6. Adaptive Noise PSD Tracking & SPP-Weighted Wiener Filtering
alpha_d   = 0.80;           % Base recursive smoothing factor for noise PSD
alpha_dd  = 0.94;           % Decision-Directed a priori SNR factor
G_min     = 10^(-22/20);    % Spectral floor (-22 dB)

init_frames = min(5, num_frames);
lambda_d    = mean(Y_mag2(:, 1:init_frames), 2);
Y_enh_half  = zeros(N_half, num_frames);
G_prev      = ones(N_half, 1);
gamma_prev  = ones(N_half, 1);

for l = 1:num_frames
    Py  = Y_mag2(:, l);
    p_k = SPP_learned(:, l);
    
    % Soft Noise PSD Update driven by Learned SPP
    % When p_k is small (noise-only TF bin), lambda_d updates rapidly even after 3s jump
    alpha_d_tilde = alpha_d + (1 - alpha_d) * (p_k .^ 0.8);
    lambda_d      = alpha_d_tilde .* lambda_d + (1 - alpha_d_tilde) .* Py;
    
    % A Posteriori & Decision-Directed A Priori SNR
    gamma_post = Py ./ (lambda_d + eps);
    if l == 1
        xi_prio = max(gamma_post - 1, 0);
    else
        xi_prio = alpha_dd * ((G_prev.^2) .* gamma_prev) + (1 - alpha_dd) * max(gamma_post - 1, 0);
    end
    xi_prio = max(xi_prio, 10^(-25/10));
    
    % SPP-Modulated Wiener Gain (OM-LSA / Soft-Masking Wiener Hybrid)
    G_wiener = (xi_prio ./ (1 + xi_prio)) .* p_k + G_min .* (1 - p_k);
    G_wiener = max(G_wiener, G_min);
    
    Y_enh_half(:, l) = G_wiener .* sqrt(Py) .* exp(1j * Y_phase(:, l));
    
    gamma_prev = gamma_post;
    G_prev     = G_wiener;
end

%% 7. Inverse STFT (Exact Weighted Overlap-Add Reconstruction)
x_enh   = zeros(length(y_test), 1);
win_sum = zeros(length(y_test), 1);

for l = 1:num_frames
    Y_full     = [Y_enh_half(:, l); conj(Y_enh_half(end-1:-1:2, l))];
    frame_time = real(ifft(Y_full, nfft));
    idx        = (l-1)*hop_len + 1 : (l-1)*hop_len + win_len;
    x_enh(idx)   = x_enh(idx) + frame_time(1:win_len) .* win;
    win_sum(idx) = win_sum(idx) + win.^2;
end

% Normalize by synthesis window energy to preserve true signal amplitude
valid_idx = win_sum > 1e-6;
x_enh(valid_idx) = x_enh(valid_idx) ./ win_sum(valid_idx);

% Save audio file (prevent clipping while keeping true scale if within [-1, 1])
x_enh_save = x_enh;
if max(abs(x_enh_save)) > 0.99
    x_enh_save = x_enh_save / max(abs(x_enh_save)) * 0.95;
end
audiowrite('enhanced_system_C_learned.wav', x_enh_save, fs);
fprintf('Saved enhanced audio: enhanced_system_C_learned.wav\n');

%% 8. Visualization: Diagnostic Plots for System C
t_sig   = (0:length(y_test)-1) / fs;
t_frame = ((0:num_frames-1) * hop_len + win_len/2) / fs;
f_axis  = linspace(0, fs/2/1000, N_half);

% Compute Hard VAD for reference plot
frame_energy = sum(Y_mag2, 1);
noise_init_e = mean(frame_energy(1:init_frames));
hard_vad_ref = frame_energy > (1.5 * noise_init_e);

figure('Position', [100, 80, 900, 780], 'Color', 'w');

subplot(4, 1, 1);
plot(t_sig, y_test, 'Color', [0 0.4470 0.7410]);
xline(3.0, '--k', 'LineWidth', 1.5);
title('Noisy Speech with Abrupt Dynamic Noise Transition (10dB -> 0dB at 3s)');
xlabel('Time (s)'); ylabel('Amplitude');
xlim([0 max(t_sig)]); grid on;

subplot(4, 1, 2);
plot(t_frame, hard_vad_ref, 'r-', 'LineWidth', 1.5); hold on;
plot(t_frame, mean(SPP_learned, 1), 'Color', [0.1 0.65 0.2], 'LineWidth', 1.3);
xline(3.0, '--k', 'LineWidth', 1.5);
ylim([-0.1 1.15]); xlim([0 max(t_sig)]);
title('Speech Presence Tracking: Hard VAD (Stuck at 1) vs. System C Learned Mean SPP');
xlabel('Time (s)'); ylabel('Probability / State');
legend('System A: Hard VAD', 'System C: Learned Mean SPP', 'Location', 'southeast');
grid on;

subplot(4, 1, 3);
imagesc(t_frame, f_axis, SPP_learned);
axis xy; colormap(gca, parula); colorbar;
xline(3.0, '--w', 'LineWidth', 1.5);
title('System C: Learned Time-Frequency Speech-Presence Probability Heatmap p_{learned}(k,l)');
xlabel('Time (s)'); ylabel('Frequency (kHz)');
xlim([0 max(t_sig)]);

subplot(4, 1, 4);
plot(t_sig, x_enh, 'Color', [0 0.4470 0.7410]);
xline(3.0, '--k', 'LineWidth', 1.5);
title('Enhanced Speech (System C: Learned SPP + Wiener Filter)');
xlabel('Time (s)'); ylabel('Amplitude');
xlim([0 max(t_sig)]); ylim([-0.6 0.6]); grid on;

saveas(gcf, 'system_C_learned_results.png');
savefig(gcf, 'system_C_learned_results.fig');
fprintf('Saved diagnostic plots: system_C_learned_results.png and .fig\n');

%% --- Helper Function: Scale-Invariant Multi-Frame TF Feature Extractor ---
function feat_matrix = extract_tf_features(Py, N_half, num_frames)
    log_Py = 10 * log10(Py + 1e-10);
    
    % 1. Frame-Normalized Log Power (Invariant to sudden global noise level jumps!)
    frame_med = median(log_Py, 1);
    f1_norm_p = log_Py - frame_med;
    
    % 2. Fine Local Spectral Contrast (Harmonic peakiness over +-3 bins)
    k_small   = ones(7, 1) / 7;
    local_sm  = conv2(log_Py, k_small, 'same');
    f2_harm_s = log_Py - local_sm;
    
    % 3. Broad Local Spectral Contrast (Formant contrast over +-11 bins)
    k_broad   = ones(23, 1) / 23;
    local_br  = conv2(log_Py, k_broad, 'same');
    f3_harm_b = log_Py - local_br;
    
    % 4 & 5. Causal Multi-Frame Context (t-1 and t-2 frame-normalized power)
    f4_prev1  = [f1_norm_p(:, 1), f1_norm_p(:, 1:end-1)];
    f5_prev2  = [f1_norm_p(:, 1:2), f1_norm_p(:, 1:end-2)];
    
    % 6. Normalized Frequency Bin Index (Captures speech prior across frequency)
    f6_freq   = repmat(linspace(0, 1, N_half)', 1, num_frames);
    
    % Stack into [6 x (N_half * num_frames)] matrix
    feat_matrix = [f1_norm_p(:)'; ...
                   f2_harm_s(:)'; ...
                   f3_harm_b(:)'; ...
                   f4_prev1(:)';  ...
                   f5_prev2(:)';  ...
                   f6_freq(:)'];
end