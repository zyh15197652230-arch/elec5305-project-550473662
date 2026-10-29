%% ELEC5305 Project: System B - IMCRA Soft-SPP Wiener Filter
% Student ID: 550473662
% Description: Implements Improved Minima Controlled Recursive Averaging (IMCRA)
%              for continuous soft Speech-Presence Probability (SPP) estimation
%              and noise PSD tracking under abrupt non-stationary noise transitions.

clc; clear; close all;

%% 1. Load Dynamic Noisy Mixture (10dB -> 0dB at 3s)
target_file = 'dev_spk1_01_dynamic_10dB_to_0dB.wav';

if isfile(target_file)
    [y, fs] = audioread(target_file);
    fprintf('Successfully loaded: %s\n', target_file);
else
    % Automatically search current folder and subfolders for any dynamic wav file
    file_list = dir('**/*dynamic*.wav');
    if ~isempty(file_list)
        target_file = fullfile(file_list(1).folder, file_list(1).name);
        [y, fs] = audioread(target_file);
        fprintf('Found and loaded: %s\n', target_file);
    else
        error('Cannot find the dynamic noisy wav file! Please run generate_dynamic_dataset.m first.');
    end
end
y = y(:, 1); % Ensure mono

%% 2. STFT Parameters (Matched with System A Backend)
frame_dur = 0.032;                  % 32 ms frame length
win_len   = round(frame_dur * fs);
if mod(win_len, 2) ~= 0
    win_len = win_len + 1;          % Ensure even window length
end
hop_len   = win_len / 2;            % 50% overlap (16 ms hop)
nfft      = 2^nextpow2(win_len);
win       = hamming(win_len, 'periodic');

% Framing and FFT
num_frames = floor((length(y) - win_len) / hop_len) + 1;
N_half     = nfft / 2 + 1;          % Positive frequencies

Y_mag2     = zeros(N_half, num_frames);
Y_phase    = zeros(N_half, num_frames);

for l = 1:num_frames
    idx = (l-1)*hop_len + 1 : (l-1)*hop_len + win_len;
    frame_sig = y(idx) .* win;
    Y_fft = fft(frame_sig, nfft);
    Y_Half = Y_fft(1:N_half);
    Y_mag2(:, l)  = abs(Y_Half).^2;
    Y_phase(:, l) = angle(Y_Half);
end

%% 3. IMCRA & Wiener Filter Hyperparameters
alpha_s   = 0.8;    % Time smoothing factor for power spectrum
alpha_d   = 0.85;   % Base noise PSD recursive smoothing factor
alpha_dd  = 0.96;   % Decision-Directed a priori SNR smoothing factor
B_min     = 1.66;   % Minimum tracking bias compensation factor
gamma_0   = 4.6;    % Local minimum SNR threshold
zeta_0    = 1.67;   % Smoothed local minimum threshold
V_win     = round(0.8 / (hop_len / fs)); % Sliding window (~0.8s) for minimum tracking
G_min     = 10^(-20/20); % Spectral floor (-20 dB) to reduce musical noise

% Frequency smoothing kernel: [0.25, 0.5, 0.25]
b_kernel  = [0.25; 0.5; 0.25];

% Initialization (using first 5 frames)
init_frames = min(5, num_frames);
lambda_d    = mean(Y_mag2(:, 1:init_frames), 2); % Initial noise PSD estimate
S_smooth    = lambda_d;
S_min_buf   = repmat(lambda_d, 1, V_win);        % Buffer for sliding minimum
xi_prev     = ones(N_half, 1);                   % Previous a priori SNR
G_prev      = ones(N_half, 1);                   % Previous Wiener gain

% Diagnostic storage
SPP_matrix       = zeros(N_half, num_frames);    % Soft SPP p(k,l)
noise_psd_track  = zeros(N_half, num_frames);    % Tracked noise PSD
Y_enh_half       = zeros(N_half, num_frames);    % Enhanced spectrum

% Also compute System A Hard VAD on the same frames for direct comparison
frame_energy_dB  = 10 * log10(sum(Y_mag2, 1) + eps);
vad_thresh_dB    = mean(frame_energy_dB(1:init_frames)) + 3.5; % Standard energy threshold
hard_vad_decision = frame_energy_dB > vad_thresh_dB;

%% 4. Frame-by-Frame IMCRA SPP Estimation & Wiener Enhancement
for l = 1:num_frames
    Py = Y_mag2(:, l);
    
    % Step 4.1: Frequency & Time Smoothing of Noisy Spectrum
    Sf = conv(Py, b_kernel, 'same');
    if l == 1
        S_smooth = Sf;
    else
        S_smooth = alpha_s * S_smooth + (1 - alpha_s) * Sf;
    end
    
    % Update sliding window buffer and track local minimum S_min
    S_min_buf = [S_min_buf(:, 2:end), S_smooth];
    S_min     = min(S_min_buf, [], 2);
    
    % Step 4.2: Compute Local Minimum Ratios (Cohen 2003)
    gamma_min = Py ./ (B_min * S_min + eps);
    zeta      = S_smooth ./ (B_min * S_min + eps);
    
    % Step 4.3: Estimate A Priori Speech Absence Probability q(k,l)
    q_k = zeros(N_half, 1);
    % Condition 1: Pure noise region
    idx_noise = (gamma_min <= 1) & (zeta < zeta_0);
    q_k(idx_noise) = 1.0;
    % Condition 2: Soft transition region
    idx_trans = (gamma_min > 1) & (gamma_min < gamma_0) & (zeta < zeta_0);
    q_k(idx_trans) = (gamma_0 - gamma_min(idx_trans)) ./ (gamma_0 - 1);
    % Clamp q_k to avoid division by zero in Bayes formula
    q_k = min(max(q_k, 0.01), 0.99);
    
    % Step 4.4: A Posteriori & Decision-Directed A Priori SNR
    gamma_post = Py ./ (lambda_d + eps);
    if l == 1
        xi_prio = alpha_dd + (1 - alpha_dd) * max(gamma_post - 1, 0);
    else
        xi_prio = alpha_dd * ((G_prev.^2) .* gamma_prev) + (1 - alpha_dd) * max(gamma_post - 1, 0);
    end
    xi_prio = max(xi_prio, 1e-4);
    
    % Step 4.5: Bayes Soft Speech-Presence Probability p(k,l)
    v_k = (gamma_post .* xi_prio) ./ (1 + xi_prio);
    p_k = 1 ./ (1 + (q_k ./ (1 - q_k)) .* (1 + xi_prio) .* exp(-min(v_k, 50)));
    
    % Step 4.6: Soft Recursive Noise PSD Update
    alpha_d_tilde = alpha_d + (1 - alpha_d) * p_k;
    lambda_d      = alpha_d_tilde .* lambda_d + (1 - alpha_d_tilde) .* Py;
    
    % Step 4.7: Common Wiener Filter Backend
    G_wiener = xi_prio ./ (1 + xi_prio);
    G_wiener = max(G_wiener, G_min);
    
    % Apply gain
    Y_enh_half(:, l) = G_wiener .* sqrt(Py) .* exp(1j * Y_phase(:, l));
    
    % Store state & diagnostics
    gamma_prev            = gamma_post;
    G_prev                = G_wiener;
    SPP_matrix(:, l)      = p_k;
    noise_psd_track(:, l) = lambda_d;
end

%% 5. Inverse STFT (Weighted Overlap-Add Reconstruction)
x_enh = zeros((num_frames - 1) * hop_len + win_len, 1);
win_sum = zeros(size(x_enh));

for l = 1:num_frames
    % Reconstruct conjugate symmetric full FFT spectrum
    Y_full = [Y_enh_half(:, l); conj(Y_enh_half(end-1:-1:2, l))];
    frame_time = real(ifft(Y_full, nfft));
    frame_time = frame_time(1:win_len) .* win;
    
    idx = (l-1)*hop_len + 1 : (l-1)*hop_len + win_len;
    x_enh(idx)   = x_enh(idx) + frame_time;
    win_sum(idx) = win_sum(idx) + win.^2;
end

% Normalize overlap-add window and match original signal length
x_enh = x_enh ./ (win_sum + eps);
if length(x_enh) < length(y)
    x_enh = [x_enh; zeros(length(y) - length(x_enh), 1)];
else
    x_enh = x_enh(1:length(y));
end

% Save enhanced audio (normalized to prevent clipping)
x_enh_norm = x_enh / (max(abs(x_enh)) + eps) * 0.95;
audiowrite('enhanced_system_B_imcra.wav', x_enh_norm, fs);
fprintf('Saved enhanced audio: enhanced_system_B_imcra.wav\n');

%% 6. Visualization: Diagnostic Plots for Report & GitHub
t_sig   = (0:length(y)-1) / fs;
t_frame = ((0:num_frames-1) * hop_len + win_len/2) / fs;
f_axis  = linspace(0, fs/2/1000, N_half); % Frequency in kHz

figure('Position', [100, 80, 900, 780], 'Color', 'w');

% Subplot 1: Noisy Speech
subplot(4, 1, 1);
plot(t_sig, y, 'Color', [0 0.4470 0.7410]);
xline(3.0, '--k', 'LineWidth', 1.5);
title('Noisy Speech with Abrupt Dynamic Noise Transition (10dB -> 0dB at 3s)');
xlabel('Time (s)'); ylabel('Amplitude');
xlim([0 max(t_sig)]); grid on;

% Subplot 2: Hard VAD vs. IMCRA Frame-Averaged Soft SPP
subplot(4, 1, 2);
plot(t_frame, hard_vad_decision, 'r-', 'LineWidth', 1.5); hold on;
plot(t_frame, mean(SPP_matrix, 1), 'b-', 'LineWidth', 1.2);
xline(3.0, '--k', 'LineWidth', 1.5);
ylim([-0.1 1.15]); xlim([0 max(t_sig)]);
title('Speech Presence Tracking: Hard VAD (Stuck at 1) vs. IMCRA Mean Soft SPP');
xlabel('Time (s)'); ylabel('Probability / State');
legend('System A: Hard VAD', 'System B: IMCRA Mean SPP', 'Location', 'southeast');
grid on;

% Subplot 3: Time-Frequency Soft SPP Heatmap p(k,l)
subplot(4, 1, 3);
imagesc(t_frame, f_axis, SPP_matrix);
axis xy; colormap(gca, parula); colorbar;
xline(3.0, '--w', 'LineWidth', 1.5);
title('System B: Time-Frequency Soft Speech-Presence Probability Heatmap p(k,l)');
xlabel('Time (s)'); ylabel('Frequency (kHz)');
xlim([0 max(t_sig)]);

% Subplot 4: Enhanced Speech (System B)
subplot(4, 1, 4);
plot(t_sig, x_enh, 'Color', [0 0.4470 0.7410]);
xline(3.0, '--k', 'LineWidth', 1.5);
title('Enhanced Speech (System B: IMCRA Soft-SPP + Wiener Filter)');
xlabel('Time (s)'); ylabel('Amplitude');
xlim([0 max(t_sig)]); ylim([-1 1]); grid on;

% Save both PNG and FIG formats
saveas(gcf, 'system_B_imcra_results.png');
savefig(gcf, 'system_B_imcra_results.fig');
fprintf('Saved diagnostic plots: system_B_imcra_results.png and .fig\n');