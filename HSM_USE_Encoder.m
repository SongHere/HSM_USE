% -----------------------------------------------------------------------
% The MATLAB programs of (encoder)
% Hybrid Sampling Mode Design for Compressive Imaging Based on Unified Sparsity Estimation

% Author: Zhen Song
% The results within the paper are test on MATLAB R2018a
% -----------------------------------------------------------------------

%%
clear;
addpath('.\data');
addpath('.\encoder');

%% data load
file_list = ["cameraman"; "jetplane"; "lake"; "pirate"; "livingroom"; "mandril_gray"];
file_name = [char(file_list(1)), '.tif'];
img = imread(file_name);
img = double(img(:, :, 1));

%% params
block_size = 8;
M_init = 27; % number of initial measurements
Mc = 24; % number of estimated coefficients for sparsity estimation

% logarithmic model for adaptive allocation
alpha = 1.7; % measurement allocation factor
a = 9.62;
b = 0.85;

qf = 40; % quality factor
num_DCT = round(qf/10); % number of DCT coefficients for DCT-block
if (num_DCT <= 0)
    num_DCT = 1;
end

% block classification
tau_p = 0.5;
tau_k = round(num_DCT/2) + 1;

% calculable params
[height, width] = size(img);
N = block_size^2; % length of block in vector
block_num = height * width / N;

%% matrix construction
tic;

% random measruement matrix
Phi_state = 0; % the random seed
randn('seed', Phi_state);
Phi_full = randn(N, N);
Phi_full = orth(Phi_full)';
Phi_full(1, :) = 1 / N;

% JPEG standard luminance quantization table
Q_JPEG = JPEG88QTable();
[~, q_table, idx_col2zig] = zigzagScanning(Q_JPEG);
% q_table: JPEG 8*8 luminance quantization table in raster scanning order
% idx_col2zig: index relationship between raster scanning (value) and zig-zag scanning order (position)
Q = qTableAdjustment(q_table, qf); % threshold adjustment

% DCT basis for saprse representation
load Psi_DCT88;

% linear reconstruction matrix construction
num_Rxx = 0.95;
Phi_init = Phi_full(1:M_init, :); % initial sampling matrix
Rxx = ImageCorr(block_size, block_size, num_Rxx); % pixel domian correlation matrix
P = Rxx * Phi_init' * (Phi_init * Rxx * Phi_init')^(-1);

I_N = eye(N); % identity matrix
idx_zig = idx_col2zig(1:Mc); % the first Mc DCT coefficients in the zig-zag scanning order
Lambda = I_N(idx_zig, :); % permutation matrix
Gamma = I_N .* Q; % threshold matrix
U = Lambda * (Gamma \ (Psi * P)); % USE matrix

%% sampling
% BCS initial sampling
img_block = im2col(img, [block_size, block_size], 'distinct');
y_init = Phi_init * img_block;

% USE
kappa = U(2:end, :) * y_init;
kappa_abs = abs(kappa);
k_list = sum(kappa_abs >= 1)' + 1; % the number of significant coefficients

% block classification
% is_DCT_block: identifier for block types (1 bit per block)
% 1: k_i<tau_k (smooth blocks) -> DCT-block
is_DCT_block = k_list < tau_k;
% 2: (k_i=tau_k) && (p_i>=tau_p) -> DCT-block
is_tau_k_block = k_list == tau_k; % relatively complex blocks of k_i=tau_k
s_AC_ahead = sum(kappa_abs(1:tau_k-1, is_tau_k_block));
s_AC_all = s_AC_ahead + sum(kappa_abs(tau_k:end, is_tau_k_block));
is_DCT_block_p = (s_AC_ahead ./ s_AC_all) >= tau_p;
idx_tau_k = find(is_tau_k_block == 1);
is_DCT_block(idx_tau_k(is_DCT_block_p)) = true;
% 3: others (complex blocks) -> CS-block where is_DCT_block==false

% adaptive rate allocation for CS-block
M_list = zeros(block_num, 1);
M_list(~is_DCT_block) = round(alpha*(a * log2(k_list(~is_DCT_block)) + b));
M_list(M_list > N) = N;
M_list(M_list < M_init & ~is_DCT_block) = M_init;
% DC estimation for DCT-block
s_DC = U(1, :) * y_init(:, is_DCT_block);

%% resampling for supplementary
Phi_add = Phi_full(M_init+1:end, :);

M_add = M_list - M_init;
y_stream_add = resampling(img_block, Phi_add, M_add);
y_init_transmitted = y_init(:, ~is_DCT_block); % only for CS-block
y_stream = [y_init_transmitted(:); y_stream_add]; % CS measurements acquired with an appropriate sampling rate

time_encoding = toc * 1000;

%% tranmition
s_DCT = [s_DC; kappa(1:num_DCT-1, is_DCT_block)]; % estimated DCT coefficients
data_stream = [s_DCT(:); y_stream];

%%
sampling_rate = length(data_stream) / block_num / N;
DCT_block_rate = sum(is_DCT_block) / block_num;
fprintf('\n%s\n Mean Sampling Rate: %.4f\n DCT Block Rate: %.2f\n Encoding time: %.4f ms\n', ...
    file_name, sampling_rate, DCT_block_rate, time_encoding);