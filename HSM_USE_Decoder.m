% -----------------------------------------------------------------------
% The MATLAB programs of (decoder)
% Unified Sparsity Estimation-Based Distributed Compressive Video Sensing

% Author: Zhen Song
% The results within the paper are test on MATLAB R2018a
% -----------------------------------------------------------------------

%%
% running the encoder first
addpath('.\decoder');
addpath('.\decoder\SPL-DDWT');
%% params
% params require transmitted from the encoder

% block_size = 8;
% M_init = 27;
% Mc = 24;

% alpha = 1.7;
% a = 9.62;
% b = 0.85;

% qf = 40;
% num_DCT = round(qf/10);
% if (num_DCT <= 0)
%     num_DCT = 1;
% end

% tau_p = 0.5;
% tau_k = round(num_DCT/2) + 1;

% above params should be modified along with the encoder

% calculable params
[height, width] = size(img);
N = block_size^2; % length of block in vector
block_num = height * width / N;

%% matrix construction through random seed synchronization
% the construction methods are the same as the encoder
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

%% measurement stream decoding
% input: data_stream, identifier, and related params
[s_DCT_decoding, y_cell] = streamDecoding(data_stream, U, M_init, N, alpha, a, b, is_DCT_block, num_DCT);

%% linear+SPL-DDWT reconstruction
% pre-reconstruction for DCT-block
recons_x_DCT = Psi \ Gamma * Lambda(1:num_DCT, :)' * s_DCT_decoding;

% linear reconstruction
Phi_set = PhiSetMaker(Phi_full, M_list);
recons_block_linear = zeros(N, block_num);
recons_block_linear(:, ~is_DCT_block) = ARec(y_cell, Phi_set, Rxx, block_size);
recons_block_linear(:, is_DCT_block) = recons_x_DCT;

% resampling for DCT-block
y_cell_spl = y_cell;
Phi_set_spl = Phi_set;
M_list_spl = M_list;

idx_DCT = 1;
num = M_init;
for idx_block = 1:block_num
    if (is_DCT_block(idx_block))
        y_cell_spl{idx_block} = Phi_full(1:num, :) * recons_x_DCT(:, idx_DCT);
        Phi_set_spl{idx_block} = Phi_full(1:num, :);
        M_list_spl(idx_block) = num;
        idx_DCT = idx_DCT + 1;
    end
end

% SPL-DDWT reconstruction
% params config
epsilon = 0.001;
params.lambda = 20;
params.TOL = 0.01;
params.level = 3;

recons_img_spl = AdaptiveBlockCS_SPL_DDWT(y_cell_spl, Phi_set_spl, params, recons_block_linear, height, width);
recons_img_spl(recons_img_spl > 255) = 255;
recons_img_spl(recons_img_spl < 0) = 0;

% evaluate
fprintf(' PSNR = %.2f dB, SSIM = %.4f\n', Psnr(img, recons_img_spl), ssim(img, recons_img_spl));
