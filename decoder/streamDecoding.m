function [s_DCT_decoding, y_cell] = streamDecoding(data_transmitted, U, M_init, N, alpha, a, b, is_DCT_block, num_DCT)

num_DCT_block = sum(is_DCT_block);
idx_read = num_DCT * num_DCT_block;

s_DCT_decoding = reshape(data_transmitted(1:idx_read), [num_DCT, num_DCT_block]);

block_num = length(is_DCT_block);
num_CS_block=block_num-num_DCT_block;
y_init = zeros(M_init, block_num);
y_init(:, ~is_DCT_block) = reshape(data_transmitted(idx_read+1:idx_read+num_CS_block*M_init), [M_init, num_CS_block]);
idx_read = idx_read + num_CS_block * M_init;

y_cell = cell(block_num, 1);
for idx_block = 1:block_num
    if (is_DCT_block(idx_block))
        continue;
    end

    kappa = U(2:end, :) * y_init(:, idx_block);
    k = sum(fix(kappa) ~= 0)';
    M = round(alpha*(a * log2(k+1) + b));

    if (M < M_init)
        M = M_init;
    elseif (M > N)
        M = N;
    end

    y_cell{idx_block} = [y_init(:, idx_block); data_transmitted(idx_read+1:idx_read+M-M_init)];
    idx_read = idx_read + M - M_init;
end

end