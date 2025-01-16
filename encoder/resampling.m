function y_stream_add = resampling(img_block, Phi_add, M_add)
% resampling

block_num = length(M_add);
y_stream_add = zeros(sum(M_add(M_add>0)), 1); % measurements stream

idx_samples = 1;
for idx_block = 1:block_num
    M = M_add(idx_block); % number of measurements of current block
    if (M <= 0)
        continue;
    end
    y_add = Phi_add(1:M, :) * img_block(:, idx_block);
    y_stream_add(idx_samples: idx_samples+M-1) = y_add;
    idx_samples = idx_samples + M;
end

end
