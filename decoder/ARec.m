function x_block = ARec(y_cell, Phi_set, Rxx, block_size)
% linear reconstruction
x_block = [];
block_num = length(y_cell);
for idx_block = 1:block_num
    Phi = Phi_set{idx_block};
    if (size(Phi, 1) == block_size^2)
        H = inv(Phi);
    else
        H = Rxx * Phi' * (Phi * Rxx * Phi')^(-1);
    end
    x = H * y_cell{idx_block};
    x_block = [x_block, x];
end
end