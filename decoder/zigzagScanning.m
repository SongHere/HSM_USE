function [zigzag_data, col_data, idx_col2zig] = zigzagScanning(matrix)
[rows, cols] = size(matrix);
idx_col2zig = [];

solution = zeros(rows*cols, 1);
index = 1;

for sum = 1:(rows + cols)
    if mod(sum, 2) == 0
        i = min(sum-1, rows);
        j = sum - i;
        while i >= 1 && j <= cols
            solution(index) = matrix(i, j);
            idx_col2zig = [idx_col2zig; i + (j - 1) * rows];
            index = index + 1;
            i = i - 1;
            j = j + 1;
        end
    else
        j = min(sum-1, cols);
        i = sum - j;
        while j >= 1 && i <= rows
            solution(index) = matrix(i, j);
            idx_col2zig = [idx_col2zig; i + (j - 1) * rows];
            index = index + 1;
            j = j - 1;
            i = i + 1;
        end
    end
end

zigzag_data = solution;

col_data = matrix(:);
end
