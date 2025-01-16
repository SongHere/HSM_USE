function reconstructed_image = AdaptiveBlockCS_SPL_DDWT(y, PhiSet, params, x0_blk, num_rows, num_cols)

num_levels = params.level;
lambda = params.lambda;
max_iterations = 200;

block_size = sqrt(size(x0_blk, 1));

TOL = params.TOL;
D_prev = 0;

x = x0_blk;

for i = 1:max_iterations
    %     disp(['The ', num2str(i), 'th Iteration!']);
    [x, D] = SPLIteration(y, x, PhiSet, block_size, num_rows, num_cols, ...
        lambda, num_levels);

    if ((D_prev ~= 0) && (abs(D-D_prev) < TOL))
        break;
    end
    D_prev = D;
end

end_level = 1;
[x, D] = SPLIteration(y, x, PhiSet, block_size, num_rows, num_cols, ...
    lambda, num_levels, 'last');

reconstructed_image = col2im(x, [block_size, block_size], ...
    [num_rows, num_cols], 'distict');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [x, D] = SPLIteration(y, x, PhiSet, block_size, num_rows, num_cols, ...
    lambda, num_levels, last)

if (exist('nor_dualtree.mat'))
    load nor_dualtree
else
    normaliz_coefcalc_dual_tree
end

[Faf, Fsf] = AntonB;
[af, sf] = dualfilt1;

x = col2im(x, [block_size, block_size], ...
    [num_rows, num_cols], 'distinct');
x_hat = wiener2(x, [3, 3]);
x_hat = im2col(x_hat, [block_size, block_size], 'distinct');

for ii = 1:size(x_hat, 2)
    Phi_B = PhiSet{ii};
    x_hat(:, ii) = x_hat(:, ii) + Phi_B' * ((Phi_B * Phi_B')^(-1)) * ...
        (y{ii} - Phi_B * x_hat(:, ii));
end

x1 = col2im(x_hat, [block_size, block_size], ...
    [num_rows, num_cols], 'distinct');

x_check = normcoef(cplxdual2D(symextend(x1, 2^(num_levels - 1)), ...
    num_levels, Faf, af), num_levels, nor);

if (nargin == 9)
    end_level = 1;
else
    end_level = num_levels - 1;
end
x_check = SPLBivariateShrinkage(x_check, end_level, lambda);

x_bar = icplxdual2D(unnormcoef(x_check, num_levels, nor), num_levels, Fsf, sf);
Irow = (2^(num_levels - 1) + 1):(2^(num_levels - 1) + num_rows);
Icol = (2^(num_levels - 1) + 1):(2^(num_levels - 1) + num_cols);
x_bar = x_bar(Irow, Icol);
x_bar = im2col(x_bar, [block_size, block_size], 'distinct');

for ii = 1:size(x_bar, 2)
    Phi_B = PhiSet{ii};
    x_bar(:, ii) = x_bar(:, ii) + Phi_B' * ((Phi_B * Phi_B')^(-1)) * ...
        (y{ii} - Phi_B * x_bar(:, ii));
end
x = x_bar;

x2 = col2im(x, [block_size, block_size], ...
    [num_rows, num_cols], 'distinct');

D = RMS(x1, x2);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function x_check = SPLBivariateShrinkage(x_check, end_level, lambda)

windowsize = 3;
windowfilt = ones(1, windowsize) / windowsize;

tmp = x_check{1}{1}{1}{1};
Nsig = median(abs(tmp(:))) / 0.6745;

for scale = 1:end_level
    for dir = 1:2
        for dir1 = 1:3
            Y_coef_real = x_check{scale}{1}{dir}{dir1};
            Y_coef_imag = x_check{scale}{2}{dir}{dir1};
            Y_parent_real = x_check{scale+1}{1}{dir}{dir1};
            Y_parent_imag = x_check{scale+1}{2}{dir}{dir1};
            Y_parent_real = expand(Y_parent_real);
            Y_parent_imag = expand(Y_parent_imag);

            Wsig = conv2(windowfilt, windowfilt, (Y_coef_real).^2, 'same');
            Ssig = sqrt(max(Wsig-Nsig.^2, eps));

            T = sqrt(3) * Nsig^2 ./ Ssig;

            Y_coef = Y_coef_real + sqrt(-1) * Y_coef_imag;
            Y_parent = Y_parent_real + sqrt(-1) * Y_parent_imag;
            Y_coef = bishrink(Y_coef, Y_parent, T*lambda);

            x_check{scale}{1}{dir}{dir1} = real(Y_coef);
            x_check{scale}{2}{dir}{dir1} = imag(Y_coef);
        end
    end
end
