function psnr = Psnr(signal_1, signal_2)
% PSNR in dB calculation function of two signals
signal_1 = double(signal_1);
signal_2 = double(signal_2);
error = signal_1 - signal_2;
psnr = 10 * log10(255^2/mean(error(:).^2)); % 8-bit
end