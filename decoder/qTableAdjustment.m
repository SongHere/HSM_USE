% JPEG quantization table adjustment function

function output = qTableAdjustment(quantization_table, quality_factor)
% Input:
% quantization_table: the orignal q_table at 80 quality factor
% quality_factor: the target quality factor

if quality_factor < 50
    scale_factor = 5000 / quality_factor;
else
    scale_factor = 200 - 2 * quality_factor;
end
output = floor((scale_factor * quantization_table + 50)/100);
output(output < 1) = 1;
output(output > 255) = 255;
end