function Phi_set = PhiSetMaker(Phi_full, M_list)

block_num = length(M_list);
Phi_set = cell(block_num, 1);
for idx_block = 1:block_num
    Phi_set{idx_block} = Phi_full(1:M_list(idx_block), :);
end

end
