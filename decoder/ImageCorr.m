% references

% Lu Gan, "Block Compressed Sensing of Natural Images," 
% 2007 15th Int. Conf. Digital Signal Process.,
% Cardiff, UK, 2007, pp. 403-406, doi: 10.1109/ICDSP.2007.4288604.

% Ran Li, Xiaomeng Duan, Xiaoli Guo, Wei He, Yongfeng Lv, 
% "Adaptive Compressive Sensing of Images Using Spatial Entropy", 
% Computational Intelligence and Neuroscience, vol. 2017, 
% Article ID 9059204, 9 pages, 2017. https://doi.org/10.1155/2017/9059204

function Rxx = ImageCorr(M,N,p)
% Image block autocorrelation matrix generate function
% Input  M-
%        N-
%        p-constant from 0.9 to 1
% OutPut Rxx-(M*N, M*N)autocorrelation matrix

Rxx = zeros(M*N,M*N);
for ii = 1:M*N
    for jj = ii:M*N % row order
        if ii == jj % diagonal line: distance=0, Rxx(ii,jj)=1
            Rxx(ii,jj) = 1;
        else
            % point 1 coordinate
            xii = rem(ii-1,M) + 1;
            yii = fix((ii-1)/M) + 1;
            % point 2 coordinate
            xjj = rem(jj-1,M) + 1;
            yjj = fix((jj-1)/M) + 1;
            % distance
            d = sqrt((xii-xjj)^2 + (yii-yjj)^2);
            % correlation
            Rxx(ii,jj) = p^(d);
            % symmetrically
            Rxx(jj,ii) = Rxx(ii,jj);
        end
    end
end
end
