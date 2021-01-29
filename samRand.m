function out = samRand(varargin)
    persistent sRand_state;
    
    if isempty(sRand_state)
        sRand_state = uint32(0x31415926);
    end
    
    if nargin == 0
        N = {1};
    elseif nargin == 1
        N = [varargin, {1}];
    else
        N = varargin;
    end
    
    out = zeros(N{:});
    n_nums = prod(cell2mat(N));
    for i_num = 1:n_nums
        % x^32 + x^22 + x^2 + x^1 + 1
        bit = bitxor(bitxor(bitxor(bitshift(sRand_state, 0), bitshift(sRand_state, -10)), bitshift(sRand_state, -30)), bitshift(sRand_state, -31));
        sRand_state = bitor(bitshift(sRand_state, -1), bitshift(bit, 31));

        out(i_num) = single(double(sRand_state) / (2^32 - 1));
    end
end
