module SamRandom

const MAX_VAL = 2^32 - 1
global state = 0x31415926

function rand(N...)
  global state

  if isempty(N)
    N = 1
  end

  n_nums = prod(N)

  out = Array{Float32, length(N)}(undef, N...)
  for i_num = 1:n_nums
    # x^32 + x^22 + x^2 + x^1 + 1
    bit = (state >> 0) ⊻ (state >> 10) ⊻ (state >> 30) ⊻ (state >> 31)
    state = (state >> 1) | (bit << 31)

    out[i_num] = state / MAX_VAL
  end

  out
end

end