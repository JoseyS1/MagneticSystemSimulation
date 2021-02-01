module MagneticSimulation

using Random, Statistics, Printf

const IntRange{T} = StepRange{T, T} where T <: Integer
const FloatRange{T, K} = StepRangeLen{T, K, K} where {T <: AbstractFloat, K <: AbstractFloat}

struct JValues{T}
  p1::Array{T, 3}
  m1::Array{T, 3}
  p2::Array{T, 3}
  m2::Array{T, 3}
  p3::Array{T, 3}
  m3::Array{T, 3}
end

struct Configuration{T}
  mag::Array{T, 2}
  mag2::Array{T, 2}
  mag4::Array{T, 2}
  susc::Array{T, 2}
  bin::Array{T, 2}
  en::Array{T, 2}
  en2::Array{T, 2}
  sph::Array{T, 2}
end

function test()
  Random.seed!(314)

  # Define constants
  L_RANGE = 10:1:10
  LT_RANGE = 20:10:25
  T_RANGE = 2f0:-0.2f0:1f0

  N_CONF = 5
  N_EQ = 200
  N_MESS = 200

  IMP_CONC = 0f0

  DIST() = flatDist(1f0, 1f0)

  runSimulation(L_RANGE, LT_RANGE, T_RANGE, N_CONF, N_EQ, N_MESS, IMP_CONC, DIST)
end

function runSimulation(l_range::IntRange{S}, lt_range::IntRange{S}, t_range::FloatRange{T, K}, 
    n_conf::S, n_eq::S, n_mess::S, imp_conc::T, j_dist::Function) where {S, T, K}

  n_temp = length(t_range)
  conf = Configuration(
    zeros(T, n_temp, n_conf),
    zeros(T, n_temp, n_conf),
    zeros(T, n_temp, n_conf),
    zeros(T, n_temp, n_conf),
    zeros(T, n_temp, n_conf),
    zeros(T, n_temp, n_conf),
    zeros(T, n_temp, n_conf),
    zeros(T, n_temp, n_conf)
  )
  cell = Array{Configuration{T}, 2}(undef, length(l_range), length(lt_range))

  for (i_l, l) = enumerate(l_range)
    for (i_lt, lt) = enumerate(lt_range)
      checkerboard = genCheckerboard(l, lt)

      l3 = lt*l*l
      q_space = T(2.0 * pi / l)
      q_time = T(2.0 * pi / lt)

      for k = 1:n_conf
        println("L = $l , Lt = $lt: Disorder configuration $k of $n_conf")

        j_vals = initSiteCouplings(l, lt, T, j_dist)

        occupied = initSiteImpurities(l, lt, imp_conc)

        hs = dropdims(getRSphere(T, l, lt, 1), dims=4)

        s_block = occupied .* hs

        for (i_temp, temp) = enumerate(t_range)
          beta = T(1.0 / temp)

          metroSweep!(s_block, l, lt, beta, occupied, j_vals, checkerboard, n_eq)
          

          # write("phase_post.bin", s_block)

          en, en2, mag, mag2, mag4 = measurement(s_block, l, lt, beta, occupied, j_vals, checkerboard, n_mess)

          conf.en[i_temp, k] = en
          conf.en2[i_temp, k] = en2
          conf.mag[i_temp, k] = mag
          conf.mag2[i_temp, k] = mag2
          conf.mag4[i_temp, k] = mag4
          conf.susc[i_temp, k] = (mag2 - mag^2) * l3 * beta
          conf.bin[i_temp, k] = 1 - mag4/(3 * mag2^2)
          conf.sph[i_temp, k] = (en2 - en^2) * l3 * beta^2
        end
      end

      cell[i_l, i_lt] = deepcopy(conf)
    end
  end

  binder = mean(1 .- cell[1, 1].mag4[:, 1:n_conf] ./ (3 .* cell[1, 1].mag2[:, 1:n_conf].^2), dims=2)

  println(collect(t_range))
  println()
  println(binder)
end

function metroSweep!(s_block::Array{T, 4}, l::S, lt::S, beta::T, occupied::Array{Bool, 3}, 
    j_vals::JValues{T}, checkerboard::Array{Bool, 3}, n_eq::S)::Nothing where {T <: AbstractFloat, S <: Integer}

  step_size = 25

  p1 = circshift(1:l, -1)
  m1 = circshift(1:l, 1)

  tp1 = circshift(1:lt, -1)
  tm1 = circshift(1:lt, 1)

  jrs = cat(
    repeat(j_vals.p1, outer=(1, 1, 1, 3)),
    repeat(j_vals.p2, outer=(1, 1, 1, 3)),
    repeat(j_vals.p3, outer=(1, 1, 1, 3)),
    repeat(j_vals.m1, outer=(1, 1, 1, 3)),
    repeat(j_vals.m2, outer=(1, 1, 1, 3)),
    repeat(j_vals.m3, outer=(1, 1, 1, 3)),
    dims=5
  )

  # write("phase_sBlock_0_0.bin", s_block)

  j = nothing
  rands = zeros(T, l, l, lt, 2*step_size)
  n_spins_test = zeros(T, l, l, lt, 3, 2*step_size)
  netxyz = zeros(T, l, l, lt, 3)
  dE = zeros(T, l, l, lt)
  swap = zeros(Bool, l, l, lt)
  for i = 1:n_eq
    if mod(i, step_size) == 1
      permutedims!(n_spins_test, getRSphere(T, l, lt, 2*step_size), [1, 2, 3, 5, 4])
      n_spins_test .*= occupied

      rands .= rand(T, l, l, lt, 2*step_size)
      j = 1
    end

    # Checkerboard
    n_spins = @view n_spins_test[:, :, :, :, j]

    netxyz .= dropdims(
      sum(
        cat(
          s_block[p1, :, :, :],
          s_block[m1, :, :, :],
          s_block[:, p1, :, :],
          s_block[:, m1, :, :],
          s_block[:, :, tp1, :],
          s_block[:, :, tm1, :],
          dims=5
        ) .* jrs,
        dims=5
      ),
      dims=5
    )

    dE .= dropdims(sum(netxyz .* (s_block .- n_spins), dims=4), dims=4)
    swap .= ((dE .< 0f0) .| (exp.(-dE.*beta) .> rands[:, :, :, j])) .& checkerboard
    s_block .= swap .* n_spins .+ .!swap .* s_block
    # write("phase_sBlock_$(i)_$(j).bin", s_block)
    j += 1

    # Inverse Checkerboard
    n_spins = @view n_spins_test[:, :, :, :, j]

    netxyz .= dropdims(
      sum(
        cat(
          s_block[p1, :, :, :],
          s_block[m1, :, :, :],
          s_block[:, p1, :, :],
          s_block[:, m1, :, :],
          s_block[:, :, tp1, :],
          s_block[:, :, tm1, :],
          dims=5
        ) .* jrs,
        dims=5
      ),
      dims=5
    )

    dE .= dropdims(sum(netxyz .* (s_block .- n_spins), dims=4), dims=4)
    swap .= ((dE .< 0f0) .| (exp.(-dE.*beta) .> rands[:, :, :, j])) .& .!checkerboard
    s_block .= swap .* n_spins .+ .!swap .* s_block
    # write("phase_sBlock_$(i)_$(j).bin", s_block)
    j += 1
  end
end

function measurement(s_block::Array{T, 4}, l::S, lt::S, beta::T, occupied::Array{Bool, 3}, 
    j_vals::JValues{T}, checkerboard::Array{Bool, 3}, n_mess::S)::Tuple{T, T, T, T, T} where {T <: AbstractFloat, S <: Integer}

  step_size = 25

  p1 = circshift(1:l, -1)
  m1 = circshift(1:l, 1)

  tp1 = circshift(1:lt, -1)
  tm1 = circshift(1:lt, 1)

  jrs = cat(
    repeat(j_vals.p1, outer=(1, 1, 1, 3)),
    repeat(j_vals.p2, outer=(1, 1, 1, 3)),
    repeat(j_vals.p3, outer=(1, 1, 1, 3)),
    repeat(j_vals.m1, outer=(1, 1, 1, 3)),
    repeat(j_vals.m2, outer=(1, 1, 1, 3)),
    repeat(j_vals.m3, outer=(1, 1, 1, 3)),
    dims=5
  )

  en = T(0)
  en2 = T(0)
  mag = T(0)
  mag2 = T(0)
  mag4 = T(0)

  # write("phase_sBlock_0_0.bin", s_block)

  j = nothing
  rands = zeros(T, l, l, lt, 2*step_size)
  n_spins_test = zeros(T, l, l, lt, 3, 2*step_size)
  netxyz = zeros(T, l, l, lt, 3)
  dE = zeros(T, l, l, lt)
  swap = zeros(Bool, l, l, lt)
  for i = 1:n_mess-1
    if mod(i, step_size) == 1
      permutedims!(n_spins_test, getRSphere(T, l, lt, 2*step_size), [1, 2, 3, 5, 4])
      n_spins_test .*= occupied

      rands .= rand(T, l, l, lt, 2*step_size)
      j = 1
    end

    netxyz .= j_vals.p1 .* s_block[p1, :, :, :] .+ j_vals.p2 .* s_block[:, p1, :, :] .+ j_vals.p3 .* s_block[:, :, tp1, :]

    sweepen = reduce(+, netxyz .* s_block)

    en_inc = sweepen / T(l*l*lt)
    en2_inc = en_inc^2

    m_xyz = [
      reduce(+, s_block[:, :, :, 1]),
      reduce(+, s_block[:, :, :, 2]),
      reduce(+, s_block[:, :, :, 3])
    ]

    mag_inc = sqrt(sum(m_xyz.^2)) / T(l*l*lt)
    mag2_inc = mag_inc^2
    mag4_inc = mag_inc^4

    en   += (en_inc   - en) / i
    en2  += (en2_inc  - en2) / i
    mag  += (mag_inc  - mag) / i
    mag2 += (mag2_inc - mag2) / i
    mag4 += (mag4_inc - mag4) / i

    # Metro Sweep
    # Checkerboard
    n_spins = @view n_spins_test[:, :, :, :, j]

    netxyz .= dropdims(
      sum(
        cat(
          s_block[p1, :, :, :],
          s_block[m1, :, :, :],
          s_block[:, p1, :, :],
          s_block[:, m1, :, :],
          s_block[:, :, tp1, :],
          s_block[:, :, tm1, :],
          dims=5
        ) .* jrs,
        dims=5
      ),
      dims=5
    )

    dE .= dropdims(sum(netxyz .* (s_block .- n_spins), dims=4), dims=4)
    swap .= ((dE .< 0f0) .| (exp.(-dE.*beta) .> rands[:, :, :, j])) .& checkerboard
    s_block .= swap .* n_spins .+ .!swap .* s_block
    # write("phase_sBlock_$(i)_$(j).bin", s_block)
    j += 1

    # Inverse Checkerboard
    n_spins = @view n_spins_test[:, :, :, :, j]

    netxyz .= dropdims(
      sum(
        cat(
          s_block[p1, :, :, :],
          s_block[m1, :, :, :],
          s_block[:, p1, :, :],
          s_block[:, m1, :, :],
          s_block[:, :, tp1, :],
          s_block[:, :, tm1, :],
          dims=5
        ) .* jrs,
        dims=5
      ),
      dims=5
    )

    dE .= dropdims(sum(netxyz .* (s_block .- n_spins), dims=4), dims=4)
    swap .= ((dE .< 0f0) .| (exp.(-dE.*beta) .> rands[:, :, :, j])) .& .!checkerboard
    s_block .= swap .* n_spins .+ .!swap .* s_block
    # write("phase_sBlock_$(i)_$(j).bin", s_block)
    j += 1
  end

  (en, en2, mag, mag2, mag4)
end

# Define helperfunctions
function genCheckerboard(l::T, lt::T)::Array{Bool, 3} where T <: Integer
  checkerboard = zeros(Bool, l, l, lt)

  for i = 1:l
    for j = 1:l
      for k = 1:lt
        @inbounds checkerboard[i, j, k] = mod(i+j+k, 2) == 0
      end
    end
  end

  checkerboard
end

function initSiteCouplings(l::T, lt::T, S::DataType, j_dist::Function)::JValues{S} where T
  j_vals = JValues(zeros(S, l, l, lt), zeros(S, l, l, lt), ones(S, l, l, lt), zeros(S, l, l, lt), zeros(S, l, l, lt), zeros(S, l, l, lt))

  for i = 1:l
    for j = 1:l
      j_vals.p1[i, j, :] .= j_dist()
      j_vals.p2[i, j, :] .= j_dist()
      j_vals.p3[i, j, :] .= j_dist()
    end
  end

  circshift!(j_vals.m1, j_vals.p1, (1, 0, 0))
  circshift!(j_vals.m2, j_vals.p2, (0, 1, 0))
  circshift!(j_vals.m3, j_vals.p3, (0, 0, 1))

  j_vals
end

function initSiteImpurities(l::T, lt::T, imp_conc::S)::Array{Bool, 3} where {T <: Integer, S <: AbstractFloat}
  occu = zeros(Bool, l, l, lt)
  rs = rand(S, l, l)

  rs_bool = rs .> imp_conc
  for i = 1:l
    for j = 1:l
      occu[i, j, :] .= rs_bool[i, j]
    end
  end

  occu
end

function getRSphere(S::DataType, l::T, lt::T, n::T)::Array{S, 5} where {T <: Integer}
  i = l * l * lt * n

  rs = 2.0 .* rand(S, i, 2)
  elev = asin.(rs[:, 1] .- 1.0)
  az = pi .* rs[:, 2]

  rcos_elev = cos.(elev)

  firstPass = cat(
    rcos_elev .* cos.(az), 
    rcos_elev .* sin.(az),
    sin.(elev),
    dims=2
  )

  reshape(firstPass, l, l, lt, n, 3)
end

function tailHeavyDist(y::T)::T where T <: AbstractFloat
  rand(T)^y
end

function flatDist(min_value::T, max_value::T)::T where T <: AbstractFloat
  max_value != min_value ? rand(T) * (max_value-min_value) + min_value : max_value
end
end