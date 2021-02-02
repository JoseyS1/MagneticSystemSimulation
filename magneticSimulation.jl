module MagneticSimulation

using Random, Statistics, Printf, InteractiveUtils

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
  L_RANGE = 30:1:30
  LT_RANGE = 40:10:40
  T_RANGE = 2f0:-0.2f0:1f0

  N_CONF = 1
  N_EQ = 200
  N_MESS = 200

  IMP_CONC = 0f0

  DIST() = flatDist(1f0, 1f0)

  runSimulation(L_RANGE, LT_RANGE, T_RANGE, N_CONF, N_EQ, N_MESS, IMP_CONC, DIST)
end

function runSimulation(l_range::IntRange{S}, lt_range::IntRange{S}, t_range::FloatRange{T, K}, 
    n_conf::S, n_eq::S, n_mess::S, imp_conc::T, j_dist::Function)::Nothing where {S, T, K}

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

      s_block = zeros(T, l, l, lt, 3)
      sphere_buffer = zeros(T, l, l, lt, 3)

      for k = 1:n_conf
        println("L = $l , Lt = $lt: Disorder configuration $k of $n_conf")

        j_vals = initSiteCouplings(l, lt, T, j_dist)

        occupied = initSiteImpurities(l, lt, imp_conc)

        getRSphere!(s_block, sphere_buffer)
        s_block .*= occupied

        for (i_temp, temp) = enumerate(t_range)
          beta = T(1.0 / temp)

          metroSweep!(s_block, l, lt, beta, occupied, j_vals, checkerboard, n_eq, sphere_buffer)

          en, en2, mag, mag2, mag4 = measurement(s_block, l, lt, beta, occupied, j_vals, checkerboard, n_mess, sphere_buffer)

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
  println(binder)

  nothing
end

function metroSweep!(s_block::Array{T, 4}, l::S, lt::S, beta::T, occupied::Array{Bool, 3}, 
    j_vals::JValues{T}, checkerboard::Array{Bool, 3}, n_eq::S, sphere_buffer::Array{T, 4})::Nothing where {T <: AbstractFloat, S <: Integer}

  step_size = 25

  p1 = circshift(1:l, -1)
  m1 = circshift(1:l, 1)

  tp1 = circshift(1:lt, -1)
  tm1 = circshift(1:lt, 1)

  rands = zeros(T, l, l, lt)
  n_spins = zeros(T, l, l, lt, 3)
  temp_spins = zeros(T, l, l, lt, 3)
  netxyz = zeros(T, l, l, lt, 3)
  dE = zeros(T, l, l, lt)
  swap = zeros(Bool, l, l, lt)
  for i = 1:n_eq
    # Checkerboard
    getRSphere!(n_spins, sphere_buffer)
    n_spins .*= occupied
    rands .= rand(T, l, l, lt)

    netxyz .= s_block[p1, :, :, :] .* j_vals.p1 .+ 
              s_block[m1, :, :, :] .* j_vals.m1 .+
              s_block[:, p1, :, :] .* j_vals.p2 .+ 
              s_block[:, m1, :, :] .* j_vals.m2 .+
              s_block[:, :, tp1, :] .* j_vals.p3 .+ 
              s_block[:, :, tm1, :] .* j_vals.m3

    temp_spins .= s_block .- n_spins
    dE .= netxyz[:, :, :, 1] .* temp_spins[:, :, :, 1] .+
          netxyz[:, :, :, 2] .* temp_spins[:, :, :, 2] .+
          netxyz[:, :, :, 3] .* temp_spins[:, :, :, 3]
    swap .= ((dE .< 0f0) .| (exp.(-dE.*beta) .> rands)) .& checkerboard
    s_block .= swap .* n_spins .+ .!swap .* s_block

    # Inverse Checkerboard
    getRSphere!(n_spins, sphere_buffer)
    n_spins .*= occupied
    rands .= rand(T, l, l, lt)

    netxyz .= s_block[p1, :, :, :] .* j_vals.p1 .+ 
              s_block[m1, :, :, :] .* j_vals.m1 .+
              s_block[:, p1, :, :] .* j_vals.p2 .+ 
              s_block[:, m1, :, :] .* j_vals.m2 .+
              s_block[:, :, tp1, :] .* j_vals.p3 .+ 
              s_block[:, :, tm1, :] .* j_vals.m3

    temp_spins .= s_block .- n_spins
    dE .= netxyz[:, :, :, 1] .* temp_spins[:, :, :, 1] .+
          netxyz[:, :, :, 2] .* temp_spins[:, :, :, 2] .+
          netxyz[:, :, :, 3] .* temp_spins[:, :, :, 3]
    swap .= ((dE .< 0f0) .| (exp.(-dE.*beta) .> rands)) .& .!checkerboard
    s_block .= swap .* n_spins .+ .!swap .* s_block
  end
end

function measurement(s_block::Array{T, 4}, l::S, lt::S, beta::T, occupied::Array{Bool, 3}, 
    j_vals::JValues{T}, checkerboard::Array{Bool, 3}, n_mess::S, sphere_buffer::Array{T, 4})::Tuple{T, T, T, T, T} where {T <: AbstractFloat, S <: Integer}

  step_size = 25

  p1 = circshift(1:l, -1)
  m1 = circshift(1:l, 1)

  tp1 = circshift(1:lt, -1)
  tm1 = circshift(1:lt, 1)

  en = T(0)
  en2 = T(0)
  mag = T(0)
  mag2 = T(0)
  mag4 = T(0)

  rands = zeros(T, l, l, lt)
  n_spins = zeros(T, l, l, lt, 3)
  temp_spins = zeros(T, l, l, lt, 3)
  netxyz = zeros(T, l, l, lt, 3)
  dE = zeros(T, l, l, lt)
  swap = zeros(Bool, l, l, lt)
  for i = 1:n_mess-1
    # Measurement
    netxyz .= j_vals.p1 .* s_block[p1, :, :, :] .+ j_vals.p2 .* s_block[:, p1, :, :] .+ j_vals.p3 .* s_block[:, :, tp1, :]

    temp_spins .= netxyz .* s_block
    en_inc = sum(temp_spins) / T(l*l*lt)
    en2_inc = en_inc^2

    mag_inc = sqrt(
      sum(@view s_block[:, :, :, 1])^2 + 
      sum(@view s_block[:, :, :, 2])^2 + 
      sum(@view s_block[:, :, :, 3])^2
    ) / T(l*l*lt)
    mag2_inc = mag_inc^2
    mag4_inc = mag_inc^4

    en   += (en_inc   - en) / i
    en2  += (en2_inc  - en2) / i
    mag  += (mag_inc  - mag) / i
    mag2 += (mag2_inc - mag2) / i
    mag4 += (mag4_inc - mag4) / i

    # Metro Sweep
    # Checkerboard
    getRSphere!(n_spins, sphere_buffer)
    n_spins .*= occupied
    rands .= rand(T, l, l, lt)

    netxyz .= s_block[p1, :, :, :] .* j_vals.p1 .+ 
              s_block[m1, :, :, :] .* j_vals.m1 .+
              s_block[:, p1, :, :] .* j_vals.p2 .+ 
              s_block[:, m1, :, :] .* j_vals.m2 .+
              s_block[:, :, tp1, :] .* j_vals.p3 .+ 
              s_block[:, :, tm1, :] .* j_vals.m3

    temp_spins .= s_block .- n_spins
    dE .= netxyz[:, :, :, 1] .* temp_spins[:, :, :, 1] .+
          netxyz[:, :, :, 2] .* temp_spins[:, :, :, 2] .+
          netxyz[:, :, :, 3] .* temp_spins[:, :, :, 3]
    swap .= ((dE .< 0f0) .| (exp.(-dE.*beta) .> rands)) .& checkerboard
    s_block .= swap .* n_spins .+ .!swap .* s_block

    # Inverse Checkerboard
    getRSphere!(n_spins, sphere_buffer)
    n_spins .*= occupied
    rands .= rand(T, l, l, lt)

    netxyz .= s_block[p1, :, :, :] .* j_vals.p1 .+ 
              s_block[m1, :, :, :] .* j_vals.m1 .+
              s_block[:, p1, :, :] .* j_vals.p2 .+ 
              s_block[:, m1, :, :] .* j_vals.m2 .+
              s_block[:, :, tp1, :] .* j_vals.p3 .+ 
              s_block[:, :, tm1, :] .* j_vals.m3

    temp_spins .= s_block .- n_spins
    dE .= netxyz[:, :, :, 1] .* temp_spins[:, :, :, 1] .+
          netxyz[:, :, :, 2] .* temp_spins[:, :, :, 2] .+
          netxyz[:, :, :, 3] .* temp_spins[:, :, :, 3]
    swap .= ((dE .< 0f0) .| (exp.(-dE.*beta) .> rands)) .& .!checkerboard
    s_block .= swap .* n_spins .+ .!swap .* s_block
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

function getRSphere!(r_sphere::Array{T, 4}, buffer::Array{T, 4})::Nothing where T <: AbstractFloat
  buffer[:, :, :, 1] .= T.(asin.(2 .* rand(T, size(r_sphere)[1:end-1]...) .- 1))
  buffer[:, :, :, 2] .= T.(2 .* pi .* rand(T, size(r_sphere)[1:end-1]...))
  buffer[:, :, :, 3] .= cos.(@view buffer[:, :, :, 1])

  @inbounds r_sphere[:, :, :, 1] .= buffer[:, :, :, 3] .* cos.(@view buffer[:, :, :, 2])
  @inbounds r_sphere[:, :, :, 2] .= buffer[:, :, :, 3] .* sin.(@view buffer[:, :, :, 2])
  @inbounds r_sphere[:, :, :, 3] .= sin.(@view buffer[:, :, :, 1])

  nothing
end

function tailHeavyDist(y::T)::T where T <: AbstractFloat
  rand(T)^y
end

function flatDist(min_value::T, max_value::T)::T where T <: AbstractFloat
  max_value != min_value ? rand(T) * (max_value-min_value) + min_value : max_value
end
end