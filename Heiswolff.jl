using Random, Statistics, Printf, BenchmarkTools, Plots
include("samRandom.jl")

# Define main function
function Heiswolff()
  # Seed random number generator
  Random.seed!(314)

  # Define constants
  LMin = 25
  LMax = 25
  Lsz = 1

  LtMin = 40
  LtMax = 40
  Ltsz = 10

  NCONF = 5
  NEQ = 200
  NMESS = 200 # Monte Carlo sweeps
  TMIN = 1f0
  TMAX = 2f0
  DT = -0.05f0 # Temperature control
  NTEMP = Int(ceil(1+(TMIN-TMAX) / DT))

  impconc = 0f0 # 0.407253 # Impurities as decimal

  Flat_Dist = true
  Tail_Heavy_Dist = false # Distribution of site coupling strenth if both are set to true the tail heavy distribution is always used
  MIN = 1f0 # Minimum value for flat distribution
  MAX = 1f0 # Maximum value for flat distribution
  y_exponent = 0f0 # Exponent value for tail heavy distribution

  # Number of disorder configs
  CANON_DIS = false
  PRINT_ALL = false # Print data about individual sweeps
  WRITE_ALL = false # Write individual configs

  NCLUSTER0 = 10 # Initial cluster flips per sweep

  # Allocate arrays
  conf = Conf(NTEMP, NCONF)
  cell = Array{Conf, 2}(undef, LMax, LtMax)

  # Run main loop
  for L = LMin:Lsz:LMax
    for Lt = LtMin:Ltsz:LtMax
      checkerboard = generateCheckerboard(L, Lt)
      checkerboardOther = collect(.!checkerboard)

      L2 = L*L
      L3 = Lt*L2          # L = linear system size
      qspace = 2f0*pi/L
      qtime = 2f0*pi/Lt   # Minimum q values for correlation length

      for k = 1:NCONF
        println("L = $L , Lt = $Lt: Disorder configuration $k of $NCONF")

        # Reset spins

        # Initialize stie couplings
        j1p, j1m, j2p, j2m, j3p, j3m = initializeSiteCouplings_Block(
          L, Lt, Flat_Dist, Tail_Heavy_Dist, MIN, MAX, y_exponent)

        # Initialize impurities
        occu = initializeSiteImpurities(L, Lt, impconc)

        # Initialize spins
        hs = getRSphere_WholeArray(L, Lt)
        sBlock = occu .* hs

        # Loop over temperatures
        ncluster = NCLUSTER0
        T = TMAX
        for itemp = 1:NTEMP
          beta = 1f0 / T

          # Equilibration, carry out NEQ full MC steps
          metro_sweep_faster_Combined_allGPU!(
            sBlock, L, Lt, beta, occu, j1p, j1m, j2p, j2m, j3p, j3m, checkerboard, checkerboardOther, NEQ)

          # for is = 1:1
          #   wolff_sweep_notGPU!(sBlock, L, Lt, beta, occu, j1p, j1m, j2p, j2m, j3p, j3m, NCLUSTER0)
          # end

          totalflip = 0f0

          avclsize = totalflip / ncluster / (NEQ / 2f0)
          ncluster = L3 / avclsize + 0.5f0 + 0.2f0

          # Measuring loop, carry out NMESS full MC sweeps
          mag = 0f0
          mag2 = 0f0
          mag4 = 0f0
          en = 0f0
          en2 = 0f0
          magqt = 0f0
          magqs = 0f0
          Gt = 0f0
          Gs = 0f0

          totalflip = 0f0

          en, en2, mag, mag2, mag4 = measurement_faster_combinedMetro(
            deepcopy(sBlock), L, Lt, beta, occu, j1p, j1m, j2p, j2m, j3p, j3m, checkerboard, checkerboardOther, NMESS)

          magqt = magqt / NMESS
          magqs = magqs / NMESS
          Gt = Gt / NMESS
          Gs = Gs / NMESS

          susc = (mag2 - mag^2) * L3 * beta
          bin = 1 - mag4/(3 * mag2^2)
          sph = (en2 - en^2) * L3 * beta^2
          Gtcon = Gt - magqt^2
          Gscon = Gs - magqs^2

          xit = (mag2 - Gt) / (Gt * qtime * qtime)
          xit = sqrt(abs(xit))
          xis = (mag2 - Gs) / (Gs * qspace * qspace)
          xis = sqrt(abs(xis))
          xitcon = ((mag2 - mag^2) - Gtcon) / (Gtcon * qtime * qtime)
          xitcon = sqrt(abs(xitcon))
          xiscon = ((mag2 - mag^2) - Gscon) / (Gscon * qspace * qspace)
          xiscon = sqrt(abs(xiscon))

          # Collect data
          conf.temperature[itemp, k] = T
          conf.mag[itemp, k] = mag
          conf.sq_mag[itemp, k] = mag^2
          conf.mag2[itemp, k] = mag2
          conf.mag4[itemp, k] = mag4
          conf.logmag[itemp, k] = log(mag)
          conf.susc[itemp, k] = susc
          conf.bin[itemp, k] = bin
          conf.sq_susc[itemp, k] = susc^2
          conf.sq_bin[itemp, k] = bin^2
          conf.en[itemp, k] = en
          conf.sph[itemp, k] = sph
          conf.sq_en[itemp, k] = en^2
          conf.sq_sph[itemp, k] = sph^2
          conf.Gt[itemp, k] = Gt
          conf.Gs[itemp, k] = Gs
          conf.Gtcon[itemp, k] = Gtcon
          conf.Gscon[itemp, k] = Gscon
          conf.xit[itemp, k] = xit
          conf.xis[itemp, k] = xis
          conf.xitcon[itemp, k] = xitcon
          conf.xiscon[itemp, k] = xiscon

          avclsize = totalflip/ncluster/NMESS
          ncluster = L3/avclsize + 0.5 + 2.0

          T += DT
        end # Temperatur
      end

      cell[L, Lt] = deepcopy(conf)
    end
  end

  temp = cell[LMin, LtMin].temperature[:, NCONF]
  binder = mean(1f0 .- cell[LMin, LtMin].mag4[:,1:NCONF] ./ (3f0.*cell[LMin, LtMin].mag2[:,1:NCONF].^2), dims=2);
  # err = std(1f0 .- cell[LMIN, LTMIN].mag4[:,1:NCONF] ./ (3f0.*cell[LMIN, LTMIN].mag2[:,1:NCONF].^2),0,2) ./ sqrt(NCONF);


  minB = 4 * ones(Float32, size(temp, 1)) / 9
  maxB = 2 * ones(Float32, size(temp, 1)) / 3

  display(plot(temp, binder))
  display(plot!(temp, minB))
  display(plot!(temp, maxB))
  display(plot!([3 / 2, 3 / 2], LinRange(4 / 9, 2 / 3, 2)))







  println(temp)
  println()
  println(binder)

  cell
end

# Define helper functions
function generateCheckerboard(L::Int, Lt::Int)::Array{Bool, 3}
  checkerboard = zeros(Bool, L, L, Lt)

  for i = 1:L
    for j = 1:L
      for k = 1:Lt
        checkerboard[i, j, k] = mod(i+j+k, 2) == 0
      end
    end
  end

  # Return
  checkerboard
end

function initializeSiteCouplings_Block(L::Int, Lt::Int, Flat_Dist::Bool, Tail_Heavy_Dist::Bool,
    MIN::Float32, MAX::Float32,
    y_exponent::Float32)::Tuple{Array{Float32, 3}, Array{Float32, 3}, Array{Float32, 3}, Array{Float32, 3}, Array{Float32, 3}, Array{Float32, 3}}
  j1p = zeros(Float32, L, L, Lt)
  j2p = zeros(Float32, L, L, Lt)
  j3p = ones(Float32, L, L, Lt)

  for i = 1:L
    for j = 1:L
      j1p[i, j, :] .= jvalue(Flat_Dist, Tail_Heavy_Dist, MIN, MAX, y_exponent)
      j2p[i, j, :] .= jvalue(Flat_Dist, Tail_Heavy_Dist, MIN, MAX, y_exponent)
      # j3p[i, j, :] .= jvalue(Flat_Dist, Tail_Heavy_Dist, MIN, MAX, y_exponent)
    end
  end

  j1m = circshift(j1p, (1, 0, 0))
  j2m = circshift(j2p, (0, 1, 0))
  j3m = circshift(j3p, (0, 0, 1))

  # Return
  (j1p, j1m, j2p, j2m, j3p, j3m)
end

function jvalue(Flat_Dist::Bool, Tail_Heavy_Dist::Bool, MIN::Float32, MAX::Float32, y_exponent::Float32)::Float32
  if Tail_Heavy_Dist
    return SamRandom.rand()^y_exponent
  elseif Flat_Dist
    if MAX != MIN
      return SamRandom.rand() * (MAX-MIN) + MIN
    else
      return MAX
    end
  end
end

function initializeSiteImpurities(L::Int, Lt::Int, impconc::Float32)::Array{Bool, 3}
  occu = zeros(Bool, L, L, Lt)
  rs = SamRandom.rand(L, L)

  rs_bool = rs .> impconc
  for i = 1:L
    for j = 1:L
      occu[i, j, :] .= rs_bool[i, j]
    end
  end

  # Return
  occu
end

function getRSphere_WholeArray(L::Int, Lt::Int)::Array{Float32, 4}
  i = L^2 * Lt

  rs = 2f0 .* SamRandom.rand(i, 2)
  elev = asin.(rs[:, 1] .- 1f0)
  az = pi .* rs[:, 2]

  rcoselev = cos.(elev)

  firstPass = cat(
    rcoselev .* cos.(az),
    rcoselev .* sin.(az),
    sin.(elev),
    dims=2
  )

  # Return
  reshape(firstPass, L, L, Lt, 3)
end

function getRSphere_WholeArray_N(L::Int, Lt::Int, N::Int)::Array{Float32, 5}
  i = L^2 * Lt * N

  rs = 2f0 .* SamRandom.rand(i, 2)
  elev = asin.(rs[:, 1] .- 1f0)
  az = pi .* rs[:, 2]

  rcoselev = cos.(elev)

  firstPass = cat(
    rcoselev .* cos.(az),
    rcoselev .* sin.(az),
    sin.(elev),
    dims=2
  )

  # Return
  reshape(firstPass, L, L, Lt, N, 3)
end

function metro_sweep_faster_Combined_allGPU!(sBlock::Array{Float32, 4}, L::Int, Lt::Int,
  beta::Float32, occu::Array{Bool, 3}, j1p::Array{Float32, 3}, j1m::Array{Float32, 3},
  j2p::Array{Float32, 3}, j2m::Array{Float32, 3}, j3p::Array{Float32, 3}, j3m::Array{Float32, 3},
  checkerboard::Array{Bool, 3}, checkerboardOther::Array{Bool, 3}, NEQ::Int)::Nothing

  p1 = circshift(1:L, -1)
  m1 = circshift(1:L, 1)

  tp1 = circshift(1:Lt, -1)
  tm1 = circshift(1:Lt, 1)

  j1pr = repeat(j1p, outer=(1, 1, 1, 3))
  j2pr = repeat(j2p, outer=(1, 1, 1, 3))
  j3pr = repeat(j3p, outer=(1, 1, 1, 3))

  j1mr = repeat(j1m, outer=(1, 1, 1, 3))
  j2mr = repeat(j2m, outer=(1, 1, 1, 3))
  j3mr = repeat(j3m, outer=(1, 1, 1, 3))

  jrs = cat(j1pr, j2pr, j3pr, j1mr, j2mr, j3mr, dims=5)

  j = 1
  neqStepSize = 25

  rands = []
  nSpinsTest = []
  for i = 1:NEQ
    if mod(i, neqStepSize) == 1
      nSpinsTest = getRSphere_WholeArray_N(L, Lt, 2*NEQ)
      nSpinsTest = permutedims(nSpinsTest, [1, 2, 3, 5, 4])
      nSpinsTest = occu .* nSpinsTest

      rands = SamRandom.rand(L, L, Lt, 2*NEQ)
      j = 1
    end

    nSpins = nSpinsTest[:, :, :, :, j]

    s1p = sBlock[p1, :, :, :]
    s1m = sBlock[m1, :, :, :]
    s2p = sBlock[:, p1, :, :]
    s2m = sBlock[:, m1, :, :]
    s3p = sBlock[:, :, tp1, :]
    s3m = sBlock[:, :, tm1, :]

    netxyz = sum(cat(s1p, s2p, s3p, s1m, s2m, s3m, dims=5) .* jrs, dims=5)

    ds = sBlock .- nSpins
    dE = sum(netxyz .* ds, dims=4)

    swap = ((dE .< 0f0) .| (exp.(-dE.*beta) .> rands[:, :, :, j])) .& checkerboard
    j += 1
    notswap = .!swap
    sBlock .= dropdims(swap .* nSpins .+ notswap .* sBlock, dims=5)

    # Other checkerboard
    nSpins = nSpinsTest[:, :, :, :, j]

    s1p = sBlock[p1, :, :, :]
    s1m = sBlock[m1, :, :, :]
    s2p = sBlock[:, p1, :, :]
    s2m = sBlock[:, m1, :, :]
    s3p = sBlock[:, :, tp1, :]
    s3m = sBlock[:, :, tm1, :]

    netxyz = sum(cat(s1p, s2p, s3p, s1m, s2m, s3m, dims=5) .* jrs, dims=5)

    ds = sBlock .- nSpins
    dE = sum(netxyz .* ds, dims=4)

    swap = ((dE .< 0f0) .| (exp.(-dE.*beta) .> rands[:, :, :, j])) .& checkerboardOther
    j += 1
    notswap = .!swap
    sBlock .= dropdims(swap .* nSpins .+ notswap .* sBlock, dims=5)
  end

  # Return
  nothing
end

function wolff_sweep_notGPU!(sBlock::Array{Float32, 4}, L::Int, Lt::Int,
  beta::Float32, occu::Array{Bool, 3}, j1p::Array{Float32, 3}, j1m::Array{Float32, 3},
  j2p::Array{Float32, 3}, j2m::Array{Float32, 3}, j3p::Array{Float32, 3}, j3m::Array{Float32, 3}, NCLUSTER::Int)::Nothing

  o1p, o1m, o2p, o2m, o3p, o3m = updateNeighborSpins(occu)

  c1p = collect(1:L) .+ 1
  c1p[end] = 1
  c1m = collect(1:L) .- 1
  c1m[1] = L

  c2p = c1p
  c2m = c1m

  c3p = collect(1:Lt) .+ 1
  c3p[end] = 1
  c3m = collect(1:Lt) .- 1
  c3m[1] = Lt

  stack = zeros(Int, L^2 * Lt, 3)

  nflip = 0
  icluster = 0

  while icluster < NCLUSTER
    addedTo = zeros(Bool, L, L, Lt)
    is = cat(rand(1:L, 2), rand(1:Lt, 1), dims=1)

    if occu[is[1], is[2], is[3]]
      addedTo[is[1], is[2], is[3]] = 1
      icluster += 1

      sp = 1
      stack[sp, :] .= is

      oldSpins = [sBlock[is[1], is[2], is[3], 1], sBlock[is[1], is[2], is[3], 2], sBlock[is[1], is[2], is[3], 3]]
      nSpins = getRSphere_WholeArray(1, 1)

      isize = 1
      scalar2 = sum(nSpins .* oldSpins)

      sBlock[is[1], is[2], is[3], :] .= oldSpins .- 2f0 * scalar2 .* nSpins[:]
      helpS = rand(Float64, 6)

      while sp > 0
        c = stack[sp, :]
        cS = [sBlock[c[1], c[2], c[3], 1], sBlock[c[1], c[2], c[3], 2], sBlock[c[1], c[2], c[3], 3]]
        scalar1 = -sum(nSpins .* cS)

        sp -= 1
        c1 = c[1]
        c2 = c[2]
        c3 = c[3]
        allC = [c1p[c1] c2 c3;c1m[c1] c2 c3;c1 c2p[c2] c3;c1 c2m[c2] c3;c1 c2 c3p[c3];c1 c2 c3m[c3]]

        if o1p[c[1], c[2], c[3]] && !addedTo[allC[1,1], allC[1, 2], allC[1, 3]]
          lo = allC[1, :]
          cS = [sBlock[lo[1], lo[2], lo[3], 1], sBlock[lo[1], lo[2], lo[3], 2], sBlock[lo[1], lo[2], lo[3], 3]]
          scalar2 = sum(nSpins .* cS)
          snsn = scalar1 * scalar2

          if snsn > 0
            padd = 1f0 - exp(-2f0*j1p[c[1], c[2], c[3]]*beta*snsn)
            help = helpS[1]
            if help < padd
              ### NEED TO ACTUALLY UPDATE THE SPIN
              sBlock[lo[1], lo[2], lo[3], :] .= cS .- 2f0*scalar2.*nSpins[:]
              addedTo[lo[1], lo[2], lo[3]] = true

              sp += 1
              stack[sp, :] .= lo

              isize += 1
            end
          end
        end

        if o1m[c[1], c[2], c[3]] && !addedTo[allC[2,1], allC[2, 2], allC[2, 3]]
          lo = allC[2, :]
          cS = [sBlock[lo[1], lo[2], lo[3], 1], sBlock[lo[1], lo[2], lo[3], 2], sBlock[lo[1], lo[2], lo[3], 3]]
          scalar2 = sum(nSpins .* cS)
          snsn = scalar1 * scalar2

          if snsn > 0
            padd = 1f0 - exp(-2f0*j1m[c[1], c[2], c[3]]*beta*snsn)
            help = helpS[2]
            if help < padd
              ### NEED TO ACTUALLY UPDATE THE SPIN
              sBlock[lo[1], lo[2], lo[3], :] .= cS .- 2f0*scalar2.*nSpins[:]
              addedTo[lo[1], lo[2], lo[3]] = true

              sp += 1
              stack[sp, :] .= lo

              isize += 1
            end
          end
        end

        if o2p[c[1], c[2], c[3]] && !addedTo[allC[3,1], allC[3, 2], allC[3, 3]]
          lo = allC[3, :]
          cS = [sBlock[lo[1], lo[2], lo[3], 1], sBlock[lo[1], lo[2], lo[3], 2], sBlock[lo[1], lo[2], lo[3], 3]]
          scalar2 = sum(nSpins .* cS)
          snsn = scalar1 * scalar2

          if snsn > 0
            padd = 1f0 - exp(-2f0*j2p[c[1], c[2], c[3]]*beta*snsn)
            help = helpS[3]
            if help < padd
              ### NEED TO ACTUALLY UPDATE THE SPIN
              sBlock[lo[1], lo[2], lo[3], :] .= cS .- 2f0*scalar2.*nSpins[:]
              addedTo[lo[1], lo[2], lo[3]] = true

              sp += 1
              stack[sp, :] .= lo

              isize += 1
            end
          end
        end

        if o2m[c[1], c[2], c[3]] && !addedTo[allC[4,1], allC[4, 2], allC[4, 3]]
          lo = allC[4, :]
          cS = [sBlock[lo[1], lo[2], lo[3], 1], sBlock[lo[1], lo[2], lo[3], 2], sBlock[lo[1], lo[2], lo[3], 3]]
          scalar2 = sum(nSpins .* cS)
          snsn = scalar1 * scalar2

          if snsn > 0
            padd = 1f0 - exp(-2f0*j2m[c[1], c[2], c[3]]*beta*snsn)
            help = helpS[4]
            if help < padd
              ### NEED TO ACTUALLY UPDATE THE SPIN
              sBlock[lo[1], lo[2], lo[3], :] .= cS .- 2f0*scalar2.*nSpins[:]
              addedTo[lo[1], lo[2], lo[3]] = true

              sp += 1
              stack[sp, :] .= lo

              isize += 1
            end
          end
        end

        if o3p[c[1], c[2], c[3]] && !addedTo[allC[5,1], allC[5, 2], allC[5, 3]]
          lo = allC[5, :]
          cS = [sBlock[lo[1], lo[2], lo[3], 1], sBlock[lo[1], lo[2], lo[3], 2], sBlock[lo[1], lo[2], lo[3], 3]]
          scalar2 = sum(nSpins .* cS)
          snsn = scalar1 * scalar2

          if snsn > 0
            padd = 1f0 - exp(-2f0*j3p[c[1], c[2], c[3]]*beta*snsn)
            help = helpS[5]
            if help < padd
              ### NEED TO ACTUALLY UPDATE THE SPIN
              sBlock[lo[1], lo[2], lo[3], :] .= cS .- 2f0*scalar2.*nSpins[:]
              addedTo[lo[1], lo[2], lo[3]] = true

              sp += 1
              stack[sp, :] .= lo

              isize += 1
            end
          end
        end

        if o3m[c[1], c[2], c[3]] && !addedTo[allC[6,1], allC[6, 2], allC[6, 3]]
          lo = allC[6, :]
          cS = [sBlock[lo[1], lo[2], lo[3], 1], sBlock[lo[1], lo[2], lo[3], 2], sBlock[lo[1], lo[2], lo[3], 3]]
          scalar2 = sum(nSpins .* cS)
          snsn = scalar1 * scalar2

          if snsn > 0
            padd = 1f0 - exp(-2f0*j3m[c[1], c[2], c[3]]*beta*snsn)
            help = helpS[6]
            if help < padd
              ### NEED TO ACTUALLY UPDATE THE SPIN
              sBlock[lo[1], lo[2], lo[3], :] .= cS .- 2f0*scalar2.*nSpins[:]
              addedTo[lo[1], lo[2], lo[3]] = true

              sp += 1
              stack[sp, :] .= lo

              isize += 1
            end
          end
        end

      end

      nflip += isize
    end
  end

  println(@sprintf("Temperature = %.4f, cluster size = %.3f", 1f0 / beta, Float32(nflip) / Float32(NCLUSTER*L^2*Lt)))

  # Return
  nothing
end

function updateNeighborSpins(s::Array{Bool, 3})::Tuple{Array{Bool, 3}, Array{Bool, 3}, Array{Bool, 3}, Array{Bool, 3}, Array{Bool, 3}, Array{Bool, 3}}
  # Return
  ( circshift(s, (-1, 0, 0)),
    circshift(s, (1, 0, 0)),
    circshift(s, (0, -1, 0)),
    circshift(s, (0, 1, 0)),
    circshift(s, (0, 0, -1)),
    circshift(s, (0, 0, 1))
  )
end

function measurement_faster_combinedMetro(sBlock::Array{Float32, 4}, L::Int, Lt::Int,
  beta::Float32, occu::Array{Bool, 3}, j1p::Array{Float32, 3}, j1m::Array{Float32, 3},
  j2p::Array{Float32, 3}, j2m::Array{Float32, 3}, j3p::Array{Float32, 3}, j3m::Array{Float32, 3},
  checkerboard::Array{Bool, 3}, checkerboardOther::Array{Bool, 3}, NEQ::Int)::Tuple{Float32, Float32, Float32, Float32, Float32}

  p1 = circshift(1:L, -1)
  m1 = circshift(1:L, 1)

  tp1 = circshift(1:Lt, -1)
  tm1 = circshift(1:Lt, 1)

  j1pr = repeat(j1p, outer=(1, 1, 1, 3))
  j2pr = repeat(j2p, outer=(1, 1, 1, 3))
  j3pr = repeat(j3p, outer=(1, 1, 1, 3))

  j1mr = repeat(j1m, outer=(1, 1, 1, 3))
  j2mr = repeat(j2m, outer=(1, 1, 1, 3))
  j3mr = repeat(j3m, outer=(1, 1, 1, 3))

  jrs = cat(j1pr, j2pr, j3pr, j1mr, j2mr, j3mr, dims=5)

  j = 1
  neqStepSize = 25

  en = 0f0
  en2 = 0f0
  mag = 0f0
  mag2 = 0f0
  mag4 = 0f0

  rands = []
  nSpinsTest = []
  for i = 1:NEQ-1
    if mod(i, neqStepSize) == 1
      nSpinsTest = getRSphere_WholeArray_N(L, Lt, 2*NEQ)
      nSpinsTest = permutedims(nSpinsTest, [1, 2, 3, 5, 4])
      nSpinsTest = occu .* nSpinsTest

      rands = SamRandom.rand(L, L, Lt, 2*NEQ)
      j = 1
    end

    mxGPU = reduce(+, sBlock[:, :, :, 1])
    myGPU = reduce(+, sBlock[:, :, :, 2])
    mzGPU = reduce(+, sBlock[:, :, :, 3])

    s1p = sBlock[p1, :, :, :]
    s1m = sBlock[m1, :, :, :]
    s2p = sBlock[:, p1, :, :]
    s2m = sBlock[:, m1, :, :]
    s3p = sBlock[:, :, tp1, :]
    s3m = sBlock[:, :, tm1, :]

    netxyz = j1pr.*s1p .+ j2pr.*s2p .+ j3pr.*s3p

    sweepen = reduce(+, netxyz .* sBlock)

    en_inc = sweepen / Float32(L^2 * Lt)
    en2_inc = en_inc^2

    mag_inc = sqrt(mxGPU^2 + myGPU^2 + mzGPU^2) / Float32(L^2 * Lt)
    mag2_inc = mag_inc^2
    mag4_inc = mag_inc^4

    # Metro sweep
    nSpins = nSpinsTest[:, :, :, :, j]
    ds = sBlock .- nSpins
    netxyz = sum(cat(s1p, s2p, s3p, s1m, s2m, s3m, dims=5) .* jrs, dims=5)

    dE = sum(netxyz .* ds, dims=4)

    swap = ((dE .< 0f0) .| (exp.(-dE.*beta) .> rands[:, :, :, j])) .& checkerboard
    j += 1
    notswap = .!swap
    sBlock .= dropdims(swap .* nSpins .+ notswap .* sBlock, dims=5)

    # Other checkerboard
    nSpins = nSpinsTest[:, :, :, :, j]
    ds = sBlock .- nSpins

    s1p = sBlock[p1, :, :, :]
    s1m = sBlock[m1, :, :, :]
    s2p = sBlock[:, p1, :, :]
    s2m = sBlock[:, m1, :, :]
    s3p = sBlock[:, :, tp1, :]
    s3m = sBlock[:, :, tm1, :]

    netxyz = sum(cat(s1p, s2p, s3p, s1m, s2m, s3m, dims=5) .* jrs, dims=5)

    dE = sum(netxyz .* ds, dims=4)

    swap = ((dE .< 0f0) .| (exp.(-dE.*beta) .> rands[:, :, :, j])) .& checkerboardOther
    j += 1
    notswap = .!swap
    sBlock .= dropdims(swap .* nSpins .+ notswap .* sBlock, dims=5)

    en   += (en_inc   - en) / i
    en2  += (en2_inc  - en2) / i
    mag  += (mag_inc  - mag) / i
    mag2 += (mag2_inc - mag2) / i
    mag4 += (mag4_inc - mag4) / i

  end

  # Return
  (en, en2, mag, mag2, mag4)
end

# Define structs
struct Conf
  temperature::Array{Float32, 2}
  mag::Array{Float32, 2}
  mag2::Array{Float32, 2}
  sq_mag::Array{Float32, 2}
  mag4::Array{Float32, 2}
  logmag::Array{Float32, 2}
  susc::Array{Float32, 2}
  bin::Array{Float32, 2}
  sq_susc::Array{Float32, 2}
  sq_bin::Array{Float32, 2}
  en::Array{Float32, 2}
  sph::Array{Float32, 2}
  sq_en::Array{Float32, 2}
  sq_sph::Array{Float32, 2}
  Gt::Array{Float32, 2}
  Gs::Array{Float32, 2}
  Gtcon::Array{Float32, 2}
  Gscon::Array{Float32, 2}
  xit::Array{Float32, 2}
  xis::Array{Float32, 2}
  xitcon::Array{Float32, 2}
  xiscon::Array{Float32, 2}
end
Conf(N::Int, M::Int) = Conf(
  zeros(Float32, N, M),
  zeros(Float32, N, M),
  zeros(Float32, N, M),
  zeros(Float32, N, M),
  zeros(Float32, N, M),
  zeros(Float32, N, M),
  zeros(Float32, N, M),
  zeros(Float32, N, M),
  zeros(Float32, N, M),
  zeros(Float32, N, M),
  zeros(Float32, N, M),
  zeros(Float32, N, M),
  zeros(Float32, N, M),
  zeros(Float32, N, M),
  zeros(Float32, N, M),
  zeros(Float32, N, M),
  zeros(Float32, N, M),
  zeros(Float32, N, M),
  zeros(Float32, N, M),
  zeros(Float32, N, M),
  zeros(Float32, N, M),
  zeros(Float32, N, M)
)

out = Heiswolff()
