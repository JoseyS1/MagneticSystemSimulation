# Import modules
using BenchmarkTools, Plots, CUDA

# Define constants
MAX_ITERATIONS = 500
GRID_SIZE = 1000
X_LIM = [-0.748766713922161 -0.748766707771757]
Y_LIM = [ 0.123640844894862  0.123640851045266]

# Setup
x = collect(range(X_LIM[1], X_LIM[2], length=GRID_SIZE))
y = collect(range(Y_LIM[1], Y_LIM[2], length=GRID_SIZE))

z0 = zeros(ComplexF64, GRID_SIZE, GRID_SIZE)
for i = 1:GRID_SIZE
  for j = 1:GRID_SIZE
    @inbounds z0[i, j] = x[i] + 1im * y[j]
  end
end
z = deepcopy(z0)
count = zeros(Float64, GRID_SIZE, GRID_SIZE)

# GPU Setup
z0_d = CuArray(z0)
z_d = CuArray(z)
count_d = CUDA.zeros(Float64, GRID_SIZE, GRID_SIZE)

function mandelbrot!(
    count::Union{CuArray{Float64, 2}, Array{Float64, 2}}, 
    z0::Union{CuArray{ComplexF64, 2}, Array{ComplexF64, 2}}, 
    z::Union{CuArray{ComplexF64, 2}, Array{ComplexF64, 2}}, 
    MAX_ITER::Int)::Nothing

  z .= z0
  count .= 0.0
  for n = 0:MAX_ITER
    z .= z.*z .+ z0
    count .+= abs.(z) .<= 2
  end
  count .= log.(count)

  nothing
end

# mandelbrot!(count, z0, z, MAX_ITERATIONS)
# @btime mandelbrot!(count, z0, z, MAX_ITERATIONS)

mandelbrot!(count_d, z0_d, z_d, MAX_ITERATIONS)
copyto!(count, count_d)
display(heatmap(x, y, count, framestyle=:none, colorbar=false))