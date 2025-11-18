using Pkg
Pkg.activate(@__DIR__)
Pkg.update()

# Make sure that this is the version from the treechild-galled branch of the
# JuliaPhylo/SNaQ.jl GitHub repo (this version is specified in the Project.toml
# but requires Julia v1.11 or higher)
Pkg.status("SNaQ")

using PhyloCoalSimulations
include("Zygote-funcs.jl")

# Select a network to optimize
net = readnewick("networks/n10h1.netfile");

# Gather the set of equations that are used to compute expected values
eqns, _, trueparameters, _ = SNaQ.find_quartet_equations(net);

# Generate some input data to optimize over
gts = simulatecoalescent(net, 100, 1);  # 100 gene trees, 1 of each taxa
observed_values = gt2CFs(gts);

# This is how we would optimize with the current approach
loss = SNaQ.optimize_bls!(net, eqns, observed_values)
optimizedparameters = SNaQ.find_quartet_equations(net)[3];
hcat(trueparameters, optimizedparameters)

# How to compute just the gradient with the hard-coded implementation for a fixed set of parameters
gradientstorage = zeros(length(trueparameters));
SNaQ.compute_loss_and_gradient!(eqns, optimizedparameters, gradientstorage, observed_values)

# How to compute the loss, gradient, and hessian with Zygote
# NOTE: compuitng the loss with Zygote in this way is quite fast, but when using
#       the most complex example (n30h4) the gradient computation takes ~1 minute
#       on my machine and the hessian computation takes ~5 minutes on my machine
zygloss = Zygote_loss(net, observed_values, optimizedparameters)
zyggrad = Zygote_gradient(net, observed_values, optimizedparameters)[1]
zyghess = Zygote_hessian(net, observed_values, optimizedparameters)

# Double-check that the hard-coded loss and gradient are within rounding error
abs(zygloss - loss)
maximum(abs.(zyggrad .- gradientstorage))
