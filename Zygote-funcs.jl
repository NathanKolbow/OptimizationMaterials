using Zygote, SNaQ, PhyloNetworks, DataFrames

# eqn is the data objects that store the equation to compute the expected concordance factors
# obsCFs are the observed concordance factors (input data)
struct Datum
    eqn::SNaQ.RecursiveCFEquation
    obsCFs::Vector{Float64}
end

# Recursively computes the expected concordance factor of a given equation for parameters θ
function compute_expectedCF_recur(curr_eq::SNaQ.RecursiveCFEquation, qidx::Int, θ)
    if curr_eq.division_H == -1
        # Treelike
        totalloss = 0.0
        expsum = exp(-sum(θ[j] for j in curr_eq.coal_edges; init=0.0))
        if qidx == curr_eq.which_coal
            return 1 - 2/3 * expsum
        else
            return 1/3 * expsum
        end
    elseif length(curr_eq.divisions) == 2
        # Hybrid with 2-way split
        # curr_eq.can_coalesce_here is NEVER true when length(div) == 2
        return θ[curr_eq.division_H] * compute_expectedCF_recur(curr_eq.divisions[1], qidx, θ) +
            (1.0 - θ[curr_eq.division_H]) * compute_expectedCF_recur(curr_eq.divisions[2], qidx, θ)
    else
        # Hybrid with 4-way split
        γ = θ[curr_eq.division_H]
        early_coal_exp_sum = exp(-sum(θ[j] for j in curr_eq.coal_edges; init=0.0))
        splitprobs = early_coal_exp_sum * (
            γ^2 * compute_expectedCF_recur(curr_eq.divisions[1], qidx, θ) +
            (1.0 - γ)^2 * compute_expectedCF_recur(curr_eq.divisions[2], qidx, θ) +
            (1.0 - γ)*γ * compute_expectedCF_recur(curr_eq.divisions[3], qidx, θ) +
            (1.0 - γ)*γ * compute_expectedCF_recur(curr_eq.divisions[4], qidx, θ)
        )

        if qidx == curr_eq.which_coal
            return (1 - early_coal_exp_sum) + splitprobs
        else
            return splitprobs
        end
    end
end

# Computes the contribution to the total loss made by datum `d`
function contribution(d::Datum, θ)
    return sum(d.obsCFs[qidx] * log(compute_expectedCF_recur(d.eqn, qidx, θ) / d.obsCFs[qidx]) for qidx=1:3)
end


# Functions for computing the loss (composite log likelihood) with Zygote
function Zygote_loss(eqns::Vector{SNaQ.QuartetData}, obsCF::Matrix{Float64}, θ)
    data = [Datum(eq.eqn, max.(r, 1e-12)) for (eq, r) in zip(eqns, eachrow(obsCF))]
    return sum(contribution(d, θ) for d in data)
end
Zygote_loss(net::HybridNetwork, obsCF::Matrix{Float64}, θ) = Zygote_loss(SNaQ.find_quartet_equations(net)[1], obsCF, θ)

# Functions for computing the gradient of the loss with Zygote
function Zygote_gradient(eqns::Vector{SNaQ.QuartetData}, obsCF::Matrix{Float64}, θ)
    data = [Datum(eq.eqn, max.(r, 1e-12)) for (eq, r) in zip(eqns, eachrow(obsCF))]
    f = t -> sum(contribution(dat, t) for dat in data)
    return Zygote.gradient(f, θ)
end
function Zygote_gradient(net::HybridNetwork, gts::Vector{HybridNetwork})
    eqs, _, params, _ = SNaQ.find_quartet_equations(net);
    q = gt2CFs(gts);
    return Zygote_gradient(eqs, q, params)
end
Zygote_gradient(net::HybridNetwork, obsCF::Matrix{Float64}, θ) = Zygote_gradient(SNaQ.find_quartet_equations(net)[1], obsCF, θ)

# Functions for computing the hessian of the loss with Zygote
function Zygote_hessian(eqns::Vector{SNaQ.QuartetData}, obsCF::Matrix{Float64}, θ)
    data = [Datum(eq.eqn, max.(r, 1e-12)) for (eq, r) in zip(eqns, eachrow(obsCF))]
    f = t -> sum(contribution(dat, t) for dat in data)
    return Zygote.hessian(f, θ)
end
function Zygote_hessian(net::HybridNetwork, gts::Vector{HybridNetwork})
    eqs, _, params, _ = SNaQ.find_quartet_equations(net);
    q = gt2CFs(gts);
    return Zygote_hessian(eqs, q, params)
end
Zygote_hessian(net::HybridNetwork, obsCF::Matrix{Float64}, θ) = Zygote_hessian(SNaQ.find_quartet_equations(net)[1], obsCF, θ)

# Helper function that converts a set of gene trees into CFs (input data)
function gt2CFs(gts::Vector{HybridNetwork})
    q, t = countquartetsintrees(gts; showprogressbar=false);
    og_stdout = stdout
    redirect_stdout(devnull)
    try
        return SNaQ.gather_expectedCF_matrix(readtableCF(DataFrame(tablequartetCF(q, t))))
    catch e
        rethrow(e)
    finally
        redirect_stdout(og_stdout)
    end
end
