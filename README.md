# Optimization Materials

See `example.jl` for a well documented walkthrough on how to simulate data and then compute the loss, gradient, and hessian with this data. The hessian can only be computed here with Zygote.jl, whereas the loss and gradient can also be computed by Zygote.jl as well as by functions in SNaQ.jl.

Examples of ground-truth models to simulate data from are provided in `networks/` with `networks/n10h1.netfile` loaded by default (`n10h1` meaning 10 taxa, or leaves, and 1 hybrid node). We would consider `n10h1` to be a relatively small example, `n20h2` to be moderately sized, and `n30h4` to be large in terms of what SNaQ can feasibly infer at present.

Gene trees (`gts` in `example.jl`) are the data users are expected to have (or are expected to be able to infer from DNA sequences), but concordance factors calculated from these gene trees (`observed_CFs` in `examples.jl`) are the input data used for SNaQ. Generally speaking, we would typically consider less than 100 gene trees to be relatively little input data, 100-1000 gene trees to be a moderate amount of input data, and more than 1000 gene trees to be a large but still realistic amount of input data.
