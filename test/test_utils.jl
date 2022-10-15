
"""
Test gradients through zygote. 

# Arguments

- `f`: function to test
- `xs`: inputs to `f`

# Keyword Arguments
Keyword arguments are passed to `rrule`.

- `fkwargs`: keyword arguments to `f`
"""
function test_zygote(f, xs...; kws...)
    config = ZygoteRuleConfig()
    test_rrule(config, f, xs...; kws..., rrule_f = rrule_via_ad)
end

function rr(seed)
    rng = Random.seed!(seed)
    return rng
end
