using JuMP
using Gurobi
using JSON
using CSV
using DataFrames

# 1. LOAD DATA
parameters = JSON.parsefile("Part 2/Model data/parameter_data.json")

demand_data = CSV.read(
    "Part 2/Model data/demand_norway.csv",
    DataFrame
)

wind_data = CSV.read(
    "Part 2/Model data/wind_onshore_3_days.csv",
    DataFrame
)

# 2. DEFINE SETS
N = ["NO1", "NO2", "NO3", "NO4", "NO5"]
H = collect(1:24)
Ω = ["low", "regular", "high"]

# Directed arcs
A = [
    (item["FromNode"], item["ToNode"])
    for item in parameters["n_tran"]
]

# Undirected physical transmission lines
L = unique([
    Tuple(sort([a[1], a[2]]))
    for a in A
])

# Incoming arcs of node n
A_in = Dict(
    n => [a for a in A if a[2] == n]
    for n in N
)

# Outgoing arcs of node n
A_out = Dict(
    n => [a for a in A if a[1] == n]
    for n in N
)

# Directed arcs associated with each line
A_l = Dict(
    l => [a for a in A if Set(a) == Set(l)]
    for l in L
)

# 3. LOAD PARAMETERS
c_gen = Float64(parameters["c_gen"])
c_tran = Float64(parameters["c_tran"])
q_gen = Float64(parameters["q_gen"])

x_bar_gen = Float64(parameters["x_bar_gen"])
V_gen = Float64(parameters["V_gen"])
alpha = Float64(parameters["alpha"])

wind_demand_share = 0.10

prob = Dict(
    omega => Float64(parameters["prob"])
    for omega in Ω
)

# Transmission efficiency indexed by directed arc
eta_tran = Dict(
    (item["FromNode"], item["ToNode"]) =>
        Float64(item["lineEfficiency"])
    for item in parameters["n_tran"]
)

# Initial transmission capacity indexed by physical line.
x_bar_tran = Dict(
    Tuple(sort([item["FromNode"], item["ToNode"]])) =>
        Float64(item["capacity"])
    for item in parameters["x_bar_tran"]
)

# Scale demand to wind_demand_share% of the original values
xi_load = Dict(
    (n, h) =>
        wind_demand_share * Float64(demand_data[h, Symbol(n)])
    for n in N, h in H
)

#Wind availability indexed by time and scenario
xi_gen = Dict{Tuple{Int, String}, Float64}()

for omega in Ω
    scenario_rows = wind_data[
        string.(wind_data.scenario) .== omega,
        :
    ]

    sort!(scenario_rows, :time)

    for h in H
        xi_gen[(h, omega)] =
            Float64(scenario_rows[h, :wind_availability])
    end
end

# 4. STOCHASTIC MODEL

function build_stochastic_model(
    N, L, A, H, Ω,
    A_in, A_out, A_l,
    c_gen, c_tran, prob, q_gen,
    eta_tran, xi_load, xi_gen,
    x_bar_gen, x_bar_tran,
    V_gen, alpha
)

    model = Model(Gurobi.Optimizer)

    # First-stage investment decisions
    @variable(model, x_gen >= 0)
    @variable(model, x_tran[l in L] >= 0)

    # Second-stage operating decisions
    @variable(model, y_gen[n in N, h in H, omega in Ω] >= 0)
    @variable(model, y_flow[a in A, h in H, omega in Ω] >= 0)

    # Objective function
    @objective(
        model,
        Min,

        c_gen * x_gen

        + c_tran * sum(x_tran[l] for l in L)

        + alpha * sum(
            prob[omega] * q_gen * y_gen[n, h, omega]
            for n in N, h in H, omega in Ω
        )
    )

    # Power balance
    @constraint(
        model,
        power_balance[n in N, h in H, omega in Ω],

        y_gen[n, h, omega]

        + sum(
            eta_tran[a] * y_flow[a, h, omega]
            for a in A_in[n]
        )

        - sum(
            y_flow[a, h, omega]
            for a in A_out[n]
        )

        == xi_load[(n, h)]
    )

    # Aggregate generation capacity
    @constraint(
        model,
        generation_capacity[h in H, omega in Ω],

        sum(y_gen[n, h, omega] for n in N)

        <= xi_gen[(h, omega)] * (x_bar_gen + x_gen)
    )

    # Transmission capacity for each directed arc
    @constraint(
        model,
        transmission_capacity[l in L, a in A_l[l],
                              h in H, omega in Ω],

        y_flow[a, h, omega] <=
            x_bar_tran[l] + x_tran[l]
    )

    # Maximum installed generation capacity
    @constraint(
        model,
        max_inst_gen,

        x_bar_gen + x_gen <= V_gen
    )

    return model
end

# 5. BUILD AND SOLVE THE STOCHASTIC MODEL
model = build_stochastic_model(
    N, L, A, H, Ω,
    A_in, A_out, A_l,
    c_gen, c_tran, prob, q_gen,
    eta_tran, xi_load, xi_gen,
    x_bar_gen, x_bar_tran,
    V_gen, alpha
)

optimize!(model)

println("Termination status: ", termination_status(model))
println("Primal status: ", primal_status(model))

println("Objective value: ", objective_value(model))

println("\nGeneration investment:")
println("New capacity: ", value(model[:x_gen]), " MW")

println("\nTransmission investments:")

for l in L
    println(
        l,
        ": new = ", value(model[:x_tran][l]), " MW",
        ", total = ",
        x_bar_tran[l] + value(model[:x_tran][l]), " MW"
    )
end

println("\nTotal generation by scenario:")

for omega in Ω
    total_generation = sum(
        value(model[:y_gen][n, h, omega])
        for n in N, h in H
    )

    println(
        omega,
        ": ", total_generation, " MWh over the 24-hour horizon"
    )
end
