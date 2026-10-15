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

# Directed arcs associated with each physical line
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

# Transmission efficiency indexed by directed arc
eta_tran = Dict(
    (item["FromNode"], item["ToNode"]) =>
        Float64(item["lineEfficiency"])
    for item in parameters["n_tran"]
)

# Initial transmission capacity indexed by physical line
x_bar_tran = Dict(
    Tuple(sort([item["FromNode"], item["ToNode"]])) =>
        Float64(item["capacity"])
    for item in parameters["x_bar_tran"]
)

# Demand scaled to 10% of original values
xi_load = Dict(
    (n, h) =>
        wind_demand_share * Float64(demand_data[h, Symbol(n)])
    for n in N, h in H
)

# Wind availability indexed by hour and scenario
xi_gen = Dict{Tuple{Int, String}, Float64}()

for omega in Ω
    scenario_rows = wind_data[
        string.(wind_data.scenario) .== omega,
        :
    ]

    sort!(scenario_rows, :time)

    nrow(scenario_rows) == length(H) ||
        error(
            "Expected 24 wind observations for scenario $omega, " *
            "found $(nrow(scenario_rows))"
        )

    for h in H
        xi_gen[(h, omega)] =
            Float64(scenario_rows[h, :wind_availability])
    end
end

# 4. BUILD A MODEL FOR ONE KNOWN SCENARIO
function build_scenario_model(
    N, L, A, H,
    A_in, A_out, A_l,
    c_gen, c_tran, q_gen,
    eta_tran, xi_load, xi_gen,
    x_bar_gen, x_bar_tran,
    V_gen, alpha,
    omega
)

    model = Model(Gurobi.Optimizer)

    # Investment decisions
    @variable(model, x_gen >= 0)
    @variable(model, x_tran[l in L] >= 0)

    # Operating decisions for the known scenario
    @variable(model, y_gen[n in N, h in H] >= 0)
    @variable(model, y_flow[a in A, h in H] >= 0)

    # Objective function
    @objective(
        model,
        Min,

        c_gen * x_gen

        + c_tran * sum(x_tran[l] for l in L)

        + alpha * q_gen * sum(
            y_gen[n, h]
            for n in N, h in H
        )
    )

    # Power balance
    @constraint(
        model,
        power_balance[n in N, h in H],

        y_gen[n, h]

        + sum(
            eta_tran[a] * y_flow[a, h]
            for a in A_in[n]
        )

        - sum(
            y_flow[a, h]
            for a in A_out[n]
        )

        == xi_load[(n, h)]
    )

    # Generation capacity
    @constraint(
        model,
        generation_capacity[h in H],

        sum(y_gen[n, h] for n in N)

        <= xi_gen[(h, omega)] * (x_bar_gen + x_gen)
    )

    # Transmission capacity for each directed arc
    @constraint(
        model,
        transmission_capacity[l in L, a in A_l[l], h in H],

        y_flow[a, h] <=
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

# 5. SOLVE EACH SCENARIO INDEPENDENTLY
results = Dict{String, Any}()

for omega in Ω

    println("SCENARIO: ", uppercase(omega))

    model = build_scenario_model(
        N, L, A, H,
        A_in, A_out, A_l,
        c_gen, c_tran, q_gen,
        eta_tran, xi_load, xi_gen,
        x_bar_gen, x_bar_tran,
        V_gen, alpha,
        omega
    )

    optimize!(model)

    println("Termination status: ", termination_status(model))
    println("Primal status: ", primal_status(model))

    obj = objective_value(model)
    new_gen = value(model[:x_gen])

    println("Objective value: ", obj)

    println("\nGeneration investment:")
    println("New capacity: ", new_gen, " MW")
    println(
        "Total installed capacity: ",
        x_bar_gen + new_gen, " MW"
    )

    println("\nTransmission investments:")

    transmission_investments = Dict{Any, Float64}()

    for l in L
        new_capacity = value(model[:x_tran][l])
        transmission_investments[l] = new_capacity

        println(
            l,
            ": new = ", new_capacity, " MW",
            ", total = ",
            x_bar_tran[l] + new_capacity, " MW"
        )
    end

    total_generation = sum(
        value(model[:y_gen][n, h])
        for n in N, h in H
    )

    println("\nTotal generation: ", total_generation, " MWh")

    # Store results for comparison
    results[omega] = (
        objective = obj,
        new_generation_capacity = new_gen,
        total_generation_capacity = x_bar_gen + new_gen,
        transmission_investments = transmission_investments,
        total_generation = total_generation
    )
end

# 6. COMPARE THE SCENARIO-SPECIFIC SOLUTIONS
println("SCENARIO COMPARISON")

for omega in Ω
    if haskey(results, omega)
        r = results[omega]

        println("\nScenario: ", omega)
        println("Objective value: ", r.objective)
        println(
            "New generation capacity: ",
            r.new_generation_capacity, " MW"
        )
        println(
            "Total generation capacity: ",
            r.total_generation_capacity, " MW"
        )
        println("Total generation: ", r.total_generation, " MWh")
    else
        println("\nScenario: ", omega, " — no solution available")
    end
end

