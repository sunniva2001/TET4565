using JuMP
using Gurobi
using JSON
using CSV
using DataFrames
using Plasmo
using PlasmoBenders


# 1. LOAD DATA
parameters = JSON.parsefile("Part 3/Model data/parameter_data.json")

demand_data = CSV.read(
    "Part 3/Model data/demand_norway.csv",
    DataFrame
)

wind_data = CSV.read(
    "Part 3/Model data/wind_onshore_3_days.csv",
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

# 4. GROUP SETS AND PARAMETERS
sets = (
    N = N,
    H = H,
    Ω = Ω,
    A = A,
    L = L,
    A_in = A_in,
    A_out = A_out,
    A_l = A_l
)

par = (
    c_gen = c_gen,
    c_tran = c_tran,
    q_gen = q_gen,
    x_bar_gen = x_bar_gen,
    x_bar_tran = x_bar_tran,
    V_gen = V_gen,
    alpha = alpha,
    prob = prob,
    eta_tran = eta_tran,
    xi_load = xi_load,
    xi_gen = xi_gen
)

# Scenario probabilities should sum to 1
@assert isapprox(
    sum(values(par.prob)),
    1.0;
    atol = 1e-8
) "Scenario probabilities must sum to 1."




#5. CREATE MASTER NODE 

function create_master_node!(
    master, 
    sets, #nodes, generators, storage, technologies, etc.
    par, #model parameters
)
    #Sets
    L = sets.L

    #First-stage investment variables   
    @variable(master, x_gen >= 0)
    @variable(master, x_tran[l in L] >= 0)

    # Maximum installed generation capacity
    @constraint(
        master,
        max_inst_gen,
        par.x_bar_gen + x_gen <= par.V_gen
    )

    # Investment costs part from objective function
    @objective(
        master,
        Min,
        par.c_gen * x_gen
        + par.c_tran * sum(x_tran[l] for l in L)
    )

    return nothing
end

#6. CREATE SUBPROBLEMs NODE

function create_subproblem_node!(
    node,
    sets,
    par,
    scenarios
)

    N = sets.N
    H = sets.H
    A = sets.A
    A_in = sets.A_in
    A_out = sets.A_out


    # Second-stage operational variables
    @variable(node, y_gen[n in N, h in H] >= 0)
    @variable(node, y_flow[a in A, h in H] >= 0)

    # Power balance for each node and hour
    @constraint(
        node,
        power_balance[n in N, h in H],

        y_gen[n, h]

        + sum(
            par.eta_tran[a] * y_flow[a, h]
            for a in A_in[n]
        )

        - sum(
            y_flow[a, h]
            for a in A_out[n]
        )

        == par.xi_load[(n, h)]
    )


    #objective function
    # Objective function
    @objective(
        node,
        Min,
         par.alpha *  par.prob[scenarios] * par.q_gen * sum(
            y_gen[n, h]
            for n in N, h in H)
    )

    return nothing
end

#7. CREATE LINKING CONSTRAINTS
function create_linking_constraints!(
    graph,
    master,
    scenario_node,
    sets,
    par,
    scenarios
)

    N = sets.N
    H = sets.H
    L = sets.L
    A_l = sets.A_l

    # Aggregate generation capacity
    @linkconstraint(
        graph,
        [h in H],

        sum(
            scenario_node[:y_gen][n, h]
            for n in N
        )

        <=
        par.xi_gen[(h, scenarios)] *
        (
            par.x_bar_gen +
            master[:x_gen]
        )
    )

    # Transmission capacity for each directed arc
    @linkconstraint(
        graph,
        [l in L, a in A_l[l], h in H],

        scenario_node[:y_flow][a, h]

        <=
        par.x_bar_tran[l] +
        master[:x_tran][l]
    )

    return nothing
end

#8. CREATE BENDERS GRAPH

function create_benders_graph(sets, par)

    graph = OptiGraph()

    # ==========================================
    # MASTER
    # ==========================================

    master_graph = OptiGraph()
    master = add_node(master_graph)

    set_name(master, :master)

    create_master_node!(
        master,
        sets,
        par
    )

    set_to_node_objectives(master_graph)
    add_subgraph!(graph, master_graph)

    # ==========================================
    # SCENARIO SUBPROBLEMS
    # ==========================================

    scenario_graphs = Dict{String, Any}()
    scenario_nodes = Dict{String, Any}()

    for scenarios in sets.Ω

        println("Creating scenario: ", scenarios)

        # Create one OptiGraph per scenario
        scenario_graph = OptiGraph()
        scenario_node = add_node(scenario_graph)

        set_name(
            scenario_node,
            Symbol("scenario_$(scenarios)")
        )

        # Add operational model
        create_subproblem_node!(
            scenario_node,
            sets,
            par,
            scenarios
        )

        set_to_node_objectives(scenario_graph)

        # Add scenario graph to main graph
        add_subgraph!(
            graph,
            scenario_graph
        )

        # Connect scenario to investment master
        create_linking_constraints!(
            graph,
            master,
            scenario_node,
            sets,
            par,
            scenarios
        )

        scenario_graphs[scenarios] = scenario_graph
        scenario_nodes[scenarios] = scenario_node
    end

    return (
        graph,
        master_graph,
        master,
        scenario_graphs,
        scenario_nodes
    )
end



# 9. BUILD GRAPH AND RUN BENDERS

graph, master_graph, master,
scenario_graphs, scenario_nodes =
    create_benders_graph(sets, par)

println("\nGraph structure:")
println(graph)

# Gurobi solver for master and subproblems
solver = optimizer_with_attributes(
    Gurobi.Optimizer,
    "OutputFlag" => 0,
    "InfUnbdInfo" => 1,
    "DualReductions" => 0
)

# Initialize Benders decomposition
benders = BendersAlgorithm(
    graph,
    master_graph;
    solver = solver,
    multicut = true,
    feasibility_cuts = true,
    add_slacks = false,
    max_iters = 500,
    tol = 1e-6,
    M = 0.0
)

println(
    "Feasibility cuts enabled: ",
    PlasmoBenders.get_feasibility_cuts(benders)
)

# Solve using Benders decomposition
run_algorithm!(benders)

# ==========================================
# RESULTS
# ==========================================

println("\nBenders results:")

println(
    "Objective value: ",
    objective_value(benders)
)

println(
    "Lower bound: ",
    dual_objective_value(benders)
)

"""println(
    "Relative gap: ",
    relative_gap(benders)
)"""

println(
    "\nGeneration investment: ",
    value(benders, master[:x_gen]),
    " MW"
)

println("\nTransmission investments:")

for l in sets.L

    investment = value(
        benders,
        master[:x_tran][l]
    )

    println(
        l,
        ": new = ", investment,
        " MW, total = ",
        par.x_bar_tran[l] + investment,
        " MW"
    )
end

println("\nTotal generation by scenario:")

for scenarios in sets.Ω

    node = scenario_nodes[scenarios]

    total_generation = sum(
        value(benders, node[:y_gen][n, h])
        for n in sets.N, h in sets.H
    )

    println(
        scenarios,
        ": ",
        total_generation,
        " MWh over the 24-hour horizon"
    )
end
