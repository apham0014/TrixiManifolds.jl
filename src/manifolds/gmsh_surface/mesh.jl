
"""
    P4estMeshGmshSurface(meshfile; kwargs...)

Construct a `Trixi.P4estMesh{2,3}` from a Gmsh-generated Abaqus `.inp`
surface mesh.

Supported elements:
- linear 4-node quadrilaterals
- quadratic 8-node quadrilaterals (e.g. S8 / M3D8)
- quadratic 9-node quadrilaterals (e.g. M3D9)

The p4est topology is two-dimensional, while the physical node
coordinates are retained in R^3.

This implementation is written for Trixi.jl v0.17.3 and uses some
non-public Trixi functions.
"""
function P4estMeshGmshSurface(meshfile::AbstractString;
                              RealT = Float64,
                              initial_refinement_level = 0,
                              boundary_symbols = nothing,
                              unsaved_changes = true,
                              p4est_partition_allow_for_coarsening = true)

    isfile(meshfile) ||
        throw(ArgumentError("Mesh file does not exist: $meshfile"))

    endswith(lowercase(meshfile), ".inp") ||
        throw(ArgumentError("P4estMeshGmshSurface currently requires an Abaqus .inp file"))

    # This first implementation is serial.
    Trixi.mpi_isparallel() &&
        error("P4estMeshGmshSurface currently supports serial execution only")

    # These are the same basic element families recognized by
    # Trixi v0.17.3's standard Abaqus importer.
    linear_quads =
        r"^(CPE4|CPEG4|CPS4|M3D4|S4|SFM3D4).*$"

    quadratic_quads =
        r"^(CPE8|CPS8|CAX8|S8|M3D8|M3D9).*$"

    linear_hexes = r"^(C3D8).*$"
    quadratic_hexes = r"^(C3D27).*$"

    # ------------------------------------------------------------------
    # 1. Let Trixi preprocess the Gmsh Abaqus file.
    #
    # This:
    #   - removes irrelevant lower-dimensional elements
    #   - keeps quadrilateral surface elements
    #   - renumbers the retained elements
    # ------------------------------------------------------------------

    elements_begin_idx, sets_begin_idx =
        Trixi.preprocess_standard_abaqus(
            meshfile,
            linear_quads,
            linear_hexes,
            quadratic_quads,
            quadratic_hexes,
            2, # topological dimension
        )

    meshfile_preproc = replace(meshfile, ".inp" => "_preproc.inp")

    # ------------------------------------------------------------------
    # 2. Produce a p4est-readable file.
    #
    # p4est "can only handle linear elements". We replace quadratic elements
    # with linear elements and handle the higher-order (quadratic) boundaries
    # internally.
    # ------------------------------------------------------------------

    mesh_order =
        Trixi.preprocess_standard_abaqus_for_p4est(
            meshfile_preproc,
            linear_quads,
            linear_hexes,
            quadratic_quads,
            quadratic_hexes,
            elements_begin_idx,
            sets_begin_idx,
        )

    meshfile_p4est =
        replace(meshfile, ".inp" => "_p4est_ready.inp")

    # ------------------------------------------------------------------
    # 3. Have p4est construct the 2D forest connectivity.
    # ------------------------------------------------------------------

    connectivity =
        Trixi.read_inp_p4est(meshfile_p4est, Val(2))

    connectivity_pw = Trixi.PointerWrapper(connectivity)

    n_trees = Int(connectivity_pw.num_trees[])

    # ------------------------------------------------------------------
    # 4. Geometry interpolation nodes.
    #
    # For a Gmsh second-order mesh the reference locations are
    #
    #       -1, 0, 1
    #
    # in each parametric direction.
    # ------------------------------------------------------------------

    if mesh_order == 2
        nodes = SVector{3, RealT}(
            -one(RealT),
            zero(RealT),
            one(RealT),
        )
    else
        nodes = SVector{2, RealT}(
            -one(RealT),
            one(RealT),
        )
    end

    nnodes = length(nodes)


    # Each tree (original quadrilateral patch) is defined by (nnodes x nnodes) 3d coordinates.
    tree_node_coordinates =
        Array{RealT, 4}(
            undef,
            3,
            nnodes,
            nnodes,
            n_trees,
        )

    # ------------------------------------------------------------------
    # 5. Read ALL THREE coordinates of every Gmsh/Abaqus node (starting
    #    from the *NODE section).
    # ------------------------------------------------------------------

    # Our helper function defines a dictionary which maps an 
    # Abaqus node ID to its 3D coordinate, i.e., node ID -> (x, y, z)
    mesh_nodes =
        _gmsh_surface_read_nodes(
            meshfile_preproc,
            elements_begin_idx,
            RealT,
        )

    # Retained element section (geometry defined between nodes) after 
    # Trixi preprocessing.
    element_lines =
        readlines(meshfile_preproc)[
            elements_begin_idx:(sets_begin_idx - 1)
        ]

    # ------------------------------------------------------------------
    # 6. Construct the curved R^3 geometry of every p4est tree.
    # ------------------------------------------------------------------

    _gmsh_surface_tree_coordinates!(
        tree_node_coordinates,
        element_lines,
        mesh_nodes,
        linear_quads,
    )

    # ------------------------------------------------------------------
    # 7. Boundary names.
    #
    # Reuse Trixi's existing node-set machinery.
    # ------------------------------------------------------------------

    if boundary_symbols === nothing
        boundary_names = fill(:all, 4, n_trees)
    else
        node_set_dict =
            Trixi.parse_node_sets(
                meshfile_p4est,
                boundary_symbols,
            )

        element_node_matrix =
            Trixi.parse_elements(
                meshfile_p4est,
                n_trees,
                2,
                elements_begin_idx,
                sets_begin_idx,
            )

        boundary_names =
            fill(Symbol("---"), 4, n_trees)

        Trixi.assign_boundaries_standard_abaqus!(
            boundary_names,
            n_trees,
            element_node_matrix,
            node_set_dict,
            Val(2),
        )
    end

    # ------------------------------------------------------------------
    # 8. Construct the actual p4est forest.
    # ------------------------------------------------------------------

    # We create the forest.
    p4est =
        Trixi.new_p4est(
            connectivity,
            initial_refinement_level,
        )

    # Trixi v0.17.3 determines NDIMS_AMBIENT using
    #
    #     size(tree_node_coordinates, 1)
    #
    # Since that is 3 here, this inner constructor returns
    #
    #     P4estMesh{2,3}
    #

    return Trixi.P4estMesh{2}(
        p4est,
        tree_node_coordinates,
        nodes,
        boundary_names,
        "",
        unsaved_changes,
        p4est_partition_allow_for_coarsening,
    )
end


"""
Read the Abaqus *NODE section while preserving all three spatial
coordinates.
"""
function _gmsh_surface_read_nodes(meshfile,
                                  elements_begin_idx,
                                  ::Type{RealT}) where {RealT}

    mesh_nodes = Dict{Int, SVector{3, RealT}}()

    in_node_section = false

    open(meshfile, "r") do io
        for (line_idx, line) in enumerate(eachline(io))

            line_idx >= elements_begin_idx && break

            stripped = strip(line)

            if startswith(uppercase(stripped), "*NODE")
                in_node_section = true
                continue
            end

            if in_node_section && startswith(stripped, "*")
                in_node_section = false
                continue
            end

            in_node_section || continue
            isempty(stripped) && continue

            parts = strip.(split(stripped, ','))

            length(parts) >= 4 ||
                error("Expected a three-dimensional Abaqus node, got:\n$line")

            node_id = parse(Int, parts[1])

            mesh_nodes[node_id] =
                SVector{3, RealT}(
                    parse(RealT, parts[2]),
                    parse(RealT, parts[3]),
                    parse(RealT, parts[4]),
                )
        end
    end

    isempty(mesh_nodes) &&
        error("No nodes were found in $meshfile")

    return mesh_nodes
end

"""
Fill `tree_node_coordinates` using the Abaqus/Gmsh quadrilateral
node ordering.

For an 8-node quadratic quad:

       4 ----- 7 ----- 3
       |               |
       8               6
       |               |
       1 ----- 5 ----- 2

For a 9-node quad, node 9 is the center.

The tensor-product storage used by Trixi is

                  η
                  ↑

       (1,3)   (2,3)   (3,3)

       (1,2)   (2,2)   (3,2)

       (1,1)   (2,1)   (3,1)  → ξ
"""
function _gmsh_surface_tree_coordinates!(
    tree_node_coordinates,
    element_lines,
    mesh_nodes,
    linear_quads,
)
    tree = 0
    element_order = 1

    for raw_line in element_lines
        line = strip(raw_line)

        isempty(line) && continue

        # --------------------------------------------------------------
        # New Abaqus element section
        # --------------------------------------------------------------
        if startswith(uppercase(line), "*ELEMENT")

            match_type =
                match(r"(?i)\*ELEMENT,\s*TYPE=([^,\s]+)", line)

            match_type === nothing &&
                error("Could not determine Abaqus element type from:\n$line")

            element_type = match_type.captures[1]

            element_order =
                occursin(linear_quads, element_type) ? 1 : 2

            continue
        end

        startswith(line, "*") && continue

        # --------------------------------------------------------------
        # Element connectivity
        # --------------------------------------------------------------
        parts = strip.(split(line, ','))

        # Ignore empty fields caused by trailing commas.
        filter!(!isempty, parts)

        length(parts) >= 5 ||
            error("Invalid quadrilateral element line:\n$line")

        element_nodes = parse.(Int, parts[2:end])

        tree += 1

        tree <= size(tree_node_coordinates, 4) ||
            error("More quadrilateral elements found than p4est trees")

        # ==============================================================
        # LINEAR Q4 ELEMENT
        # ==============================================================

        if element_order == 1
            length(element_nodes) >= 4 ||
                error("Linear quad requires four nodes:\n$line")

            X1 = mesh_nodes[element_nodes[1]]
            X2 = mesh_nodes[element_nodes[2]]
            X3 = mesh_nodes[element_nodes[3]]
            X4 = mesh_nodes[element_nodes[4]]

            if size(tree_node_coordinates, 2) == 2
                # Q1 tensor layout
                tree_node_coordinates[:, 1, 1, tree] .= X1
                tree_node_coordinates[:, 2, 1, tree] .= X2
                tree_node_coordinates[:, 1, 2, tree] .= X4
                tree_node_coordinates[:, 2, 2, tree] .= X3
            else
                # A mixed linear/quadratic mesh has quadratic geometry
                # nodes globally, so interpolate the straight element
                # onto the 3x3 tensor grid.

                tree_node_coordinates[:, 1, 1, tree] .= X1
                tree_node_coordinates[:, 2, 1, tree] .= (X1 + X2) / 2
                tree_node_coordinates[:, 3, 1, tree] .= X2

                tree_node_coordinates[:, 1, 2, tree] .= (X1 + X4) / 2
                tree_node_coordinates[:, 2, 2, tree] .=
                    (X1 + X2 + X3 + X4) / 4
                tree_node_coordinates[:, 3, 2, tree] .= (X2 + X3) / 2

                tree_node_coordinates[:, 1, 3, tree] .= X4
                tree_node_coordinates[:, 2, 3, tree] .= (X4 + X3) / 2
                tree_node_coordinates[:, 3, 3, tree] .= X3
            end

            continue
        end

        # ==============================================================
        # QUADRATIC Q8 / Q9 ELEMENT
        # ==============================================================

        length(element_nodes) >= 8 ||
            error("Quadratic quad requires at least eight nodes:\n$line")

        X1 = mesh_nodes[element_nodes[1]]
        X2 = mesh_nodes[element_nodes[2]]
        X3 = mesh_nodes[element_nodes[3]]
        X4 = mesh_nodes[element_nodes[4]]

        X5 = mesh_nodes[element_nodes[5]]
        X6 = mesh_nodes[element_nodes[6]]
        X7 = mesh_nodes[element_nodes[7]]
        X8 = mesh_nodes[element_nodes[8]]

        # Gmsh/Abaqus boundary nodes map directly onto the Q2 tensor grid.
        #
        # Bottom
        tree_node_coordinates[:, 1, 1, tree] .= X1
        tree_node_coordinates[:, 2, 1, tree] .= X5
        tree_node_coordinates[:, 3, 1, tree] .= X2

        # Left/right midside points
        tree_node_coordinates[:, 1, 2, tree] .= X8
        tree_node_coordinates[:, 3, 2, tree] .= X6

        # Top
        tree_node_coordinates[:, 1, 3, tree] .= X4
        tree_node_coordinates[:, 2, 3, tree] .= X7
        tree_node_coordinates[:, 3, 3, tree] .= X3

        # --------------------------------------------------------------
        # Center
        # --------------------------------------------------------------

        if length(element_nodes) >= 9
            # Complete Q2 / 9-node element
            X9 = mesh_nodes[element_nodes[9]]

            tree_node_coordinates[:, 2, 2, tree] .= X9

        else
            # 8-node serendipity element.
            #
            # Evaluate the standard S8 mapping at ξ = η = 0:
            #
            # X(0,0) =
            #   -1/4 (X1 + X2 + X3 + X4)
            #   +1/2 (X5 + X6 + X7 + X8)
            #
            Xcenter =
                -(X1 + X2 + X3 + X4) / 4 +
                (X5 + X6 + X7 + X8) / 2

            tree_node_coordinates[:, 2, 2, tree] .= Xcenter
        end
    end

    tree == size(tree_node_coordinates, 4) ||
        error(
            "Found $tree quadrilateral elements, but p4est contains " *
            "$(size(tree_node_coordinates, 4)) trees",
        )

    return nothing
end


function validate_surface_edges(meshfile::AbstractString)
    edge_counts = Dict{Tuple, Int}()
    element_count = 0
    in_surface_section = false
    element_order = 0

    for raw_line in eachline(meshfile)
        line = strip(raw_line)

        isempty(line) && continue
        startswith(line, "**") && continue

        if startswith(uppercase(line), "*ELEMENT")
            match_type = match(
                r"(?i)\*ELEMENT,\s*TYPE=([^,\s]+)",
                line,
            )

            in_surface_section = false
            element_order = 0

            if match_type !== nothing
                element_type = uppercase(match_type.captures[1])

                if startswith(element_type, "M3D9") ||
                   startswith(element_type, "S8")
                    in_surface_section = true
                    element_order = 2
                elseif startswith(element_type, "CPS4") ||
                       startswith(element_type, "M3D4") ||
                       startswith(element_type, "S4")
                    in_surface_section = true
                    element_order = 1
                end
            end

            continue
        end

        if startswith(line, "*")
            in_surface_section = false
            continue
        end

        in_surface_section || continue

        fields = strip.(split(line, ','))
        filter!(!isempty, fields)

        # Element ID plus at least four node IDs
        length(fields) >= 5 ||
            error("Invalid surface element line:\n$line")

        node_ids = parse.(Int, fields[2:end])
        element_count += 1

        if element_order == 1
            corners = node_ids[1:4]

            edges = (
                (corners[1], corners[2]),
                (corners[2], corners[3]),
                (corners[3], corners[4]),
                (corners[4], corners[1]),
            )

            for edge in edges
                key = Tuple(sort(collect(edge)))
                edge_counts[key] = get(edge_counts, key, 0) + 1
            end
        else
            # Abaqus/Gmsh M3D9 ordering:
            #
            # corners: 1, 2, 3, 4
            # midsides: 5, 6, 7, 8
            corners = node_ids[1:4]
            midsides = node_ids[5:8]

            edges = (
                (corners[1], corners[2], midsides[1]),
                (corners[2], corners[3], midsides[2]),
                (corners[3], corners[4], midsides[3]),
                (corners[4], corners[1], midsides[4]),
            )

            for edge in edges
                # Reverse orientation does not change the edge identity.
                key = (
                    min(edge[1], edge[2]),
                    max(edge[1], edge[2]),
                    edge[3],
                )
                edge_counts[key] = get(edge_counts, key, 0) + 1
            end
        end
    end

    external_edges = [
        edge for (edge, count) in edge_counts if count == 1
    ]

    nonmanifold_edges = [
        (edge, count)
        for (edge, count) in edge_counts
        if count > 2
    ]

    result = (
        elements = element_count,
        edges = length(edge_counts),
        external_edges = external_edges,
        nonmanifold_edges = nonmanifold_edges,
        closed = isempty(external_edges) && isempty(nonmanifold_edges),
    )

    println("Surface elements: ", result.elements)
    println("Unique surface edges: ", result.edges)
    println("External edges: ", length(result.external_edges))
    println("Non-manifold edges: ", length(result.nonmanifold_edges))
    println("Closed surface topology: ", result.closed)

    return result
end