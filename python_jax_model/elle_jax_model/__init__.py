"""JAX-based prototype modules for a grain-growth style ELLE rewrite."""

from .artifacts import dominant_grain_map, save_snapshot_artifacts, snapshot_statistics
from .elle_export import extract_flynn_topology, write_unode_elle
from .elle_phasefield import (
    EllePhaseFieldTemplate,
    EllePhaseFieldConfig,
    initialize_elle_phasefield,
    load_elle_phasefield_state,
    phasefield_statistics,
    run_elle_phasefield_simulation,
    save_elle_phasefield_artifacts,
    write_elle_phasefield_state,
)
from .elle_html_viewer import write_elle_html_viewer
from .elle_visualize import render_elle_file
from .phasefield_compare import (
    collect_elle_phasefield_snapshots,
    compare_elle_phasefield_files,
    compare_elle_phasefield_sequences,
    compare_elle_phasefield_states,
    inspect_elle_phasefield_binary,
    run_original_elle_phasefield_sequence,
    run_python_elle_phasefield_sequence,
    write_phasefield_comparison_report,
)
from .mesh import (
    MeshFeedbackConfig,
    MeshRelaxationConfig,
    apply_mesh_feedback,
    apply_mesh_transport,
    build_mesh_state,
    compute_mesh_motion_velocity,
    couple_mesh_to_order_parameters,
    mesh_motion_field,
    rasterize_mesh_labels,
    relax_mesh_state,
)
from .simulation import GrainGrowthConfig, initialize_order_parameters, run_simulation
from .topology import TopologyTracker, run_simulation_with_topology

__all__ = [
    "GrainGrowthConfig",
    "MeshFeedbackConfig",
    "MeshRelaxationConfig",
    "TopologyTracker",
    "apply_mesh_feedback",
    "apply_mesh_transport",
    "build_mesh_state",
    "collect_elle_phasefield_snapshots",
    "compare_elle_phasefield_files",
    "compare_elle_phasefield_sequences",
    "compare_elle_phasefield_states",
    "compute_mesh_motion_velocity",
    "couple_mesh_to_order_parameters",
    "dominant_grain_map",
    "EllePhaseFieldConfig",
    "EllePhaseFieldTemplate",
    "extract_flynn_topology",
    "initialize_elle_phasefield",
    "initialize_order_parameters",
    "inspect_elle_phasefield_binary",
    "load_elle_phasefield_state",
    "mesh_motion_field",
    "phasefield_statistics",
    "rasterize_mesh_labels",
    "render_elle_file",
    "relax_mesh_state",
    "run_elle_phasefield_simulation",
    "run_original_elle_phasefield_sequence",
    "run_python_elle_phasefield_sequence",
    "run_simulation",
    "run_simulation_with_topology",
    "save_elle_phasefield_artifacts",
    "save_snapshot_artifacts",
    "snapshot_statistics",
    "write_phasefield_comparison_report",
    "write_elle_phasefield_state",
    "write_elle_html_viewer",
    "write_unode_elle",
]
