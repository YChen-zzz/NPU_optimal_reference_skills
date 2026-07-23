"""Centralized thresholds for all parse scripts.

Edit values here instead of hunting through individual scripts.
Loaded via common.threshold(script, key).
"""

THRESHOLDS = {
    "step_trace": {
        "severe_host_bound_util": 20,       # % — below this = severe host-bound
        "moderate_host_bound_util": 50,     # % — below this = moderate host-bound
        "step_util_variance": 20,           # % — max-min util difference across steps
        "step_duration_spread": 2.0,        # max/min ratio for step duration outlier
        "large_optimizable_space": 30,      # % — Free/Total above this = large optimizable space
    },

    "op_statistic": {
        "top3_concentration": 80,           # % — top-3 ops > this = concentrated bottleneck
        "move_keywords": [                  # op types classified as data movement
            "Transpose", "Cast", "Copy", "Contiguous", "Reshape", "MemSet", "Format",
        ],
        "data_movement_ratio": 3,           # % — data movement > this = signal
        "frag_count_multiplier": 3,         # x avg count — fragmentation signal
        "frag_max_avg_us": 10,              # us — avg below this + high count = fragmented
        "heavy_max_count": 10,              # count <= this = heavy single-invocation
        "heavy_min_avg_us": 100,            # us — avg above this = heavy
        "heavy_min_ratio": 0.01,            # total/total ratio above this = heavy
    },

    "kernel_details": {
        "suspect_min_duration_us": 10,      # us — kernels above this considered for suspect
        "suspect_mac_ratio": 0.2,           # mac_ratio below this = low compute (AI_CORE)
        "suspect_vec_ratio": 0.05,          # vec_ratio below this = low compute (AI_VECTOR)
        "block_dim_buckets": [8, 28],       # boundaries: 1, 2-8, 9-28, 29+
        "wait_buckets_us": [100, 500, 2000],# boundaries for wait time distribution
        "cube_low_util": 50,                # % — cube utilization below this = low
        "low_parallelism_ratio": 0.1,       # % — block_dim=1 ratio above this = signal
        "hw_dominance_ratio": 1.5,          # x — mte>mac*1.5 = memory-dominated, vice versa
        "fusible_small_us": 10.0,           # us — kernels below this = fusible candidate
        "fusible_min_length": 5,            # min consecutive small kernels for a sequence
        "fusible_min_total_us": 100,        # us — min cumulative duration for a sequence
        "compute_bound_mac_ratio": 0.5,     # mac_ratio above this + high dur = true compute-bound (replace/quant target)
        "comm_keywords": [                  # AI_CPU ops that are communication (excluded from fallback)
            "broadcast", "allgather", "alltoall", "allreduce", "hcom", "send", "recv", "reducescatter",
        ],
        "short_kernel_dominant": 60,        # % — short kernel (<20us) ratio above this = dominant
    },

    "trace_view": {
        "gap_buckets_us": [10, 50, 200],    # boundaries for gap distribution
        "compute_task_types": [             # task types classified as compute
            "AI_CORE", "AI_VECTOR", "AICORE", "AIVEC", "MIX", "VECTOR",
        ],
        "compile_early_window": 0.2,        # fraction of timeline considered "early"
        "compile_early_frac": 0.8,          # % of compile in early window = Type A (warmup)
        "freq_decrease_ratio": 0.05,        # % — frequency degradation above this = signal
        "sync_co_ratio": 10,                # % — sync/launch ratio above this = signal
        "prefetch_keywords": [              # op names classified as prefetch/prealloc candidates
            "aten::to", "copy_", "aten::copy", "::empty", "empty_",
            "aten::empty", "memcpy", "to_copy", "_to_copy", "pin_memory",
        ],
        "stack_lib_markers": [              # call stack frames matching these are filtered out
            "site-packages", "dist-packages", "/lib/python",
            "torch/nn/modules", "torch/_ops", "autograd/profiler", "torch_npu/profiler",
        ],
        "stack_max_frames": 6,              # max project frames shown in condensed call stack
        "disp_lat_sample_cap": 200000,      # max dispatch latency samples stored
        "compile_ts_cap": 500000,           # max compile timestamps stored
        "dispatch_kernel_ratio": 50,        # % — dispatch/kernel-active above this = significant
        "h2d_gap_threshold_us": 50,         # us — device starts within this of launch = host-bound op
        "h2d_min_run_len": 3,               # min consecutive host-bound ops to form a region
        "h2d_max_runs": 15,                 # max regions reported (sorted by device idle time)
        "h2d_callstack_per_run": 2,         # max distinct op call stacks shown per region
    },

    "memory_record": {
        "frag_gap_mb": 1000,                # MB — large fragmentation gap above this = signal
        "growth_min_records": 20,           # min records for growth trend analysis
        "growth_mb": 100,                   # MB — growth above this between early/late = signal
        "churn_jump_mb": 50,                # MB — jumps above this = large
        "churn_count": 20,                  # count of large jumps above this = high churn
        "frag_growth_mb": 50,               # MB — fragmentation growth above this = signal
        "oom_risk_mb": 60000,               # MB — reserved above this = OOM risk
    },

    "operator_memory": {
        "short_life_us": 1000000,           # 1s — tensors alive < this = short-lived (waste)
        "short_lived_min_kb": 100,          # KB — size above this for short-lived tracking
        "short_lived_max_life_us": 1000,    # us — lifetime below this for short-lived tracking
        "size_track_min_kb": 10,            # KB — size above this for repeated alloc tracking
        "repeated_count": 10,               # count above this = repeated alloc signal
        "churn_total_kb": 10000,            # KB — short-lived total above this = churn signal
        "dominate_ratio": 0.5,              # ratio above this = single op dominates
        "parallelism_ratio": 0.8,           # projected peak / HBM above this = parallelism trigger
    },

    "operator_details": {
        "pure_host_pct": 50,                # % — pure host ops above this = DEFINITE signal
        "extreme_hd_ratio": 10,             # x — host > device * this = extreme ratio
        "extreme_host_us": 5000,            # us — host above this for extreme ratio
        "hd_ratio_display_cap": 10000,      # display "∞" above this
    },

    "communication": {
        "wait_dominant_ratio": 0.8,         # wait/total above this = DEFINITE sync-bound
        "per_type_wait_ratio": 0.9,         # per-type wait ratio above this = SIGNAL
        "per_type_min_count": 10,           # min count for per-type signal
        "low_bw_ratio": 0.3,                # bandwidth below avg*this = low bandwidth link
        "low_bw_min_size_mb": 1,            # MB — min size for low bandwidth link signal
        "small_packet_mb": 1.0,             # MB — packets below this = small
        "small_packet_ratio": 0.3,          # small packet ratio above this = SIGNAL
    },
}
