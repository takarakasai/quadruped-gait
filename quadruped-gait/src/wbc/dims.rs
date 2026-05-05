//! Dimension bookkeeping for the WBC decision vector layout.

/// Sizes for the WBC's decision variable layout
/// `x = [q̈ (nv); f_GRF (3·nc); τ (na)]`.
#[derive(Debug, Clone, Copy)]
pub struct WbcDims {
    /// Number of generalized velocities. For a floating-base robot this
    /// is `6 + actuated_joints` (6 floating-base DoF + revolute count).
    pub nv: usize,
    /// Number of 3-DoF contact points. Always 4 for a quadruped.
    pub nc: usize,
    /// Number of actuated joints (motor count). `nv − 6` for floating
    /// base; `nv` for fixed base.
    pub na: usize,
}

impl WbcDims {
    /// Total number of decision variables.
    pub fn n_decision(&self) -> usize {
        self.nv + 3 * self.nc + self.na
    }

    /// Offset of the `q̈` block.
    pub fn q_offset(&self) -> usize {
        0
    }

    /// Offset of the `f_GRF` block.
    pub fn f_offset(&self) -> usize {
        self.nv
    }

    /// Offset of the `τ` block.
    pub fn tau_offset(&self) -> usize {
        self.nv + 3 * self.nc
    }
}
