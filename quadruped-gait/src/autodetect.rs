//! Auto-detect a quadruped [`KinematicsConfig`] (plus the per-joint IK→model
//! sign corrections) directly from a [`misarta`] kinematic model.
//!
//! This is the headless, `misarta`-backed port of articara's
//! `gait::auto_detect_kinematics_config` and the sign-table half of
//! `gait::GaitController::build`. It lets a gait be configured for a real
//! robot from a `.misa` file (`misarta::native::load` → `build_model`)
//! without pulling in the articara application crate.
//!
//! Topology assumption (same as the analytical IK): each leg is a 3-DOF
//! Roll-Pitch-Pitch chain `hip(X) → thigh(Y) → calf(Y) → foot`.

use misarta::fk::forward_kinematics;
use misarta::joint::JointType;
use misarta::model::Model;
use nalgebra::Vector3;

use crate::config::{KinematicsConfig, LegId, LegKinematics};

fn slot_of(id: LegId) -> usize {
    match id {
        LegId::FL => 0,
        LegId::FR => 1,
        LegId::RL => 2,
        LegId::RR => 3,
    }
}

/// misarta joint index whose *child* link is `link_name`.
fn joint_of_child_link(model: &Model<f64>, link_name: &str) -> Option<usize> {
    model.link_names.iter().position(|n| n == link_name)
}

fn revolute_axis(jt: &JointType<f64>) -> Option<Vector3<f64>> {
    match jt {
        JointType::Revolute { axis } => Some(*axis),
        _ => None,
    }
}

/// Climb from the joint carrying `foot_link` up the parent chain, collecting
/// the nearest `n` revolute joints. Returned foot→hip, e.g. `[calf, thigh, hip]`.
fn climb_to_active_joints(
    model: &Model<f64>,
    foot_link: &str,
    n: usize,
) -> Result<Vec<usize>, String> {
    let foot_joint = joint_of_child_link(model, foot_link)
        .ok_or_else(|| format!("foot link {foot_link:?} not found in model"))?;
    let mut chain = Vec::with_capacity(n);
    let mut idx = foot_joint;
    // misarta joints are 1-based; index 0 is the universe/root sentinel.
    while chain.len() < n && idx != 0 {
        idx = model.joints[idx].parent;
        if idx != 0 && revolute_axis(&model.joints[idx].joint_type).is_some() {
            chain.push(idx);
        }
    }
    if chain.len() < n {
        return Err(format!(
            "foot link {foot_link:?}: found {} active joints climbing to root (need {n})",
            chain.len()
        ));
    }
    Ok(chain)
}

/// Auto-detect one leg's [`LegKinematics`] from the model.
///
/// `home_q` (length `model.nq`) is the standing configuration; link lengths
/// and `hip_offset` are pose-invariant, but `nominal_foot_body` is sampled at
/// `home_q` so the gait's stance plane sits at the standing height rather than
/// at full leg extension.
pub fn auto_detect_leg_kinematics(
    model: &Model<f64>,
    foot_link: &str,
    leg: LegId,
    home_q: &[f64],
) -> Result<LegKinematics, String> {
    let chain = climb_to_active_joints(model, foot_link, 3)?;
    let (calf_idx, thigh_idx, hip_idx) = (chain[0], chain[1], chain[2]);
    let foot_joint = joint_of_child_link(model, foot_link).unwrap();

    // Classify axes in the body (root) frame at the neutral pose. A joint's
    // axis is stored in its own frame, so rotate it by the joint's rest-pose
    // world rotation before checking the dominant component.
    let neutral = model.neutral_q();
    let data0 = forward_kinematics(model, &neutral);
    let axis_in_body = |j: usize| -> Vector3<f64> {
        let raw = revolute_axis(&model.joints[j].joint_type).unwrap_or_else(Vector3::x);
        data0.oMi[j].rotation * raw
    };
    let hip_axis = axis_in_body(hip_idx);
    let thigh_axis = axis_in_body(thigh_idx);
    let calf_axis = axis_in_body(calf_idx);
    if hip_axis.x.abs() < hip_axis.y.abs() || hip_axis.x.abs() < hip_axis.z.abs() {
        return Err(format!(
            "hip joint {} axis {hip_axis:?} doesn't look like a Roll (X) axis — \
             the analytical IK assumes RPP topology",
            model.joints[hip_idx].name
        ));
    }
    if thigh_axis.y.abs() < thigh_axis.x.abs() || thigh_axis.y.abs() < thigh_axis.z.abs() {
        return Err(format!(
            "thigh joint {} axis {thigh_axis:?} doesn't look like a Pitch (Y) axis",
            model.joints[thigh_idx].name
        ));
    }
    if calf_axis.y.abs() < calf_axis.x.abs() || calf_axis.y.abs() < calf_axis.z.abs() {
        return Err(format!(
            "calf joint {} axis {calf_axis:?} doesn't look like a Pitch (Y) axis",
            model.joints[calf_idx].name
        ));
    }

    // Joint-origin world positions at the standing pose.
    let data = forward_kinematics(model, home_q);
    let pos = |j: usize| -> Vector3<f64> { data.oMi[j].translation.vector };
    let hip_pos = pos(hip_idx);
    let thigh_pos = pos(thigh_idx);
    let calf_pos = pos(calf_idx);
    let foot_pos = pos(foot_joint);
    let body_idx = model.joints[hip_idx].parent;
    let body_pos = data.oMi[body_idx].translation.vector;

    let hip_offset = hip_pos - body_pos;
    let hip_to_thigh_y = (thigh_pos.y - hip_pos.y).abs();
    let upper_leg = (thigh_pos - calf_pos).norm();
    let lower_leg = (calf_pos - foot_pos).norm();
    if upper_leg < 1e-6 || lower_leg < 1e-6 {
        return Err(format!(
            "degenerate link lengths: upper={upper_leg:.6} lower={lower_leg:.6} \
             — check the foot link and joint frames don't coincide"
        ));
    }

    let mut kin = LegKinematics::new(
        leg,
        model.joints[hip_idx].name.clone(),
        model.joints[thigh_idx].name.clone(),
        model.joints[calf_idx].name.clone(),
        foot_link.to_string(),
        hip_offset,
        hip_to_thigh_y,
        upper_leg,
        lower_leg,
    );
    // Override the straight-leg default with the actual standing foot.
    kin.nominal_foot_body = foot_pos - body_pos;
    Ok(kin)
}

/// Auto-detect a full [`KinematicsConfig`] for all four legs. On failure
/// returns the per-leg `(LegId, message)` errors.
pub fn auto_detect_kinematics_config(
    model: &Model<f64>,
    foot_links: &[(LegId, &str); 4],
    home_q: &[f64],
) -> Result<KinematicsConfig, Vec<(LegId, String)>> {
    let mut errors = Vec::new();
    let mut detected: [Option<LegKinematics>; 4] = [None, None, None, None];
    for (leg, name) in foot_links.iter() {
        match auto_detect_leg_kinematics(model, name, *leg, home_q) {
            Ok(kin) => detected[slot_of(*leg)] = Some(kin),
            Err(e) => errors.push((*leg, e)),
        }
    }
    if !errors.is_empty() {
        return Err(errors);
    }
    Ok(KinematicsConfig {
        fl: detected[0].take().unwrap(),
        fr: detected[1].take().unwrap(),
        rl: detected[2].take().unwrap(),
        rr: detected[3].take().unwrap(),
    })
}

/// Per-`(leg, joint)` multiplier that maps the analytical IK's joint output to
/// the model's joint-axis (URDF) sign convention. Slot order FL/FR/RL/RR ×
/// (hip, thigh, calf). Apply as `q_model = q_ik * sign`.
///
/// The IK uses URDF convention for the hip (roll about +X) but the *opposite*
/// of URDF's right-hand rule for thigh/calf (pitch about +Y), so a +Y model
/// axis means the IK output is negated.
pub fn joint_signs(model: &Model<f64>, kin: &KinematicsConfig) -> Result<[[f64; 3]; 4], String> {
    use std::collections::HashMap;
    let name_to_joint: HashMap<&str, usize> = model
        .joints
        .iter()
        .enumerate()
        .map(|(i, j)| (j.name.as_str(), i))
        .collect();

    let legs = [&kin.fl, &kin.fr, &kin.rl, &kin.rr];
    let ik_to_urdf_factor = [1.0, -1.0, -1.0];
    let mut signs = [[1.0_f64; 3]; 4];
    for (slot, lk) in legs.iter().enumerate() {
        let names = [&lk.hip_joint, &lk.thigh_joint, &lk.calf_joint];
        for (k, nm) in names.iter().enumerate() {
            let idx = *name_to_joint
                .get(nm.as_str())
                .ok_or_else(|| format!("joint {nm:?} (from kinematics) not in model"))?;
            let axis = revolute_axis(&model.joints[idx].joint_type)
                .ok_or_else(|| format!("joint {nm:?} is not revolute"))?;
            // hip uses the X component, thigh/calf the Y component.
            let comp = if k == 0 { axis.x } else { axis.y };
            let urdf_sign = if comp >= 0.0 { 1.0 } else { -1.0 };
            signs[slot][k] = ik_to_urdf_factor[k] * urdf_sign;
        }
    }
    Ok(signs)
}
