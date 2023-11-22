use nalgebra::{DMatrix, DVector};

use crate::act_fn::ActivationFn;

#[derive(Debug)]
pub struct Layer {
    pub layer_index: usize,
    pub neuron_count: usize,
    pub weights: DMatrix<f32>,
    pub biases: DVector<f32>,
    pub activation_fn: ActivationFn,
}

impl Layer {
    /// Compute the output activation values from input vector
    pub fn compute(&self, input: &DVector<f32>) -> DVector<f32> {
        let raw = &self.weights * input + &self.biases;
        return raw.map(|val| return self.activation_fn.compute(val));
    }
}
