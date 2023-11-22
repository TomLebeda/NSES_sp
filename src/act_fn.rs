use std::{fmt::Display, str::FromStr};

/// Activation function for neural layers
#[derive(Debug, Clone, Copy)]
pub enum ActivationFn {
    BinaryBipolar,
    BinaryUnipolar,
    SigmoidBipolar { param: f32 },
    SigmoidUnipolar { param: f32 },
    ReLU { param: f32 },
    Linear { param: f32 },
}

impl ActivationFn {
    /// Computes the value of f(x) for given x, where f() is the activation function (self)
    pub fn compute(&self, val: f32) -> f32 {
        match &self {
            Self::BinaryBipolar => return val.signum(),
            Self::BinaryUnipolar => return val.signum().max(0.0),
            Self::SigmoidBipolar { param } => return 2.0 / (1.0 + (-param * val).exp()) - 1.0,
            Self::SigmoidUnipolar { param } => return 1.0 / (1.0 + (-param * val).exp()),
            Self::ReLU { param } => return (param * val).max(0.0),
            Self::Linear { param } => return param * val,
        };
    }

    /// Computes the value of f'(x) for given x, where f'() is derivative (prime) of the activation function (self)
    /// For binary functions the derivative is approximated as sigmoid with parameter 10.0
    pub fn compute_prime(&self, val: f32) -> f32 {
        match &self {
            Self::BinaryBipolar => {
                let temp = 2.0 / (1.0 + (-10.0 * val).exp()) - 1.0;
                return temp * (1.0 - temp);
            }
            Self::BinaryUnipolar => {
                let temp = 1.0 / (1.0 + (-10.0 * val).exp());
                return temp * (1.0 - temp);
            }
            Self::SigmoidBipolar { param: _ } | Self::SigmoidUnipolar { param: _ } => {
                let temp = self.compute(val);
                return temp * (1.0 - temp);
            }
            Self::ReLU { param } => {
                if val <= 0.0 {
                    return 0.0;
                } else {
                    return *param;
                }
            }
            Self::Linear { param } => return *param,
        };
    }
}

impl Display for ActivationFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return write!(f, "{:?}", self);
    }
}

impl FromStr for ActivationFn {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut pieces = s.split(':');
        let layer_str = match pieces.next() {
            None => return Err(String::from("Missing layer activation function type")),
            Some(piece) => piece,
        };
        let param = match pieces.next() {
            None => None,
            Some(piece) => Some(match piece.parse::<f32>() {
                Ok(n) => n,
                Err(_) => return Err(format!("Can't parse act_fn parameter from {:?}", piece)),
            }),
        };

        match layer_str {
            "BinaryBipolar" | "bibi" | "bb" => return Ok(ActivationFn::BinaryBipolar),
            "BinaryUnipolar" | "biun" | "bu" => return Ok(ActivationFn::BinaryUnipolar),
            "SigmoidBipolar" | "sibi" | "sb" => match param {
                None => return Err(String::from("SigmoidBipolar requires a f32 parameter")),
                Some(param) => return Ok(ActivationFn::SigmoidBipolar { param }),
            },
            "SigmoidUnipolar" | "siun" | "su" => match param {
                None => return Err(String::from("SigmoidUnipolar requires a f32 parameter")),
                Some(param) => return Ok(ActivationFn::SigmoidUnipolar { param }),
            },
            "ReLU" | "relu" | "r" => match param {
                None => return Err(String::from("ReLU requires a f32 parameter")),
                Some(param) => return Ok(ActivationFn::ReLU { param }),
            },
            "Linear" | "line" | "lin" | "l" => match param {
                None => return Err(String::from("Linear requires a f32 parameter")),
                Some(param) => return Ok(ActivationFn::Linear { param }),
            },
            _ => {
                return Err(format!("Unrecognized activation function: {:?}", layer_str));
            }
        }
    }
}
