use std::{path::PathBuf, str::FromStr};

use clap::{Parser, ValueEnum};

use crate::act_fn::ActivationFn;

#[derive(Parser, Debug)]
#[command(name = "nses_sp")]
#[command(version = "1.1.0")]
#[command(author = "Tomáš Lebeda <tom.lebeda@gmail.com>")]
#[command(about = "This is simple piece of software that implements parametric fully-connected neural network for point classification and real-time visualizations.")]
#[command(help_template = "{name} v{version}\nAuthor: {author-with-newline}{about-section}\nUSAGE: '{usage}'\n\nOPTIONS:\n{options}")]
pub struct Cli {
    /// level of logging details (into stderr)
    #[arg(short, long, value_enum, default_value_t = LogLevel::Trace)]
    pub log_level: LogLevel,

    /// path to file with input (training) data
    ///
    /// Expected format is space-separated-matrix of numbers where:
    ///
    /// - each line represents one training point
    ///
    /// - last column are natural numbers representing class label
    ///
    /// - other columns are real numbers representing features
    #[arg(short, long)]
    pub input_file: String,

    /// number of features that will be present in the data, leave 0 for "auto"
    ///
    /// this may speed up the data parsing and you can use it to check whether the input file
    /// contains what your are expecting it to contain, but it's mostly unnecessary since this program
    /// can figure out the number of features automatically
    #[arg(long, default_value_t = 0)]
    pub num_features: usize,

    /// number of classes that will be present in the data, leave 0 for "auto"
    ///
    /// this may speed up the data parsing and you can use it to check whether the input file
    /// contains what your are expecting it to contain, but it's mostly unnecessary since this program
    /// can figure out the number of classes automatically
    #[arg(long, default_value_t = 0)]
    pub num_classes: usize,

    /// Hidden layer configuration separated by commas in format
    /// [size]:[fn_type]:[parameter]
    ///
    /// Function type can be: BinaryBipolar (bibi, bb), BinaryUnipolar (biun, bu), SigmoidUnipolar
    /// (siun, su), SigmoidBipolar (sibi, sb), ReLU (relu, r), Linear (line, lin, l)
    ///
    /// Parameter can be omitted for Binary functions, otherwise they must be floating point number
    ///
    /// Example:
    ///
    /// '10:sibi:0.6' will add hidden layer with 10 neurons using SigmoidBipolar activation
    /// function with parameter lambda=0.6
    ///
    /// '10:sibi:0.6,5:bb,4:l:0.2' will add hidden layer with 10 neurons using SigmoidBipolar
    /// activation function with parameter lambda=0.6, after that it will add hidden layer with 5
    /// neurons using BinaryBipolar activation function, after that it will add hidden layer with 4
    /// neurons using Linear activation function with parameter lambda=0.2. In this case, the
    /// network will have 4 layers in total (3 hidden + output) where the first (closest to inputs)
    /// will have 10 neurons.
    #[arg(long, value_parser = clap::value_parser!(LayerConfig), value_delimiter=',')]
    pub hidden_layers: Vec<LayerConfig>,

    /// activation functions to be used in final layer in format
    /// [fn_type]:[parameter], see --hidden-layers for possible values
    ///
    /// this has the same syntax as --hidden-layers but you don't specify the number of neurons
    #[arg(long, default_value_t = ActivationFn::BinaryUnipolar, value_parser = clap::value_parser!(ActivationFn))]
    pub final_layer_af: ActivationFn,

    /// batch size for training, leave empty to use all data in each batch
    #[arg(long)]
    pub batch_size: Option<usize>,

    /// learning rate
    #[arg(long, default_value_t = 0.1)]
    pub learning_rate: f32,

    /// learning momentum, must be larger or equal to 0 and smaller than 1
    #[arg(long, default_value_t = 0.0)]
    pub momentum: f32,

    /// file where loss values will be logged
    #[arg(long)]
    pub log_file_costs: Option<PathBuf>,

    /// file where accuracy values will be logged
    #[arg(long)]
    pub log_file_acc: Option<PathBuf>,

    /// file where the last points grid values will be saved
    #[arg(long)]
    pub log_file_last_grid: Option<PathBuf>,

    /// file where the last points classes will be saved
    #[arg(long)]
    pub log_file_last_points: Option<PathBuf>,

    /// density of point-grid for visualisation of class regions
    #[arg(long, default_value_t = 0.2)]
    pub grid_step: f32,

    /// percentage of space to pad around actual data when drawing grid
    #[arg(long, default_value_t = 1.0)]
    pub grid_padding: f32,

    /// radius of dots on grid for visualisation of areas
    #[arg(long, default_value_t = 0.05)]
    pub grid_dot_radius: f32,

    /// radius of datapoints for visualisation of areas
    #[arg(long, default_value_t = 0.2)]
    pub dot_radius: f32,

    /// index of the feature that should be used for X axis while doing visualisation
    #[arg(long, default_value_t = 0)]
    pub feature_x: usize,

    /// index of the feature that should be used for Y axis while doing visualisation
    #[arg(long, default_value_t = 1)]
    pub feature_y: usize,

    /// enable visualisations via Rerun.io
    #[arg(short, long, default_value_t = false)]
    pub show: bool,

    /// target accuracy of the classification
    #[arg(long, default_value_t = 99.5)]
    pub target_acc: f32,

    /// max number of training epochs
    #[arg(long, default_value_t = 10_000)]
    pub max_epochs: u32,

    /// file where the data for visualization will be saved as .rrd file
    ///
    /// when --show is enabled, this is used only as a emergency backup plan if the viewer fails to spawn
    #[arg(long)]
    pub data_log: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct LayerConfig {
    pub size: usize,
    pub act_fn: ActivationFn,
}

impl FromStr for LayerConfig {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut pieces = s.split(':');
        let size = match pieces.next() {
            None => return Err(String::from("Can't parse layer size from empty string")),
            Some(piece) => match piece.parse::<usize>() {
                Ok(n) => n,
                Err(_) => return Err(format!("Can't parse layer size from {:?}", piece)),
            },
        };
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
            "BinaryBipolar" | "bibi" | "bb" => {
                return Ok(LayerConfig {
                    size,
                    act_fn: ActivationFn::BinaryBipolar,
                })
            }
            "BinaryUnipolar" | "biun" | "bu" => {
                return Ok(LayerConfig {
                    size,
                    act_fn: ActivationFn::BinaryUnipolar,
                })
            }
            "SigmoidBipolar" | "sibi" | "sb" => match param {
                None => return Err(String::from("SigmoidBipolar requires a f32 parameter")),
                Some(param) => {
                    return Ok(LayerConfig {
                        size,
                        act_fn: ActivationFn::SigmoidBipolar { param },
                    })
                }
            },
            "SigmoidUnipolar" | "siun" | "su" => match param {
                None => return Err(String::from("SigmoidUnipolar requires a f32 parameter")),
                Some(param) => {
                    return Ok(LayerConfig {
                        size,
                        act_fn: ActivationFn::SigmoidUnipolar { param },
                    })
                }
            },
            "ReLU" | "relu" | "r" => match param {
                None => return Err(String::from("ReLU requires a f32 parameter")),
                Some(param) => {
                    return Ok(LayerConfig {
                        size,
                        act_fn: ActivationFn::ReLU { param },
                    })
                }
            },
            "Linear" | "line" | "lin" | "l" => match param {
                None => return Err(String::from("Linear requires a f32 parameter")),
                Some(param) => {
                    return Ok(LayerConfig {
                        size,
                        act_fn: ActivationFn::Linear { param },
                    })
                }
            },
            _ => {
                return Err(format!("Unrecognized activation function: {:?}", layer_str));
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
    Off,
}
