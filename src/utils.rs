use bimap::BiMap;
use clap::Parser;
use env_logger::Env;
use itertools::Itertools;
use log::{info, trace, warn};
use nalgebra::DVector;

use crate::{
    cli::{Cli, LogLevel},
    net::{ClassID, ClassIndex},
};

#[derive(Debug, Clone)]
pub struct Point {
    pub features: Vec<f32>,
    pub class: ClassID,
}

/// Load data from file while checking configuration.
pub fn load_data(cli: &mut Cli) -> Result<Vec<Point>, String> {
    trace!("Loading data from {:?}", cli.input_file);
    let path = &cli.input_file;
    let mut unique_classes: Vec<u32> = vec![];
    let mut points: Vec<Point> = vec![];
    let Ok(data) = std::fs::read_to_string(path) else {
        return Err(format!("Can't load file {:?}", path));
    };
    for (line_idx, line) in data.lines().filter(|l| return !l.is_empty()).enumerate() {
        let mut columns = line.split_whitespace();
        let mut p = Point { features: vec![], class: ClassID(0) };

        // If the number of features is not specified, try to guess it from the first line.
        // For the following lines this will also serve as a check if the data have consistent number of features.
        if cli.num_features == 0 {
            // last column is class, so subtract 1 from the number of columns
            let n = columns.clone().count() - 1;
            if n < 1 {
                // in case there is only single column in the data file
                return Err(format!("Detected {} features, but need at least 1.", n));
            };
            info!("Detected number of features: {}", n);
            cli.num_features = n;
        };

        // for each feature try to parse the current line further
        for i in 0..cli.num_features {
            let Some(s) = columns.next() else {
                return Err(format!("line {:?} doesn't have enough columns, expected {}", line, cli.num_features + 1));
            };
            let Ok(n) = s.parse::<f32>() else {
                return Err(format!("Can't parse {}. column of {}. line ({:?}) into f32", i + 1, line_idx + 1, s));
            };
            p.features.push(n);
        }
        let Some(s) = columns.next() else {
            return Err(format!("line {:?} doesn't have enough columns, expected {}", line, cli.num_features + 1));
        };
        let Ok(n) = s.parse::<f32>() else {
            return Err(format!("Can't parse {}. column of {}. line ({:?}) into f32", cli.num_features + 1, line_idx + 1, s));
        };
        let n = n as u32;
        if !unique_classes.contains(&n) {
            unique_classes.push(n);
        }
        p.class = ClassID(n);
        if columns.next().is_some() {
            warn!("{}. line of {:?} contains more columns than expected ({})", line_idx + 1, path, cli.num_features + 1)
        }
        points.push(p);
    }
    if cli.num_classes == 0 {
        let n = unique_classes.len();
        if n < 1 {
            return Err(format!("Detected {} classes, but need at least 1.", n));
        };
        info!("Detected number of classes: {}", n);
        cli.num_classes = n;
    } else if unique_classes.len() != cli.num_classes {
        warn!("Number of classes is {}, but expected {}.", unique_classes.len(), cli.num_classes);
    }
    if cli.num_classes < 2 && cli.show {
        warn!("To visualise the trainig, the data need at least two features. Disabling visualisation.");
        cli.show = false;
    }
    let feature_count = points.first().unwrap().features.len();
    if cli.feature_x >= feature_count {
        warn!("Configured feature_x can't be {}, the number of features is only {}, overwriting to 0", cli.feature_x, feature_count);
        cli.feature_x = 0;
    }
    if cli.feature_y >= feature_count {
        warn!("Configured feature_y can't be {}, the number of features is only {}, overwriting to 1", cli.feature_y, feature_count);
        cli.feature_y = 1;
    }
    info!("Loaded {} datapoints from {:?}", points.len(), cli.input_file);
    return Ok(points);
}

pub fn init() -> Cli {
    let mut cli = Cli::parse();
    let loglevel = match cli.log_level {
        LogLevel::Warn => "RUST_LOG=off,nses_sp=warn",
        LogLevel::Info => "RUST_LOG=off,nses_sp=info",
        LogLevel::Error => "RUST_LOG=off,nses_sp=error",
        LogLevel::Debug => "RUST_LOG=off,nses_sp=debug",
        LogLevel::Trace => "RUST_LOG=off,nses_sp=trace",
        LogLevel::Off => "RUST_LOG=off",
    };
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", loglevel)
    }
    env_logger::init_from_env(Env::default());
    if cli.batch_size.is_some_and(|v| return v < 1) {
        cli.batch_size = Some(64);
        warn!("Invalid batch size, must be at least 1, overwriting to {}", cli.batch_size.unwrap());
    }
    if cli.grid_padding <= 0.0 {
        cli.grid_padding = 1.0;
        warn!("Padding must be greater than 0, overwriting to {}", cli.grid_padding);
    }
    if cli.momentum < 0.0 {
        cli.momentum = 0.0;
        warn!("Momentum can't be negative, overwriting to {}", cli.momentum);
    } else if cli.momentum >= 1.0 {
        cli.momentum = 0.5;
        warn!("Momentum must be smaller than 1, overwriting to {}", cli.momentum);
    }
    return cli;
}

pub fn construct_class_1hot_mapping(data: &[Point]) -> BiMap<ClassID, ClassIndex> {
    trace!("Constructing one-hot class mapping...");
    let mut map = BiMap::new();
    let unique_classes = data.iter().map(|d| return d.class).unique().sorted();
    for (cls_idx, cls_val) in unique_classes.enumerate() {
        map.insert(cls_val, ClassIndex(cls_idx as u32));
    }
    trace!("Successfully constructed one-hot mapping for all classes.");
    return map;
}

pub fn compute_cost(data: &DVector<f32>, target: &DVector<f32>) -> f32 {
    return (data - target).map(|v| return v * v).sum();
}
