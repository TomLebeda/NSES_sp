use std::collections::HashMap;

use itertools::Itertools;
use log::trace;
use rerun::{Boxes2D, Color, Points2D, RecordingStream, TimeSeriesScalar};
use rgb_hsv::hsv_to_rgb;

use crate::{cli::Cli, net::ClassID, utils::Point};

/// Construct boxes around each class in training data to indicate correct classes
pub fn get_training_boxes(cli: &Cli, data: &[Point], color_map: &HashMap<ClassID, Color>) -> Boxes2D {
    let unique_classes = data.iter().map(|d| return d.class).unique().sorted();
    let mut colors: Vec<Color> = vec![];
    let mut mins: Vec<(f32, f32)> = vec![];
    let mut sizes: Vec<(f32, f32)> = vec![];
    unique_classes.for_each(|class_val| {
        let filtered: Vec<Point> = data.iter().filter(|p| return p.class == class_val).cloned().collect();
        let (left, right, top, bottom) = get_data_boundaries(cli, &filtered);
        mins.push((left, bottom));
        sizes.push((right - left, top - bottom));
        colors.push(*color_map.get(&class_val).unwrap())
    });
    return Boxes2D::from_mins_and_sizes(mins, sizes).with_colors(colors);
}

/// Construct grid of points around the provided data to visualize class areas
pub fn get_point_grid(cli: &Cli, data: &[Point]) -> Vec<Point> {
    let (left, right, top, bottom) = get_data_boundaries(cli, data);
    let (center_x, center_y) = ((left + right) / 2.0, (top + bottom) / 2.0);
    let (width, height) = ((right - left).abs(), (top - bottom).abs());
    let from_x = center_x - width * cli.grid_padding;
    let to_x = center_x + width * cli.grid_padding;
    let from_y = center_y - height * cli.grid_padding;
    let to_y = center_y + height * cli.grid_padding;
    let mut points: Vec<Point> = vec![];
    let mut x = from_x;
    while x <= to_x {
        let mut y = from_y;
        while y <= to_y {
            points.push(Point {
                features: vec![x, y],
                class: ClassID(0),
            });
            y += cli.grid_step;
        }
        x += cli.grid_step;
    }
    return points;
}

pub fn log_points(rec: &RecordingStream, label: &str, radius: f32, points: &[Point], color_map: &HashMap<ClassID, Color>, cli: &mut Cli) {
    let points = Points2D::new(points.iter().map(|p| {
        return (p.features[cli.feature_x], p.features[cli.feature_y]);
    }))
    .with_colors(points.iter().map(|p| {
        return color_map.get(&p.class).unwrap().to_owned();
    }))
    .with_radii(points.iter().map(|_| {
        return radius;
    }));
    rec.log(label, &points).unwrap();
}

pub fn log_scalar(rec: &RecordingStream, value: f64, label: &str) {
    rec.log(label, &TimeSeriesScalar::new(value)).unwrap();
}

fn get_data_boundaries(cli: &Cli, data: &[Point]) -> (f32, f32, f32, f32) {
    let left = data.iter().filter_map(|p| return p.features.get(cli.feature_x)).fold(f32::INFINITY, |a, &b| return a.min(b));
    let right = data.iter().filter_map(|p| return p.features.get(cli.feature_x)).fold(f32::NEG_INFINITY, |a, &b| return a.max(b));
    let bottom = data.iter().filter_map(|p| return p.features.get(cli.feature_y)).fold(f32::INFINITY, |a, &b| return a.min(b));
    let top = data.iter().filter_map(|p| return p.features.get(cli.feature_y)).fold(f32::NEG_INFINITY, |a, &b| return a.max(b));
    return (left, right, top, bottom);
}

pub fn construct_class_color_mapping(data: &[Point]) -> HashMap<ClassID, Color> {
    trace!("Constructing class color mapping...");
    let mut map: HashMap<ClassID, Color> = HashMap::new();
    let unique_classes = data.iter().map(|d| return d.class).unique().sorted();
    let num_colors = unique_classes.len();
    let hsv_colors: Vec<(f32, f32, f32)> = (0..num_colors).map(|i| return (i as f32 / num_colors as f32, 1.0, 1.0)).collect();
    for (cls, hsv) in unique_classes.zip(hsv_colors) {
        let (r, g, b) = hsv_to_rgb(hsv);
        map.insert(cls, Color::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8));
    }
    trace!("Successfully constructed color mapping for all classes.");
    return map;
}
