use std::path::PathBuf;

use bimap::BiMap;
use log::{info, trace, warn};
use nalgebra::{DMatrix, DVector};
use rand::seq::SliceRandom;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use rerun::RecordingStream;

use crate::{
    cli::Cli,
    layer::Layer,
    plot::{construct_class_color_mapping, get_point_grid, get_training_boxes, log_points, log_scalar},
    utils::{compute_cost, construct_class_1hot_mapping, Point},
};

#[derive(Debug)]
pub struct Network {
    pub layers: Vec<Layer>,
    pub input_feature_count: usize,
    pub output_class_count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ClassID(pub u32);
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ClassIndex(pub u32);

impl Network {
    /// Creates a new network based on configuration from CLI
    pub fn new(cli: &Cli) -> Self {
        // prepare vector for storing layers
        let mut layers: Vec<Layer> = vec![];
        // iterate over hidden layer configurations (from CLI) and create the actual layers
        trace!("creating new network: constructing hidden layers");
        for (layer_idx, conf) in cli.hidden_layers.iter().enumerate() {
            // create new layer based on the current configuration
            let new_layer = Layer {
                layer_index: layer_idx,
                neuron_count: conf.size,
                weights: if layer_idx == 0 {
                    // first layer must have the dimensions based on number of features
                    DMatrix::identity(conf.size, cli.num_features)
                } else {
                    // other layers must have the dimensions based on previous layers
                    let previous_layer = layers.last().unwrap();
                    DMatrix::identity(conf.size, previous_layer.neuron_count)
                },
                biases: DVector::from_element(conf.size, 0.0),
                activation_fn: conf.act_fn,
            };
            // add the layer to the list
            layers.push(new_layer);
        }
        trace!("creating new network: adding final layer");
        // finally add single extra layer -> final output layer
        layers.push(Layer {
            layer_index: layers.len(),
            neuron_count: cli.num_classes,
            weights: if layers.is_empty() {
                // if there are no previous layers, determine the size based on input features
                DMatrix::identity(cli.num_classes, cli.num_features)
            } else {
                // otherwise determine the size based on previous layer
                let previous_layer = layers.last().unwrap();
                DMatrix::identity(cli.num_classes, previous_layer.neuron_count)
            },
            biases: DVector::from_element(cli.num_classes, 0.0),
            activation_fn: cli.final_layer_af,
        });
        // construct the actual Network object and return it
        return Network {
            layers,
            input_feature_count: cli.num_features,
            output_class_count: cli.num_classes,
        };
    }

    /// Randomize all weights in biases in the network.
    pub fn randomize(&mut self) {
        info!("randomizing weights and biases");
        for layer in &mut self.layers {
            layer.weights = DMatrix::new_random(layer.weights.nrows(), layer.weights.ncols());
            layer.biases = DVector::new_random(layer.biases.nrows());
        }
    }

    /// Pass set of points through the network and classify each of them.
    /// The 'class' attribute of each point will be overwritten by the class with highest score as
    /// output from the network.
    pub fn classify_points(&self, points: &mut [Point], class_map: &BiMap<ClassID, ClassIndex>) {
        // classification is independent of other points => we can do it in parallel
        points.par_iter_mut().for_each(|point| {
            // compute the output of the network for given point
            let output = self.compute(DVector::from_vec(point.features.clone()));
            // find max value of the input and it's index
            let (max_idx, _max_val) = output.argmax();
            // wrap the index of maximum output as ClassIndex for type safety
            let class_idx = ClassIndex(max_idx as u32);
            // get the class ID based on the class index
            match class_map.get_by_right(&class_idx) {
                Some(class_id) => point.class = *class_id,
                None => {
                    warn!("Can't find class ID for class index {}, will not classify the point.", class_idx.0)
                }
            }
        })
    }

    /// Pass the input vector through the whole network and return the output vector
    fn compute(&self, input: DVector<f32>) -> DVector<f32> {
        // keep track of the last layer activation
        let mut last_activation: DVector<f32> = input;
        for layer in self.layers.iter() {
            // iteratively propagate the vector through the network's layers
            last_activation = layer.compute(&last_activation);
        }
        // the last activation is the network's output vector
        return last_activation;
    }

    /// run the training routine with given marked points
    pub fn train(&mut self, data: &[Point], cli: &mut Cli) {
        info!("starting training routine");
        // if visualisation is enabled, run the Rerun viewer or try to save the data into file
        let mut rec: Option<RecordingStream> = None;
        if cli.show {
            trace!("starting Rerun viewer");
            match rerun::RecordingStreamBuilder::new("nses_sp_visual").spawn() {
                Ok(viewer) => {
                    rec = Some(viewer);
                }
                Err(e) => {
                    warn!("Failed to start Rerun viewver: {e}");
                    if cli.data_log.is_none() {
                        let timestamp = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S%.3f");
                        let log_file_name = format!("autosave_{}.rrd", timestamp);
                        cli.data_log = Some(PathBuf::from(log_file_name));
                    }
                    info!("Can't do realtime visualisation, I will try to save the data into file.");
                    info!("You can open the saved data afterwards on \"https://app.rerun.io\" or with 'rerun path/to/file' command.");
                    cli.show = false;
                }
            }
        }
        if rec.is_none() && cli.data_log.is_some() {
            info!("Data will be saved here: {:?}", &cli.data_log);
            match rerun::RecordingStreamBuilder::new("nses_sp_visual").save(cli.data_log.as_ref().unwrap()) {
                Ok(saver) => {
                    rec = Some(saver);
                }
                Err(e) => {
                    warn!("Can't save the data: {e}")
                }
            }
        }
        // prepare the class mapping between class IDs and their indexes
        let class_map = construct_class_1hot_mapping(data);
        // prepare mapping between classes and their colors
        let color_map = construct_class_color_mapping(data);
        // prepare grid of artificial points for class-area visualisation
        let mut point_grid = match rec.is_some() {
            true => Some(get_point_grid(cli, data)),
            // don't bother with it if visualisations are turned off
            false => None,
        };

        // prepare accumulators for file-logging values
        let mut costs: Vec<f32> = vec![];
        let mut accs: Vec<f32> = vec![];

        // copy all the data for classification (to not corrupt training data)
        let mut data_copy: Vec<Point> = data.to_vec();

        // if visualisation is enabled get the class-boxes and log them
        if let Some(rec) = &rec {
            let boxes = get_training_boxes(cli, data, &color_map);
            rec.log("boxes", &boxes).unwrap();
        }

        // prepare randomizer for thread-save shuffling between epochs
        let mut rng = rand::thread_rng();
        // prepare variable for terminating training process
        let mut should_train = true;
        // get the batch size from config or set it to be full dataset if not set
        let batch_size = cli.batch_size.unwrap_or(data.len());
        // prepare the counter for epochs
        let mut epochs = 0;
        // prepare the variable for keeping track of last differences for momentum-based updates
        let mut last_diffs_w: Vec<DMatrix<f32>> = self
            .layers
            .iter()
            .rev()
            .map(|l| {
                let mut m = l.weights.clone();
                m.fill(0.0);
                return m;
            })
            .collect();
        let mut last_diffs_b: Vec<DVector<f32>> = self
            .layers
            .iter()
            .rev()
            .map(|l| {
                let mut m = l.biases.clone();
                m.fill(0.0);
                return m;
            })
            .collect();
        // prepare the variable for keeping track of last accuracy
        let mut last_logged_acc = 0.0;
        // enter the training loop
        trace!("entering training loop");
        while should_train {
            // get the indexes of points into a list
            // (we use indexes instead of the actual vector of points because of performance reasons)
            let mut point_indexes: Vec<usize> = (0..data.len()).collect();
            // shuffle the list of indexes => later will pick numbers at random
            point_indexes.shuffle(&mut rng);

            // run EPOCH:
            for (batch_idx, batch) in point_indexes.chunks(batch_size).enumerate() {
                // process current batch and get it's gradients and cost
                let (grads_w, grads_b, batch_cost) = process_batch(batch, data, &class_map, &self.layers);
                costs.push(batch_cost);

                // if visualisation is enabled, log the batch cost
                if let Some(rec) = &rec {
                    log_scalar(rec, batch_cost as f64, "batch_cost")
                }
                // update weights and biases
                (last_diffs_w, last_diffs_b) = self.update_params(cli, last_diffs_w, last_diffs_b, grads_w, grads_b);
                // if visualisation is enabled...
                if let Some(rec) = &rec {
                    // classify the grid points
                    self.classify_points(point_grid.as_mut().unwrap(), &class_map);
                    // and log them
                    log_points(rec, "grid_points", cli.grid_dot_radius, point_grid.as_ref().unwrap(), &color_map, cli);
                    // and also classify the copy of training data
                    self.classify_points(&mut data_copy, &class_map);
                    // and log them
                    log_points(rec, "data", cli.dot_radius, &data_copy, &color_map, cli);
                }
                // get the current classification accuracy
                // (we use the same training data which is a bad practice)
                let acc = self.get_accuracy(data, &class_map);
                accs.push(acc);
                // if visualisation is enabled, log the current accuracy
                if let Some(rec) = &rec {
                    log_scalar(rec, acc as f64, "train_accuracy")
                }
                // get the accuracy change since last update
                let acc_change = (acc - last_logged_acc).abs();
                // log the accuracy if changed since last time
                if acc_change > 0.01 {
                    trace!("accuracy: {acc:3.2}% (on {}. batch of {}. epoch)", batch_idx + 1, epochs + 1);
                    // update the last LOGGED accuracy, otherwise small changes could accumulate unnoticed
                    last_logged_acc = acc;
                }
                // check if training should continue
                if cli.target_acc - acc < 0.001 {
                    info!("accuracy {}% -> reached target point {}% => exiting training loop", acc, cli.target_acc);
                    // this will prevent the start of next epoch
                    should_train = false;
                    // this will interrupt current epoch
                    break;
                } else if epochs >= cli.max_epochs {
                    info!("reached maximum number of epochs ({}), exiting training loop with acc {}%", cli.max_epochs, acc);
                    // this will prevent the start of next epoch
                    should_train = false;
                    // this will interrupt current epoch
                    break;
                }
            }
            // increment the epoch counter after the end of epoch
            epochs += 1;
        }

        // print out the final weights and biases
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            info!("Layer {} - weights: {}", layer_idx, layer.weights);
            info!("Layer {} - biases^T: {}", layer_idx, layer.biases.transpose());
        }

        if let Some(file) = &cli.log_file_costs {
            trace!("writing costs into file {}", file.to_string_lossy());
            let write_res = std::fs::write(file, format!("{:?}", costs).replace(']', "").replace('[', "").replace(", ", "\n"));
            if write_res.is_err() {
                warn!("Failed to write costs into {}", file.to_string_lossy());
            } else {
                info!("Written costs into file {}", file.to_string_lossy());
            }
        }
        if let Some(file) = &cli.log_file_acc {
            trace!("writing accuracies into file {}", file.to_string_lossy());
            let write_res = std::fs::write(file, format!("{:?}", accs).replace(']', "").replace('[', "").replace(", ", "\n"));
            if write_res.is_err() {
                warn!("Failed to write accuracies into {}", file.to_string_lossy());
            } else {
                info!("Written accuracies into file {}", file.to_string_lossy());
            }
        }
        if let Some(file) = &cli.log_file_last_grid {
            trace!("writing last grid points into file {}", file.to_string_lossy());
            let grid = match point_grid {
                Some(existing_grid) => existing_grid,
                None => {
                    let mut new_grid = get_point_grid(cli, data);
                    self.classify_points(&mut new_grid, &class_map);
                    new_grid
                }
            };
            let string_buf: Vec<String> = grid
                .iter()
                .map(|p| return format!("{} {} {}", p.features[cli.feature_x], p.features[cli.feature_y], p.class.0))
                .collect();
            let write_res = std::fs::write(file, format!("x y label\n{:?}", string_buf).replace(['[', ']', '"'], "").replace(", ", "\n"));
            if write_res.is_err() {
                warn!("Failed to write grid points into {}", file.to_string_lossy());
            } else {
                info!("Written grid points into file {}", file.to_string_lossy());
            }
        }
        if let Some(file) = &cli.log_file_last_points {
            trace!("writing last points into file {}", file.to_string_lossy());
            let grid = data_copy;
            let string_buf: Vec<String> = grid
                .iter()
                .map(|p| return format!("{} {} {}", p.features[cli.feature_x], p.features[cli.feature_y], p.class.0))
                .collect();
            let write_res = std::fs::write(file, format!("x y label\n{:?}", string_buf).replace(['[', ']', '"'], "").replace(", ", "\n"));
            if write_res.is_err() {
                warn!("Failed to write grid points into {}", file.to_string_lossy());
            } else {
                info!("Written grid points into file {}", file.to_string_lossy());
            }
        }
    }

    /// Computes the accuracy of classification from the training points, return percentage (0.0 to 100.0)
    fn get_accuracy(&self, training_data: &[Point], class_map: &BiMap<ClassID, ClassIndex>) -> f32 {
        // prepare the counter of correct classifications
        let mut correct = 0;
        // re-classify all points (without changing the labels)
        for point in training_data {
            // get the input vector for this point
            let input = DVector::from_vec(point.features.clone());
            // pass the vector through the network to get the output
            let output = self.compute(input);
            // find the index of maximum output
            let (max_idx, _) = output.argmax();
            // find the target index for the correct class
            let target_idx = class_map.get_by_left(&point.class).unwrap().0;
            // if the indexes match, count as correct
            if max_idx == target_idx as usize {
                correct += 1;
            }
        }
        // return the final accuracy as percentage
        return (correct as f32 / (training_data.len() as f32)) * 100.0;
    }

    // update the weights and biases of the network with the momentum-based gradient descent
    fn update_params(
        &mut self,
        cli: &Cli,
        last_diffs_w: Vec<DMatrix<f32>>,
        last_diffs_b: Vec<DVector<f32>>,
        grads_w: Vec<DMatrix<f32>>,
        grads_b: Vec<DVector<f32>>,
    ) -> (Vec<DMatrix<f32>>, Vec<DVector<f32>>) {
        // prepare the differences which should have the same shape as gradients
        let mut diffs_w: Vec<DMatrix<f32>> = grads_w.clone();
        let mut diffs_b: Vec<DVector<f32>> = grads_b.clone();
        // for each layer...
        self.layers.iter_mut().rev().enumerate().for_each(|(layer_idx, layer)| {
            // compute the differences based on momentum and gradients
            diffs_w[layer_idx] = cli.momentum * &last_diffs_w[layer_idx] + (1.0 - cli.momentum) * cli.learning_rate * &grads_w[layer_idx];
            diffs_b[layer_idx] = cli.momentum * &last_diffs_b[layer_idx] + (1.0 - cli.momentum) * cli.learning_rate * &grads_b[layer_idx];
            // and actually update the layer's weights and biases
            layer.weights -= &diffs_w[layer_idx];
            layer.biases -= &diffs_b[layer_idx];
        });
        // return the applied changes for next iteration (needed for momentum)
        return (diffs_w, diffs_b);
    }
}

/// get the differential of cost function based on the layer's activation values
fn get_act_diff(layer_idx: usize, layers: &Vec<Layer>, zets: &Vec<DVector<f32>>, activations: &Vec<DVector<f32>>, target: &DVector<f32>) -> DVector<f32> {
    // if we are at the last layer, the recursion ends
    if layer_idx == layers.len() - 1 {
        // this is derivative of cost function
        return activations[layer_idx + 1].clone() - target.clone();
    } else {
        // otherwise we need to recursively compute values of previous layers
        let next_weights = &layers[layer_idx + 1].weights;
        let next_zets = &zets[layer_idx + 1];
        let sig_prime_of_next_zets = next_zets.map(|v| return layers[layer_idx].activation_fn.compute_prime(v));
        let next_activation_diff = get_act_diff(layer_idx + 1, layers, zets, activations, target);
        // evil matrix sorcery
        return next_weights.transpose() * sig_prime_of_next_zets.component_mul(&next_activation_diff);
    };
}

// Process single batch of training points and return the gradients and costs
fn process_batch(batch: &[usize], data: &[Point], class_map: &BiMap<ClassID, ClassIndex>, layers: &Vec<Layer>) -> (Vec<DMatrix<f32>>, Vec<DVector<f32>>, f32) {
    // prepare the accumulators for gradients and cost
    let mut accum_weight_grads: Vec<DMatrix<f32>> = vec![];
    let mut accum_bias_grads: Vec<DVector<f32>> = vec![];
    let mut accum_cost = 0.0;

    // prepare batch size for convenience
    let batch_size = batch.len();
    // iterate all point indexes in current batch
    for point_idx in batch {
        // prepare helper variables
        let mut current_activations: Vec<DVector<f32>> = vec![];
        let mut current_zets: Vec<DVector<f32>> = vec![];

        // get the actual point
        let point = &data[*point_idx];
        // construct the input vector from the point
        let input: DVector<f32> = DVector::from_vec(point.features.clone());
        // construct the target vector based on the label
        let target_idx = class_map.get_by_left(&point.class).unwrap().0;
        let mut target = DVector::<f32>::zeros(class_map.len());
        target[target_idx as usize] = 1.0;
        // append the input as the first activation (as zeroth layer)
        current_activations.push(input);

        // FORWARD PASS:
        for (_layer_idx, layer) in layers.iter().enumerate() {
            // store the zet value (before activation function) for gradient calculation
            let zet = &layer.weights * current_activations.last().unwrap() + &layer.biases;
            current_zets.push(zet.clone());
            // pass the zet through the activation function and also store the output
            let activation = zet.map(|val| return layer.activation_fn.compute(val));
            current_activations.push(activation);
        }

        // get the cost of current point classification
        let cost = compute_cost(current_activations.last().unwrap(), &target);
        // don't forget to accumulate the cost
        accum_cost += cost;

        // BACK PROPAGATION:
        // prepare the variables
        let mut weight_grads: Vec<DMatrix<f32>> = vec![];
        let mut bias_grads: Vec<DVector<f32>> = vec![];
        // iterate the layers in reverse
        for (layer_idx, layer) in layers.iter().enumerate().rev() {
            // extract the variables for convenience
            let act = &current_activations[layer_idx];
            let zet = &current_zets[layer_idx];
            let sig_prime_of_zet = zet.map(|v| return layer.activation_fn.compute_prime(v));
            let act_diff = get_act_diff(layer_idx, layers, &current_zets, &current_activations, &target);
            // compute the actual element of gradient for biases
            let bias_grad_item = sig_prime_of_zet.component_mul(&act_diff);
            // compute the actual element of gradient for weights, which is more complex with evil matrix sorcery
            let mut weight_grad_item = sig_prime_of_zet * act.transpose();
            for (row_idx, mut row) in weight_grad_item.row_iter_mut().enumerate() {
                row *= act_diff[row_idx];
            }
            // append the gradient elements to the total gradient vectors
            weight_grads.push(weight_grad_item);
            bias_grads.push(bias_grad_item);
        }
        // accumulate the gradients with the previous by summing them
        if accum_weight_grads.is_empty() {
            accum_weight_grads = weight_grads;
        } else {
            accum_weight_grads.iter_mut().zip(weight_grads).for_each(|(acc, curr)| *acc += curr)
        }
        if accum_bias_grads.is_empty() {
            accum_bias_grads = bias_grads;
        } else {
            accum_bias_grads.iter_mut().zip(bias_grads).for_each(|(acc, curr)| *acc += curr)
        }
    }
    // now divide everything by batch size => average gradient for current batch
    let weight_grads: Vec<DMatrix<f32>> = accum_weight_grads.iter_mut().map(|v| return &*v / batch_size as f32).collect();
    let bias_grads: Vec<DVector<f32>> = accum_bias_grads.iter_mut().map(|v| return &*v / batch_size as f32).collect();
    return (weight_grads, bias_grads, accum_cost);
}
