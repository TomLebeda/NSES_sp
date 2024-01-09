nses_sp v1.1.0
Author: Tomáš Lebeda <tom.lebeda@gmail.com>

This is simple piece of software that implements parametric fully-connected neural network for point classification and real-time visualizations.

USAGE: 'nses_sp [OPTIONS] --input-file <INPUT_FILE>'

OPTIONS:
  -l, --log-level <LOG_LEVEL>
          level of logging details (into stderr)
          
          [default: trace]
          [possible values: error, warn, info, debug, trace, off]

  -i, --input-file <INPUT_FILE>
          path to file with input (training) data
          
          Expected format is space-separated-matrix of numbers where:
          
          - each line represents one training point
          
          - last column are natural numbers representing class label
          
          - other columns are real numbers representing features

      --num-features <NUM_FEATURES>
          number of features that will be present in the data, leave 0 for "auto"
          
          this may speed up the data parsing and you can use it to check whether the input file contains what your are expecting it to contain, but it's mostly unnecessary since this program can figure out the number of features automatically
          
          [default: 0]

      --num-classes <NUM_CLASSES>
          number of classes that will be present in the data, leave 0 for "auto"
          
          this may speed up the data parsing and you can use it to check whether the input file contains what your are expecting it to contain, but it's mostly unnecessary since this program can figure out the number of classes automatically
          
          [default: 0]

      --hidden-layers <HIDDEN_LAYERS>
          Hidden layer configuration separated by commas in format [size]:[fn_type]:[parameter]
          
          Function type can be: BinaryBipolar (bibi, bb), BinaryUnipolar (biun, bu), SigmoidUnipolar (siun, su), SigmoidBipolar (sibi, sb), ReLU (relu, r), Linear (line, lin, l)
          
          Parameter can be omitted for Binary functions, otherwise they must be floating point number
          
          Example:
          
          '10:sibi:0.6' will add hidden layer with 10 neurons using SigmoidBipolar activation function with parameter lambda=0.6
          
          '10:sibi:0.6,5:bb,4:l:0.2' will add hidden layer with 10 neurons using SigmoidBipolar activation function with parameter lambda=0.6, after that it will add hidden layer with 5 neurons using BinaryBipolar activation function, after that it will add hidden layer with 4 neurons using Linear activation function with parameter lambda=0.2. In this case, the network will have 4 layers in total (3 hidden + output) where the first (closest to inputs) will have 10 neurons.

      --final-layer-af <FINAL_LAYER_AF>
          activation functions to be used in final layer in format [fn_type]:[parameter], see --hidden-layers for possible values
          
          this has the same syntax as --hidden-layers but you don't specify the number of neurons
          
          [default: BinaryUnipolar]

      --batch-size <BATCH_SIZE>
          batch size for training, leave empty to use all data in each batch

      --learning-rate <LEARNING_RATE>
          learning rate
          
          [default: 0.1]

      --momentum <MOMENTUM>
          learning momentum, must be larger or equal to 0 and smaller than 1
          
          [default: 0]

      --log-file-costs <LOG_FILE_COSTS>
          file where loss values will be logged

      --log-file-acc <LOG_FILE_ACC>
          file where accuracy values will be logged

      --log-file-last-grid <LOG_FILE_LAST_GRID>
          file where the last points grid values will be saved

      --log-file-last-points <LOG_FILE_LAST_POINTS>
          file where the last points classes will be saved

      --grid-step <GRID_STEP>
          density of point-grid for visualisation of class regions
          
          [default: 0.2]

      --grid-padding <GRID_PADDING>
          percentage of space to pad around actual data when drawing grid
          
          [default: 1]

      --grid-dot-radius <GRID_DOT_RADIUS>
          radius of dots on grid for visualisation of areas
          
          [default: 0.05]

      --dot-radius <DOT_RADIUS>
          radius of datapoints for visualisation of areas
          
          [default: 0.2]

      --feature-x <FEATURE_X>
          index of the feature that should be used for X axis while doing visualisation
          
          [default: 0]

      --feature-y <FEATURE_Y>
          index of the feature that should be used for Y axis while doing visualisation
          
          [default: 1]

  -s, --show
          enable visualisations via Rerun.io

      --target-acc <TARGET_ACC>
          target accuracy of the classification
          
          [default: 99.5]

      --max-epochs <MAX_EPOCHS>
          max number of training epochs
          
          [default: 10000]

      --data-log <DATA_LOG>
          file where the data for visualization will be saved as .rrd file
          
          when --show is enabled, this is used only as a emergency backup plan if the viewer fails to spawn

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version
