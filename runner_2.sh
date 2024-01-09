#!/bin/bash
input_file="./data/data2_Lebeda.txt"
final_layer_af="siun:0.6"
batch_size=500
momentum=0.8
learning_rate=0.2
target_acc=98.0
grid_step=0.2
grid_padding=0.6
hidden_layer="6:relu:0.6"
./target/release/nses_sp \
	--show \
	--input-file $input_file \
	--final-layer-af $final_layer_af \
	--batch-size $batch_size \
	--momentum $momentum \
	--grid-step $grid_step \
	--learning-rate $learning_rate \
	--target-acc $target_acc \
	--grid-padding $grid_padding \
	--hidden-layers $hidden_layer
