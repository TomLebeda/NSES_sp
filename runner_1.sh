#!/bin/bash
input_file="./data/data1_Lebeda.txt"
final_layer_af="siun:0.4"
batch_size=500
momentum=0.8
learning_rate=0.2
target_acc=100.0
grid_step=0.05
grid_padding=0.6
grid_dot_radius=0.01
dot_radius=0.07
hidden_layer="4:lin:0.6"
./target/release/nses_sp \
	--show \
	--input-file $input_file \
	--final-layer-af $final_layer_af \
	--batch-size $batch_size \
	--momentum $momentum \
	--grid-step $grid_step \
	--learning-rate $learning_rate \
	--target-acc $target_acc \
	--grid-dot-radius $grid_dot_radius \
	--dot-radius $dot_radius \
	--grid-padding $grid_padding
