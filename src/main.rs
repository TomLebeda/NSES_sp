use log::*;
mod act_fn;
mod cli;
mod layer;
mod net;
mod plot;
mod utils;

use net::*;
use utils::*;

fn main() {
    let mut cli = init();
    let data = load_data(&mut cli).unwrap_or_else(|e| {
        error!("{}", e);
        std::process::exit(1);
    });
    let mut net = Network::new(&cli);
    net.randomize();
    net.train(&data, &mut cli)
}
