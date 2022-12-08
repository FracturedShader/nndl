mod mnist;
mod network;

use std::path::PathBuf;

use clap::Parser;
use mnist::MNISTData;
use network::Network;

/// Program implementing bits from http://neuralnetworksanddeeplearning.com/
#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Args {
    /// Path to MNIST folder with ubyte gzips
    mnist_folder: PathBuf,
}

fn train_network(data: MNISTData) {
    let mut network = Network::new_random(&[data.training.images.shape().0, 30, 10]);

    network.learn_sgd(&data.training, Some(&data.test), 30, 10, 3.0);
}

fn main() {
    let args = Args::parse();

    println!("Loading MNIST data");

    match MNISTData::parse(&args.mnist_folder) {
        Ok(data) => train_network(data),
        Err(e) => eprint!("Error loading MNIST data: {}", e),
    }
}
