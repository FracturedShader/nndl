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
    /// Path to MNIST folder with ubyte.gz files
    mnist_folder: PathBuf,

    /// What sizes the hidden layers of the network should have
    #[arg(default_values_t = [30])]
    middle_layer_sizes: Vec<usize>,

    /// Number of complete passes of the training data when learning
    #[arg(short, long, default_value_t = 30)]
    epochs: u32,

    /// How many images to do back-propagation on before teaching network
    #[arg(short, long, default_value_t = 10)]
    nabla_batch_size: usize,

    /// Scale when applying the computed gradient per batch
    #[arg(short, long, default_value_t = 3.0)]
    learning_rate: f32,
}

fn train_network(args: &Args, data: MNISTData) {
    let mut layer_sizes = Vec::with_capacity(args.middle_layer_sizes.len() + 2);

    layer_sizes.push(data.training.images.shape().0);
    layer_sizes.extend(&args.middle_layer_sizes);
    layer_sizes.push(10);

    let mut network = Network::new_random(&layer_sizes);

    network.learn_sgd(
        &data.training,
        Some(&data.test),
        args.epochs,
        args.nabla_batch_size,
        args.learning_rate,
    );
}

fn main() {
    let args = Args::parse();

    println!("Loading MNIST data");

    match MNISTData::parse(&args.mnist_folder) {
        Ok(data) => train_network(&args, data),
        Err(e) => eprint!("Error loading MNIST data: {}", e),
    }
}
