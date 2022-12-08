use std::{
    io::{self, Write},
    iter::zip,
    time::{Duration, Instant},
};

use itertools::{izip, Itertools};
use nalgebra::{DMatrix, DVector};
use permutation_iterator::Permutor;
use rand::Rng;
use rayon::prelude::*;

use crate::mnist::MNISTDataSet;

fn sigmoid(x: f32) -> f32 {
    if x < -40.0 {
        0.0
    } else if x > 40.0 {
        1.0
    } else {
        1.0 / (1.0 + f32::exp(-x))
    }
}

fn sigmoid_prime(x: f32) -> f32 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

pub struct Network {
    layer_sizes: Vec<usize>,
    biases: Vec<DVector<f32>>,
    weights: Vec<DMatrix<f32>>,
}

impl Network {
    pub fn new_random(layer_sizes: &[usize]) -> Self {
        let mut rng = rand::thread_rng();

        let biases: Vec<_> = layer_sizes
            .iter()
            .skip(1)
            .map(|&nr| DVector::from_fn(nr, |_, _| rng.gen::<f32>() * 2.0 - 1.0))
            .collect();

        let weights: Vec<_> = zip(layer_sizes.iter(), layer_sizes.iter().skip(1))
            .map(|(&left_layer_len, &right_layer_len)| {
                DMatrix::from_fn(right_layer_len, left_layer_len, |_, _| {
                    rng.gen::<f32>() * 2.0 - 1.0
                })
            })
            .collect();

        Network {
            layer_sizes: Vec::from(layer_sizes),
            biases,
            weights,
        }
    }

    pub fn feed_forward(&self, input: DVector<f32>) -> DVector<f32> {
        zip(&self.weights, &self.biases).fold(input, |a, (w, b)| (w * a + b).map(sigmoid))
    }

    pub fn learn_sgd(
        &mut self,
        training_data: &MNISTDataSet,
        testing_data: Option<&MNISTDataSet>,
        epochs: u32,
        batch_size: usize,
        eta: f32,
    ) {
        let data_len = training_data.images.ncols() as u64;
        let test_data_len = testing_data.map(|td| td.images.ncols()).unwrap_or_default();
        let batches_len = (data_len as usize + batch_size - 1) / batch_size;

        let print_freq = Duration::from_millis(200);
        let mut last_print = Instant::now() - print_freq;

        let mut parallel_time = Option::<u128>::None;
        let mut serial_time = Option::<u128>::None;

        for epoch_idx in 0..epochs {
            let batch_start = Instant::now();
            let parallel = match (parallel_time, serial_time) {
                (None, _) => true,
                (_, None) => false,
                (Some(a), Some(b)) => a < b,
            };

            for (batch_idx, batch) in Permutor::new(data_len)
                .chunks(batch_size)
                .into_iter()
                .enumerate()
            {
                if last_print.elapsed() > print_freq {
                    print!("\rUpdating batch {}/{}", batch_idx, batches_len);
                    io::stdout().flush().unwrap();
                    last_print = Instant::now();
                }

                if parallel {
                    self.update_batch_parallel(
                        training_data,
                        batch.map(|i| i as usize).collect(),
                        eta,
                    );
                } else {
                    self.update_batch(training_data, batch.map(|i| i as usize).collect(), eta);
                }
            }

            let elapsed = batch_start.elapsed().as_millis();

            if parallel {
                &mut parallel_time
            } else {
                &mut serial_time
            }
            .get_or_insert(elapsed);

            println!(
                "\rUpdating batch {}/{0} completed in {} in {}ms",
                batches_len,
                if parallel { "parallel" } else { "serial" },
                elapsed
            );

            if let Some(td) = testing_data {
                print!("Evaluating network");
                io::stdout().flush().unwrap();

                let correct_count = self.evaluate(td);

                println!(
                    "\rEpoch {}/{} score: {}% ({}/{})",
                    epoch_idx + 1,
                    epochs,
                    100.0 * correct_count as f32 / test_data_len as f32,
                    correct_count,
                    test_data_len
                );
            } else {
                println!("Epoch {}/{} complete", epoch_idx + 1, epochs);
            }
        }
    }
}

impl Network {
    fn back_prop(
        &self,
        training_data: &MNISTDataSet,
        src_idx: usize,
    ) -> (Vec<DVector<f32>>, Vec<DMatrix<f32>>) {
        let img = training_data.images.column(src_idx);

        // Feed forward, but track raw + clamped values
        let mut activation = DVector::from_row_slice(img.as_slice());
        let mut activations = Vec::<DVector<f32>>::with_capacity(self.layer_sizes.len());
        let mut zs = Vec::<DVector<f32>>::with_capacity(self.layer_sizes.len());

        for (b, w) in zip(&self.biases, &self.weights) {
            let z = w * &activation + b;
            let mut a2 = z.map(sigmoid);

            zs.push(z);

            std::mem::swap(&mut activation, &mut a2);

            activations.push(a2);
        }

        // Convert number to output vector
        let mut expected = DVector::<f32>::zeros(*self.layer_sizes.last().unwrap());

        expected[training_data.labels[src_idx] as usize] = 1.0;

        let mut delta =
            (&activation - expected).component_mul(&zs.last().unwrap().map(sigmoid_prime));

        activations.push(activation);

        let (mut dnb, mut dnw) = Self::make_nablas(&self.biases, &self.weights);

        *dnb.last_mut().unwrap() = delta.clone();

        *dnw.last_mut().unwrap() = &delta * &activations[activations.len() - 2].transpose();

        for (z, w, nb, nw, a) in izip!(
            zs.iter().rev().skip(1),
            self.weights.iter().rev(),
            dnb.iter_mut().rev().skip(1),
            dnw.iter_mut().rev().skip(1),
            activations.iter().rev().skip(2)
        ) {
            let sp = z.map(sigmoid_prime);
            delta = (w.transpose() * delta).component_mul(&sp);

            *nb = delta.clone();
            *nw = &delta * a.transpose();
        }

        (dnb, dnw)
    }

    fn evaluate(&self, test_data: &MNISTDataSet) -> usize {
        test_data
            .images
            .column_iter()
            .zip(test_data.labels.row_iter())
            .par_bridge()
            .map(|(img, expected)| {
                let res = self.feed_forward(img.clone_owned());

                usize::from(res.argmax().0 == expected.x as usize)
            })
            .sum()
    }

    fn make_nablas(
        biases: &[DVector<f32>],
        weights: &[DMatrix<f32>],
    ) -> (Vec<DVector<f32>>, Vec<DMatrix<f32>>) {
        (
            biases.iter().map(|v| DVector::zeros(v.nrows())).collect(),
            weights
                .iter()
                .map(|m| {
                    let (r, c) = m.shape();

                    DMatrix::zeros(r, c)
                })
                .collect(),
        )
    }

    fn update_batch(&mut self, training_data: &MNISTDataSet, batch: Vec<usize>, eta: f32) {
        let (nabla_b, nabla_w) = batch
            .iter()
            .map(|&idx| self.back_prop(training_data, idx))
            .reduce(|(mut nb, mut nw), (dnb, dnw)| {
                nb.iter_mut().zip(dnb).for_each(|(n, dn)| *n += dn);
                nw.iter_mut().zip(dnw).for_each(|(n, dn)| *n += dn);

                (nb, nw)
            })
            .expect("Batch size shouldn't be zero");

        let rate = eta / (batch.len() as f32);

        self.biases
            .iter_mut()
            .zip(nabla_b)
            .for_each(|(b, nb)| *b -= nb * rate);

        self.weights
            .iter_mut()
            .zip(nabla_w)
            .for_each(|(w, nw)| *w -= nw * rate);
    }

    fn update_batch_parallel(&mut self, training_data: &MNISTDataSet, batch: Vec<usize>, eta: f32) {
        let (nabla_b, nabla_w) = batch
            .par_iter()
            .map(|&idx| self.back_prop(training_data, idx))
            .reduce_with(|(mut nb, mut nw), (dnb, dnw)| {
                nb.par_iter_mut().zip(dnb).for_each(|(n, dn)| *n += dn);
                nw.par_iter_mut().zip(dnw).for_each(|(n, dn)| *n += dn);

                (nb, nw)
            })
            .expect("Batch size shouldn't be zero");

        let rate = eta / (batch.len() as f32);

        self.biases
            .par_iter_mut()
            .zip(nabla_b)
            .for_each(|(b, nb)| *b -= nb * rate);

        self.weights
            .par_iter_mut()
            .zip(nabla_w)
            .for_each(|(w, nw)| *w -= nw * rate);
    }
}
