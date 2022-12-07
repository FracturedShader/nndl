use std::iter::zip;

use itertools::{izip, Itertools};
use nalgebra::{DMatrix, DVector, Dynamic, Matrix, RawStorage, U1};
use permutation_iterator::Permutor;
use rand::Rng;

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
    nabla_b: Vec<DVector<f32>>,
    nabla_w: Vec<DMatrix<f32>>,
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

        let (nabla_b, nabla_w) = Self::make_nablas(&biases, &weights);

        Network {
            layer_sizes: Vec::from(layer_sizes),
            biases,
            weights,
            nabla_b,
            nabla_w,
        }
    }

    pub fn feed_forward(&self, input: DVector<f32>) -> DVector<f32> {
        zip(&self.weights, &self.biases).fold(input, |a, (w, b)| (w * a + b).map(sigmoid))
    }

    pub fn learn_sgd(
        &mut self,
        training_data: &MNISTDataSet,
        epochs: u32,
        batch_size: usize,
        eta: f32,
    ) {
        let data_len = training_data.images.len() as u64;
        let batches_len = (data_len as usize + batch_size - 1) / batch_size;

        for i in 0..epochs {
            for (i, batch) in Permutor::new(data_len)
                .chunks(batch_size)
                .into_iter()
                .enumerate()
            {
                print!("Updating batch {}/{}\r", i, batches_len);
                self.update_batch(training_data, batch, batch_size, eta);
            }

            println!("Epoch {} complete", i);
        }
    }

    pub fn learn_sgd_tested(
        &mut self,
        training_data: &MNISTDataSet,
        testing_data: &MNISTDataSet,
        epochs: u32,
        batch_size: usize,
        eta: f32,
    ) {
        let data_len = training_data.images.ncols() as u64;
        let test_data_len = testing_data.images.ncols();
        let batches_len = (data_len as usize + batch_size - 1) / batch_size;

        for i in 0..epochs {
            for (i, batch) in Permutor::new(data_len)
                .chunks(batch_size)
                .into_iter()
                .enumerate()
            {
                print!("Updating batch {}/{}\r", i, batches_len);
                self.update_batch(training_data, batch, batch_size, eta);
            }

            println!("Evaluating network against test data.");

            let correct_count = self.evaluate(testing_data);

            println!(
                "Epoch {} score: {}% ({}/{})",
                i,
                100.0 * correct_count as f32 / test_data_len as f32,
                correct_count,
                test_data_len
            );
        }
    }
}

fn ascii_ramp(v: f32) -> char {
    let gradient = r#"$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,"^`'. "#;
    let ci = v * ((gradient.len() - 1) as f32);

    gradient.chars().nth(ci as usize).unwrap()
}

fn display_image<'a, I>(data: I)
where
    I: Iterator<Item = &'a f32>,
{
    for (i, &v) in data.enumerate() {
        let c = ascii_ramp(1.0 - v);

        if i % 28 == 27 {
            println!("{}{}", c, c);
        } else {
            print!("{}{}", c, c);
        }
    }
}

impl Network {
    fn back_prop(
        &mut self,
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
            .zip(test_data.labels.iter())
            .map(|(img, expected)| {
                let res = self.feed_forward(img.clone_owned());

                if res.argmax().0 == *expected as usize {
                    1
                } else {
                    0
                }
            })
            .sum()
    }

    fn make_nablas(
        biases: &Vec<DVector<f32>>,
        weights: &Vec<DMatrix<f32>>,
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

    fn update_batch<I>(
        &mut self,
        training_data: &MNISTDataSet,
        batch: I,
        batch_size: usize,
        eta: f32,
    ) where
        I: Iterator<Item = u64>,
    {
        self.zero_nabla();

        for idx in batch {
            let (delta_nabla_bias, delta_nabla_weights) =
                self.back_prop(training_data, idx as usize);

            for (nb, dnb) in self.nabla_b.iter_mut().zip(delta_nabla_bias) {
                *nb += dnb;
            }

            for (nw, dnw) in self.nabla_w.iter_mut().zip(delta_nabla_weights) {
                *nw += dnw;
            }
        }

        let rate = eta / (batch_size as f32);

        for (b, nb) in self.biases.iter_mut().zip(&self.nabla_b) {
            *b -= nb * rate;
        }

        for (w, nw) in self.weights.iter_mut().zip(&self.nabla_w) {
            *w -= nw * rate;
        }
    }

    fn zero_nabla(&mut self) {
        for v in self.nabla_b.iter_mut() {
            v.fill(0.0);
        }

        for v in self.nabla_w.iter_mut() {
            v.fill(0.0);
        }
    }
}
