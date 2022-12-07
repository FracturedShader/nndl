use std::{
    fs::File,
    io::{self, BufReader, Read},
    path::Path,
};

use byteorder::{BigEndian, ReadBytesExt};
use nalgebra::{DMatrix, DVector};

#[derive(Clone, Copy, Debug, PartialEq)]
enum DataType {
    UnsignedByte,
    SignedByte,
    Short,
    Int,
    Float,
    Double,
    Unknown,
}

impl From<u32> for DataType {
    fn from(v: u32) -> Self {
        match (v >> 8) & 0xFF {
            0x08 => Self::UnsignedByte,
            0x09 => Self::SignedByte,
            0x0B => Self::Short,
            0x0C => Self::Int,
            0x0D => Self::Float,
            0x0E => Self::Double,
            _ => Self::Unknown,
        }
    }
}

#[derive(Debug)]
struct RawImages {
    dims: (u32, u32),
    data: Vec<f32>,
}

impl RawImages {
    pub fn parse(src: &mut impl Read) -> io::Result<Self> {
        let magic = src.read_u32::<BigEndian>()?;

        let data_type = DataType::from(magic);
        let data_dims = magic & 0xFF;

        assert_eq!(data_type, DataType::UnsignedByte);
        assert_eq!(data_dims, 3);

        let count = src.read_u32::<BigEndian>()?;
        let d1 = src.read_u32::<BigEndian>()?;
        let d2 = src.read_u32::<BigEndian>()?;
        let dims = (d1, d2);

        let mut data: Vec<u8> = vec![0; (count * d1 * d2) as usize];

        src.read_exact(&mut data)?;

        Ok(RawImages {
            dims,
            data: data.into_iter().map(|v| (v as f32) / 255.0).collect(),
        })
    }

    pub fn split_off(&mut self, at: usize) -> Self {
        RawImages {
            dims: self.dims,
            data: self
                .data
                .split_off(at * (self.dims.0 * self.dims.1) as usize),
        }
    }
}

#[derive(Debug)]
struct RawLabels(Vec<u8>);

impl RawLabels {
    pub fn parse(src: &mut impl Read) -> io::Result<Self> {
        let magic = src.read_u32::<BigEndian>()?;

        let data_type = DataType::from(magic);
        let data_dims = magic & 0xFF;

        assert_eq!(data_type, DataType::UnsignedByte);
        assert_eq!(data_dims, 1);

        let count = src.read_u32::<BigEndian>()?;

        let mut data = vec![0; count as usize];

        src.read_exact(&mut data)?;

        Ok(RawLabels(data))
    }

    pub fn split_off(&mut self, at: usize) -> RawLabels {
        RawLabels(self.0.split_off(at))
    }
}

#[derive(Debug)]
pub struct MNISTDataSet {
    pub images: DMatrix<f32>,
    pub labels: DVector<u8>,
}

impl MNISTDataSet {
    fn from_raw_parts(images: RawImages, labels: RawLabels) -> Self {
        let im_size = (images.dims.0 * images.dims.1) as usize;

        Self {
            images: DMatrix::from_vec(im_size, images.data.len() / im_size, images.data),
            labels: DVector::from_vec(labels.0),
        }
    }
}

#[derive(Debug)]
pub struct MNISTData {
    pub training: MNISTDataSet,
    pub validation: MNISTDataSet,
    pub test: MNISTDataSet,
}

impl MNISTData {
    pub fn parse(dir: &Path) -> io::Result<Self> {
        let mut train_images = BufReader::new(File::open(dir.join("train-images-idx3-ubyte.gz"))?);
        let mut train_labels = BufReader::new(File::open(dir.join("train-labels-idx1-ubyte.gz"))?);
        let mut test_images = BufReader::new(File::open(dir.join("t10k-images-idx3-ubyte.gz"))?);
        let mut test_labels = BufReader::new(File::open(dir.join("t10k-labels-idx1-ubyte.gz"))?);

        let mut train_images = RawImages::parse(&mut train_images)?;
        let mut train_labels = RawLabels::parse(&mut train_labels)?;
        let validation_images = train_images.split_off(50_000);
        let validation_labels = train_labels.split_off(50_000);
        let test_images = RawImages::parse(&mut test_images)?;
        let test_labels = RawLabels::parse(&mut test_labels)?;

        Ok(Self {
            training: MNISTDataSet::from_raw_parts(train_images, train_labels),
            validation: MNISTDataSet::from_raw_parts(validation_images, validation_labels),
            test: MNISTDataSet::from_raw_parts(test_images, test_labels),
        })
    }
}
