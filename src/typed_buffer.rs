use std::marker::PhantomData;

use wgpu::Buffer;

// #[derive(Debug, Hash, PartialEq, Eq)]
pub struct Array<T> {
    pub inner: wgpu::Buffer,
    _phatom: PhantomData<T>,
}

impl<T: Sized> Array<T> {
    pub fn new(buffer: Buffer) -> Self {
        Self {
            inner: buffer,
            _phatom: Default::default(),
        }
    }
    pub fn size(&self) -> u64 {
        self.inner.size() / std::mem::size_of::<T>() as u64
    }
}
