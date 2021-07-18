use rayon::prelude::*;
use std::iter::FromIterator;

/// Struct that we use to threat sequential and parallel iterators in an 
/// homogeneus way.
///
/// # Example
/// ```ignore
/// let x = true;
/// let v = 69420;
/// let iter = if x {
///     ItersWrapper::Sequential((0..100).into_iter())
/// } else {
///     ItersWrapper::Parallel((0..100).into_par_iter().map(|x| x + 1))
/// };
/// 
/// println!("{:?}", iter.map(|x: i32| -> i32 {v + x * 2}).collect::<Vec<_>>());
/// ```
pub enum ItersWrapper<
    Item,
    I:Iterator<Item=Item>, 
    P:ParallelIterator<Item=Item>,
>{
    Sequential(I),
    Parallel(P),
}

impl<Item, I, P> ItersWrapper<Item, I, P> 
where
    Item: Send,
    I: Iterator<Item=Item>,
    P: ParallelIterator<Item=Item>,
{
    pub fn map<F, R>(self, op: F) -> ItersWrapper<R, std::iter::Map<I, F>, rayon::iter::Map<P, F>>
    where
        R: Send,
        F: Fn(Item) -> R  + Sync + Send
        {
            match self {
                Self::Parallel(p) => ItersWrapper::Parallel(p.map(op)),
                Self::Sequential(i) => ItersWrapper::Sequential(i.map(op)),
            }
    }

    pub fn filter<F>(self, op: F) -> ItersWrapper<Item, std::iter::Filter<I, F>, rayon::iter::Filter<P, F>>
    where
        F: Fn(&Item) -> bool + Sync + Send
        {
            match self {
                Self::Parallel(p) => ItersWrapper::Parallel(p.filter(op)),
                Self::Sequential(i) => ItersWrapper::Sequential(i.filter(op)),
            }
    }

    pub fn for_each<F>(self, op: F)
    where
        F: Fn(Item) + Sync + Send
        {
            match self {
                Self::Parallel(p) => p.for_each(op),
                Self::Sequential(i) => i.for_each(op),
            }
    }

    pub fn reduce<ID, F>(self,identity: ID, op: F) -> Item
    where
        F: Fn(Item, Item) -> Item + Sync + Send,
        ID: Fn() -> Item + Sync + Send,
        {
            match self {
                Self::Parallel(p) => p.reduce(identity, op),
                Self::Sequential(i) => i.chain(vec![identity()].into_iter()).reduce(op).unwrap(),
            }
    }

    pub fn all<F>(self, op: F) -> bool
    where
        F: Fn(Item) -> bool + Sync + Send,
        {
            match self {
                Self::Parallel(p) => p.all(op),
                Self::Sequential(mut i) => i.all(op),
            }
    }

    pub fn any<F>(self, op: F) -> bool
    where
        F: Fn(Item) -> bool + Sync + Send,
        {
            match self {
                Self::Parallel(p) => p.any(op),
                Self::Sequential(mut i) => i.any(op),
            }
    }

    pub fn collect<B: FromIterator<Item> + FromParallelIterator<Item>>(self) -> B {
            match self {
                Self::Parallel(p) => p.collect::<B>(),
                Self::Sequential(i) => i.collect::<B>(),
            }
    } 
}