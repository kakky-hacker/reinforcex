use std::collections::VecDeque;

pub struct BoundedVecDeque<T> {
    deque: VecDeque<T>,
    max_len: usize,
}

impl<T> BoundedVecDeque<T>
where
    T: Clone,
{
    pub fn new(max_len: usize) -> Self {
        Self {
            deque: VecDeque::new(),
            max_len,
        }
    }

    pub fn push_back(&mut self, item: T) -> Option<T> {
        let mut res: Option<T> = None;
        if self.deque.len() == self.max_len {
            res = self.deque.pop_front();
        }
        self.deque.push_back(item);
        res
    }

    pub fn empty(&mut self) {
        self.deque = VecDeque::new();
    }

    pub fn is_empty(&self) -> bool {
        self.deque.is_empty()
    }

    pub fn clone(&self) -> VecDeque<T> {
        self.deque.clone()
    }

    pub fn front_mut(&mut self) -> &mut T {
        self.deque.front_mut().unwrap()
    }

    pub fn len(&self) -> usize {
        self.deque.len()
    }

    pub fn to_vec(&self) -> Vec<T> {
        self.deque.clone().into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounded_vec_deque_new() {
        let deque: BoundedVecDeque<i32> = BoundedVecDeque::new(3);
        assert_eq!(deque.clone().len(), 0);
        assert!(deque.is_empty());
    }

    #[test]
    fn test_bounded_vec_deque_push_back() {
        let mut deque = BoundedVecDeque::new(3);
        assert!(deque.push_back(1).is_none());
        assert!(deque.push_back(2).is_none());
        assert!(deque.push_back(3).is_none());
        assert_eq!(deque.push_back(4).unwrap(), 1);
        assert_eq!(deque.clone().len(), 3);
        assert_eq!(deque.clone().into_iter().collect::<Vec<_>>(), vec![2, 3, 4]);
        assert!(!deque.is_empty());
    }

    #[test]
    fn test_bounded_vec_deque_empty() {
        let mut deque = BoundedVecDeque::new(3);
        deque.push_back(1);
        deque.push_back(2);
        deque.empty();
        assert_eq!(deque.clone().len(), 0);
        assert!(deque.is_empty());
    }
}
