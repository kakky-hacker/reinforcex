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

    pub fn push_back(&mut self, item: T) {
        if self.deque.len() == self.max_len {
            self.deque.pop_front();
        }
        self.deque.push_back(item);
    }

    pub fn empty(&mut self) {
        self.deque = VecDeque::new();
    }

    pub fn clone(&self) -> VecDeque<T> {
        self.deque.clone()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounded_vec_deque_new() {
        let deque: BoundedVecDeque<i32> = BoundedVecDeque::new(3);
        assert_eq!(deque.clone().len(), 0);
    }

    #[test]
    fn test_bounded_vec_deque_push_back() {
        let mut deque = BoundedVecDeque::new(3);
        deque.push_back(1);
        deque.push_back(2);
        deque.push_back(3);
        deque.push_back(4);
        assert_eq!(deque.clone().len(), 3);
        assert_eq!(deque.clone().into_iter().collect::<Vec<_>>(), vec![2, 3, 4]);
    }

    #[test]
    fn test_bounded_vec_deque_empty() {
        let mut deque = BoundedVecDeque::new(3);
        deque.push_back(1);
        deque.push_back(2);
        deque.empty();
        assert_eq!(deque.clone().len(), 0);
    }
}
