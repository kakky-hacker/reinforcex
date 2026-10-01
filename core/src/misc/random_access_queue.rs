use rand::prelude::SliceRandom;
use rand::Rng;
use std::collections::VecDeque;

pub struct RandomAccessQueue<T> {
    queue_front: Vec<T>,
    queue_back: VecDeque<T>,
    maxlen: usize,
}

impl<T> RandomAccessQueue<T> {
    pub fn new(maxlen: usize) -> Self {
        RandomAccessQueue {
            queue_front: Vec::new(),
            queue_back: VecDeque::new(),
            maxlen,
        }
    }

    pub fn clear(&mut self) {
        // Replay experiences can own their successor through Arc. Release
        // oldest entries first so dropping the reversed front vector cannot
        // recursively destroy an entire long trajectory on one stack.
        while self.len() > 0 {
            self.popleft();
        }
    }

    pub fn len(&self) -> usize {
        self.queue_front.len() + self.queue_back.len()
    }

    pub fn append(&mut self, item: T) {
        self.queue_back.push_back(item);
        if self.len() > self.maxlen {
            self.popleft();
        }
    }

    pub fn popleft(&mut self) -> T {
        if self.queue_front.is_empty() {
            if self.queue_back.is_empty() {
                panic!("pop from empty RandomAccessQueue")
            }
            self.queue_front = self.queue_back.drain(..).collect();
            self.queue_front.reverse();
        }
        self.queue_front.pop().unwrap()
    }

    pub fn get(&self, index: isize) -> Option<&T> {
        if index >= 0 {
            let index: usize = index as usize;
            if index < self.queue_front.len() {
                self.queue_front.get(self.queue_front.len() - 1 - index)
            } else {
                self.queue_back.get(index - self.queue_front.len())
            }
        } else {
            let index = index.unsigned_abs() - 1;
            if index < self.queue_back.len() {
                self.queue_back.get(self.queue_back.len() - 1 - index)
            } else {
                self.queue_front.get(index - self.queue_back.len())
            }
        }
    }

    pub fn sample_with_replacement(&self, k: usize) -> Vec<&T> {
        let mut rng = rand::thread_rng();
        let length = self.len();
        let indices: Vec<usize> = (0..k).map(|_| rng.gen_range(0..length)).collect();
        indices
            .into_iter()
            .filter_map(|i: usize| self.get(i as isize))
            .collect()
    }

    pub fn sample_without_replacement(&self, k: usize) -> Vec<&T> {
        let length = self.len();
        if k > length {
            panic!("Cannot sample more elements than available in the queue");
        }

        let mut indices: Vec<usize> = (0..length).collect();
        let mut rng = rand::thread_rng();
        indices.shuffle(&mut rng);
        indices
            .into_iter()
            .take(k)
            .filter_map(|i| self.get(i as isize))
            .collect()
    }
}

impl<T> Drop for RandomAccessQueue<T> {
    fn drop(&mut self) {
        self.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_access_queue_new() {
        let queue: RandomAccessQueue<i32> = RandomAccessQueue::new(5);
        assert_eq!(queue.len(), 0);
    }

    #[test]
    fn test_negative_indices_follow_logical_queue_order() {
        let mut queue = RandomAccessQueue::new(5);
        for value in 0..8 {
            queue.append(value);
        }
        for index in 0..5 {
            assert_eq!(queue.get(index), Some(&(index + 3)));
            assert_eq!(queue.get(-(index + 1)), Some(&(7 - index)));
        }
        assert_eq!(queue.get(-6), None);
        assert_eq!(queue.get(isize::MIN), None);
    }

    #[test]
    fn test_linked_entries_are_cleared_and_dropped_without_recursive_chain() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc, Mutex,
        };

        struct Node {
            next: Mutex<Option<Arc<Node>>>,
            depth: Arc<AtomicUsize>,
            max_depth: Arc<AtomicUsize>,
        }
        impl Drop for Node {
            fn drop(&mut self) {
                let depth = self.depth.fetch_add(1, Ordering::SeqCst) + 1;
                self.max_depth.fetch_max(depth, Ordering::SeqCst);
                drop(self.next.lock().unwrap().take());
                self.depth.fetch_sub(1, Ordering::SeqCst);
            }
        }
        for explicit_clear in [false, true] {
            let depth = Arc::new(AtomicUsize::new(0));
            let max_depth = Arc::new(AtomicUsize::new(0));
            let mut queue = RandomAccessQueue::new(63);
            let mut previous: Option<Arc<Node>> = None;
            for _ in 0..64 {
                let node = Arc::new(Node {
                    next: Mutex::new(None),
                    depth: depth.clone(),
                    max_depth: max_depth.clone(),
                });
                if let Some(previous) = previous.take() {
                    *previous.next.lock().unwrap() = Some(node.clone());
                }
                queue.append(node.clone());
                previous = Some(node);
            }
            drop(previous);
            if explicit_clear {
                queue.clear();
            }
            drop(queue);
            assert_eq!(depth.load(Ordering::SeqCst), 0);
            assert_eq!(max_depth.load(Ordering::SeqCst), 1);
        }
    }

    #[test]
    fn test_random_access_queue_append() {
        let mut queue = RandomAccessQueue::new(3);
        queue.append(1);
        queue.append(2);
        queue.append(3);
        queue.append(4);
        assert_eq!(queue.len(), 3);
    }

    #[test]
    fn test_random_access_queue_popleft() {
        let mut queue = RandomAccessQueue::new(5);
        queue.append(1);
        queue.append(2);
        let first = queue.popleft();
        assert_eq!(first, 1);
        assert_eq!(queue.len(), 1);
    }

    #[test]
    fn test_random_access_queue_sample_with_replacement() {
        let mut queue = RandomAccessQueue::new(5);
        for i in 1..=5 {
            queue.append(i);
        }
        let samples = queue.sample_with_replacement(3);
        assert_eq!(samples.len(), 3);
        let samples = queue.sample_with_replacement(6);
        assert_eq!(samples.len(), 6);
    }

    #[test]
    fn test_random_access_queue_sample_without_replacement() {
        let mut queue = RandomAccessQueue::new(5);
        for i in 1..=5 {
            queue.append(i);
        }
        let samples = queue.sample_without_replacement(3);
        assert_eq!(samples.len(), 3);
    }

    #[test]
    #[should_panic]
    fn test_random_access_queue_sample_without_replacement_should_panic() {
        let mut queue = RandomAccessQueue::new(5);
        for i in 1..=5 {
            queue.append(i);
        }
        queue.sample_without_replacement(6);
    }
}
