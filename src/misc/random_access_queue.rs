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
            let index: usize = (-index - 1) as usize;
            if index < self.queue_back.len() {
                self.queue_back.get(index)
            } else {
                self.queue_front.get(index - self.queue_back.len())
            }
        }
    }

    pub fn sample(&self, k: usize) -> Vec<&T> {
        let mut rng = rand::thread_rng();
        let length = self.len();
        let indices: Vec<usize> = (0..k).map(|_| rng.gen_range(0..length)).collect();
        indices
            .into_iter()
            .filter_map(|i: usize| self.get(i as isize))
            .collect()
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
    fn test_random_access_queue_sample() {
        let mut queue = RandomAccessQueue::new(5);
        for i in 1..=5 {
            queue.append(i);
        }
        let samples = queue.sample(3);
        assert_eq!(samples.len(), 3);
    }
}
