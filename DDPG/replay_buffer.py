"""
replay_buffer.py
================

一个轻量级的单文件 DQN 经验回放池。

- `RingBuffer`  : 固定容量的环形双向队列（底层是定长数组 + head 指针，索引 O(1)）。
- `ReplayBuffer`: 基于 RingBuffer 的经验池，存放任意元组，支持随机采样 batch。

特点：
    * 纯标准库，无第三方依赖。
    * 满了之后 append 自动丢弃最旧的一条（FIFO），容量恒定。
    * 随机采样无放回，O(batch_size) 复杂度。
    * 提供 `sample()` 返回 List[tuple]，以及 `sample_transposed()` 返回按字段分组的 batch。

用法示例（DQN 训练循环）::

    buffer = ReplayBuffer(capacity=100_000, seed=42)

    # 存经验
    buffer.push((state, action, reward, next_state, done))
    # 或者
    buffer.add(state, action, reward, next_state, done)

    # 采样
    batch = buffer.sample(32)                       # List[Tuple]
    s, a, r, s2, d = buffer.sample_transposed(32)   # 按列拆好，直接丢进网络
"""

from __future__ import annotations

import random
from typing import Generic, Iterator, List, Optional, Tuple, TypeVar

__all__ = ["RingBuffer", "ReplayBuffer"]

T = TypeVar("T")


class RingBuffer(Generic[T]):
    """固定容量的环形双向队列。

    当缓冲区已满时：
        - `append(item)`     会从头部弹出最旧的元素，返回被弹出的元素；
        - `appendleft(item)` 会从尾部弹出最旧的元素，返回被弹出的元素。
    这样 `len(buffer)` 永远 <= capacity。
    """

    __slots__ = ("_capacity", "_data", "_head", "_size")

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity 必须为正整数，收到 {capacity!r}")
        self._capacity = int(capacity)
        self._data: List[Optional[T]] = [None] * self._capacity
        self._head = 0          # 指向最旧的元素
        self._size = 0          # 当前元素个数

    # ------------------------------------------------------------------ #
    # 基本信息
    # ------------------------------------------------------------------ #
    @property
    def capacity(self) -> int:
        return self._capacity

    def is_full(self) -> bool:
        return self._size == self._capacity

    def __len__(self) -> int:
        return self._size

    def __bool__(self) -> bool:
        return self._size > 0

    # ------------------------------------------------------------------ #
    # 入队 / 出队
    # ------------------------------------------------------------------ #
    def append(self, item: T) -> Optional[T]:
        """尾部入队。若已满，则弹出头部最旧元素并返回它，否则返回 None。"""
        if self._size < self._capacity:
            self._data[(self._head + self._size) % self._capacity] = item
            self._size += 1
            return None

        evicted = self._data[self._head]
        self._data[self._head] = item
        self._head = (self._head + 1) % self._capacity
        return evicted

    def appendleft(self, item: T) -> Optional[T]:
        """头部入队。若已满，则弹出尾部最旧元素并返回它，否则返回 None。"""
        evicted: Optional[T] = None
        if self._size == self._capacity:
            evicted = self.pop()
        self._head = (self._head - 1) % self._capacity
        self._data[self._head] = item
        self._size += 1
        return evicted

    def popleft(self) -> T:
        """从头部弹出最旧元素。"""
        if self._size == 0:
            raise IndexError("pop from empty RingBuffer")
        item = self._data[self._head]
        self._data[self._head] = None
        self._head = (self._head + 1) % self._capacity
        self._size -= 1
        return item  # type: ignore[return-value]

    def pop(self) -> T:
        """从尾部弹出最新的元素。"""
        if self._size == 0:
            raise IndexError("pop from empty RingBuffer")
        idx = (self._head + self._size - 1) % self._capacity
        item = self._data[idx]
        self._data[idx] = None
        self._size -= 1
        return item  # type: ignore[return-value]

    def clear(self) -> None:
        self._data = [None] * self._capacity
        self._head = 0
        self._size = 0

    # ------------------------------------------------------------------ #
    # 索引 / 迭代
    # ------------------------------------------------------------------ #
    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self._size))]
        if index < 0:
            index += self._size
        if not 0 <= index < self._size:
            raise IndexError(f"index {index} out of range (size={self._size})")
        return self._data[(self._head + index) % self._capacity]

    def __iter__(self) -> Iterator[T]:
        head, size, cap = self._head, self._size, self._capacity
        for i in range(size):
            yield self._data[(head + i) % cap]  # type: ignore[misc]

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(capacity={self._capacity}, "
            f"size={self._size}, data={list(self)!r})"
        )


class ReplayBuffer:
    """DQN 经验回放池。

    存放任意元组（通常为 `(state, action, reward, next_state, done)`），
    容量满时新元素进来会自动丢掉最旧的一条。
    """

    def __init__(self, capacity: int, seed: Optional[int] = None) -> None:
        self._buffer: RingBuffer[Tuple] = RingBuffer(capacity)
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------ #
    # 写入
    # ------------------------------------------------------------------ #
    def push(self, transition: Tuple) -> None:
        """压入一条经验（一个元组）。缓冲区已满时自动丢弃最旧的一条。"""
        self._buffer.append(transition)

    def add(self, *transition) -> None:
        """`push` 的便利写法：`buffer.add(s, a, r, s2, done)`。"""
        self._buffer.append(tuple(transition))

    # ------------------------------------------------------------------ #
    # 采样
    # ------------------------------------------------------------------ #
    def sample(self, batch_size: int) -> List[Tuple]:
        """随机无放回采样 batch_size 条经验，返回 List[Tuple]。"""
        n = len(self._buffer)
        if batch_size <= 0:
            raise ValueError(f"batch_size 必须为正整数，收到 {batch_size!r}")
        if batch_size > n:
            raise ValueError(
                f"batch_size({batch_size}) 大于当前缓存大小({n})"
            )
        indices = self._rng.sample(range(n), batch_size)
        return [self._buffer[i] for i in indices]

    def sample_transposed(self, batch_size: int) -> Tuple:
        """随机采样并转置，返回按字段分组的元组。

        例：``s, a, r, s2, d = buffer.sample_transposed(32)``
        """
        batch = self.sample(batch_size)
        return tuple(zip(*batch))

    # ------------------------------------------------------------------ #
    # 其它
    # ------------------------------------------------------------------ #
    @property
    def capacity(self) -> int:
        return self._buffer.capacity

    def clear(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)

    def __iter__(self) -> Iterator[Tuple]:
        return iter(self._buffer)

    def __getitem__(self, index):
        return self._buffer[index]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(capacity={self.capacity}, size={len(self)})"


# ---------------------------------------------------------------------- #
# 自测 / 演示
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    # --- RingBuffer 自测 ---
    rb = RingBuffer(3)
    for i in range(3):
        rb.append(i)
    assert list(rb) == [0, 1, 2]
    assert rb.append(3) == 0                 # 满了，弹出最旧的 0
    assert list(rb) == [1, 2, 3]
    assert rb[0] == 1 and rb[-1] == 3
    assert rb.popleft() == 1
    assert rb.pop() == 3
    assert list(rb) == [2]
    print("RingBuffer OK:", rb)

    # --- ReplayBuffer 自测 ---
    buf = ReplayBuffer(capacity=5, seed=0)
    for i in range(5):
        buf.add(i, i * 10, i * 100)          # 存元组
    assert len(buf) == 5
    print("初始缓存:", list(buf))

    for i in range(5, 8):                    # 超过容量，自动丢弃最旧的
        buf.push((i, i * 10, i * 100))
    print("滚动后缓存:", list(buf))
    assert len(buf) == 5 and list(buf)[0][0] == 3

    batch = buf.sample(3)
    print("随机 batch:", batch)
    assert len(batch) == 3

    s, a, r = buf.sample_transposed(4)
    print("转置 batch -> s:", s, " a:", a, " r:", r)

    try:
        buf.sample(10)
    except ValueError as e:
        print("预期报错:", e)