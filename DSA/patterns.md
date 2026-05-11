# 🚀 DSA Master Revision — Amazon SDE 2 Edition
### *Not just code. Understand it. Own it. Solve anything.*

---

## 📖 How to Use This Guide

This is a **study material**, not a cheatsheet.  
Each section follows this structure:

```
📌 What is it?          — The core idea in plain English
🧠 Intuition            — The "aha" moment. WHY does this work?
⚠️  Common Mistakes     — What trips people up
🔁 Pattern Template     — Reusable skeleton
📝 Problem Statement    — What LeetCode/Amazon asks
🐌 Brute Force          — The obvious O(n²) / O(n³) answer
⚡ Optimized            — One step better
🏆 Optimal              — The real solution
🔑 Key Tricks           — Things to memorize
```

---

## 📋 Table of Contents

1. [Complexity Analysis — Think Before You Code](#1-complexity-analysis--think-before-you-code)
2. [Arrays & Two Pointers](#2-arrays--two-pointers)
3. [Sliding Window](#3-sliding-window)
4. [Prefix Sum](#4-prefix-sum)
5. [Hashing & Frequency Maps](#5-hashing--frequency-maps)
6. [Strings](#6-strings)
7. [Linked Lists](#7-linked-lists)
8. [Stacks & Monotonic Stack](#8-stacks--monotonic-stack)
9. [Binary Search](#9-binary-search)
10. [Trees (DFS & BFS)](#10-trees-dfs--bfs)
11. [Heaps / Priority Queues](#11-heaps--priority-queues)
12. [Graphs](#12-graphs)
13. [Dynamic Programming](#13-dynamic-programming)
14. [Recursion & Backtracking](#14-recursion--backtracking)
15. [Sorting](#15-sorting)
16. [Trie](#16-trie)
17. [Amazon Pattern Recognition Guide](#17-amazon-pattern-recognition-guide)

---

---

# 1. Complexity Analysis — Think Before You Code

## 📌 What is it?

Before writing a single line, you should **know the target complexity** based on the input size `n`. Amazon OA gives you `n` constraints — use them.

## 🧠 The Golden Rule Table

| Input Size (n)     | Max Acceptable Complexity | Typical Algorithm             |
|--------------------|--------------------------|-------------------------------|
| n ≤ 10             | O(n!) or O(2ⁿ)           | Backtracking, brute force     |
| n ≤ 20             | O(2ⁿ)                    | Bitmask DP, subsets           |
| n ≤ 100            | O(n³)                    | Triple nested loops           |
| n ≤ 1,000          | O(n²)                    | Double nested loops           |
| n ≤ 100,000        | O(n log n)               | Sorting, heap, divide & conquer |
| n ≤ 1,000,000      | O(n)                     | Two pointers, sliding window, hash |
| n > 1,000,000      | O(log n) or O(1)         | Binary search, math           |

> ⚡ **Trick:** If n = 10⁵ and you write O(n²), that's 10¹⁰ operations. Computer does ~10⁸/sec. That's 100 seconds. **TLE.**

## 📐 Space Complexity Rules

- **O(1)** — Only a fixed number of variables. No extra arrays.
- **O(n)** — You store a copy of the input or a hash map of size n.
- **O(h)** — Tree recursion. h = height = log n for balanced, n for skewed.
- **O(n²)** — A 2D DP table. Rare but happens (LCS, edit distance).

## ⚠️ Common Mistakes

- Forgetting that **recursion uses stack space** — recursive DFS on a graph of n nodes = O(n) space even if no extra array.
- Thinking sorted() is free — it's O(n log n) time + O(n) space.
- Forgetting that Python dict operations are **O(1) average** but O(n) worst case (hash collision).

---

---

# 2. Arrays & Two Pointers

## 📌 What is it?

Two Pointers is a technique where you place two indices (left, right) in an array and **move them intelligently** based on a condition — instead of checking every pair with two nested loops.

## 🧠 Intuition

Imagine you're looking for two people in a sorted lineup whose heights add to exactly 10 feet.

- **Brute force:** Ask every person to stand next to every other person. O(n²).
- **Two pointers:** Stand the shortest person and tallest person together. If too tall → move tallest left. If too short → move shortest right. You eliminate an entire side at each step. O(n).

**The key insight:** When the array is **sorted** (or has some ordering invariant), moving a pointer doesn't just check one pair — it **rules out all pairs on that side**.

## 🔁 Pattern Templates

```
Template 1: Opposite ends (sorted array, pairs)
  left = 0, right = n-1
  while left < right:
      if valid: record, both move inward
      elif need_bigger: left++
      else: right--

Template 2: Same direction (fast/slow, remove duplicates)
  slow = 0
  for fast in range(n):
      if condition: arr[slow] = arr[fast]; slow++

Template 3: Three pointers (3Sum style)
  Fix one, use two pointers for the remaining
```

---

### 🔴 Problem: Two Sum II (Sorted Array)

**Statement:** Given a **1-indexed sorted array**, find two numbers that add up to `target`. Return their indices. Exactly one solution guaranteed.

**Example:**
```
Input:  nums = [2, 7, 11, 15], target = 9
Output: [1, 2]   (nums[0] + nums[1] = 2 + 7 = 9)
```

**🧠 Think:** The array is sorted. If I pick any two elements and their sum is too big, I need a smaller number — so move the right pointer left. If too small, move left pointer right. At each step we're eliminating one candidate for sure.

```python
# 🐌 BRUTE FORCE — O(n²) time, O(1) space
# Check every pair of indices
def two_sum_brute(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i + 1, j + 1]

# 🏆 OPTIMAL — Two Pointers — O(n) time, O(1) space
def two_sum_optimal(nums, target):
    left, right = 0, len(nums) - 1
    while left < right:
        s = nums[left] + nums[right]
        if s == target:
            return [left + 1, right + 1]
        elif s < target:
            left += 1    # need bigger sum → move left forward
        else:
            right -= 1   # need smaller sum → move right backward
```

---

### 🔴 Problem: Container With Most Water

**Statement:** Given n vertical lines on a graph at positions 0..n-1 with heights `height[i]`, find two lines that together with the x-axis forms a container with the most water.

**Example:**
```
Input:  height = [1,8,6,2,5,4,8,3,7]
Output: 49
(Lines at index 1 (height=8) and index 8 (height=7): width=7, min_height=7, area=49)
```

**🧠 Think:** Water held = `min(height[left], height[right]) * (right - left)`. 

Why move the shorter wall? Because:
- Width can only **decrease** as pointers move inward.
- So we MUST try to increase height.
- Moving the taller wall inward can only maintain or decrease height (bounded by shorter wall). Useless.
- Moving the shorter wall inward **might** find a taller wall. That's our only hope.

```python
# 🐌 BRUTE FORCE — O(n²) — Check all pairs
def max_water_brute(height):
    max_water = 0
    for i in range(len(height)):
        for j in range(i + 1, len(height)):
            max_water = max(max_water, min(height[i], height[j]) * (j - i))
    return max_water

# 🏆 OPTIMAL — Two Pointers — O(n) time, O(1) space
def max_water_optimal(height):
    left, right = 0, len(height) - 1
    max_water = 0
    while left < right:
        water = min(height[left], height[right]) * (right - left)
        max_water = max(max_water, water)
        # Always move the shorter wall — only way to possibly do better
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    return max_water
```

---

### 🔴 Problem: 3Sum

**Statement:** Find all unique triplets in an unsorted array that sum to zero.

**Example:**
```
Input:  nums = [-1, 0, 1, 2, -1, -4]
Output: [[-1, -1, 2], [-1, 0, 1]]
```

**🧠 Think:** Reduce 3Sum → 2Sum. Sort the array. Fix one element `nums[i]`, then run two pointers on the rest looking for `target = -nums[i]`. Sorting helps with deduplication too — skip duplicate values of `i`, `left`, and `right`.

```python
# 🐌 BRUTE FORCE — O(n³) — Three nested loops + set to dedup
def three_sum_brute(nums):
    result = set()
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            for k in range(j + 1, len(nums)):
                if nums[i] + nums[j] + nums[k] == 0:
                    result.add(tuple(sorted([nums[i], nums[j], nums[k]])))
    return [list(t) for t in result]

# 🏆 OPTIMAL — Sort + Fix one + Two Pointers — O(n²) time, O(1) space
def three_sum_optimal(nums):
    nums.sort()
    result = []

    for i in range(len(nums) - 2):
        # Skip duplicate values for the fixed element
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        # Early exit: smallest possible sum already > 0
        if nums[i] > 0:
            break

        left, right = i + 1, len(nums) - 1
        while left < right:
            s = nums[i] + nums[left] + nums[right]
            if s == 0:
                result.append([nums[i], nums[left], nums[right]])
                # Skip duplicates for left and right
                while left < right and nums[left] == nums[left + 1]: left += 1
                while left < right and nums[right] == nums[right - 1]: right -= 1
                left += 1
                right -= 1
            elif s < 0:
                left += 1
            else:
                right -= 1
    return result
```

---

### 🔴 Problem: Trapping Rain Water

**Statement:** Given an array of bar heights, compute how much water can be trapped between them after it rains.

**Example:**
```
Input:  height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6

Visual:
    #
  # ##  #
# ####  ##  ← bars
  ↑↑    ↑↑ trapped water between bars
```

**🧠 Think:** Water at position `i` = `min(max_left[i], max_right[i]) - height[i]`. It's bounded by the shorter of the two tallest walls on either side.

Two Pointer insight: If `max_left < max_right`, we know the left side is the bottleneck. We can calculate water for the left pointer without knowing exact right max — we already know right side is taller.

```python
# 🐌 BRUTE FORCE — O(n²) — For each i, scan left and right for max walls
def trap_brute(height):
    n = len(height)
    total = 0
    for i in range(1, n - 1):
        left_max = max(height[:i+1])
        right_max = max(height[i:])
        total += min(left_max, right_max) - height[i]
    return total

# ⚡ OPTIMIZED — Prefix arrays — O(n) time, O(n) space
def trap_prefix(height):
    n = len(height)
    left_max = [0] * n
    right_max = [0] * n
    left_max[0] = height[0]
    right_max[n-1] = height[n-1]
    for i in range(1, n):
        left_max[i] = max(left_max[i-1], height[i])
    for i in range(n-2, -1, -1):
        right_max[i] = max(right_max[i+1], height[i])
    return sum(min(left_max[i], right_max[i]) - height[i] for i in range(n))

# 🏆 OPTIMAL — Two Pointers — O(n) time, O(1) space
def trap_optimal(height):
    left, right = 0, len(height) - 1
    left_max = right_max = 0
    total = 0
    while left < right:
        if height[left] < height[right]:
            # Right side is guaranteed taller — left_max is the true bottleneck
            if height[left] >= left_max:
                left_max = height[left]
            else:
                total += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                total += right_max - height[right]
            right -= 1
    return total
```

## 🔑 Key Tricks for Two Pointers

```
✅ Sort first if you need to use two pointers (usually required)
✅ Use "fix one element + two pointers" to reduce O(n³) → O(n²)
✅ To skip duplicates: while left < right and arr[left] == arr[left+1]: left += 1
✅ When confused which pointer to move: move the one that limits the answer
✅ Works great when: sorted array, pairs/triplets, palindromes, partitioning
```

---

---

# 3. Sliding Window

## 📌 What is it?

A technique to process a **contiguous subarray or substring** efficiently by maintaining a "window" that slides from left to right — instead of re-computing the whole window from scratch each time.

## 🧠 Intuition

Imagine you're reading a newspaper through a small frame (window). Instead of starting from the beginning each time, you just slide the frame one step and update based on what enters and what leaves.

**The key insight:** When you move the window one step right:
- One element **enters** from the right → add it to window state
- One element **leaves** from the left → remove it from window state
- This is O(1) per move, instead of O(k) to recompute

## 🔁 Two Types

**Type 1: Fixed Size Window** — window size `k` is given
```
Useful for: max sum of k elements, avg of k elements, etc.
Move: right moves → add arr[right]; left moves → remove arr[right-k]
```

**Type 2: Variable Size Window** — window grows/shrinks to satisfy a condition
```
Useful for: longest/shortest substring with constraint
Expand right to try including more; shrink left when constraint violated
```

## ⚠️ Common Mistakes

- Not shrinking the window when constraint is violated (infinite loop).
- Shrinking too aggressively (while instead of if, or vice versa).
- Updating the answer inside the shrink loop instead of after it.
- Off-by-one: window size = `right - left + 1`.

---

### 🔴 Problem: Longest Substring Without Repeating Characters

**Statement:** Find the length of the longest substring (contiguous) without repeating characters.

**Example:**
```
Input:  s = "abcabcbb"
Output: 3   ("abc" is the longest with no repeats)

Input:  s = "pwwkew"
Output: 3   ("wke")
```

**🧠 Think:** Use a window [left, right]. Expand right by adding `s[right]` to a set. If we see a duplicate, shrink left until the duplicate is removed. At each step, `right - left + 1` is the current window size.

**Better trick:** Instead of a set, use a hashmap `char → last_seen_index`. When we see `s[right]` again, we can jump `left` directly to `last_seen + 1` instead of crawling one by one.

```python
# 🐌 BRUTE FORCE — O(n³) — Check every substring
def length_of_longest_brute(s):
    def all_unique(sub): return len(set(sub)) == len(sub)
    max_len = 0
    for i in range(len(s)):
        for j in range(i, len(s)):
            if all_unique(s[i:j+1]):
                max_len = max(max_len, j - i + 1)
    return max_len

# ⚡ OPTIMIZED — Sliding Window with Set — O(n) time, O(min(n,m)) space
def length_of_longest_set(s):
    seen = set()
    left = max_len = 0
    for right in range(len(s)):
        while s[right] in seen:
            seen.remove(s[left])
            left += 1
        seen.add(s[right])
        max_len = max(max_len, right - left + 1)
    return max_len

# 🏆 OPTIMAL — Sliding Window with HashMap (jump) — O(n) time, O(min(n,m)) space
# Key improvement: instead of crawling left one step at a time,
# JUMP directly to last_seen[char] + 1
def length_of_longest_optimal(s):
    char_index = {}   # char → its most recent index
    left = 0
    max_len = 0
    for right, char in enumerate(s):
        # If char is in window (last seen index >= left), move left past it
        if char in char_index and char_index[char] >= left:
            left = char_index[char] + 1
        char_index[char] = right
        max_len = max(max_len, right - left + 1)
    return max_len
```

---

### 🔴 Problem: Minimum Window Substring

**Statement:** Given strings `s` and `t`, find the minimum length substring of `s` that contains **all characters of `t`** (including duplicates). Return `""` if none.

**Example:**
```
Input:  s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"

Explanation: "BANC" is the smallest window containing A, B, and C.
```

**🧠 Think:** 
- Track what we *need* (from `t`) and what we *have* (in current window).
- `formed` = number of unique chars in window that match required frequency.
- When `formed == required`, we have a valid window → try to shrink from left.
- When shrinking breaks validity → expand right again.

```python
from collections import Counter

# 🏆 OPTIMAL — Sliding Window with two counters — O(|s| + |t|) time
def min_window(s, t):
    if not t or not s:
        return ""

    t_count = Counter(t)
    required = len(t_count)          # # of unique chars we need

    left = 0
    formed = 0                       # # of unique chars meeting required freq
    window_counts = {}

    best = float("inf"), None, None  # (length, left, right)

    for right in range(len(s)):
        char = s[right]
        window_counts[char] = window_counts.get(char, 0) + 1

        # Check if this char now meets its required frequency
        if char in t_count and window_counts[char] == t_count[char]:
            formed += 1

        # While window is valid, try to shrink from left
        while left <= right and formed == required:
            # Update best answer
            if right - left + 1 < best[0]:
                best = (right - left + 1, left, right)

            # Remove left char from window
            left_char = s[left]
            window_counts[left_char] -= 1
            if left_char in t_count and window_counts[left_char] < t_count[left_char]:
                formed -= 1
            left += 1

    return "" if best[0] == float("inf") else s[best[1]: best[2] + 1]
```

---

### 🔴 Problem: Sliding Window Maximum

**Statement:** Given array `nums` and window size `k`, return the max of each window as it slides from left to right.

**Example:**
```
Input:  nums = [1,3,-1,-3,5,3,6,7], k = 3
Output: [3, 3, 5, 5, 6, 7]

Windows: [1,3,-1]=3, [3,-1,-3]=3, [-1,-3,5]=5, [-3,5,3]=5, [5,3,6]=6, [3,6,7]=7
```

**🧠 Think:** We want max of each window efficiently. Use a **Monotonic Deque** (decreasing order). 
- Add new element: pop from back anything smaller (they can never be the max while this element exists).
- Remove old element: if front of deque is the outgoing element (index = i - k), pop it.
- Front of deque is always the window's max.

```python
from collections import deque

# 🐌 BRUTE FORCE — O(n*k) — Recompute max for each window
def max_sliding_brute(nums, k):
    return [max(nums[i:i+k]) for i in range(len(nums) - k + 1)]

# 🏆 OPTIMAL — Monotonic Deque — O(n) time, O(k) space
def max_sliding_optimal(nums, k):
    dq = deque()   # stores indices; values are decreasing
    result = []

    for i, num in enumerate(nums):
        # Remove indices outside window
        if dq and dq[0] < i - k + 1:
            dq.popleft()

        # Maintain decreasing order: remove smaller elements from back
        # (they can never be the max while num is in the window)
        while dq and nums[dq[-1]] < num:
            dq.pop()

        dq.append(i)

        # Window is fully formed after k-1 iterations
        if i >= k - 1:
            result.append(nums[dq[0]])  # front is always the max

    return result
```

## 🔑 Key Tricks for Sliding Window

```
✅ Fixed window:  right - left = k-1 always. Update: add arr[right], remove arr[right-k].
✅ Variable window: shrink while INVALID. Update answer after shrink loop.
✅ Use deque for sliding window max/min (monotonic deque pattern).
✅ "At most K distinct" problems: use shrink-until-valid approach.
✅ Answer is usually: max(right - left + 1) or min(right - left + 1).
✅ Initialize: left = 0, window_state = {}, result = 0 or float('inf').
```

---

---

# 4. Prefix Sum

## 📌 What is it?

Precompute cumulative sums so that the sum of any subarray `[i, j]` can be computed in O(1) instead of O(n).

## 🧠 Intuition

Think of it like odometer readings on a car:
- `prefix[i]` = total distance traveled up to point `i`
- Distance from point `a` to `b` = `prefix[b] - prefix[a-1]`
- You don't re-drive the whole trip — you just subtract two readings.

```
arr    =  [3, 1, 4, 1, 5]
prefix =  [0, 3, 4, 8, 9, 14]    (prefix[0] = 0 is a sentinel)

Sum from index 1 to 3 = prefix[4] - prefix[1] = 9 - 3 = 6 ✅
(arr[1] + arr[2] + arr[3] = 1 + 4 + 1 = 6)
```

---

### 🔴 Problem: Subarray Sum Equals K

**Statement:** Given array `nums` and integer `k`, count the number of contiguous subarrays whose sum equals `k`.

**Example:**
```
Input:  nums = [1, 1, 1], k = 2
Output: 2   → [1,1] at index [0,1] and [1,1] at index [1,2]

Input:  nums = [1, 2, 3], k = 3
Output: 2   → [3] and [1,2]
```

**🧠 Think:**
```
If prefix_sum[j] - prefix_sum[i] = k
Then the subarray from i+1 to j sums to k.
Rearranged: prefix_sum[i] = prefix_sum[j] - k

So at each index j, we ask:
"How many previous prefix sums equal (current_prefix_sum - k)?"
Store prefix sums in a hashmap as we go.
```

**Why prefix[0] = 1 in the map?** Because if `prefix_sum[j] == k` itself, we need to count the subarray from 0 to j. That's like saying prefix[i] = 0 occurred once (before the array started).

```python
# 🐌 BRUTE FORCE — O(n²) — Check all subarrays
def subarray_sum_brute(nums, k):
    count = 0
    for i in range(len(nums)):
        total = 0
        for j in range(i, len(nums)):
            total += nums[j]
            if total == k:
                count += 1
    return count

# 🏆 OPTIMAL — Prefix Sum + HashMap — O(n) time, O(n) space
def subarray_sum_optimal(nums, k):
    # prefix_count[s] = how many times prefix sum 's' has appeared
    prefix_count = {0: 1}   # empty prefix has sum 0, seen once
    prefix_sum = 0
    count = 0

    for num in nums:
        prefix_sum += num
        # How many subarrays ending here sum to k?
        count += prefix_count.get(prefix_sum - k, 0)
        prefix_count[prefix_sum] = prefix_count.get(prefix_sum, 0) + 1

    return count
```

---

### 🔴 Problem: Product of Array Except Self

**Statement:** Return array `output` where `output[i]` is the product of all elements except `nums[i]`. **No division allowed.**

**Example:**
```
Input:  nums = [1, 2, 3, 4]
Output: [24, 12, 8, 6]

output[0] = 2*3*4 = 24
output[1] = 1*3*4 = 12
```

**🧠 Think:** For each position, the answer = (product of everything left of i) × (product of everything right of i). Build left prefix product array, then traverse right-to-left maintaining a running right product.

```python
# 🏆 OPTIMAL — Left and Right prefix products — O(n) time, O(1) extra space
def product_except_self(nums):
    n = len(nums)
    result = [1] * n

    # Pass 1: result[i] = product of all elements to the LEFT of i
    left_product = 1
    for i in range(n):
        result[i] = left_product
        left_product *= nums[i]

    # Pass 2: multiply each result[i] by product of all elements to the RIGHT
    right_product = 1
    for i in range(n - 1, -1, -1):
        result[i] *= right_product
        right_product *= nums[i]

    return result
```

---

---

# 5. Hashing & Frequency Maps

## 📌 What is it?

Hash maps store key-value pairs with **O(1) average** lookup, insert, and delete. The core idea: **trade memory for speed** by precomputing and storing intermediate results.

## 🧠 Intuition

The classic insight: instead of searching for a match (expensive), **store what you've seen** and check if the current element completes a valid pair/group/sum.

Pattern: "For each element X, ask: does Y exist in what I've already seen?"

## Python Tools

```python
from collections import Counter, defaultdict

# Counter — frequency map in one line
freq = Counter([1, 2, 2, 3, 3, 3])   # → {3:3, 2:2, 1:1}
freq.most_common(2)                   # → [(3,3), (2,2)]

# defaultdict — auto-initializes missing keys
graph = defaultdict(list)
graph['A'].append('B')               # No KeyError even if 'A' didn't exist

# Plain dict with .get()
d = {}
d[key] = d.get(key, 0) + 1          # Safe increment
```

---

### 🔴 Problem: Two Sum (Unsorted Array)

**Statement:** Find two numbers in an array that add up to `target`. Return their indices.

**Example:**
```
Input:  nums = [2, 7, 11, 15], target = 9
Output: [0, 1]   (nums[0] + nums[1] = 9)
```

**🧠 Think:** For each number `x`, we need `target - x`. Instead of scanning the whole array for it, store every number we've seen so far in a hashmap. When we see `x`, immediately check if its complement is already in the map.

```python
# 🐌 BRUTE FORCE — O(n²) — Check every pair
def two_sum_brute(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]

# 🏆 OPTIMAL — HashMap — O(n) time, O(n) space
def two_sum_optimal(nums, target):
    seen = {}   # value → index
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
        # Note: we add AFTER checking, so we don't use the same element twice
```

---

### 🔴 Problem: Longest Consecutive Sequence

**Statement:** Find the length of the longest consecutive sequence in an unsorted array. Must run in O(n).

**Example:**
```
Input:  nums = [100, 4, 200, 1, 3, 2]
Output: 4   → sequence [1, 2, 3, 4]
```

**🧠 Think:** Don't sort (that's O(n log n)). Instead, put everything in a set. For each number, only start counting a sequence if it's the **beginning** of a sequence (i.e., `num - 1` is NOT in the set). Then extend the sequence upward as far as possible.

This way each number is processed at most twice → O(n).

```python
# 🐌 BRUTE FORCE — O(n³) — For each num, check each consecutive sequence
# ⚡ OPTIMIZED — Sort + scan — O(n log n)
def longest_consecutive_sort(nums):
    if not nums: return 0
    nums = sorted(set(nums))
    best = curr = 1
    for i in range(1, len(nums)):
        if nums[i] == nums[i-1] + 1:
            curr += 1
            best = max(best, curr)
        else:
            curr = 1
    return best

# 🏆 OPTIMAL — HashSet — O(n) time, O(n) space
def longest_consecutive_optimal(nums):
    num_set = set(nums)
    best = 0

    for num in num_set:
        if num - 1 not in num_set:      # Only start from sequence beginning
            length = 1
            while num + length in num_set:
                length += 1
            best = max(best, length)

    return best
```

---

---

# 6. Strings

## 📌 Key Techniques

- **Palindrome:** Two pointers from center outward, or reverse comparison.
- **Anagram:** Sort both strings OR compare character frequency arrays.
- **Substring search:** Sliding window + hashmap.
- **Pattern matching:** KMP algorithm (rarely needed in OA, know conceptually).

## 🧠 Important String Facts in Python

```python
s = "hello"
s[::-1]              # Reverse → "olleh"  — O(n) time and space
ord('a')             # ASCII value → 97
chr(97)              # Character → 'a'
s.split()            # Split by whitespace
" ".join(["a","b"])  # Join list → "a b"
s.count('l')         # Count occurrences
```

---

### 🔴 Problem: Valid Palindrome

**Statement:** A string is a palindrome if it reads the same forwards and backwards, considering only alphanumeric characters and ignoring case. Return True/False.

**Example:**
```
Input:  s = "A man, a plan, a canal: Panama"
Output: True   (cleaned: "amanaplanacanalpanama")
```

```python
# 🏆 OPTIMAL — Two Pointers — O(n) time, O(1) space
def is_palindrome(s):
    left, right = 0, len(s) - 1
    while left < right:
        # Skip non-alphanumeric from left
        while left < right and not s[left].isalnum():
            left += 1
        # Skip non-alphanumeric from right
        while left < right and not s[right].isalnum():
            right -= 1
        if s[left].lower() != s[right].lower():
            return False
        left += 1
        right -= 1
    return True
```

---

### 🔴 Problem: Longest Palindromic Substring

**Statement:** Return the longest palindromic substring in `s`.

**Example:**
```
Input:  s = "babad"
Output: "bab" (or "aba")

Input:  s = "cbbd"
Output: "bb"
```

**🧠 Think:** Every palindrome "expands" from its center. There are 2n-1 possible centers (n single chars + n-1 between chars). For each center, expand outward while characters match.

```python
# 🐌 BRUTE FORCE — O(n³) — Check all substrings
def longest_palindrome_brute(s):
    def is_palindrome(sub): return sub == sub[::-1]
    best = ""
    for i in range(len(s)):
        for j in range(i, len(s)):
            if is_palindrome(s[i:j+1]) and j-i+1 > len(best):
                best = s[i:j+1]
    return best

# 🏆 OPTIMAL — Expand Around Center — O(n²) time, O(1) space
def longest_palindrome_optimal(s):
    result = ""

    def expand(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        # After loop: s[left+1 : right] is the palindrome
        return s[left + 1: right]

    for i in range(len(s)):
        odd = expand(i, i)          # Odd-length: single center char
        even = expand(i, i + 1)     # Even-length: center between two chars
        if len(odd) > len(result):  result = odd
        if len(even) > len(result): result = even

    return result
```

---

### 🔴 Problem: Group Anagrams

**Statement:** Group strings that are anagrams of each other.

**Example:**
```
Input:  strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"], ["nat","tan"], ["ate","eat","tea"]]
```

**🧠 Think:** Two strings are anagrams iff they have the same **character frequency**. We need a hashable key representing the frequency. Two options:
1. Sort the string → `"eat" → "aet"` (canonical form). O(L log L) per string.
2. Use a tuple of 26 counts → O(L) per string.

```python
from collections import defaultdict

# ⚡ OPTION 1 — Sort as key — O(n * L log L)
def group_anagrams_sort(strs):
    groups = defaultdict(list)
    for s in strs:
        key = tuple(sorted(s))
        groups[key].append(s)
    return list(groups.values())

# 🏆 OPTION 2 — Count array as key — O(n * L)
def group_anagrams_count(strs):
    groups = defaultdict(list)
    for s in strs:
        count = [0] * 26
        for c in s:
            count[ord(c) - ord('a')] += 1
        groups[tuple(count)].append(s)
    return list(groups.values())
```

---

---

# 7. Linked Lists

## 📌 What is it?

A sequence of nodes where each node has a `val` and a `next` pointer. Unlike arrays, nodes are **not contiguous in memory** — no index access, must traverse.

## 🧠 Key Patterns

1. **Dummy node** — Add a dummy head before the real head. Simplifies edge cases (inserting at head, deleting head).
2. **Fast & Slow pointers** — Fast moves 2x speed. When fast reaches end, slow is at middle. Used for: middle of list, cycle detection, kth from end.
3. **Reverse in-place** — Use three pointers: prev, curr, next_node.

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# Build a linked list from a Python list (useful for testing)
def build_list(values):
    dummy = ListNode(0)
    curr = dummy
    for v in values:
        curr.next = ListNode(v)
        curr = curr.next
    return dummy.next

# Convert linked list to Python list (useful for verifying)
def to_list(head):
    result = []
    while head:
        result.append(head.val)
        head = head.next
    return result
```

---

### 🔴 Problem: Reverse a Linked List

**Statement:** Reverse a singly linked list and return the new head.

**Example:**
```
Input:  1 → 2 → 3 → 4 → 5 → None
Output: 5 → 4 → 3 → 2 → 1 → None
```

**🧠 Think:** Walk through the list. Before moving forward, redirect the current node's `next` to point backwards. You need 3 pointers: `prev` (what we've reversed so far), `curr` (current node), `next_node` (save before we overwrite curr.next).

```
Step by step with [1 → 2 → 3]:
prev=None  curr=1  → redirect 1.next = None  → prev=1, curr=2
prev=1     curr=2  → redirect 2.next = 1     → prev=2, curr=3
prev=2     curr=3  → redirect 3.next = 2     → prev=3, curr=None
Return prev (3) → 3 → 2 → 1 → None ✅
```

```python
# 🏆 ITERATIVE — O(n) time, O(1) space
def reverse_list(head):
    prev = None
    curr = head
    while curr:
        next_node = curr.next   # Save next before overwriting
        curr.next = prev        # Reverse the pointer
        prev = curr             # Move prev forward
        curr = next_node        # Move curr forward
    return prev   # prev is the new head

# RECURSIVE — O(n) time, O(n) space (call stack)
def reverse_list_recursive(head):
    if not head or not head.next:
        return head                         # Base: single node is already reversed
    new_head = reverse_list_recursive(head.next)  # Reverse the rest
    head.next.next = head    # Make next node point back to current
    head.next = None         # Cut current node's forward pointer
    return new_head
```

---

### 🔴 Problem: Linked List Cycle Detection

**Statement:** Detect if a linked list has a cycle. If yes, return the node where the cycle begins; otherwise return None.

**Example:**
```
Input: 3 → 2 → 0 → -4 → (back to 2)
Output: Node with value 2 (cycle starts here)
```

**🧠 Think (Floyd's Algorithm):**

**Phase 1 — Detect:** Move `slow` 1 step, `fast` 2 steps. If they meet → cycle exists.

**Phase 2 — Find start:** Reset `slow` to head. Move both `slow` and `fast` one step at a time. They'll meet exactly at the cycle start.

*Why does Phase 2 work?* When they first meet inside the cycle, the distance from that meeting point to the cycle start equals the distance from the head to the cycle start. (Mathematical proof with modular arithmetic — just trust and memorize the pattern.)

```python
# 🐌 BRUTE — HashSet — O(n) time, O(n) space
def detect_cycle_set(head):
    seen = set()
    curr = head
    while curr:
        if id(curr) in seen:
            return curr
        seen.add(id(curr))
        curr = curr.next
    return None

# 🏆 OPTIMAL — Floyd's Algorithm — O(n) time, O(1) space
def detect_cycle_optimal(head):
    slow = fast = head

    # Phase 1: Detect cycle
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    else:
        return None   # No cycle (fast reached end)

    # Phase 2: Find cycle start
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next
    return slow   # Both point to cycle start
```

---

### 🔴 Problem: Merge Two Sorted Lists

**Statement:** Merge two sorted linked lists and return as one sorted list.

**Example:**
```
Input:  l1 = 1 → 2 → 4,   l2 = 1 → 3 → 4
Output:       1 → 1 → 2 → 3 → 4 → 4
```

**🧠 Think:** Use a **dummy node** to avoid special-casing the head. Compare the heads of both lists, attach the smaller one, advance that pointer. Attach remaining list at the end.

```python
# 🏆 ITERATIVE — O(m+n) time, O(1) space
def merge_two_lists(l1, l2):
    dummy = ListNode(0)     # Dummy head simplifies logic
    curr = dummy

    while l1 and l2:
        if l1.val <= l2.val:
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next

    curr.next = l1 or l2   # Attach the remaining non-empty list
    return dummy.next
```

## 🔑 Key Tricks for Linked Lists

```
✅ Always use a dummy head node to simplify insertion/deletion at head
✅ Fast & Slow: when fast reaches end, slow is at middle
✅ Find kth node from end: move fast k steps ahead, then move both
✅ Never lose your reference: save next_node BEFORE modifying curr.next
✅ Cycle: if slow == fast → cycle detected. Reset slow to head, move 1 step each → cycle start
✅ Draw it out! Linked list bugs are almost always pointer errors
```

---

---

# 8. Stacks & Monotonic Stack

## 📌 What is it?

**Stack:** Last In, First Out (LIFO). Push/pop from the top. O(1) operations.

**Monotonic Stack:** A stack that maintains elements in **strictly increasing or decreasing order**. When you add a new element, pop all elements that violate the order first. Powerful for "next greater/smaller element" type problems.

## 🧠 Intuition for Monotonic Stack

Imagine you're standing in a line looking for the next taller person ahead of you. You scan left-to-right. Every time you see someone taller, they become the "next greater" for everyone currently waiting in your stack who is shorter than them.

**Pattern:** Process each element. While stack is not empty AND stack top is less than current element → the current element is the "next greater" for stack top. Pop and record.

---

### 🔴 Problem: Daily Temperatures

**Statement:** Given daily temperatures, for each day return the number of days until a warmer temperature. If no warmer day exists, return 0.

**Example:**
```
Input:  temps = [73, 74, 75, 71, 69, 72, 76, 73]
Output:         [ 1,  1,  4,  2,  1,  1,  0,  0]

Day 0 (73°): Next warmer is day 1 (74°) → 1 day
Day 2 (75°): Next warmer is day 6 (76°) → 4 days
```

**🧠 Think:** Use a stack of indices (not values). For each temperature, pop all days from the stack that are cooler — the current day is their "next warmer day." If no warmer day found before array ends → 0 (already initialized).

```python
# 🐌 BRUTE FORCE — O(n²) — For each day, scan forward until warmer found
def daily_temps_brute(temps):
    result = [0] * len(temps)
    for i in range(len(temps)):
        for j in range(i + 1, len(temps)):
            if temps[j] > temps[i]:
                result[i] = j - i
                break
    return result

# 🏆 OPTIMAL — Monotonic Decreasing Stack — O(n) time, O(n) space
def daily_temps_optimal(temps):
    result = [0] * len(temps)
    stack = []   # stores indices; temps[stack] is decreasing

    for i, temp in enumerate(temps):
        # Current temp is warmer than stack top → resolve those days
        while stack and temps[stack[-1]] < temp:
            idx = stack.pop()
            result[idx] = i - idx    # days to wait = current index - that day's index
        stack.append(i)

    return result   # Unresolved indices stay 0 (already initialized)
```

---

### 🔴 Problem: Largest Rectangle in Histogram

**Statement:** Given an array of bar heights, find the area of the largest rectangle that fits inside the histogram.

**Example:**
```
Input:  heights = [2,1,5,6,2,3]
Output: 10

  ##
  ##
  ####   ← largest rectangle area = 5*2 = 10 (bars 2,3 with height 5)
#######
```

**🧠 Think:** For each bar, find how far left and right it can extend while maintaining its height. That gives width. This is the "previous smaller element" + "next smaller element" problem.

Monotonic increasing stack: when we encounter a bar shorter than the top of the stack, the top bar can't extend right anymore. Calculate its max area and pop it.

```python
# 🐌 BRUTE FORCE — O(n²) — For each pair (i,j), find min height
def largest_rect_brute(heights):
    max_area = 0
    for i in range(len(heights)):
        min_h = heights[i]
        for j in range(i, len(heights)):
            min_h = min(min_h, heights[j])
            max_area = max(max_area, min_h * (j - i + 1))
    return max_area

# 🏆 OPTIMAL — Monotonic Increasing Stack — O(n) time, O(n) space
def largest_rect_optimal(heights):
    stack = []       # stores indices, heights are increasing
    max_area = 0
    heights = heights + [0]   # Append 0 as sentinel to flush remaining bars

    for i, h in enumerate(heights):
        start = i
        while stack and heights[stack[-1]] > h:
            idx = stack.pop()
            # Width: from current index to the new stack top (exclusive)
            width = i - (stack[-1] + 1 if stack else 0)
            max_area = max(max_area, heights[idx] * width)
            start = idx   # This bar could extend back to idx
        stack.append(start)

    return max_area
```

---

### 🔴 Problem: LRU Cache

**Statement:** Design a data structure for Least Recently Used cache. Support `get(key)` and `put(key, value)` in O(1). When capacity is full, evict the least recently used item.

**🧠 Think:** We need:
- O(1) lookup → HashMap
- O(1) eviction of LRU + promotion on access → Doubly Linked List (head=LRU, tail=MRU)

Together: **HashMap + Doubly Linked List** = the classic LRU design.

In Python, `OrderedDict` gives us this out of the box.

```python
from collections import OrderedDict

# 🏆 PYTHON — Using OrderedDict (interview acceptable, mention underlying idea)
class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)    # Mark as most recently used
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)   # Remove least recently used (leftmost)
```

---

---

# 9. Binary Search

## 📌 What is it?

A technique to find a target in a **sorted** (or monotonic) search space by repeatedly halving the search space. Each comparison eliminates half the remaining candidates.

## 🧠 Intuition

Think of a phone book. To find "Smith", you don't read every name. You open to the middle — if "Smith" comes after, ignore the left half. Repeat. That's binary search.

**The deeper pattern:** Binary search works on ANY problem where the solution space is **monotonic** — meaning if value X satisfies a condition, all values > X also satisfy it (or all < X do). This opens up "binary search on the answer" — a powerful technique.

## ⚠️ The Tricky Part: Boundary Conditions

```
Two templates — memorize one and stick to it:

Template 1: left <= right (for finding exact target)
  - Exits when left > right
  - Use when you know the element exists or want -1

Template 2: left < right (for finding boundary/minimum)
  - Exits when left == right (converged to answer)
  - Use when looking for first True / last False
```

```python
# Template 1 — Find exact target
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2   # ← ALWAYS use this, not (l+r)//2
        if arr[mid] == target: return mid
        elif arr[mid] < target: left = mid + 1
        else: right = mid - 1
    return -1

# Template 2 — Find leftmost position where condition is True
def binary_search_boundary(arr, condition):
    left, right = 0, len(arr) - 1
    result = -1
    while left <= right:
        mid = (left + right) // 2
        if condition(arr[mid]):
            result = mid
            right = mid - 1   # keep looking left for earlier True
        else:
            left = mid + 1
    return result
```

---

### 🔴 Problem: Search in Rotated Sorted Array

**Statement:** A sorted array was rotated at some unknown pivot. Find the target in O(log n).

**Example:**
```
Input:  nums = [4,5,6,7,0,1,2], target = 0
Output: 4   (index of 0)

Input:  nums = [4,5,6,7,0,1,2], target = 3
Output: -1
```

**🧠 Think:** Even after rotation, one half is always sorted. Check which half is sorted, then determine if the target falls in that half. If yes → search there. If no → search the other half.

```python
# 🏆 OPTIMAL — Modified Binary Search — O(log n) time
def search_rotated(nums, target):
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = (left + right) // 2

        if nums[mid] == target:
            return mid

        # Left half is sorted (no rotation in this half)
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1     # Target in sorted left half
            else:
                left = mid + 1      # Target in right half
        # Right half is sorted
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1      # Target in sorted right half
            else:
                right = mid - 1     # Target in left half

    return -1
```

---

### 🔴 Problem: Koko Eating Bananas (Binary Search on Answer)

**Statement:** Koko has `piles` of bananas and `h` hours. She eats at speed `k` bananas/hour (one pile per hour). Find the minimum `k` such that she can finish all bananas in `h` hours.

**Example:**
```
Input:  piles = [3,6,7,11], h = 8
Output: 4

At speed 4: ceil(3/4)+ceil(6/4)+ceil(7/4)+ceil(11/4) = 1+2+2+3 = 8 hours ✅
At speed 3: 1+2+3+4 = 10 hours ❌
```

**🧠 Think:** Instead of searching through piles, **binary search on the answer** (the speed k).
- Range of k: [1, max(piles)]
- Condition: `can_finish(k)` → total hours ≤ h
- Find minimum k where condition is true → binary search!

```python
import math

# 🏆 OPTIMAL — Binary Search on Answer — O(n log m) where m = max(piles)
def min_eating_speed(piles, h):
    def can_finish(speed):
        # Total hours needed at this speed
        return sum(math.ceil(p / speed) for p in piles) <= h

    left, right = 1, max(piles)
    while left < right:
        mid = (left + right) // 2
        if can_finish(mid):
            right = mid        # This speed works, try slower
        else:
            left = mid + 1     # Too slow, try faster
    return left
```

**Other "binary search on answer" examples:** Minimum days to make bouquets, Capacity to ship packages, Cutting wood to height H.

## 🔑 Key Tricks for Binary Search

```
✅ Always use mid = left + (right - left) // 2 to avoid integer overflow
✅ When finding minimum valid value: right = mid (not mid-1), exit when left == right
✅ When finding maximum valid value: left = mid+1, track best separately
✅ Rotated array: one half is ALWAYS sorted — determine which, then decide where to go
✅ "Binary search on answer": if you can verify in O(n), the answer range is binary searchable
✅ Think: is the condition function MONOTONIC? (once true, always true for larger/smaller values?)
```

---

---

# 10. Trees (DFS & BFS)

## 📌 What is it?

**Tree:** Hierarchical data structure. Each node has at most one parent (except root) and zero or more children.

**Binary Tree:** Each node has at most 2 children (left, right).

**BST:** Binary Search Tree — left child < node < right child. Inorder traversal gives sorted order.

## 🧠 DFS vs BFS — When to use which?

```
DFS (Depth First Search):
  → Goes deep before going wide
  → Uses: recursion (call stack) or explicit stack
  → Best for: path problems, tree shape, ancestor problems, all-paths
  → Space: O(h) where h = height

BFS (Breadth First Search):
  → Explores level by level
  → Uses: queue (deque)
  → Best for: shortest path, level-by-level problems, "closest" problems
  → Space: O(w) where w = max width (can be O(n) for wide trees)
```

## Tree Traversals

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val; self.left = left; self.right = right

# INORDER (Left → Root → Right) → gives sorted values for BST
def inorder(root):
    return inorder(root.left) + [root.val] + inorder(root.right) if root else []

# PREORDER (Root → Left → Right) → used for tree serialization/cloning
def preorder(root):
    return [root.val] + preorder(root.left) + preorder(root.right) if root else []

# POSTORDER (Left → Right → Root) → used when processing children before parent
def postorder(root):
    return postorder(root.left) + postorder(root.right) + [root.val] if root else []

# LEVEL ORDER (BFS) → left to right, level by level
from collections import deque
def level_order(root):
    if not root: return []
    result, queue = [], deque([root])
    while queue:
        level = []
        for _ in range(len(queue)):      # ← Snapshot level size!
            node = queue.popleft()
            level.append(node.val)
            if node.left:  queue.append(node.left)
            if node.right: queue.append(node.right)
        result.append(level)
    return result
```

---

### 🔴 Problem: Lowest Common Ancestor (LCA)

**Statement:** Given a binary tree and two nodes `p` and `q`, find their lowest common ancestor (the deepest node that has both as descendants, or is one of them).

**Example:**
```
        3
       / \
      5   1
     / \ / \
    6  2 0  8
      / \
     7   4

LCA(5, 1) = 3
LCA(5, 4) = 5  ← 5 is ancestor of 4, so 5 itself is LCA
```

**🧠 Think:** Use recursion. At each node, recurse into left and right subtrees.
- If current node is null, p, or q → return it
- If both left and right returned non-null → current node is the LCA (p and q are in different subtrees)
- If only one side returned non-null → that side has both p and q (or just one of them)

```python
# 🏆 OPTIMAL — Single DFS Pass — O(n) time, O(h) space
def lca(root, p, q):
    if not root or root == p or root == q:
        return root       # Found p, q, or reached null

    left = lca(root.left, p, q)
    right = lca(root.right, p, q)

    if left and right:
        return root       # p is in left subtree, q is in right (or vice versa)
    return left or right  # Both are in one subtree
```

---

### 🔴 Problem: Binary Tree Maximum Path Sum

**Statement:** A path is any sequence of nodes from parent to child (not necessarily through root). Find the maximum sum path.

**Example:**
```
      -10
      /  \
     9   20
        /  \
       15   7

Answer: 42 (path: 15 → 20 → 7)
```

**🧠 Think:** At each node, compute the maximum "contribution" it can give to its parent = `node.val + max(left_contribution, right_contribution, 0)`. But also update the global max considering going through both children: `node.val + left_contribution + right_contribution`.

Key: **don't pass negative contributions upward** (use `max(0, contribution)`).

```python
# 🏆 OPTIMAL — DFS with global state — O(n) time, O(h) space
def max_path_sum(root):
    max_sum = [float('-inf')]   # List to allow mutation inside nested function

    def max_gain(node):
        if not node:
            return 0
        # Ignore negative paths (take 0 instead)
        left_gain  = max(0, max_gain(node.left))
        right_gain = max(0, max_gain(node.right))

        # Path that goes through current node (using both children)
        path_sum = node.val + left_gain + right_gain
        max_sum[0] = max(max_sum[0], path_sum)

        # Return max single-branch contribution to parent
        return node.val + max(left_gain, right_gain)

    max_gain(root)
    return max_sum[0]
```

---

### 🔴 Problem: Serialize and Deserialize Binary Tree

**Statement:** Design an algorithm to serialize a binary tree to a string and deserialize back. (Amazon favorite!)

**🧠 Think:** Use preorder DFS. Record `"null"` for missing nodes so we can reconstruct exactly. During deserialize, consume values from the sequence and recursively rebuild.

```python
class Codec:
    def serialize(self, root):
        # Preorder DFS → "1,2,null,null,3,null,null"
        def dfs(node):
            if not node:
                return "null,"
            return str(node.val) + "," + dfs(node.left) + dfs(node.right)
        return dfs(root)

    def deserialize(self, data):
        values = iter(data.split(","))

        def dfs():
            val = next(values)
            if val == "null":
                return None
            node = TreeNode(int(val))
            node.left  = dfs()
            node.right = dfs()
            return node

        return dfs()
```

---

---

# 11. Heaps / Priority Queues

## 📌 What is it?

A heap is a **complete binary tree** that satisfies the heap property:
- **Min-Heap:** Parent ≤ children. Root is always the minimum.
- **Max-Heap:** Parent ≥ children. Root is always the maximum.

Python's `heapq` is a **min-heap**. For max-heap: negate all values.

## 🧠 Intuition

Think of a heap as a "partially sorted" structure. You don't know the exact order of all elements — but you always know where the smallest (or largest) is: at the top. Insertion and deletion maintain this invariant in O(log n).

**When to use heaps:**
- "Top K" problems (smallest/largest K elements)
- Merge K sorted arrays/lists
- Running median (two heaps)
- Anything needing "efficiently get the best element repeatedly"

```python
import heapq

# Min-Heap operations
heap = []
heapq.heappush(heap, 5)      # Push — O(log n)
heapq.heappop(heap)          # Pop min — O(log n)
heap[0]                      # Peek min — O(1)
heapq.heapify(arr)           # Build heap from list — O(n)

# Max-Heap: negate values
heapq.heappush(heap, -5)     # Store -5 for value 5
-heapq.heappop(heap)         # Negate on retrieval

# Push with priority tuple — Python compares tuples lexicographically
heapq.heappush(heap, (priority, value))
```

---

### 🔴 Problem: Kth Largest Element in Array

**Statement:** Find the kth largest element in an unsorted array.

**Example:**
```
Input:  nums = [3,2,1,5,6,4], k = 2
Output: 5   (2nd largest)
```

**🧠 Think:** 
- Sort approach: O(n log n) — valid but not optimal
- Min-Heap of size k: maintain a heap of the k largest elements seen so far. The smallest among them (heap root) is the kth largest overall. When heap grows beyond k, pop the minimum (evict the "weakest" candidate).

```python
# 🐌 BRUTE — Sort — O(n log n)
def kth_largest_brute(nums, k):
    nums.sort(reverse=True)
    return nums[k-1]

# 🏆 OPTIMAL — Min-Heap of size k — O(n log k) time, O(k) space
def kth_largest_optimal(nums, k):
    heap = []
    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)    # Evict smallest — we only want top k
    return heap[0]   # Smallest of top-k = kth largest
```

---

### 🔴 Problem: Find Median from Data Stream

**Statement:** Design a data structure that supports: `addNum(num)` and `findMedian()` — both efficiently.

**Example:**
```
addNum(1) → median = 1
addNum(2) → median = 1.5
addNum(3) → median = 2
```

**🧠 Think:** Split numbers into two halves:
- **Lower half** in a max-heap (we want the max of lower half quickly)
- **Upper half** in a min-heap (we want the min of upper half quickly)

Median = average of both tops (even count) or top of larger half (odd count).

**Invariant to maintain:** `max_heap.size == min_heap.size` OR `max_heap.size == min_heap.size + 1`

```
 Lower half       Upper half
(max-heap)        (min-heap)
[1, 2, 3]         [4, 5, 6]
      ↑ max=3     ↑ min=4
      Median = (3 + 4) / 2 = 3.5
```

```python
class MedianFinder:
    def __init__(self):
        self.max_heap = []   # Lower half — negate for max-heap
        self.min_heap = []   # Upper half — natural min-heap

    def addNum(self, num):
        # Step 1: Always push to max_heap first
        heapq.heappush(self.max_heap, -num)

        # Step 2: Ensure max_heap's max <= min_heap's min
        if self.min_heap and -self.max_heap[0] > self.min_heap[0]:
            heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))

        # Step 3: Balance sizes (max_heap can have at most 1 extra)
        if len(self.max_heap) > len(self.min_heap) + 1:
            heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))
        elif len(self.min_heap) > len(self.max_heap):
            heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))

    def findMedian(self):
        if len(self.max_heap) > len(self.min_heap):
            return float(-self.max_heap[0])
        return (-self.max_heap[0] + self.min_heap[0]) / 2.0
```

---

---

# 12. Graphs

## 📌 Key Concepts

```
Vertex (Node): A point in the graph
Edge: Connection between two vertices
Directed: Edges have direction (A→B doesn't mean B→A)
Undirected: Edges are bidirectional
Weighted: Edges have costs/distances
Unweighted: All edges equal
Cycle: Path that starts and ends at same vertex
Connected: All vertices reachable from any vertex (undirected)
```

## 🧠 BFS vs DFS for Graphs

```
BFS → Shortest path in unweighted graph (guaranteed optimal)
DFS → Detect cycles, topological sort, all paths, connected components
Dijkstra → Shortest path in weighted graph (use min-heap + BFS-like)
Union-Find → Dynamic connectivity, MST (Kruskal's)
```

```python
from collections import defaultdict, deque

def build_graph(n, edges, directed=False):
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        if not directed:
            graph[v].append(u)
    return graph
```

---

### 🔴 Problem: Number of Islands

**Statement:** Given a 2D grid of '1's (land) and '0's (water), count the number of islands (connected groups of '1's). Connectivity is horizontal and vertical.

**Example:**
```
Input:
11110
11010
11000
00000

Output: 1  (all land connected)

Input:
11000
11000
00100
00011

Output: 3
```

**🧠 Think:** Treat grid as a graph. Each '1' cell is a node connected to its 4 neighbors. Run DFS from every unvisited '1', marking all reachable land as visited ('0'). Count how many DFS calls you make — that's the island count.

This technique is called **flood fill**.

```python
# 🏆 OPTIMAL — DFS Flood Fill — O(m*n) time, O(m*n) space
def num_islands(grid):
    if not grid: return 0
    rows, cols = len(grid), len(grid[0])
    count = 0

    def dfs(r, c):
        # Out of bounds or water → stop
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] != '1':
            return
        grid[r][c] = '0'    # Sink the land (mark visited by modifying in-place)
        dfs(r+1, c); dfs(r-1, c)   # Down, Up
        dfs(r, c+1); dfs(r, c-1)   # Right, Left

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                dfs(r, c)
                count += 1   # Each DFS call = one island

    return count
```

---

### 🔴 Problem: Course Schedule (Cycle Detection / Topological Sort)

**Statement:** There are `n` courses. `prerequisites[i] = [a, b]` means you must take course `b` before course `a`. Can you finish all courses?

**Example:**
```
Input:  n=2, prerequisites=[[1,0]]
Output: True   (take 0 then 1)

Input:  n=2, prerequisites=[[1,0],[0,1]]
Output: False  (circular dependency: 0 needs 1, 1 needs 0)
```

**🧠 Think:** Model as a directed graph. Course `b → a` means "b must come before a". A valid course schedule exists **iff the graph has no cycle**. Use DFS with 3 states:
- `0` = unvisited
- `1` = currently being visited (in current DFS path)
- `2` = fully processed (no cycle through it)

If we reach a node with state `1` during DFS → cycle detected!

```python
# 🏆 OPTIMAL — DFS with 3-state coloring — O(V + E)
def can_finish(numCourses, prerequisites):
    graph = defaultdict(list)
    for course, prereq in prerequisites:
        graph[prereq].append(course)    # prereq → course

    state = [0] * numCourses  # 0=unvisited, 1=visiting, 2=done

    def has_cycle(node):
        if state[node] == 1: return True   # Back edge! Cycle.
        if state[node] == 2: return False  # Already verified clean

        state[node] = 1   # Mark as currently visiting
        for neighbor in graph[node]:
            if has_cycle(neighbor):
                return True
        state[node] = 2   # Mark as fully processed
        return False

    return not any(has_cycle(i) for i in range(numCourses))
```

---

### 🔴 Problem: Word Ladder (BFS Shortest Path)

**Statement:** Transform `beginWord` to `endWord` by changing one letter at a time. Each intermediate word must be in the given word list. Find the minimum number of transformations.

**Example:**
```
Input:  beginWord="hit", endWord="cog", wordList=["hot","dot","dog","lot","log","cog"]
Output: 5  → hit → hot → dot → dog → cog
```

**🧠 Think:** BFS gives shortest path in unweighted graphs. Each node is a word. Two words are connected if they differ by exactly one letter. BFS from `beginWord` to `endWord` — the level at which we reach `endWord` is the answer.

**Optimization:** Instead of comparing each word to all words (O(n²)), generate all 26 possible single-char substitutions for each position and check if they're in the word set. O(26 * L) per word.

```python
from collections import deque

# 🏆 OPTIMAL — BFS — O(M² × N) where M=word length, N=wordList size
def ladder_length(beginWord, endWord, wordList):
    word_set = set(wordList)
    if endWord not in word_set:
        return 0

    queue = deque([(beginWord, 1)])   # (current_word, steps)
    visited = {beginWord}

    while queue:
        word, steps = queue.popleft()
        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                new_word = word[:i] + c + word[i+1:]
                if new_word == endWord:
                    return steps + 1
                if new_word in word_set and new_word not in visited:
                    visited.add(new_word)
                    queue.append((new_word, steps + 1))
    return 0
```

---

### Union-Find (Disjoint Set Union)

**Use when:** Checking if two elements are in the same connected component. Supports dynamic connections. Faster than BFS/DFS for repeated connectivity queries.

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))   # Initially each node is its own parent
        self.rank = [0] * n            # Rank for union by rank optimization
        self.components = n

    def find(self, x):
        # Path compression: make every node point directly to root
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False   # Already in same component
        # Union by rank: attach smaller tree under larger tree
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        self.components -= 1
        return True

    def connected(self, x, y):
        return self.find(x) == self.find(y)
```

---

---

# 13. Dynamic Programming

## 📌 What is it?

DP solves problems by breaking them into **overlapping subproblems** and storing results to avoid recomputation (memoization). Unlike divide & conquer (which solves *independent* subproblems), DP subproblems **share** work.

## 🧠 The DP Mindset — 5 Steps

```
Step 1: DEFINE STATE
  What does dp[i] represent? Be very precise.
  "dp[i] = maximum sum of subarray ending at index i"

Step 2: RECURRENCE RELATION
  How does dp[i] depend on smaller subproblems?
  dp[i] = max(dp[i-1] + nums[i], nums[i])

Step 3: BASE CASE
  What are the smallest valid inputs?
  dp[0] = nums[0]

Step 4: ORDER
  Which direction do we fill the table?
  Usually left-to-right, or i increasing.

Step 5: ANSWER
  Where in dp is the final answer?
  max(dp[i] for all i)
```

## Top-Down (Memoization) vs Bottom-Up (Tabulation)

```python
# Top-Down: Natural recursion + cache
from functools import lru_cache

@lru_cache(maxsize=None)
def dp(i, j, ...):
    # base case
    # recursive calls with memo
    pass

# Bottom-Up: Fill dp table iteratively
dp = [initial] * (n + 1)
for i in range(1, n + 1):
    dp[i] = f(dp[i-1], dp[i-2], ...)
```

---

### 🔴 Problem: Maximum Subarray (Kadane's Algorithm)

**Statement:** Find the contiguous subarray with the largest sum.

**Example:**
```
Input:  nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6   → subarray [4,-1,2,1]
```

**🧠 Think:** At each position, we decide: start a new subarray here, or extend the previous one? If previous sum is negative, it only hurts us → start fresh.

`dp[i]` = max sum of subarray **ending at** index i.
`dp[i] = max(nums[i], dp[i-1] + nums[i])`

```python
# 🐌 BRUTE FORCE — O(n²) — Try all subarrays
def max_subarray_brute(nums):
    max_sum = float('-inf')
    for i in range(len(nums)):
        curr = 0
        for j in range(i, len(nums)):
            curr += nums[j]
            max_sum = max(max_sum, curr)
    return max_sum

# 🏆 OPTIMAL — Kadane's Algorithm — O(n) time, O(1) space
def max_subarray_kadane(nums):
    max_sum = curr_sum = nums[0]
    for num in nums[1:]:
        curr_sum = max(num, curr_sum + num)   # Start fresh OR extend
        max_sum = max(max_sum, curr_sum)
    return max_sum
```

---

### 🔴 Problem: Coin Change

**Statement:** Given coin denominations and a target amount, find the minimum number of coins to make that amount. Return -1 if impossible.

**Example:**
```
Input:  coins = [1,5,11], amount = 15
Output: 3   → 5+5+5 (NOT 11+1+1+1+1 = 5 coins, greedy fails here!)

Note: Greedy (always pick largest coin) FAILS for this problem.
```

**🧠 Think:**
- `dp[i]` = minimum coins to make amount `i`
- To make amount `i`: try each coin `c`. If we use coin `c`, we need `dp[i - c]` coins for the rest + 1 for this coin.
- `dp[i] = min(dp[i - c] + 1 for each coin c where c <= i)`
- Base: `dp[0] = 0` (0 coins to make amount 0)

```python
# 🏆 OPTIMAL — Bottom-up DP — O(amount × n) time, O(amount) space
def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0   # 0 coins needed to make amount 0

    for coin in coins:
        for amt in range(coin, amount + 1):
            dp[amt] = min(dp[amt], dp[amt - coin] + 1)

    return dp[amount] if dp[amount] != float('inf') else -1

# Walk-through with coins=[1,5,11], amount=6:
# dp[0]=0, dp[1]=1(1), dp[2]=2(1+1), dp[3]=3, dp[4]=4, dp[5]=1(5), dp[6]=2(5+1)
```

---

### 🔴 Problem: Longest Common Subsequence (LCS)

**Statement:** Find the length of the longest subsequence common to two strings. (Subsequence = characters in order, not necessarily contiguous.)

**Example:**
```
Input:  text1 = "abcde", text2 = "ace"
Output: 3   → "ace" is a common subsequence

Input:  text1 = "abc", text2 = "abc"
Output: 3
```

**🧠 Think:**
- `dp[i][j]` = LCS of `text1[:i]` and `text2[:j]`
- If `text1[i-1] == text2[j-1]`: characters match! `dp[i][j] = dp[i-1][j-1] + 1`
- Else: skip one character from either string: `dp[i][j] = max(dp[i-1][j], dp[i][j-1])`

```
    ""  a  c  e
""   0  0  0  0
a    0  1  1  1
b    0  1  1  1
c    0  1  2  2
d    0  1  2  2
e    0  1  2  3  ← Answer
```

```python
# 🏆 OPTIMAL — 2D DP — O(m*n) time, O(m*n) space
def lcs(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1   # Characters match
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])  # Take best from skipping one

    return dp[m][n]
```

---

### 🔴 Problem: 0/1 Knapsack

**Statement:** Given items with weights and values, and a knapsack of capacity W, choose items (can't repeat) to maximize value without exceeding W.

**Example:**
```
weights = [1, 3, 4, 5], values = [1, 4, 5, 7], W = 7
Answer: 9  → items with weights 3 and 4 (values 4+5=9) or weight 1 and 3 (1+4=5)... 
Actually: weights 3+4=7, values 4+5=9 ✅
```

**🧠 Think:**
- `dp[i][w]` = max value using first `i` items with capacity `w`
- For each item: either **skip** it → `dp[i-1][w]`, or **take** it → `dp[i-1][w - weight[i]] + value[i]`
- Take the max of both options.

**Space optimization:** Since `dp[i]` only depends on `dp[i-1]`, compress to 1D array. Traverse capacity **right to left** to avoid using the same item twice.

```python
# 🏆 SPACE OPTIMIZED — O(n*W) time, O(W) space
def knapsack(weights, values, W):
    dp = [0] * (W + 1)

    for weight, value in zip(weights, values):
        # RIGHT TO LEFT prevents using same item twice!
        for capacity in range(W, weight - 1, -1):
            dp[capacity] = max(dp[capacity],               # Skip item
                               dp[capacity - weight] + value)  # Take item
    return dp[W]
```

## 🔑 Key Tricks for DP

```
✅ "Minimum/maximum result" → usually DP
✅ "Count number of ways" → usually DP (add instead of min/max)
✅ If you can define state, recurrence, and base case → code writes itself
✅ When you see exponential brute force → think DP or memoization
✅ Knapsack variants: 0/1 (each item once, right-to-left), Unbounded (any times, left-to-right)
✅ dp[0] = 0 is often the seed/base case (empty set/empty string/amount 0)
✅ Space optimization: 2D → 1D when dp[i] only uses dp[i-1]
✅ Top-down with @lru_cache is easiest to code in interviews
```

---

---

# 14. Recursion & Backtracking

## 📌 What is it?

Backtracking is **systematic brute force**. Try every possible choice, and if a choice leads to a dead end, **undo it (backtrack)** and try the next option.

## 🧠 Intuition

Think of a maze. At each junction, try every direction. If you hit a wall, step back to the last junction and try another direction. The key is **undoing** your move when you backtrack.

## 🔁 Universal Template

```python
def backtrack(state, choices):
    # 1. Base case: reached valid solution
    if is_complete(state):
        results.append(copy(state))  # ← ALWAYS copy, not reference
        return
    
    for choice in choices:
        # 2. Check if this choice is valid
        if not is_valid(choice, state):
            continue
        
        # 3. Make the choice
        state.add(choice)
        
        # 4. Recurse with remaining choices
        backtrack(state, next_choices)
        
        # 5. UNDO the choice (backtrack)
        state.remove(choice)   # ← This is the crucial step!
```

## ⚠️ Common Mistakes

- Forgetting to **copy** the state when adding to results (shallow copy bug).
- Not **undoing** the choice → leads to corrupted state.
- Not pruning invalid branches → TLE.
- Not handling duplicates → duplicate results.

---

### 🔴 Problem: Subsets

**Statement:** Return all possible subsets of a set (no duplicates in input).

**Example:**
```
Input:  nums = [1,2,3]
Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
```

**🧠 Think:** At each step, we include the current element or we don't. Total subsets = 2ⁿ. Use backtracking: at each index, try including element → recurse → then exclude (backtrack).

```python
# 🏆 OPTIMAL — Backtracking — O(2ⁿ * n) time (copy each subset)
def subsets(nums):
    result = []

    def backtrack(start, current):
        result.append(current[:])    # ← Copy current subset into result

        for i in range(start, len(nums)):
            current.append(nums[i])  # Choose nums[i]
            backtrack(i + 1, current)  # Recurse (no reuse: i+1)
            current.pop()            # Undo choice

    backtrack(0, [])
    return result
```

---

### 🔴 Problem: Permutations

**Statement:** Return all permutations of a list of **distinct** integers.

**Example:**
```
Input:  nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
```

**🧠 Think:** A permutation uses each element exactly once in some order. Build permutation one position at a time. At each position, try every unused element. Track used elements with a boolean array.

```python
# 🏆 OPTIMAL — Backtracking with used array — O(n! * n) time
def permutations(nums):
    result = []
    used = [False] * len(nums)

    def backtrack(current):
        if len(current) == len(nums):
            result.append(current[:])   # Complete permutation
            return
        for i in range(len(nums)):
            if used[i]:
                continue              # Skip already-used elements
            used[i] = True
            current.append(nums[i])
            backtrack(current)
            current.pop()             # Undo
            used[i] = False           # Undo

    backtrack([])
    return result
```

---

### 🔴 Problem: Combination Sum

**Statement:** Find all unique combinations of `candidates` that sum to `target`. Each number can be used unlimited times.

**Example:**
```
Input:  candidates = [2,3,6,7], target = 7
Output: [[2,2,3],[7]]
```

**🧠 Think:** Similar to subsets but we're looking for combinations that sum to target. We can reuse elements (so we pass `i` not `i+1` when recursing). Prune: if remaining < 0, stop.

```python
# 🏆 OPTIMAL — Backtracking with pruning — O(N^(T/M)) where T=target, M=min candidate
def combination_sum(candidates, target):
    candidates.sort()   # Enables pruning
    result = []

    def backtrack(start, current, remaining):
        if remaining == 0:
            result.append(current[:])
            return
        for i in range(start, len(candidates)):
            if candidates[i] > remaining:
                break             # Pruning: sorted, so rest are also too big
            current.append(candidates[i])
            backtrack(i, current, remaining - candidates[i])  # i, not i+1 (reuse!)
            current.pop()

    backtrack(0, [], target)
    return result
```

---

---

# 15. Sorting

## 📌 Key Sorting Algorithms

```
Merge Sort  — Divide & Conquer. Stable. O(n log n) always. Extra O(n) space.
Quick Sort  — Divide & Conquer. In-place. O(n log n) avg, O(n²) worst. Not stable.
Heap Sort   — Uses max-heap. In-place. O(n log n) always. Not stable.
Counting    — Only integers in known range. O(n + k). Extremely fast in practice.
```

---

```python
# MERGE SORT — O(n log n) time, O(n) space
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left  = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:      # ← <= makes it stable
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


# QUICK SORT — O(n log n) avg, O(n²) worst, O(log n) space
def quick_sort(arr, low=0, high=None):
    if high is None: high = len(arr) - 1
    if low >= high: return
    p = partition(arr, low, high)
    quick_sort(arr, low, p - 1)
    quick_sort(arr, p + 1, high)

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i+1], arr[high] = arr[high], arr[i+1]
    return i + 1
```

### 🔴 Problem: Sort Colors (Dutch National Flag)

**Statement:** Sort an array containing only 0s, 1s, and 2s in-place — one pass, O(1) space.

**Example:**
```
Input:  [2,0,2,1,1,0]
Output: [0,0,1,1,2,2]
```

**🧠 Think:** Use 3 pointers — `low`, `mid`, `high`. Everything left of `low` is 0, everything right of `high` is 2, `low` to `mid-1` is 1. `mid` is the cursor.

```python
# 🏆 OPTIMAL — Dutch National Flag — O(n) time, O(1) space
def sort_colors(nums):
    low = mid = 0
    high = len(nums) - 1

    while mid <= high:
        if nums[mid] == 0:
            nums[low], nums[mid] = nums[mid], nums[low]
            low += 1; mid += 1
        elif nums[mid] == 1:
            mid += 1            # 1 is in place, just advance
        else:                   # nums[mid] == 2
            nums[mid], nums[high] = nums[high], nums[mid]
            high -= 1           # Don't advance mid — need to re-examine swapped val
```

---

---

# 16. Trie

## 📌 What is it?

A **prefix tree** where each node represents a character. Paths from root to marked nodes spell out valid words. Ideal for prefix-based operations.

## 🧠 Intuition

Think of a filing system organized alphabetically. "cat", "can", "car" all start with "ca" — they share a branch. Trie exploits this shared structure.

```
       root
        |
        c
        |
        a
       /|\
      t  n  r
          |
          (end)
```

```python
class TrieNode:
    def __init__(self):
        self.children = {}     # char → TrieNode
        self.is_end = False    # True if a word ends here

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):          # O(L)
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True

    def search(self, word):          # O(L) — exact match
        node = self.root
        for ch in word:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return node.is_end

    def starts_with(self, prefix):   # O(L) — prefix match
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return True   # Prefix found (don't need is_end)
```

**When to use Trie:** prefix search, autocomplete, word dictionaries, "words starting with", IP routing.

---

---

# 17. Amazon Pattern Recognition Guide

## 🎯 Read the Problem → Map to Pattern

| Problem Keywords / Signal                          | Pattern                     |
|---------------------------------------------------|-----------------------------|
| "Sorted array", "pair/triplet sum"                | Two Pointers                |
| "Contiguous subarray", "window", "substring"      | Sliding Window              |
| "Subarray sum = k", "count subarrays"             | Prefix Sum + HashMap        |
| "Next greater / smaller element"                  | Monotonic Stack             |
| "Top K", "Kth largest/smallest"                   | Heap (min/max)              |
| "Running median"                                  | Two Heaps                   |
| "Shortest path, unweighted"                       | BFS                         |
| "All paths", "cycle detection"                    | DFS                         |
| "Connected components", "union"                   | Union-Find or BFS/DFS       |
| "Minimum/maximum value", "count ways"             | Dynamic Programming         |
| "All combinations/permutations/subsets"           | Backtracking                |
| "Prefix word", "autocomplete"                     | Trie                        |
| "Sorted + O(log n)"                               | Binary Search               |
| "Minimize maximum / maximize minimum"             | Binary Search on Answer     |
| "Interval overlap / merge"                        | Sort by start, then merge   |
| "Frequency / count / group"                       | HashMap / Counter           |

---

## 🔑 Universal Tricks to Remember

```python
# 1. NEVER overflow mid calculation
mid = left + (right - left) // 2

# 2. BFS level snapshot (don't let queue size change mid-loop)
for _ in range(len(queue)):   # Snapshot the level count!
    node = queue.popleft()

# 3. Prefix sum — query sum[i:j] in O(1)
prefix = [0] * (n + 1)
for i in range(n): prefix[i+1] = prefix[i] + nums[i]
range_sum = prefix[j+1] - prefix[i]   # sum from i to j inclusive

# 4. DFS on 2D grid — 4 directions
for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
    nr, nc = r + dr, c + dc
    if 0 <= nr < rows and 0 <= nc < cols:
        dfs(nr, nc)

# 5. Memoization — use @lru_cache for top-down DP
from functools import lru_cache
@lru_cache(maxsize=None)
def dp(i, remaining):
    ...

# 6. Fast & Slow pointer — find middle of linked list
slow = fast = head
while fast and fast.next:
    slow = slow.next
    fast = fast.next.next
# slow is at middle

# 7. Heap with custom priority (use tuple: (priority, value))
heapq.heappush(heap, (3, "apple"))

# 8. Count character frequency in O(1) space (26 letters)
count = [0] * 26
for c in s: count[ord(c) - ord('a')] += 1

# 9. Check power of 2 without loops
def is_power_of_two(n): return n > 0 and (n & (n-1)) == 0

# 10. Reverse a string / list slice
s[::-1]       # Reversed string
arr[::-1]     # Reversed list (new copy)
arr.reverse() # In-place reverse

# 11. Deque for O(1) operations on both ends
from collections import deque
dq = deque()
dq.append(x)      # right push
dq.appendleft(x)  # left push
dq.pop()          # right pop
dq.popleft()      # left pop

# 12. defaultdict avoids KeyError on missing keys
from collections import defaultdict
graph = defaultdict(list)    # default value is []
count = defaultdict(int)     # default value is 0
```

---

## 📊 Complexity Quick Reference

| Pattern               | Time              | Space      | When to use                        |
|-----------------------|-------------------|------------|-------------------------------------|
| Two Pointers          | O(n)              | O(1)       | Sorted array, pairs, palindrome     |
| Sliding Window        | O(n)              | O(k)       | Subarray/substring constraints      |
| Prefix Sum            | O(n) pre, O(1) query | O(n)   | Range sum queries                   |
| Binary Search         | O(log n)          | O(1)       | Sorted array or monotonic condition |
| BFS                   | O(V+E)            | O(V)       | Shortest path, level order          |
| DFS                   | O(V+E)            | O(V)       | Connectivity, paths, backtracking   |
| Heap (K ops)          | O(n log k)        | O(k)       | Top K, running min/max              |
| Hash Map              | O(n)              | O(n)       | Frequency, lookup, grouping         |
| DP (1D)               | O(n)              | O(n)→O(1)  | Subarray, subsequence problems      |
| DP (2D)               | O(m×n)            | O(m×n)     | Grid, two-string problems           |
| Backtracking          | O(2ⁿ) or O(n!)   | O(n)       | Subsets, permutations, paths        |
| Merge Sort            | O(n log n)        | O(n)       | Stable sort, divide & conquer       |
| Trie                  | O(L) per op       | O(N×L)     | Prefix matching, word search        |

---

## 🏆 Amazon OA Game Plan

### Step 1: Read + Classify (2 min)
- What does input look like? (sorted? graph? string?)
- What are we optimizing? (min? max? count? find?)
- What's n? → decide max acceptable complexity

### Step 2: Think Out Loud
- Start with brute force. State complexity.
- Ask: "Can I avoid redundant work?" → that's your optimization direction
- Mention the pattern you're using: "I'll use a sliding window here"

### Step 3: Code
- Handle edge cases first (empty input, single element, all same)
- Write helper functions if logic is complex
- Use meaningful variable names

### Step 4: Test
- Test with given examples
- Test edge cases: `[]`, `[1]`, `[1,1,1]`, negatives, large values
- Trace through manually if unsure

### 🔥 Amazon-Specific Topics (High Frequency)
1. Arrays — Two Sum, 3Sum, Container with Most Water, Trapping Rain Water
2. Strings — Minimum Window Substring, Group Anagrams, Palindrome
3. Linked Lists — Reverse, Merge K sorted, LRU Cache
4. Trees — Level Order, LCA, Max Path Sum, Serialize/Deserialize
5. Graphs — Number of Islands, Course Schedule, Word Ladder
6. Heaps — Kth Largest, Merge K Lists, Top K Frequent, Median Stream
7. DP — Coin Change, Max Subarray, Knapsack, LCS, Edit Distance
8. Backtracking — Subsets, Permutations, Combination Sum, Word Search

---

> 💡 **The real secret:** Every hard problem is just a combination of simple patterns. Master the ~15 core patterns here and you can decompose almost anything Amazon throws at you.
>
> Practice the mental model first. Code comes second. If you can't explain your approach in plain English, you're not ready to code it yet.
>
> **You've got this. Good luck! 🚀**