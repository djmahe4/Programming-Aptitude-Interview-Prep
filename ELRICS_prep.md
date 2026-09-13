# ELRICS Campus Drive – Core Concepts Cheatsheet
**Junior Software Engineer Technical Preparation**  
*Expanded, detailed, and interview-ready reference for OS, OOP, DBMS, DSA, Web, Networking & System Design*

---

## 1. Object-Oriented Programming (OOP)

### Encapsulation
- **Definition**: Bundling data (attributes) and methods that operate on that data into a single unit (class), while restricting direct external access to the internal state.
- **How**: Access modifiers — `private` / `protected` / `public` (Java/C++), or Python naming conventions (`_protected`, `__private` name-mangling).
- **Example (Python)**:
  ```python
  class BankAccount:
      def __init__(self, balance):
          self.__balance = balance          # private

      def deposit(self, amount):
          if amount > 0:
              self.__balance += amount

      def get_balance(self):
          return self.__balance
  ```
- **Interview point**: Protects invariants, reduces coupling, enables controlled mutation.

### Abstraction
- **Definition**: Hiding complex implementation details and exposing only the essential interface/contract to the client.
- **How**: Abstract classes, interfaces, pure virtual methods.
- **Python**: `abc.ABC` + `@abstractmethod`.
- **Benefit**: Client code depends on the contract, not the concrete implementation → easier to change internals later.

### Inheritance
- **Definition**: “is-a” relationship allowing a class to inherit attributes and methods from a parent class (code reuse + hierarchical modeling).
- **Best practice**: Prefer **composition over inheritance** for most cases to avoid tight coupling and fragile base-class problems. Use inheritance only when true subtype relationship exists (Liskov).
- **Multiple inheritance**: Supported in Python (via C3 MRO); avoided or carefully designed in Java/C#.

### Polymorphism
- **Compile-time (Static)**: Method overloading — same method name, different parameter lists (resolved at compile time). Not natively supported in Python the same way.
- **Runtime (Dynamic)**: Method overriding — subclass provides its own implementation of a parent method (resolved via virtual method table / dynamic dispatch).
- **Python**: Duck typing + overriding. Also supports operator overloading via special methods (`__add__`, `__eq__`, etc.).

### SOLID Principles (Design Quality)
| Principle | Meaning | Quick Check |
|-----------|---------|-------------|
| **S**ingle Responsibility | A class should have only one reason to change | Does this class do more than one job? |
| **O**pen-Closed | Open for extension, closed for modification | Can I add behavior without changing existing code? |
| **L**iskov Substitution | Subtypes must be substitutable for their base types | Can I replace parent with child without breaking clients? |
| **I**nterface Segregation | Many specific interfaces > one fat interface | Do clients depend on methods they don’t use? |
| **D**ependency Inversion | Depend on abstractions, not concretions | Are high-level modules depending on low-level details? |

### Python-Specific OOP Notes
- **Duck typing**: “If it walks like a duck and quacks like a duck…” — no need for explicit interfaces.
- `@abstractmethod` + `ABC` for true abstract base classes.
- Multiple inheritance resolved by **C3 Linearization** (Method Resolution Order – MRO). Check with `ClassName.__mro__` or `help(ClassName)`.
- `__slots__` for memory optimization (prevents `__dict__` creation).

**Project Tie-in**: In Qbrain (AST analyzer) and Py2Rust, you used Visitor pattern (classic OOP) and type systems — these are pure OOP + language-internals engineering.

---

## 2. Operating Systems (High-Yield)

### Processes vs Threads
| Aspect              | Process                          | Thread                              |
|---------------------|----------------------------------|-------------------------------------|
| Address Space       | Independent                      | Shared (same process)               |
| Creation Cost       | High                             | Low                                 |
| Context Switch      | Expensive                        | Cheap                               |
| Communication       | IPC required                     | Shared memory + sync primitives     |
| Failure Isolation   | High                             | Low (one thread crash can kill all) |

- **When to use**:  
  - CPU-bound → multiple processes or careful thread pools.  
  - I/O-bound → threads or async (event loop).  
- **Your projects**: MCP routing server uses async I/O (non-blocking concurrency) — conceptually close to cooperative multitasking / event-driven model.

### CPU Scheduling
- **Algorithms**: FCFS, SJF (optimal average waiting time), Round-Robin (time quantum), Priority, Multilevel Queue / Feedback.
- **Preemptive vs Non-preemptive**: Preemptive can interrupt a running process; non-preemptive runs to completion or voluntary yield.
- **Context switch cost**: Saving/restoring registers, page tables, TLB flush → non-trivial; too frequent switching hurts performance.

### Memory Management
- **Virtual Memory**: Each process sees a large contiguous address space; OS maps virtual → physical via page tables.
- **Paging**: Fixed-size pages. Page table + TLB (Translation Lookaside Buffer) for fast translation.
- **Segmentation**: Variable-size logical segments (code, data, stack).
- **Page Replacement**: FIFO, LRU (approximation via Clock/Second-Chance), Optimal (theoretical), Working-Set.
- **Thrashing**: Excessive paging because working set does not fit in memory → system spends most time swapping.

### Concurrency & Synchronization
- **Race condition**: Outcome depends on interleaving of concurrent accesses.
- **Critical section**: Code that accesses shared resources; must be executed atomically.
- **Primitives**:
  - **Mutex**: Mutual exclusion lock (binary).
  - **Semaphore**: Counting (can allow N concurrent accesses) or binary.
  - **Monitor**: Higher-level construct (condition variables + mutual exclusion).
- **Deadlock – Four Necessary Conditions** (all must hold):
  1. Mutual Exclusion
  2. Hold and Wait
  3. No Preemption
  4. Circular Wait
- **Handling**: Prevention (break one condition), Avoidance (Banker’s algorithm), Detection + Recovery, or just ignore (ostrich approach in many systems).

### Inter-Process Communication (IPC)
- Pipes (anonymous / named), Shared Memory, Message Queues, Sockets, Signals, Memory-mapped files.

### File Systems & Linux Essentials
- **Inodes**: Metadata structure (permissions, size, pointers to data blocks).
- **Journaling**: Write-ahead logging for crash consistency.
- **RAID**: Levels 0 (striping), 1 (mirroring), 5/6 (parity), 10 (stripe + mirror).
- **Linux process states**: Running, Interruptible Sleep, Uninterruptible Sleep, Stopped, Zombie.
- **Key syscalls / tools**: `fork` + `exec`, signals, `/proc` filesystem, `ps`, `top`, `strace`, `lsof`.

**Interview Tip**: Always relate back to your experience — “In the MCP routing server I used asyncio which is cooperative concurrency; understanding OS-level scheduling and context-switch costs helped me reason about latency.”

---

## 3. DBMS & SQL

### ACID Properties
- **Atomicity**: Transaction is all-or-nothing.
- **Consistency**: Database moves from one valid state to another (constraints, triggers, cascades).
- **Isolation**: Concurrent transactions do not interfere (levels matter).
- **Durability**: Once committed, data survives crashes (WAL – Write-Ahead Logging + fsync).

**Isolation Levels** (from weakest to strongest):
1. Read Uncommitted – dirty reads possible
2. Read Committed – no dirty reads
3. Repeatable Read – no non-repeatable reads
4. Serializable – full isolation (no phantoms)

### Indexes
- **B-Tree / B+Tree**: Ordered, excellent for range queries and equality. Most common.
- **Hash Index**: Fast equality, poor for ranges.
- **Trade-offs**: Faster SELECT, slower INSERT/UPDATE/DELETE, extra disk + memory.
- **Advanced**: Covering indexes (index-only scans), composite indexes (order of columns matters — leftmost prefix rule), selectivity (high-cardinality columns better).

### Normalization
- **1NF**: Atomic values, no repeating groups.
- **2NF**: No partial dependency on composite key.
- **3NF**: No transitive dependency.
- **BCNF**: Every determinant is a candidate key.
- **Anomalies**: Insert, Update, Delete anomalies caused by redundancy.
- **When to denormalize**: Read-heavy workloads, analytics, when join cost dominates.

### Transactions & Concurrency Control
- **Locks**: Shared (read) vs Exclusive (write). Two-phase locking (2PL).
- **MVCC** (Multi-Version Concurrency Control): Readers see a snapshot; writers create new versions (PostgreSQL, InnoDB).
- **Optimistic vs Pessimistic**: Optimistic assumes low conflict (check at commit); Pessimistic locks early.

### SQL Mastery
```sql
SELECT columns
FROM table
WHERE conditions
GROUP BY columns
HAVING aggregate_conditions
ORDER BY columns
LIMIT n OFFSET m;
```

- **JOINs**: INNER, LEFT/RIGHT/FULL OUTER, CROSS, SELF.
- **Window Functions** (must-know modern SQL):
  ```sql
  ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC)
  RANK(), DENSE_RANK(), LAG(), LEAD(), SUM() OVER (...)
  ```
- **CTEs** (`WITH ... AS`) preferred over nested subqueries for readability.
- **EXISTS** vs **IN**: EXISTS often better for correlated checks; stops at first match.
- **Transactions**:
  ```sql
  BEGIN;
  -- statements
  COMMIT;   -- or ROLLBACK;
  SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;
  ```

### NoSQL Awareness
| Type          | Examples     | Best For                     | CAP Trade-off          |
|---------------|--------------|------------------------------|------------------------|
| Document      | MongoDB      | Flexible schema, JSON-like   | Usually AP             |
| Key-Value     | Redis        | Caching, sessions, counters  | AP / CP options        |
| Wide-Column   | Cassandra    | Massive write scale          | AP                     |
| Graph         | Neo4j        | Highly connected data        | Usually CP             |

**CAP Theorem**: Consistency, Availability, Partition tolerance — pick any two under network partition.

### Common Pitfalls
- N+1 query problem (ORM classic).
- Missing or wrong indexes → full table scans.
- Low selectivity indexes.
- Ignoring the query planner (`EXPLAIN ANALYZE`).

---

## 4. Data Structures & Algorithms (Coding Round Essentials)

### Core Structures & When to Use
- **Arrays / Strings**: Two pointers, sliding window, prefix sums / difference arrays.
- **Hash Map / Set**: O(1) average lookup → frequency counting, two-sum family, uniqueness.
- **Stack**: Matching parentheses, monotonic stack, DFS (explicit), expression evaluation.
- **Queue / Deque**: BFS, sliding window maximum (monotonic deque).
- **Heap (Priority Queue)**: Top-K, Dijkstra, merge K sorted lists, median maintenance.
- **Trees**: BST operations, Traversals (in/pre/post/level), LCA, diameter, balanced trees concepts.
- **Graphs**: Adjacency list, BFS/DFS, Topological sort (Kahn / DFS), Dijkstra, Union-Find (DSU) for connected components / cycle detection.

### Complexity Mindset
- Always state **Time** and **Auxiliary Space**.
- Discuss trade-offs: “Hash map gives O(N) time / O(N) space; sorting + two pointers gives O(N log N) time / O(1) space.”

### High-Frequency Patterns
- Two Sum / Pair with given sum variants
- Sliding Window (fixed & variable size)
- Intervals (merge, insert, meeting rooms)
- Prefix Sum + Hash Map (subarray sum equals K)
- Binary Search on answer / on sorted space
- Backtracking (permutations, subsets, N-Queens style)
- Dynamic Programming (1D / 2D, knapsack family, LIS, edit distance)

---

## 5. REST APIs & Web Architecture

### HTTP Methods & Idempotency
- **Idempotent**: GET, PUT, DELETE, HEAD, OPTIONS (multiple identical calls → same state).
- **Non-idempotent**: POST, PATCH (usually).

### Status Codes (Memorize these)
- **2xx**: 200 OK, 201 Created, 204 No Content
- **4xx**: 400 Bad Request, 401 Unauthorized, 403 Forbidden, 404 Not Found, 409 Conflict, 429 Too Many Requests
- **5xx**: 500 Internal Server Error, 502 Bad Gateway, 503 Service Unavailable

### Authentication & Authorization
- **Session Cookies**: Server-side state, sticky sessions or shared session store.
- **JWT**: Stateless (Header.Payload.Signature). Pros: scalable. Cons: revocation harder, size, secret management.
- **OAuth 2.0 / OIDC**: Authorization framework (not pure authentication).
- **API Keys**: Simple but less granular.

### Other Concepts
- Statelessness (REST constraint).
- Versioning strategies (URI, header, content negotiation).
- Caching: `Cache-Control`, `ETag`, `Last-Modified`, CDN, Redis as application cache.
- HATEOAS (optional depth — hypermedia as the engine of application state).

---

## 6. Networking Basics

- **OSI Model** (7 layers) vs **TCP/IP** (4 layers) — know mapping.
- **TCP vs UDP**:
  - TCP: reliable, ordered, connection-oriented, congestion control.
  - UDP: fire-and-forget, lower latency, no guarantees (DNS, video, gaming).
- **HTTP Evolution**:
  - HTTP/1.1: persistent connections, pipelining issues.
  - HTTP/2: multiplexing, header compression (HPACK), server push.
  - HTTP/3: QUIC (UDP-based), faster connection establishment, better loss recovery.
- **DNS**: Hierarchical, recursive vs iterative resolution, records (A, AAAA, CNAME, MX, TXT).
- **TLS Handshake** (high-level): ClientHello → ServerHello + Certificate → Key exchange → Encrypted communication.
- **CORS**: Browser security mechanism controlling cross-origin requests.

---

## 7. System Design Light (Round 2 / 3 Ready)

### Scalability Building Blocks
- **Vertical** vs **Horizontal** scaling.
- **Load Balancing**: Round-robin, least connections, consistent hashing; Layer 4 vs Layer 7.
- **Caching**: Cache-aside, read-through, write-through, write-behind. Cache invalidation is hard.
- **Database**: Read replicas, sharding (key-based, range, directory), replication (sync/async).
- **Message Queues**: Decoupling, buffering, async processing (Kafka, RabbitMQ, SQS).
- **Consistency Models**: Strong, eventual, causal. CAP theorem in practice.

### Project-Specific Scaling Thought
**MCP Routing Server**:
- Async workers / process pool for concurrent provider calls.
- Circuit breaker + timeout + retry with exponential backoff.
- Rate limiting (token bucket / sliding window) per provider.
- Latency + token-budget aware routing (already implemented conceptually).
- Horizontal scale behind a load balancer; shared Redis for rate-limit state if needed.

---

## Quick Revision Checklist (Night Before)

- [ ] Can explain Encapsulation vs Abstraction in one clear sentence each.
- [ ] Can list the 4 deadlock conditions and one way to break each.
- [ ] Can write a correct window function query and explain PARTITION BY vs GROUP BY.
- [ ] Can discuss B-Tree vs Hash index trade-offs.
- [ ] Can state time/space for the main coding patterns (two pointers, sliding window, hash map).
- [ ] Can defend why you chose a particular approach in coding round using Big-O.
- [ ] Can relate at least one OS or concurrency concept back to the MCP server or AST analyzer.
- [ ] Elevator pitch + project defenses ready.

---
*Document generated for ELRICS Campus Drive preparation – Core Concepts Expanded Cheatsheet*  
*Last updated: September 2026*
