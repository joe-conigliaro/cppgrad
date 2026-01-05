// Copyright (c) 2026 Joe Conigliaro
// https://github.com/joe-conigliaro
#pragma once

#include <mutex>
#include <queue>
#include <atomic>
#include <thread>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <functional>
#include <condition_variable>

namespace cppgrad {
namespace backend {
namespace cpu {

// Simple thread pool (single shared queue).
class ThreadPool {
public:
    ThreadPool() { resize(default_threads()); }
    ~ThreadPool() { shutdown(); }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    static unsigned default_threads() {
        unsigned hw = std::thread::hardware_concurrency();
        return hw ? hw : 4u;
    }

    unsigned size() const noexcept {
        return nworkers_.load(std::memory_order_acquire);
    }

    // Stop current workers (join) and start nthreads new workers.
    void resize(unsigned nthreads) {
        nthreads = std::max(1u, nthreads);
        shutdown();

        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = false;
        }

        workers_.reserve(nthreads);
        for (unsigned i = 0; i < nthreads; ++i) {
            workers_.emplace_back([this] { worker_loop(); });
        }
        nworkers_.store(static_cast<unsigned>(workers_.size()), std::memory_order_release);
    }

    // Enqueue; throws if pool is stopping/stopped.
    void enqueue(std::function<void()> f) {
        {
            std::lock_guard<std::mutex> lk(m_);
            if (stop_) throw std::runtime_error("ThreadPool: enqueue() on stopped pool");
            q_.push(std::move(f));
        }
        cv_.notify_one();
    }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
        }
        cv_.notify_all();

        for (auto& t : workers_) {
            if (t.joinable()) t.join();
        }
        workers_.clear();

        // Drain queue under lock (prevents races if someone enqueues concurrently).
        {
            std::lock_guard<std::mutex> lk(m_);
            std::queue<std::function<void()>> empty;
            q_.swap(empty);
        }
        nworkers_.store(0u, std::memory_order_release);
    }

private:
    void worker_loop() {
        for (;;) {
            std::function<void()> task;

            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [&] { return stop_ || !q_.empty(); });

                if (stop_ && q_.empty()) return;

                task = std::move(q_.front());
                q_.pop();
            }

            // Tasks are assumed not to throw (exceptions are caught inside).
            task();
        }
    }

    std::mutex m_;
    std::condition_variable cv_;
    std::queue<std::function<void()>> q_;
    std::vector<std::thread> workers_;
    // we coud use `workers_.size()` with lock, or without lock (possible race)
    // but keeping atomic nworkers_ avoids using a lock or a race without one.
    std::atomic<unsigned> nworkers_{0};
    bool stop_ = false;
};

// Scoped task group.
class TaskGroup {
public:
    explicit TaskGroup(ThreadPool& pool) : pool_(pool) {}

    template<class Fn>
    void run(Fn&& fn) {
        {
            std::lock_guard<std::mutex> lk(m_);
            ++pending_;
        }

        pool_.enqueue([this, f = std::forward<Fn>(fn)]() mutable {
            try {
                f();
            } catch (...) {
                std::lock_guard<std::mutex> elk(m_);
                if (!ex_) ex_ = std::current_exception();
            }

            std::lock_guard<std::mutex> lk(m_);
            if (--pending_ == 0) {
                cv_.notify_one();
            }
        });
    }

    void wait() {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&]{ return pending_ == 0; });
        if (ex_) std::rethrow_exception(ex_);
    }

private:
    ThreadPool& pool_;
    std::mutex m_;
    std::condition_variable cv_;
    // TODO: revisit using atomic to avoid mutex.
    std::size_t pending_ = 0;
    std::exception_ptr ex_;
};


// Global runtime config.
struct Runtime {
    ThreadPool pool;
    std::atomic<std::size_t> grain{256};

    static Runtime& instance() {
        static Runtime rt;
        return rt;
        // Note: we can intentionally never destruct to avoid shutdown-order issues
        // with other static objects/destructors that might call into the runtime.
        // static Runtime* rt = new Runtime(); // intentionally leaked
        // return *rt;
    }

    void set_num_threads(unsigned n) {
        n = std::max(1u, n);
        pool.resize(n);
    }

    void set_grain(std::size_t g) {
        grain.store(std::max<std::size_t>(1, g), std::memory_order_release);
    }
};

// Parallel for over [begin, end). Functor signature: fn(size_t b, size_t e).
template <typename Fn>
inline void parallel_for(std::size_t begin, std::size_t end, Fn fn) {
    auto& rt = Runtime::instance();
    const std::size_t n = (end > begin) ? (end - begin) : 0;
    if (n == 0) return;

    const unsigned nt = std::max(1u, rt.pool.size());
    const std::size_t g = rt.grain.load(std::memory_order_acquire);

    if (n < g || nt == 1) {
        fn(begin, end);
        return;
    }

    const std::size_t chunk = std::max(g, (n + nt - 1) / nt);

    TaskGroup tg(rt.pool);
    for (std::size_t s = begin; s < end; s += chunk) {
        const std::size_t e = std::min(end, s + chunk);
        tg.run([=] { fn(s, e); });
    }
    tg.wait();
}

// Parallel reduce over [0, n). eval(i)->T, combine(T,T)->T.
template <typename T, typename EvalFn, typename CombineFn>
inline T parallel_reduce(std::size_t n, EvalFn eval, CombineFn combine, T init) {
    auto& rt = Runtime::instance();
    if (n == 0) return init;

    const unsigned nt = std::max(1u, rt.pool.size());
    const std::size_t g = rt.grain.load(std::memory_order_acquire);

    if (n < g || nt == 1) {
        T acc = init;
        for (std::size_t i = 0; i < n; ++i) acc = combine(acc, eval(i));
        return acc;
    }

    const std::size_t chunk = std::max(g, (n + nt - 1) / nt);
    const std::size_t num_tasks = (n + chunk - 1) / chunk;

    std::vector<T> partials(num_tasks, init);

    TaskGroup tg(rt.pool);
    for (std::size_t task_id = 0, start = 0; start < n; start += chunk, ++task_id) {
        const std::size_t end = std::min(n, start + chunk);
        tg.run([&, start, end, task_id] {
            T acc = init;
            for (std::size_t i = start; i < end; ++i) acc = combine(acc, eval(i));
            partials[task_id] = acc;
        });
    }
    tg.wait();

    T out = init;
    for (const T& p : partials) out = combine(out, p);
    return out;
}

} // namespace cpu
} // namespace backend
} // namespace cppgrad
